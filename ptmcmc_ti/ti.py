import numpy as np
from jax import numpy as jnp


def find_mu(Dk):
    return jnp.mean(Dk)


def find_Rk(mu, Dk):
    return jnp.array(mu - Dk)


def find_chi_sq(Rk):
    return jnp.sum(Rk**2)


def find_phi(Rk):
    return Rk[0]**2 + Rk[-1]**2


def find_psi(Rk):
    return jnp.sum(Rk[:-1] * Rk[1:])


def find_Q(mu, epsilon, Dk):
    Rk = find_Rk(mu, Dk)
    chi_sq = find_chi_sq(Rk)
    phi = find_phi(Rk)
    psi = find_psi(Rk)
    return chi_sq + epsilon**2 * (chi_sq - phi) - 2 * epsilon * psi


def find_mu0(epsilon, Dk):
    N = len(Dk)
    return (-(epsilon**2*Dk[1]) - epsilon**2*Dk[-1] + np.sum(Dk) + epsilon**2*np.sum(Dk) - 
            epsilon*np.sum(Dk[:-1] + Dk[1:]))/((-1 + epsilon)*(-2*epsilon - N + epsilon*N))


def find_epsilon(Dk):
    N = len(Dk)
    epsilon = 1 / (N - 2)
    return epsilon


def find_sigma(Dk):
    N = len(Dk)
    epsilon = find_epsilon(Dk)
    mu0 = find_mu0(epsilon, Dk)
    Q0 = find_Q(mu0, epsilon, Dk)
    return np.sqrt(Q0)/np.sqrt(-((-1 + epsilon**2)*(-1 + N)))



def calc_evidence_ti(chain_dir, burn_pct=0.25, verbose=True):
    """
    Compute the ln(evidence) with thermodynamic integration.
    
    For error estimates, we sort of follow https://arxiv.org/pdf/1410.3835.pdf,
    but we don't use the spline/RJMCMC approach here. So take the error estimates
    as an order of magnitude estimate.

    Input:
        chain_dir (string): folder location where the chains are stored
        burn_pct (float): percent of the start of the chain to remove
        verbose (bool): get more info

    Returns:
        ln_Z (float): natural logarithm of the evidence
        inv_temp (array): 1 / temperature of the chains
        mean_like (array): means of the ln(likelihood)
    """
    files = sorted(glob.glob(chain_dir + 'chain*'))
    beta_list = []
    mean_list = []
    var_list = []
    for fname in files:
        # get temperature from file name
        temp = fname.split('/')[-1].split('_')[-1].split('.')[:-1]
        separator = '.'
        temp = separator.join(temp)
        temp = float(temp)
        if verbose:
            print('Working on temperature', temp)
        with open(fname, 'r') as f:
            chain = np.loadtxt(f, usecols=[-3])  # only get likelihood
        burn = int(burn_pct * len(chain)) # aggressive burn in

        epsilon = find_epsilon(chain)  # measure of correlation
        mean = find_mu0(epsilon, chain)
        sigma = find_sigma(chain)  # std
        error = sigma * np.sqrt((1 + epsilon) / (len(chain)*(1 - epsilon) + 2 * epsilon))

        beta = 1 / temp  # integration variable
        mean_list.append(mean)
        beta_list.append(beta)
        var_list.append(error**2)
    # build numpy array
    betas = np.array(beta_list)
    means = np.array(mean_list)
    variances = np.array(var_list)
    # sort the data from least to greatest
    data = np.column_stack([betas, means, variances])
    data = data[data[:, 0].argsort()]
    inv_temp = data[:, 0]
    mean_like = data[:, 1]
    var = data[:, 2]
    dx = np.diff(inv_temp)
    dx = np.insert(dx, 0, 0)

    # error est. for statistical error:
    # total = 0
    # for i in range(1, len(inv_temp)):
    #     total += (1 / 4 * (var[i]**2 * dx[i]**2 +
    #               2 * var[i - 1]**2 * dx[i] * dx[i - 1] +
    #               var[i - 1]**2 * dx[i]**2))

    # error est. for statistical error:
    total = 0
    for i in range(len(inv_temp)):
        total += dx[i]**2 / 4 * (var[i]**2 + var[i - 1]**2 + 2 * var[i] * var[i - 1])

    stat_unc = np.sqrt(total)

    log_Z = np.trapz(mean_like, inv_temp)

    # discretization error estimate:
    dy = np.diff(mean_like)
    dx = np.diff(inv_temp)
    deriv = dy / dx

    d2y = np.diff(deriv)
    dx2 = np.diff(dx)
    deriv2 = d2y / dx2
    N = len(inv_temp)
    disc_unc = max(abs(-(deriv2 * (dx[:-1])**3 * N) / (12)))

    if verbose:
        print('model:')
        print('ln(evidence) =', log_Z)
        print('statistical uncertainty =', stat_unc)
        print('discretization uncertainty =', disc_unc)
        print()
    return log_Z, stat_unc, disc_unc, inv_temp, mean_like


def bayes_factor_ti(model1_dir, model2_dir, burn_pct=0.25, verbose=True):
    """
    Compute Bayes factor for two different models using thermodynamic integration.

    30-60 chains minimum are recommended with Tmax >= 1e5 to 1e7. Don't forget to
    set writeHotChains=True.

    Input:
        model1_dir (string): directory path to folder containing chains for model 1
        model2_dir (string): directory path to folder containing chains for model 2
        burn_pct (float): percent of the start of the chain to remove
        verbose (bool): get more info

    Returns:
        ln_bayes (float): natural log of the bayes factor
        stat_unc (float): statistical uncertainty
        disc_unc (float): discretization uncertainty
    """
    ln_ev_1, s_unc1, disc_unc1, temps1, means1 = calc_evidence_ti(model1_dir, burn_pct=burn_pct, verbose=True)
    ln_ev_2a, s_unc2, disc_unc2, temps2, means2 = calc_evidence_ti(model2_dir, burn_pct=burn_pct, verbose=True)

    ln_bayes = ln_ev_2a - ln_ev_1
    stat_unc = np.sqrt(s_unc1**2 + s_unc2**2)
    disc_unc = np.sqrt(disc_unc1**2 + disc_unc2**2)

    if verbose:
        print('ln(BF) =', ln_bayes, '+/-', stat_unc, '+/-', disc_unc)

    return ln_bayes, stat_unc, disc_unc