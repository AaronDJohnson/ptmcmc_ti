import numpy as np
from scipy.interpolate import UnivariateSpline

import glob
import acor.acor


def find_means(chain_dir, burn_pct=0.25, verbose=True):
    """
    Take mean of log likelihood for several temperatures and inverse temperatures.

    Input:
        chain_dir (string): folder location where the chains are stored
        burn_pct (float) [0.25]: percent of the start of the chain to remove
        verbose (bool) [True]: get more info

    Return:
        inv_temp (array): 1 / temperature of the chains
        mean_like (array): means of the ln(likelihood)
        stat_unc (float): uncertainty associated with the MCMC chain
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
        tau, mean, sigma = acor.acor(chain[burn:])

        beta = 1 / temp  # integration variable
        mean_list.append(mean)
        beta_list.append(beta)
        var_list.append(sigma**2)
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
    # with a lot of samples we expect this to be small compared to the other sources of error
    total = 0
    for i in range(len(inv_temp)):
        total += dx[i]**2 / 4 * (var[i]**2 + var[i - 1]**2 + 2 * var[i] * var[i - 1])

    stat_unc = np.sqrt(total)

    return inv_temp, mean_like, stat_unc


def calc_evidence_ti(model_dir, burn_pct=0.25, verbose=True):
    """
    Compute ln(evidence) of chains of several different temperatures.

    Input:
        model_dir1 (string): folder location where the chains are stored for first model
        model_dir2 (string): folder location where the chains are stored for second model
        burn_pct (float) [0.25]: percent of the start of the chain to remove
        verbose (bool) [True]: get more info

    Return:
        ln_Z (float): natural logarithm of the evidence
        spline_up (float): max(spline - data) -- absolute error between spline and data
        spline_dn (float): min(spline - data) -- absolute error between spline and data
        stat_unc (float): uncertainty associated with the MCMC chain
        disc_unc (float): uncertainty associated with discretization of the integral
    """
    inv_temp, mean_like, stat_unc = find_means(model_dir, burn_pct=burn_pct, verbose=verbose)

    # use splines to get discretization error:
    y_spl = UnivariateSpline(inv_temp, inv_temp * mean_like)
    y_spl_2d = y_spl.derivative(n=2)

    # max error in cubic spline:
    err_func = y_spl(inv_temp) - inv_temp * mean_like
    spline_up = abs(max(err_func))
    spline_dn = abs(min(err_func))
    
    x_new = np.linspace(inv_temp[0], 1, num=10000)
    ln_Z = np.trapz(y_spl(x_new), np.log(x_new))

    # discretization error estimate:
    N = len(x_new)
    a = x_new[0]
    b = 1
    disc_unc = max(abs(-(b - a) / (12 * N**2) * y_spl_2d(x_new)))

    if verbose:
        print()
        print('model:')
        print('ln(evidence) =', ln_Z)
        print('statistical uncertainty =', stat_unc)
        print('spline uncertainty = +', spline_up, '/-', spline_dn)
        print('discretization uncertainty =', disc_unc)
        print()
    return ln_Z, spline_up, spline_dn, stat_unc, disc_unc


def calc_bf_ti(model_dir1, model_dir2, burn_pct=0.25, verbose=True):
    """
    Compute ln(evidence) of chains of several different temperatures.

    Input:
        model_dir1 (string): folder location where the chains are stored for first model
        model_dir2 (string): folder location where the chains are stored for second model
        burn_pct (float) [0.25]: percent of the start of the chain to remove
        verbose (bool) [True]: get more info

    Return:
        ln_Z (float): natural logarithm of the evidence
        spline_up (float): max(spline - data) -- absolute error between spline and data
        spline_dn (float): min(spline - data) -- absolute error between spline and data
        stat_unc (float): uncertainty associated with the MCMC chain
        disc_unc (float): uncertainty associated with discretization of the integral
    """
    inv_temp1, mean_like1, stat_unc1 = find_means(model_dir1, burn_pct=burn_pct, verbose=verbose)
    inv_temp2, mean_like2, stat_unc2 = find_means(model_dir2, burn_pct=burn_pct, verbose=verbose)

    if not np.array_equal(inv_temp1, inv_temp2):
        print('Temperatures are mismatched somewhere!')
    else:
        inv_temp = inv_temp1
    
    mean_diff = mean_like2 - mean_like1  # difference between models
    # combine statistical uncertainties:
    stat_unc = np.sqrt(stat_unc1**2 + stat_unc2**2)

    # use splines to get discretization error:
    y_spl = UnivariateSpline(inv_temp, inv_temp * mean_diff)
    y_spl_2d = y_spl.derivative(n=2)

    # error in cubic spline:
    err_func = y_spl(inv_temp) - inv_temp * mean_diff
    spline_up = abs(max(err_func))
    spline_dn = abs(min(err_func))
    # print('err_up =', spline_up)
    # print('err_dn =', spline_dn)
    
    x_new = np.linspace(inv_temp[0], 1, num=10000)
    ln_Z = np.trapz(y_spl(x_new), np.log(x_new))

    # discretization error estimate:
    N = len(x_new)
    a = x_new[0]
    b = 1
    disc_unc = max(abs(-(b - a) / (12 * N**2) * y_spl_2d(x_new)))

    if verbose:
        print()
        print('model:')
        print('ln(evidence) =', ln_Z)
        print('statistical uncertainty =', stat_unc)
        print('spline uncertainty = +', spline_up, '/-', spline_dn)
        print('discretization uncertainty =', disc_unc)
        print()
    return ln_Z, spline_up, spline_dn, stat_unc, disc_unc
