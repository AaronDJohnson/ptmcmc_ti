import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

import glob, os
# import acor.acor
import matplotlib.pyplot as plt


def load_chain(chainfile):
    param_file = os.path.dirname(chainfile) + '/pars.txt'
    with open(param_file, 'r') as f:
        params = [line.rstrip('\n') for line in f]
    params += ['lnpost', 'lnlike']
    idx = list(params).index('lnlike')
    temp = '.'.join(chainfile.split('_')[-1].split('.')[:-1])
    if temp == 'hot':
        temp = 1e80

    # import data
    chain_raw = pd.read_csv(chainfile, sep='\t', dtype=float, header=None, usecols=[idx]).values
    chain_df = pd.DataFrame(data=chain_raw, columns=['lnlike'])
    return temp, chain_df['lnlike']


def make_chainlist(chainfolder):
    """
    Note: removes parts of the chains if all chains aren't the same size
    """
    max_nan = 0
    chains = pd.DataFrame()
    chainfiles = glob.glob(chainfolder + 'chain_*.txt')
    for chainfile in chainfiles:
        print(chainfile)
        temp, lnlike = load_chain(chainfile)
        chains[str(temp)] = lnlike
    for key in chains:
        nans = chains[key].isna().sum()
        max_nan = max(nans, max_nan)
        # print(max_nan)
    chains.drop(chains.tail(max_nan).index, inplace=True)
    # print(chains.columns.values)
    floats = chains.columns.values.astype(np.float)
    strings = chains.columns.values
    an_array = np.column_stack([floats, strings])
    sorted_array = an_array[np.argsort(an_array[:, 0])]
    chains = chains[sorted_array[:, 1]]
    return chains


def chainfolder_to_txt(chainfolder, out_file):
    """
    Save files temps in chainfolder to a text file that contains temperature
    on the first line and ln likelihood associated with each temperature
    in the columns.

    Input:
        chainfolder (string): folder containing parallel tempered chains
        out_file (string): filepath to the output
    """
    chains = make_chainlist(chainfolder)
    chains.to_csv(out_file, sep=' ', index=False, header=True)


def find_means(txt_loc, burn_pct=0.25, remove_hot=False):
    """
    Take mean of log likelihood for several temperatures and inverse temperatures.

    Input:
        txt_loc (string): location where .txt file is stored
        burn_pct (float) [0.25]: percent of the start of the chain to remove
        verbose (bool) [True]: get more info

    Return:
        inv_temp (array): 1 / temperature of the chains
        mean_like (array): means of the ln(likelihood)
        stat_unc (float): uncertainty associated with the MCMC chain
    """

    with open(txt_loc, 'r') as f:
        temps = np.loadtxt(f, max_rows=1)
        like = np.loadtxt(f, skiprows=1)

    # remove burn in phase
    burn = int(burn_pct * like.shape[1])
    like = like[burn:]

    # build numpy array
    inv_temps = 1 / temps[::-1]
    mean_like = np.average(like, axis=0)[::-1]
    std = np.std(like, axis=0)[::-1]

    if remove_hot:
        inv_temps = inv_temps[1:]
        mean_like = mean_like[1:]

    return inv_temps, mean_like, std


def ti_log_evidence(txt_loc, burn_pct=0.25, verbose=True, iterations=1000,
                    remove_hot=False, plot=False):
    """
    Compute ln(evidence) of chains of several different temperatures.

    Input:
        txt_loc (string): location of text file
        burn_pct (float) [0.25]: percent of the start of the chain to remove
        verbose (bool) [True]: get more info
        iters (int) [1000]: number of iterations to use to get error estimate
        remove_hot (bool) [False]: if a hot chain exists (T=1e80), remove it

    Return:
        ln_Z (float): natural logarithm of the evidence
        spline_up (float): max(spline - data) -- absolute error between spline and data
        spline_dn (float): min(spline - data) -- absolute error between spline and data
        stat_unc (float): uncertainty associated with the MCMC chain
        disc_unc (float): uncertainty associated with discretization of the integral
    """
    inv_temps, mean_like, std = find_means(txt_loc, burn_pct=burn_pct, remove_hot=remove_hot)

    ln_Z_arr = np.zeros(iterations)

    new_means = np.zeros((iterations, len(inv_temps)))
    x_new = np.linspace(inv_temps[0], 1, num=10000)
    for i in range(len(inv_temps)):
        mu = mean_like[i]
        sigma = std[i]
        print(sigma)
        new_means[:, i] = np.random.default_rng().normal(mu, sigma, iterations)
    for i in range(iterations):
        y_spl = UnivariateSpline(inv_temps, inv_temps * new_means[i, :])
        ln_Z = np.trapz(y_spl(x_new), np.log(x_new))
        ln_Z_arr[i] = ln_Z
    ln_Z = np.mean(ln_Z_arr)
    total_err = np.std(ln_Z_arr)

    if plot:
        plt.figure(figsize=(12, 5))
        plt.plot(np.log10(inv_temps), mean_like, 'o-')
        plt.fill_between(np.log10(inv_temps), new_means.min(axis=0), new_means.max(axis=0), color='gray', alpha=0.5)
    # use splines to get discretization error + statistical error:
    # we will do this by assuming a normal distribution around each point
    # and generating a thousand realizations using the uncertainties above
    # Then we generate the splines and get the uncertainty in the integral.

    if verbose:
        print()
        print('model:')
        print('ln(evidence) =', ln_Z)
        print('error in ln_Z =', total_err)
        print()
    return ln_Z, total_err


def calc_log_bf_ti(model_dir1, model_dir2, burn_pct=0.25, verbose=True):
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
