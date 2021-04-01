import torch
import numpy as np
from scipy.stats import entropy, chisquare


### Calculating score
def get_probs(data, bins):
    """
    This defines the binning procedure
    :param encoding: the tensor of vectors to be binned
    :return: a vector of counts for each bin, ordered the same way every time it is called
    """
    x, y = np.hsplit(data.detach().cpu().numpy(), 2)
    if isinstance(bins, int):
        counts = np.histogram2d(x[:, -1], y[:, -1], bins=bins, range=[[-4, 4], [-4, 4]], density=False)
    else:
        counts = np.histogram2d(x[:, -1], y[:, -1], bins=[bins, bins], density=False)
    return np.ndarray.flatten(counts[0])


def get_kl(base_dist, target_dist, model, n_calc, n_test, device, type='squares', glen=100, g_chi=0):
    # Define binning
    if type == 'squares':
        bins = glen
        nbins = bins ** 2
    elif type == 'equal_prob':
        # These bins only make sense if the target distribution is a normal dist, and even then it is debatable how useful this is.
        # TODO: reimplement this definition of the bins
        pre_sqrt = -2 * np.log((2 * np.pi) ** 0.5 * np.linspace(0.0001, 0.99, 101))
        pre_sqrt = pre_sqrt[pre_sqrt >= 0]
        bins = pre_sqrt ** 0.5
        bins = bins[~np.isnan(bins)]
        bins = np.concatenate((-bins, np.zeros(1), np.flip(bins)), 0)
        nbins = (len(bins) - 1) ** 2
    # Get KL estimate
    q = np.zeros(nbins)
    p = np.zeros(nbins)
    with torch.no_grad():
        for i in range(n_calc):
            data = base_dist.sample([n_test]).to(device)
            target_sample = target_dist.sample([n_test]).to(device)
            encoding = model(data)
            q += get_probs(encoding, bins)
            p += get_probs(target_sample,
                           bins)  # TODO: you should be able to get this exactly without sampling, it's only a function of the bin size and the distribution type
    mx = (np.array(q) > 0) & (np.array(p) > 0)
    if g_chi:
        chi2, p_val = chisquare(q[mx], p[mx], ddof=2)
    norm_factor = n_calc * n_test
    q = q / norm_factor
    p = p / norm_factor
    kl_div = entropy(p[mx], qk=q[mx])
    if g_chi:
        return kl_div, chi2, p_val
    else:
        return kl_div


def get_kl_and_error(base_dist, target_dist, model, n_calc, nrun, n_test, device, g_chi=0):
    store = []
    chi = []
    pvals = []
    for i in range(nrun):
        vals = get_kl(base_dist, target_dist, model, n_calc, n_test, device, g_chi=g_chi)
        if g_chi:
            store += [vals[0]]
            chi += [vals[1]]
            pvals += [vals[2]]
        else:
            store += [vals]
    kl_means = np.mean(store)
    kl_var = np.std(store)
    if g_chi:
        chi_mean = np.mean(chi)
        p_mean = np.mean(pvals)
        return kl_means, kl_var, chi_mean, p_mean
    else:
        return kl_means, kl_var


def _test():
    import matplotlib.pyplot as plt
    from utils.io import get_top_dir
    import hyperparams
    import torch.distributions as torch_distributions

    # bdist = 'normal'
    # tdist = 'normal'
    # base_dist = hyperparams.torch_dists(bdist, 2)
    # target_dist = hyperparams.torch_dists(tdist, 2)

    base_dist = torch_distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
    mean = 0.5
    var = 1
    target_dist = torch_distributions.MultivariateNormal(torch.zeros(2) + mean, torch.eye(2) * var)

    var_vector = np.ones(2) * var
    mean_vector = np.zeros(2) + mean
    true_kl = 0.5 * np.sum(mean_vector ** 2 + var_vector - 1 - np.log(var_vector))

    def id_fun(data):
        return data

    ### Get the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    n_calc_min = 6
    n_calc_max = 20
    n_test = int(1e5)

    nrun = 10

    diff_means = []
    diff_vars = []
    for n_calc in range(n_calc_min, n_calc_max):
        kl_means, kl_var = get_kl_and_error(base_dist, target_dist, id_fun, n_calc, nrun, n_test, device)
        diff_means += [kl_means]
        diff_vars += [kl_var]
    print(diff_means)
    x_axis = np.arange(n_calc_min, n_calc_max) * n_test
    plt.errorbar(x_axis, diff_means, yerr=diff_vars, label='estimates')
    plt.plot(x_axis, [true_kl] * len(diff_means), label='true')
    plt.xlabel('# points sampled')
    plt.legend
    plt.ylabel('KL divergence')
    plt.savefig(get_top_dir() + '/images/test_MC.png')

    return 0


if __name__ == '__main__':
    _test()
