import matplotlib.pyplot as plt
import numpy as np
import torch

from data.hyper_plane import HyperCheckerboardDataset

from utils.io import get_top_dir
from utils.plotting import plot2Dhist

import os


# TODO: the argument you have called vae should be a stochastic argument, so you can see encoded means etc.
def post_process_plane(model, test_data, invertible=False, implicit=True, sup_title='', bounds=[-1.5, 1.5], bins=50,
                       vae=False):
    test_data = test_data.data.to(model.device)
    model.eval()
    nm = model.exp_name
    top_dir = get_top_dir()
    sv_dir = top_dir + '/images' + '/' + model.dir
    if not os.path.exists(sv_dir):
        os.makedirs(sv_dir)

    with torch.no_grad():
        # Get all outputs from forward pass of the model
        outputs = model(test_data.data)
        # Use if statements to calculate the number of plots that will be created TODO: this is quite a silly way to do this
        if isinstance(outputs, tuple):
            nplots = sum([output.shape[1] <= 2 for output in outputs])
            # if outputs[0].shape[1] <= 2:
            #     nplots = len(outputs)
        else:
            nplots = 1
        # There is one plot if it is and if it isn't invertible
        nplots += 1
        if implicit:
            nplots += 1

        fig, axs_ = plt.subplots(1, nplots, figsize=(5 * nplots + 2, 5))
        axs = fig.axes

        # Plot samples drawn from the model
        samples = model.get_numpy(model.sample(int(1e5)))
        plot2Dhist(samples, axs[0], bins, bounds)
        axs[0].set_title('Samples')
        # Create an index for creating the other plots
        ind = 1

        # If the inner model is not invertible show the outer encoder performance
        if not invertible:
            # recons = model.get_numpy(model.autoencode(test_data.to(device)))
            encoding = model.encoder(test_data)
            # If this is a VAE there are two outputs from the encoder
            if vae:
                encoding = model.model_sample(encoding)
            recons = model.get_numpy(model.decoder(encoding))
            plot2Dhist(recons, axs[ind], bins, bounds)
            axs[ind].set_title('Reconstruction')
            ind += 1

        if invertible:
            encoding = model.get_numpy(model.encode(test_data))
            plot2Dhist(encoding, axs[ind], bins)
            axs[ind].set_title('Encoding')
            ind += 1

        # Visualize the different latent spaces
        if isinstance(outputs, tuple):
            latent_names = ['A', 'Z', "A'"]
            for i, l_spc in enumerate(outputs[:-1]):
                l_spc = model.get_numpy(l_spc)
                if l_spc.shape[1] == 2:
                    plot2Dhist(l_spc, axs[ind], bins)
                    axs[ind].set_title('Latent Space {}'.format(latent_names[i]))
                    ind += 1
                if l_spc.shape[1] == 1:
                    axs[ind].hist(l_spc, bins)
                    axs[ind].set_title('Latent Space {}'.format(latent_names[i]))
                    ind += 1

        if implicit:
            # plot samples in A space
            a_sample = model.zy(model.bdist_sample(int(1e5)))
            if isinstance(a_sample, tuple):
                a_sample = model.model_sample(a_sample)
                # a_sample = a_sample[0]
            a_sample = model.get_numpy(a_sample)

            plot2Dhist(a_sample, axs[ind], bins)
            axs[ind].set_title('A space sample')

        nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
        fig.suptitle(sup_title + ' {}'.format(nparams))
        fig.tight_layout()
        fig.savefig(sv_dir + '/post_processing_{}.png'.format(nm))

        print('There are {} trainable parameters'.format(nparams))

        if vae:
            # Plot the means and std
            encoding = model.encoder(test_data)
            fig, axs_ = plt.subplots(1, 2, figsize=(5 * nplots + 2, 5))
            axs = fig.axes
            titles = ['Means', 'Logvars']
            for i in range(2):
                plot2Dhist(model.get_numpy(encoding[i]), axs[i], bins)
                axs[i].set_title(titles[i])
            fig.tight_layout()
            fig.savefig(sv_dir + '/post_processing_means_{}.png'.format(nm))

        fig, axs = plt.subplots(1, 1, figsize=(5, 5))
        recons = model.get_numpy(model.autoencode(test_data))
        plot2Dhist(recons, axs, bins)
        fig.tight_layout()
        fig.savefig(sv_dir + '/autoencode_{}.png'.format(nm))

        # TODO: Sinkhorn distace of samples from dataset


def calculate_mass(four_vector):
    # TODO: can't take the square root without scaling (otherwise there are values < 0)
    return four_vector[:, 0] ** 2 - torch.sum(four_vector[:, 1:4] * four_vector[:, 1:4], axis=1)


def sample_(model, number, bsize=int(1e5)):
    # It is very memory ineffecient to sample all n columns just to extract some features, so do it in batches
    # Ceil divide the number into batches
    if number > bsize:
        nloops = -(number // bsize)
    else:
        nloops = 1
        bsize = number
    # TODO: transverse momenta also
    mass = torch.zeros((nloops, bsize, 5))
    for i in range(nloops):
        sample = model.sample(bsize)[:, :20]
        mass[i] = calculate_mass(sample.view(-1, 4, 5))
    return model.get_numpy(mass)[0]


def post_process_hepmass(model, test_data, sup_title=''):
    # test_data = test_data_.data.to(model.device)
    model.eval()
    nm = model.exp_name
    top_dir = get_top_dir()
    sv_dir = top_dir + '/images' + '/' + model.dir
    if not os.path.exists(sv_dir):
        os.makedirs(sv_dir)

    # # This is how you perform sampling of combinations somewhat efficiently.
    # nplots = 5
    # fig, axs_ = plt.subplots(1, nplots, figsize=(5 * nplots + 2, 5))
    # axs = fig.axes
    #
    # with torch.no_grad():
    #     nsample = int(1e3)
    #     indices = torch.randperm(len(test_data))[:nsample]
    #     orig_mass = model.get_numpy(calculate_mass(test_data.data[indices, :20].view(-1, 4, 5)))
    #     mass_sample = sample_(model, nsample)
    #     titles = ['Leading lepton mass', 'Jet 1 Mass', 'Jet 2 Mass', 'Jet 3 Mass', 'Jet 4 Mass']
    #     for i in range(nplots):
    #         axs[i].hist(orig_mass[:, i], label='original')
    #
    #         # Plot samples drawn from the model
    #         axs[i].hist(mass_sample[:, i], label='samples')
    #         axs[i].set_title(titles[i])
    #         axs[i].legend()
    #
    #     nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #     fig.suptitle(sup_title + ' {}'.format(nparams))
    #     fig.tight_layout()
    #     fig.savefig(sv_dir + '/post_processing_{}.png'.format(nm))

    fig, axs_ = plt.subplots(7, 3, figsize=(5 * 3 + 2, 5 * 7 + 2))
    axs = fig.axes

    with torch.no_grad():
        nsample = int(1e5)
        indices = torch.randperm(len(test_data))[:nsample]
        originals = (test_data.data[indices, :21] + 1) / 2
        sample = model.sample(nsample)

        for i in range(21):
            axs[i].hist(model.get_numpy(originals[:, i]), label='original', alpha=0.5)

            # Plot samples drawn from the model
            axs[i].hist(model.get_numpy(sample[:, i]), label='samples', alpha=0.5)
            axs[i].set_title('Feature {}'.format(i))
            axs[i].legend()

    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fig.suptitle(sup_title + ' {}'.format(nparams))
    fig.tight_layout()
    fig.savefig(sv_dir + '/post_processing_{}.png'.format(nm))

    print('There are {} trainable parameters'.format(nparams))

    # TODO: Sinkhorn distace of samples from dataset


def get_bins(data, nbins=20):
    max_ent = data.max().item()
    min_ent = data.min().item()
    return np.linspace(min_ent, max_ent, num=nbins)


def post_process_jets(model, test_data, anomaly_set=None, anomaly_theshold=3.5, sup_title=''):
    # TODO: set bins by hand
    model.eval()
    nm = model.exp_name
    top_dir = get_top_dir()
    sv_dir = top_dir + '/images' + '/' + model.dir
    if not os.path.exists(sv_dir):
        os.makedirs(sv_dir)

    if anomaly_set:
        with torch.no_grad():
            # Get the log probabilities of the two datasets before scaling the data
            likelihoods_data = model.log_prob(test_data.data)
            likelihoods_outliers = model.log_prob(anomaly_set.data)

    fig, axs_ = plt.subplots(2, 4, figsize=(5 * 4 + 2, 5 * 2 + 2))
    axs = fig.axes
    data_dim = test_data.data.shape[1]

    # Unnormalize the data back to original range
    test_data.unnormalize()

    with torch.no_grad():
        nsample = int(1e5)
        # Don't plot all of the originals, just nsample number
        indices = torch.randperm(len(test_data))[:nsample]
        originals = test_data.data[indices]
        # sample = model.sample(nsample if nsample <= originals.shape[0] else originals.shape[0])
        sample = model.sample(nsample)
        # Unnormalize the sample
        sample = test_data.unnormalize(sample)

        for i in range(data_dim):
            bins = get_bins(originals[:, i])
            axs[i].hist(model.get_numpy(originals[:, i]), label='original', alpha=0.5, density=True, bins=bins)
            # Plot samples drawn from the model
            axs[i].hist(model.get_numpy(sample[:, i]), label='samples', alpha=0.5, density=True, bins=bins)
            axs[i].set_title('Feature {}'.format(i))
            axs[i].legend()

    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fig.suptitle(sup_title + ' {}'.format(nparams))
    fig.tight_layout()
    fig.savefig(sv_dir + '/post_processing_{}.png'.format(nm))

    print('There are {} trainable parameters'.format(nparams))

    # Calculate some observables
    def get_mjj(data):
        d_eta = data[:, 1] - data[:, 5]
        d_phi = data[:, 2] - data[:, 6]
        pt_prod = data[:, 0] * data[:, 4]
        # return (2 * pt_prod * (torch.cosh(d_eta) - torch.cos(d_phi))) ** (0.5)
        return 2 * pt_prod * (torch.cosh(d_eta) - torch.cos(d_phi))

    def get_d_eta(data):
        return data[:, 1] - data[:, 5]

    def hist_obs(function, originals, sample, ax, title):
        orig = function(originals)
        bins = get_bins(orig)
        ax.hist(model.get_numpy(orig), label='original', alpha=0.5, density=True, bins=bins)
        ax.hist(model.get_numpy(function(sample)), label='sample', alpha=0.5, density=True, bins=bins)
        ax.set_title(title)

    fig, axs_ = plt.subplots(1, 2, figsize=(12, 7))
    axs = fig.axes
    # hist_obs(get_mjj, originals, sample, axs[0], 'Dijet Mass')
    hist_obs(get_d_eta, originals, sample, axs[1], 'D_eta')

    # mjj_orig = get_mjj(originals)
    # print(mjj_orig.min())
    # print(mjj_orig.max())
    # axs[0].hist(model.get_numpy(mjj_orig), label='original', alpha=0.5, density=True, bins=10)
    # axs[0].hist(model.get_numpy(get_mjj(sample)), label='sample', alpha=0.5, density=True, bins=10)
    # axs[0].set_title('Dijet Mass')

    fig.tight_layout()
    fig.savefig(sv_dir + '/hlo_{}.png'.format(nm))

    if anomaly_set:
        # Get the outliers from each set
        weird_data = test_data[abs(likelihoods_data) > anomaly_theshold]
        normal_data = test_data[abs(likelihoods_data) <= anomaly_theshold]
        weird_outliers = anomaly_set[abs(likelihoods_outliers) > anomaly_theshold]
        normal_outliers = test_data[abs(likelihoods_outliers) <= anomaly_theshold]

        print('There are {}% of anomalies in the testset, and {}% in the data'.format(len(weird_data) / len(test_data),
                                                                                      len(weird_outliers) / len(
                                                                                          anomaly_set)))

        fig, axs_ = plt.subplots(1, 2, figsize=(12, 7))
        axs = fig.axes
        # hist_obs(get_mjj, originals, sample, axs[0], 'Dijet Mass')
        hist_obs(get_d_eta, normal_outliers, weird_outliers, axs[0], 'D_eta')
        hist_obs(get_d_eta, normal_data, weird_data, axs[1], 'D_eta')

        fig.tight_layout()
        fig.savefig(sv_dir + '/hlo_anomalies_{}.png'.format(nm))


def get_counts(data, to_slice, bound=4, nbins=50):
    bin_edges = np.linspace(-bound, bound, nbins + 1)
    # Apply a slice to the data
    mask = torch.all((to_slice > 0) & (to_slice < 2), 1)
    data = data[mask.type(torch.bool)].cpu().numpy()
    return np.histogram2d(data[:, 0], data[:, 1], bins=bin_edges)[0]


def get_ood(model, nsamples, nrun, bound, nbins, data_generator=None, get_target=None, max_it=1000):
    percent_ood = np.zeros(nrun)
    percent_oob = np.zeros(nrun)
    counts = np.zeros((nbins, nbins))
    counts_true = np.zeros((nbins, nbins))
    it = 0

    for i in range(nrun):
        with torch.no_grad():
            # TODO: deal with this code duplication
            if data_generator:
                data = data_generator(nsamples)
                sample = model(data).detach().cpu()
            else:
                # If not generator is passed then the model must be a flow
                sample = model.sample(nsamples).detach().cpu()
        percent_ood[i] = HyperCheckerboardDataset.count_ood(sample)
        percent_oob[i] = HyperCheckerboardDataset.count_oob(sample)
        counts += get_counts(sample, sample[:, 2:], bound, nbins)
        if get_target:
            target = get_target(data)
            counts_true += get_counts(target, target[:, 2:], bound, nbins)

    print('{}% of OOD data, std {}.'.format(np.mean(percent_ood), np.std(percent_ood)))
    print('{}% of OOB data, std {}.'.format(np.mean(percent_oob), np.std(percent_oob)))

    # Plot one slice of the data to inspect the training.
    # TODO: shift any improvements back to hyper_plane.py testing, and ideally load this as a function
    # TODO: should also update percent_ood in the while loop - but it seems to be quite accurate

    while np.sum(counts) < int(1e4) and (it < max_it):
        it += 1
        with torch.no_grad():
            if data_generator:
                data = data_generator(nsamples)
                sample = model(data).detach().cpu()
            else:
                # If not generator is passed then the model must be a flow
                sample = model.sample(nsamples).detach().cpu()
        counts += get_counts(sample, sample[:, 2:], bound, nbins)
        if get_target:
            target = get_target(data)
            counts_true += get_counts(target, target[:, 2:], bound, nbins)
    if get_target:
        return percent_ood, percent_oob, counts, counts_true
    else:
        return percent_ood, percent_oob, counts


def hist_features(originals, sample, model, data_dim, axs):
    for i in range(data_dim):
        bins = get_bins(originals[:, i])
        axs[i].hist(model.get_numpy(originals[:, i]), label='original', alpha=0.5, density=True, bins=bins)
        # Plot samples drawn from the model
        axs[i].hist(model.get_numpy(sample[:, i]), label='samples', alpha=0.5, density=True, bins=bins)
        axs[i].set_title('Feature {}'.format(i))
        axs[i].legend()


def post_process_anode(model, datasets, sup_title='NSF'):
    sv_dir = get_top_dir() + '/images' + '/' + model.dir
    if not os.path.exists(sv_dir):
        os.makedirs(sv_dir)
    nm = model.exp_name
    nfeatures = datasets.nfeatures

    def hist_dataset(dataset, sv_name):
        context_valid = dataset.data[:, -1].view(-1, 1).to(model.device)
        data_valid = dataset.data[:, :-1]

        with torch.no_grad():
            valid_samples = model.sample(1, context_valid).squeeze()

        ncols = int(np.ceil(nfeatures / 4))
        fig, axs_ = plt.subplots(ncols, 4, figsize=(5 * 4 + 2, 5 * ncols + 2))
        axs = fig.axes
        hist_features(data_valid, valid_samples, model, nfeatures, axs)
        fig.savefig(sv_dir + '/post_processing_{}_{}.png'.format(nm, sv_name))

    hist_dataset(datasets.validationset, 'validation')
    hist_dataset(datasets.signalset, 'signal')

    # TODO: look at distributions across observables, and the average log prob in each of the bins

    def get_outlier_sample(dataset, mult_sampl=1, threshold=0.95):
        context = dataset.data[:, -1].view(-1, 1).to(model.device)

        with torch.no_grad():
            samples = model.sample(mult_sampl, context).squeeze()
            # TODO if mult_sample > 1 this doesn't work
            lp = model.flow.log_prob(samples, context=context)
            cut = torch.quantile(lp, 1 - threshold)
            outliers = samples[:, -1][lp < cut]

        return outliers

    # TODO: plot this for multiple different cuts instead of one fixed one.
    valid_outlier_context = get_outlier_sample(datasets.validationset)
    sample_outlier_context = get_outlier_sample(datasets.signalset)
    print('There are {}% samples in the validation set, and {}% outliers in the signal region'.format(
        len(valid_outlier_context) / len(datasets.validationset) * 100,
        len(sample_outlier_context) / len(datasets.signalset) * 100))

    samples = torch.cat((valid_outlier_context, sample_outlier_context))

    fig, ax = plt.subplots(1, 1, figsize=(5 + 2, 5 + 2))
    ax.hist(model.get_numpy(samples))
    # Label the sideband region
    bands = [elem * 4 for elem in datasets.bins]
    ax.set_xticks(np.append(ax.get_xticks(), bands))
    ax.get_xticklabels(ax.get_xticklabels() + ['sb'] * len(bands))
    fig.savefig(sv_dir + '/post_processing_{}_{}.png'.format(nm, 'outliers'))

    return 0
