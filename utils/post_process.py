import matplotlib.pyplot as plt

import numpy as np

import torch
from utils.DRE import get_auc

from .io import get_top_dir
from .plotting import getFeaturePlot, get_bins, getCrossFeaturePlot, hist_features, hist_features_single, \
    plot_single_feature_mass_diagnostic

import os


def calculate_mass(four_vector):
    return four_vector[:, 0] ** 2 - torch.sum(four_vector[:, 1:4] * four_vector[:, 1:4], axis=1)


def sample_(model, number, bsize=int(1e5)):
    # It is very memory ineffecient to sample all n columns just to extract some features, so do it in batches
    # Ceil divide the number into batches
    if number > bsize:
        nloops = -(number // bsize)
    else:
        nloops = 1
        bsize = number
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


def post_process_jets(model, test_data, anomaly_set=None, anomaly_theshold=3.5, sup_title=''):
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


def post_process_anode(model, datasets, sup_title='NSF', quantiles=True):
    sv_dir = get_top_dir() + '/images' + '/' + model.dir
    if not os.path.exists(sv_dir):
        os.makedirs(sv_dir)
    nm = model.exp_name
    nfeatures = datasets.nfeatures

    def hist_dataset(dataset, sv_name, nms):
        context_valid = dataset.data[:, -1].view(-1, 1).to(model.device)
        data_valid = dataset.data[:, :-1]

        with torch.no_grad():
            valid_samples = model.sample(1, context_valid).squeeze()

        ncols = int(np.ceil(nfeatures / 4))
        fig, axs_ = plt.subplots(ncols, 4, figsize=(5 * 5 + 2, 5 * ncols))
        axs = fig.axes
        hist_features(data_valid, valid_samples, model, nfeatures, axs, nms)
        fig.suptitle(sv_name)
        fig.savefig(sv_dir + '/post_processing_{}_{}.png'.format(nm, sv_name))

    hist_dataset(datasets.validationset, 'validation', datasets.signalset.feature_nms)
    hist_dataset(datasets.signalset, 'signal', datasets.signalset.feature_nms)

    # TODO: look at distributions across observables, and the average log prob in each of the bins

    def get_outlier_sample(dataset, mult_sampl=1, threshold=0.95, set_cut=None):
        context = dataset.data[:, -1].view(-1, 1).to(model.device)
        with torch.no_grad():
            samples = model.sample(mult_sampl, context).squeeze()
            lp = model.flow.log_prob(samples, context=context)
            if set_cut is None:
                cut = torch.quantile(lp, 1 - threshold)
            else:
                cut = set_cut
            outliers = samples[:, -1][lp < cut]

        if set_cut is None:
            return outliers, cut
        else:
            return outliers

    threshold = 0.95
    data_outlier_context, cut = get_outlier_sample(datasets.trainset, threshold=0.95)
    valid_outlier_context = get_outlier_sample(datasets.validationset, set_cut=cut)
    sample_outlier_context = get_outlier_sample(datasets.signalset, set_cut=cut)
    print(
        'There are {}% samples in the validation set, and {}% outliers in the signal region. \n'
        'There are {}% in the training data.'.format(
            len(valid_outlier_context) / len(datasets.validationset) * 100,
            len(sample_outlier_context) / len(datasets.signalset) * 100,
            (1 - threshold) * 100))

    # samples = torch.cat((valid_outlier_context, sample_outlier_context))
    # fig, ax = plt.subplots(1, 1, figsize=(5 + 2, 5 + 2))
    # ax.hist(model.get_numpy(samples))
    # if quantiles:
    #     lm = datasets.trainset.data1[:, -1]
    #     hm = datasets.trainset.data2[:, -1]
    #     bands = [lm.min(), lm.max(), hm.min(), hm.max()]
    # else:
    #     # Label the sideband region
    #     bands = [elem * 4 for elem in datasets.bins]
    # ax.set_xticks(np.append(ax.get_xticks(), bands))
    # ax.get_xticklabels(ax.get_xticklabels() + ['sb'] * len(bands))
    # fig.savefig(sv_dir + '/post_processing_{}_{}.png'.format(nm, 'outliers'))

    # Do this at the end where you won't use signalset after
    # TODO: clean this up and don't overwrite
    dl = datasets.trainset.data1
    dh = datasets.trainset.data2
    datasets.signalset.data = torch.cat((dl.data, dh.data), 0)
    hist_dataset(datasets.signalset, 'training', datasets.signalset.feature_nms)

    return 0


def post_process_curtains(model, datasets, sup_title='NSF', anomaly_data=None):
    low_mass_training = datasets.trainset.data1
    high_mass_training = datasets.trainset.data2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sv_dir = get_top_dir() + '/images' + '/' + model.dir
    if not os.path.exists(sv_dir):
        os.makedirs(sv_dir)
    nm = model.exp_name

    # TODO: move these functions somewhere nicer
    def get_samples(input_dist, target_dist, direction):
        target_dist.data = target_dist.data.to(model.device)
        s1 = input_dist.data.shape[0]
        s2 = target_dist.data.shape[0]
        nsamp = min(s1, s2)
        with torch.no_grad():
            if direction == 'forward':
                samples = model.transform_to_data(input_dist[:nsamp],
                                                  target_dist[torch.randperm(s2, device=torch.device('cpu'))][:nsamp],
                                                  batch_size=1000)
            elif direction == 'inverse':
                samples = model.inverse_transform_to_data(
                    target_dist[torch.randperm(s2, device=torch.device('cpu'))][:nsamp], input_dist[:nsamp],
                    batch_size=1000)
        return samples

    def get_maps(base_name, input_dataset, target_datasets, direction='forward'):
        for i, set in enumerate(target_datasets):
            target_sample = target_datasets[set]
            print(f"Now evaluating sample {set} from {base_name}")
            samples = get_samples(input_dataset, target_sample, direction)
            # For the feature plot we only want to look at as many samples as there are in SB1
            getFeaturePlot(model, target_sample, samples, input_dataset, nm, sv_dir, f'{base_name} to {set}',
                           datasets.signalset.feature_nms)

    # Map low mass samples to high mass
    high_mass_datasets = {'Signal Set': datasets.signalset, 'SB2': high_mass_training,
                          'Validation SB2': datasets.validationset}
    low_mass_sample = low_mass_training
    low_mass_sample.data = low_mass_sample.data.to(model.device)
    get_maps('SB1', low_mass_sample, high_mass_datasets)

    # Then map the high mass sample to the low mass samples
    low_mass_datasets = {'Signal Set': datasets.signalset, 'SB1': low_mass_training,
                         'Validation SB1': datasets.validationset_lm}
    high_mass_sample = high_mass_training
    high_mass_sample.data = high_mass_sample.data.to(model.device)
    get_maps('SB2', high_mass_sample, low_mass_datasets, direction='inverse')

    # Validation set one, SB2 to one mass bin higher
    get_maps('SB2', high_mass_training, {'Validation SB2': datasets.validationset}, direction='forward')

    # Validation set two, SB1 to one mass bin lower
    get_maps('SB1', low_mass_training, {'Validation SB1': datasets.validationset_lm}, direction='inverse')

    # # And finally, map the combined side bands into the signal region, this is the REAL deal and we will measure the
    # performance of this map by training a classifier to separate interpolated samples from real.
    sb2_samples = get_samples(high_mass_sample, datasets.signalset, 'inverse')
    sb1_samples = get_samples(low_mass_sample, datasets.signalset, 'forward')
    samples = torch.cat((sb2_samples, sb1_samples))
    # For the feature plot we only want to look at as many samples as there are in SB1
    getFeaturePlot(model, datasets.signalset, samples, high_mass_sample, nm, sv_dir, 'SB1 and SB2 to Signal ',
                   datasets.signalset.feature_nms)

    # Get the AUC of the ROC for a classifier trained to separate interpolated samples from data
    print('Benchmark classifier separating samples from anomalies')
    auc_supervised = get_auc(anomaly_data.data[:, :-1].to(device), datasets.signalset.data[:, :-1], sv_dir, nm)
    print('With anomalies injected')
    auc_anomalies = get_auc(samples, datasets.signalset.data[:, :-1], sv_dir, nm,
                            anomaly_data=anomaly_data.data[:, :-1].to(device))
    print('Without anomalies injected')
    auc = get_auc(samples, datasets.signalset.data[:, :-1], sv_dir, nm)
    with open(sv_dir + '/auc_{}.npy'.format(nm), 'wb') as f:
        np.save(f, auc_supervised)
        np.save(f, auc_anomalies)
        np.save(f, auc)

    nmass = 5
    masses = np.linspace(datasets.signalset.data[:, -1].min().item(),
                         datasets.trainset.data2.data[:, -1].max().item(), nmass)

    nfeatures = datasets.nfeatures
    fig, ax = plt.subplots(1, nfeatures, figsize=(5 * nfeatures + 2, 5))
    bns = []
    lm = low_mass_sample.unnormalize(low_mass_sample.data)
    for i in range(nfeatures):
        bns += [get_bins(lm[:, i])]
    hist_features_single(lm, model, datasets.signalset.feature_nms, ax, bns, label='SB1')
    for mass in masses:
        with torch.no_grad():
            samples = model.transform_to_data(low_mass_sample.data.to(device),
                                              mass * torch.ones((low_mass_sample.data.shape[0], 1)).to(device))
        # getCrossFeaturePlot(model, low_mass_sample.unnormalize(low_mass_sample.data), samples, nm, sv_dir, mass,
        #                     datasets.signalset.feature_nms)
        hist_features_single(samples, model, datasets.signalset.feature_nms, ax, bns,
                             label=f'Mass: {low_mass_sample.unnormalize(torch.tensor(mass).view(-1, 1)).item():.2f}')
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    fig.tight_layout()
    fig.savefig(sv_dir + f'/feature_distributions_{nm}')

    # Look at the distributions at different fixed mass values
    max_mass = datasets.validationset.data[:, -1].max().item()
    min_mass = min(datasets.signalset.data[:, -1].min().item(), datasets.validationset.data[:, -1].min().item())
    nshuffle = 100
    nsamp = low_mass_sample.shape[0]
    samples = torch.empty((nshuffle, nsamp, nfeatures))
    for i in range(nshuffle):
        mass_sample = (min_mass - max_mass) * torch.rand(nsamp, 1) + max_mass
        with torch.no_grad():
            samples[i] = model.transform_to_data(low_mass_sample, mass_sample.to(device))

    smp = low_mass_sample.unnormalize(samples.view(-1, nfeatures))
    plot_single_feature_mass_diagnostic(model, samples.view(-1, nfeatures), smp, datasets.signalset.feature_nms, sv_dir,
                                        'Mass Diagnostic', nm)


def post_process_flows_for_flows(model, datasets, sup_title='NSF'):
    sv_dir = get_top_dir() + '/images' + '/' + model.dir
    nm = model.exp_name
    low_mass_training = datasets.trainset.data1
    sample = model.sample(low_mass_training.data.shape[0])
    nplot = 1
    fig, ax = plt.subplots(nplot, datasets.nfeatures, figsize=(5 * datasets.nfeatures + 2, 5 * nplot + 2))
    hist_features(low_mass_training, sample, model, datasets.nfeatures, ax)
    fig.savefig(sv_dir + '/post_processing_{}_{}.png'.format(nm, 'base_dist_sample'))
