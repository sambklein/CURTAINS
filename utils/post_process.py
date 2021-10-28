import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.DRE import get_auc

from .io import get_top_dir
from .plotting import (get_bins, getFeaturePlot, getInputTransformedHist,
                       hist_features, hist_features_single, plot_rates_dict,
                       plot_single_feature_mass_diagnostic)
from .sampling_utils import signalMassSampler


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
        hist_features(data_valid, valid_samples, nfeatures, axs, nms)
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


# Some functions for evaluating curtains
def transform_to_mass(data, lm, hm, mass_sampler, model):
    if lm > hm:
        raise Exception('First input must be the low end of the mass window.')
    data_mass = data.data[:, -1].view(-1, 1)
    sample_mass = mass_sampler.sample(len(data_mass), limits=(lm.item(), hm.item())).to(model.device)
    if data_mass.min() >= hm:
        direction = 'inverse'
    elif data_mass.max() <= lm:
        direction = 'forward'
    else:
        raise NotImplementedError('The mass range to which you map cannot overlap with the input mass range.')

    with torch.no_grad():
        feature_sample = {'forward': model.transform_to_mass, 'inverse': model.inverse_transform_to_mass}[
            direction](data.data[:, :-1], data_mass, sample_mass)[0]
    return torch.cat((feature_sample, sample_mass), 1).cpu()


def get_samples(input_dist, target_dist, model, r_mass=False):
    target_dist.data = target_dist.data.to(model.device)
    s1 = input_dist.data.shape[0]
    s2 = target_dist.data.shape[0]
    nsamp = min(s1, s2)

    if input_dist[:, -1].min() >= target_dist[:, -1].max():
        direction = 'inverse'
    elif input_dist[:, -1].max() <= target_dist[:, -1].min():
        direction = 'forward'
    else:
        raise NotImplementedError('The mass range to which you map cannot overlap with the input mass range.')

    with torch.no_grad():
        # id = input_dist.data[input_dist.data[:, -1].sort()[1]][:nsamp]
        # td = target_dist.data[target_dist.data[:, -1].sort()[1]][:nsamp]
        id = input_dist.data[:nsamp]
        td = target_dist.data[:nsamp]

        mass = td[:, -1].view(-1, 1)
        if direction == 'forward':
            samples = model.transform_to_data(id, td, batch_size=1000)[0]
        elif direction == 'inverse':
            samples = model.inverse_transform_to_data(td, id, batch_size=1000)[0]
    if r_mass:
        return torch.cat((samples, mass), -1)
    else:
        return samples


def post_process_curtains(model, datasets, sup_title='NSF', signal_anomalies=None, load=False, use_mass_sampler=False,
                          n_sample_for_plot=-1, light_job=0, classifier_args=None, plot=True):
    if classifier_args is None:
        classifier_args = {}

    low_mass_training = datasets.trainset.data1
    high_mass_training = datasets.trainset.data2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sv_dir = get_top_dir() + '/images' + '/' + model.dir
    if not os.path.exists(sv_dir):
        os.makedirs(sv_dir)
    nm = model.exp_name

    # Fit the mass to use for sampling
    m1 = datasets.trainset.data1[:, -1]
    m2 = datasets.trainset.data2[:, -1]
    masses = torch.cat((m1, m2))
    edge1 = datasets.mass_bins[2].item()
    edge2 = datasets.mass_bins[3].item()
    mass_sampler = signalMassSampler(masses, edge1, edge2, plt_sv_dir=sv_dir,
                                     scaler=low_mass_training.unnorm_mass, unscaler=low_mass_training.norm_mass)

    def get_transformed(data, lm=None, hm=None, target_dist=None, r_mass=True, oversample=4):
        if use_mass_sampler:
            # TODO: the bin information should be stored in the class not found from data.
            if hm is None:
                hm = target_dist.data[:, -1].max()
            if lm is None:
                lm = target_dist.data[:, -1].min()
            data_lst = []
            for i in range(oversample):
                data_lst += [transform_to_mass(data, lm, hm, mass_sampler, model)]
            data = torch.cat(data_lst, 0)
            if not r_mass:
                data = data[:, :-1]
            return data
        else:
            return get_samples(data, target_dist, model, r_mass=r_mass)

    def get_maps(base_name, input_dataset, target_datasets):
        if plot:
            for i, set in enumerate(target_datasets):
                target_sample = target_datasets[set]
                print(f"Now evaluating sample {set} from {base_name}")
                samples = get_transformed(input_dataset, target_dist=target_sample, r_mass=False)
                # For the feature plot we only want to look at as many samples as there are in SB1
                getFeaturePlot(model, target_sample, samples, input_dataset, nm, sv_dir, f'{base_name} to {set}',
                               datasets.signalset.feature_nms, n_sample_for_plot=n_sample_for_plot)

    # Map low mass samples to high mass
    high_mass_datasets = {'Signal Set': datasets.signalset, 'SB2': high_mass_training,
                          'OB2': datasets.validationset}
    low_mass_sample = low_mass_training
    low_mass_sample.data = low_mass_sample.data.to(model.device)
    get_maps('SB1', low_mass_sample, high_mass_datasets)

    # Then map the high mass sample to the low mass samples
    low_mass_datasets = {'Signal Set': datasets.signalset, 'SB1': low_mass_training,
                         'OB1': datasets.validationset_lm}
    high_mass_sample = high_mass_training
    high_mass_sample.data = high_mass_sample.data.to(model.device)
    get_maps('SB2', high_mass_sample, low_mass_datasets)

    # Validation set one, SB2 to one mass bin higher
    get_maps('SB2', high_mass_training, {'OB2': datasets.validationset})
    if not light_job:
        # AUC for OB2 vs T(SB2)
        print('SB2 from OB2')
        ob2_samples = get_transformed(high_mass_training, lm=datasets.mass_bins[4], hm=datasets.mass_bins[5],
                                      target_dist=datasets.validationset)
        auc_ob2 = get_auc(ob2_samples, datasets.validationset.data, sv_dir, nm + 'OB2_vs_TSB2',
                          mscaler=low_mass_training.unnorm_mass, load=load, sup_title=f'T(SB2) vs OB2',
                          **classifier_args)

        # Validation set two, SB1 to one mass bin lower
        get_maps('SB1', low_mass_training, {'OB1': datasets.validationset_lm})
        # AUC for OB1 vs T(SB1)
        print('SB1 from OB1')
        ob1_samples = get_transformed(low_mass_training, lm=datasets.mass_bins[0], hm=datasets.mass_bins[1],
                                      target_dist=datasets.validationset_lm)
        auc_ob1 = get_auc(ob1_samples, datasets.validationset_lm.data, sv_dir, nm + 'OB1_vs_TSB1',
                          mscaler=low_mass_training.unnorm_mass, load=load, sup_title=f'T(SB1) vs OB1',
                          **classifier_args)

    if light_job <= 1 or (light_job == 3):
        # Map the combined side bands into the signal region
        sb2_samples = get_transformed(high_mass_sample, lm=datasets.mass_bins[2], hm=datasets.mass_bins[3],
                                      target_dist=datasets.signalset)
        sb1_samples = get_transformed(low_mass_sample, lm=datasets.mass_bins[2], hm=datasets.mass_bins[3],
                                      target_dist=datasets.signalset)
        samples = torch.cat((sb2_samples, sb1_samples))

        # For the feature plot we only want to look at as many samples as there are in SB1
        getFeaturePlot(model, datasets.signalset, samples, high_mass_sample, nm, sv_dir, 'SB1 and SB2 to Signal ',
                       datasets.signalset.feature_nms, n_sample_for_plot=n_sample_for_plot)

        print('Without anomalies injected')
        auc = get_auc(samples, datasets.signalset.data, sv_dir, nm + 'SB12', mscaler=low_mass_training.unnorm_mass,
                      load=False, sup_title=f'T(SB1) U T(SB2) vs SR',
                      **classifier_args)

    if light_job <= 1:
        print('SB2 from signal set')
        auc_sb2 = get_auc(sb2_samples, datasets.signalset.data, sv_dir, nm + 'SB2',
                          mscaler=low_mass_training.unnorm_mass,
                          load=load, sup_title=f'T(SB2) vs SR',
                          **classifier_args)
        print('SB1 from signal set')
        auc_sb1 = get_auc(sb1_samples, datasets.signalset.data, sv_dir, nm + 'SB1',
                          mscaler=low_mass_training.unnorm_mass,
                          load=load, sup_title=f'T(SB1) vs SR',
                          **classifier_args)

        # print('Without anomalies injected')
        # auc = get_auc(samples, datasets.signalset.data, sv_dir, nm + 'SB12', mscaler=low_mass_training.unnorm_mass,
        #               load=False, sup_title=f'T(SB1) U T(SB2) vs SR',
        #               **classifier_args)

    if (not light_job) or (light_job == 3):
        # Get the AUC of the ROC for a classifier trained to separate interpolated samples from data
        print('Benchmark classifier separating samples from anomalies')
        auc_super_info = get_auc(signal_anomalies.data.to(device), datasets.signalset.data.to(device), sv_dir,
                                 nm + 'Super', mscaler=low_mass_training.unnorm_mass, load=load,
                                 sup_title=f'QCD SR vs Anomalies SR', return_rates=True,
                                 **classifier_args)
        print('With anomalies injected')
        rates_sr_vs_transformed = {'Supervised': auc_super_info[1]}
        rates_sr_qcd_vs_anomalies = {'Supervised': auc_super_info[1]}
        # rates_sr_vs_transformed = {}
        # rates_sr_qcd_vs_anomalies = {}
        for beta in [0.5, 1, 5]:
            auc_info = get_auc(samples, datasets.signalset.data, sv_dir, nm + f'{beta}%Anomalies',
                               anomaly_data=signal_anomalies.data.to(device), beta=beta / 100,
                               sup_title=f'QCD in SR doped with {beta:.3f}% anomalies',
                               mscaler=low_mass_training.unnorm_mass,
                               load=load, return_rates=True,
                               **classifier_args)
            auc_anomalies = auc_info[0]
            rates_sr_vs_transformed[f'{beta}'] = auc_info[1]
            rates_sr_qcd_vs_anomalies[f'{beta}'] = auc_info[2]

        plot_rates_dict(sv_dir, rates_sr_qcd_vs_anomalies, 'SR QCD vs SR Anomalies')
        plot_rates_dict(sv_dir, rates_sr_vs_transformed, 'T(SB12) vs SR')

        # Transform signal region samples into SB1 (TODO: same for map to SB2)
        sb1_samples = get_transformed(datasets.signalset, lm=datasets.mass_bins[1], hm=datasets.mass_bins[2],
                                      target_dist=low_mass_sample)
        anom_sb1_samples = get_transformed(signal_anomalies.data.to(device), lm=datasets.mass_bins[1],
                                           hm=datasets.mass_bins[2], target_dist=low_mass_sample)
        auc_super_info = get_auc(sb1_samples, low_mass_sample.data, sv_dir,
                                 nm + 'SR_in_SB1', mscaler=low_mass_training.unnorm_mass, load=load,
                                 sup_title=f'SR_in_SB1', return_rates=True,
                                 **classifier_args)
        auc_supervised = auc_super_info[0]
        rates_sr_vs_transformed = {'Supervised': auc_super_info[1]}
        rates_sr_qcd_vs_anomalies = {'Supervised': auc_super_info[1]}
        for beta in [0.5, 1, 5]:
            auc_info = get_auc(sb1_samples, low_mass_sample.data, sv_dir, nm + f'SR_in_SB1_{beta}%Anomalies',
                               anomaly_data=anom_sb1_samples,
                               beta=beta / 100,
                               sup_title=f'SR in SB1 doped with {beta:.3f}% anomalies',
                               mscaler=low_mass_training.unnorm_mass,
                               load=load,
                               return_rates=True,
                               **classifier_args)
            auc_anomalies = auc_info[0]
            rates_sr_vs_transformed[f'{beta}'] = auc_info[1]
            rates_sr_qcd_vs_anomalies[f'{beta}'] = auc_info[2]

        plot_rates_dict(sv_dir, rates_sr_qcd_vs_anomalies, 'SB1 QCD vs SR Anomalies')
        plot_rates_dict(sv_dir, rates_sr_vs_transformed, 'SB1 vs T(SR)')

    if light_job < 2:
        with open(sv_dir + '/auc_{}.npy'.format(nm), 'wb') as f:
            np.save(f, auc_sb2)
            np.save(f, auc_sb1)
            if not light_job:
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
                                              mass * torch.ones((low_mass_sample.data.shape[0], 1)).to(device))[0]
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
            samples[i] = model.transform_to_data(low_mass_sample, mass_sample.to(device))[0]

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
    hist_features(low_mass_training, sample, datasets.nfeatures, ax)
    fig.savefig(sv_dir + '/post_processing_{}_{}.png'.format(nm, 'base_dist_sample'))
