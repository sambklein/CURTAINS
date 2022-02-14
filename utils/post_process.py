import json
import os
import pdb
import pickle
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch

from data.physics_datasets import preprocess_method
from utils.DRE import get_auc

from .io import get_top_dir
from .plotting import (get_bins, getFeaturePlot, getInputTransformedHist,
                       hist_features, hist_features_single, plot_rates_dict,
                       plot_single_feature_mass_diagnostic)
from .sampling_utils import signalMassSampler
from .torch_utils import tensor2numpy, shuffle_tensor


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
        if direction == 'forward':
            feature_sample = model.transform_to_mass(data.data[:, :-1], data_mass, sample_mass)[0]
        elif direction == 'inverse':
            feature_sample = model.inverse_transform_to_mass(data.data[:, :-1], sample_mass, data_mass)[0]
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
        id = shuffle_tensor(input_dist.data)[:nsamp]
        td = shuffle_tensor(target_dist.data)[:nsamp]

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
                          n_sample_for_plot=-1, light_job=0, classifier_args=None, plot=True, mass_sampler=None,
                          cathode=False, summary_writer=None, args=None, cathode_load=False):
    if classifier_args is None:
        classifier_args = {}

    oversample = args.oversample

    summary_writer_passed = summary_writer is not None
    low_mass_training = datasets.trainset.data1
    high_mass_training = datasets.trainset.data2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sv_dir = get_top_dir() + '/images' + '/' + model.dir
    if not os.path.exists(sv_dir):
        os.makedirs(sv_dir)
    nm = model.exp_name

    if mass_sampler is None:
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

    def save_samples(data_tensor, name):
        if 'data_unscaler' in classifier_args.keys():
            data_tensor = classifier_args['data_unscaler'](data_tensor)
        np.save(os.path.join(sv_dir, name), data_tensor.detach().cpu().numpy())

    def get_maps(base_name, input_dataset, target_datasets):
        if plot:
            for i, set in enumerate(target_datasets):
                target_sample = target_datasets[set]
                print(f"Now evaluating sample {set} from {base_name}")
                samples = get_transformed(input_dataset, target_dist=target_sample, r_mass=False, oversample=1)
                sv_nm = f'{base_name}_to_{set}'
                if sv_nm in ['SB1_to_SB2', 'SB2_to_SB1']:
                    save_samples(samples, f'{sv_nm}_samples')
                # For the feature plot we only want to look at as many samples as there are in SB1
                title = f'{base_name} to {set}' if not cathode else f'Sampled from {set}'
                getFeaturePlot(target_sample, samples, nm, sv_dir, title, datasets.signalset.feature_nms, input_dataset,
                               n_sample_for_plot=n_sample_for_plot, summary_writer=summary_writer)

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
    auc_dict = {}
    if not light_job:
        # AUC for OB2 vs T(SB2)
        print('SB2 from OB2')
        ob2_samples = get_transformed(high_mass_training, lm=datasets.mass_bins[4], hm=datasets.mass_bins[5],
                                      target_dist=datasets.validationset, oversample=oversample)
        save_samples(ob2_samples, 'SB2_to_OB2_samples')
        auc_ob2 = get_auc(ob2_samples, datasets.validationset.data, sv_dir, nm + 'OB2_vs_TSB2',
                          sup_title=f'T(SB2) vs OB2', load=load, **classifier_args)
        auc_dict['SB2/OB2'] = auc_ob2

        # Validation set two, SB1 to one mass bin lower
        get_maps('SB1', low_mass_training, {'OB1': datasets.validationset_lm})
        # AUC for OB1 vs T(SB1)
        print('SB1 from OB1')
        ob1_samples = get_transformed(low_mass_training, lm=datasets.mass_bins[0], hm=datasets.mass_bins[1],
                                      target_dist=datasets.validationset_lm, oversample=oversample)
        save_samples(ob2_samples, 'SB1_to_OB1_samples')
        auc_ob1 = get_auc(ob1_samples, datasets.validationset_lm.data, sv_dir, nm + 'OB1_vs_TSB1',
                          sup_title=f'T(SB1) vs OB1', load=load, **classifier_args)
        auc_dict['SB1/OB1'] = auc_ob1

    if light_job <= 1 or (light_job == 3):
        if cathode_load:
            sb2_samples = preprocess_method(
                torch.tensor(np.load(os.path.join(sv_dir, 'SB1_to_SR_samples.npy')), dtype=torch.float32),
                datasets.signalset.scale
            )[0]
            sb1_samples = preprocess_method(
                torch.tensor(np.load(os.path.join(sv_dir, 'SB2_to_SR_samples.npy')), dtype=torch.float32),
                datasets.signalset.scale
            )[0]
        else:
            # Map the combined side bands into the signal region
            sb2_samples = get_transformed(high_mass_sample, lm=datasets.mass_bins[2], hm=datasets.mass_bins[3],
                                          target_dist=datasets.signalset, oversample=oversample)
            sb1_samples = get_transformed(low_mass_sample, lm=datasets.mass_bins[2], hm=datasets.mass_bins[3],
                                          target_dist=datasets.signalset, oversample=oversample)
            save_samples(sb2_samples, 'SB1_to_SR_samples')
            save_samples(sb1_samples, 'SB2_to_SR_samples')
        samples = torch.cat((sb2_samples, sb1_samples))

        # For the feature plot we only want to look at as many samples as there are in SB1
        title = 'SB1 and SB2 to Signal' if not cathode else f'Sampled from Signal'
        getFeaturePlot(datasets.signalset, samples, nm, sv_dir, title, datasets.signalset.feature_nms, high_mass_sample,
                       n_sample_for_plot=n_sample_for_plot)

    if light_job <= 1:
        print('SB2 from signal set')
        auc_sb2 = get_auc(sb2_samples, datasets.signalset.data, sv_dir, nm + 'SB2', sup_title=f'T(SB2) vs SR',
                          load=load, **classifier_args)
        print('SB1 from signal set')
        auc_sb1 = get_auc(sb1_samples, datasets.signalset.data, sv_dir, nm + 'SB1', sup_title=f'T(SB1) vs SR',
                          load=load, **classifier_args)

        auc_dict['SB1/SR'] = auc_sb1
        auc_dict['SB2/SR'] = auc_sb2

    if (not light_job) or (light_job == 3):
        # Get the AUC of the ROC for a classifier trained to separate interpolated samples from data
        print('With anomalies injected')
        rates_sr_vs_transformed = {}
        # rates_sr_qcd_vs_anomalies = {'Supervised': auc_super_info[1]}
        rates_sr_qcd_vs_anomalies = {}

        auc_info = get_auc(samples, datasets.signalset.data, sv_dir, nm + f'Anomalies_no_eps',
                           anomaly_data=signal_anomalies.data.to(device),
                           sup_title=f'QCD in SR', load=load, return_rates=True,
                           **classifier_args)
        auc_anomalies = auc_info[0]
        print(f'AUC separation {auc_anomalies}')
        auc_dict['SB12/SR'] = auc_anomalies
        rates_sr_vs_transformed[f'0.0'] = auc_info[3]
        rates_sr_qcd_vs_anomalies[f'0.0'] = auc_info[2]
        with open(f'{sv_dir}/counts_no_eps.pkl', 'wb') as f:
            pickle.dump(auc_info[-1], f)

        plot_rates_dict(sv_dir, rates_sr_qcd_vs_anomalies, 'SR QCD vs SR Anomalies')
        plot_rates_dict(sv_dir, rates_sr_vs_transformed, 'T(SB12) vs SR')

        with open(f'{sv_dir}/rates.pkl', 'wb') as f:
            pickle.dump([rates_sr_qcd_vs_anomalies, rates_sr_vs_transformed], f)

    if summary_writer_passed:
        # summary_writer.add_hparams(vars(args), dict(auc_dict, **rates_sr_qcd_vs_anomalies))
        summary_writer.add_hparams(vars(args), auc_dict)

    with open(f'{sv_dir}/aucs.pkl', 'wb') as f:
        pickle.dump(auc_dict, f)


def post_process_flows_for_flows(model, datasets, sup_title='NSF'):
    sv_dir = get_top_dir() + '/images' + '/' + model.dir
    nm = model.exp_name
    low_mass_training = datasets.trainset.data1
    sample = model.sample(low_mass_training.data.shape[0])
    nplot = 1
    fig, ax = plt.subplots(nplot, datasets.nfeatures, figsize=(5 * datasets.nfeatures + 2, 5 * nplot + 2))
    hist_features(low_mass_training, sample, datasets.nfeatures, ax)
    fig.savefig(sv_dir + '/post_processing_{}_{}.png'.format(nm, 'base_dist_sample'))
