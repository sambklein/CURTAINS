import os
import pickle

import numpy as np
import torch

from data.physics_datasets import preprocess_method
from utils.DRE import run_classifiers

from .io import get_top_dir
from .plotting import (getFeaturePlot, plot_rates_dict)
from .sampling_utils import signalMassSampler
from .torch_utils import shuffle_tensor


def calculate_mass(four_vector):
    return four_vector[:, 0] ** 2 - torch.sum(four_vector[:, 1:4] * four_vector[:, 1:4], axis=1)


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

    def get_transformed(data, lm=None, hm=None, target_dist=None, r_mass=True, oversample=4,
                        use_mass_sampler=use_mass_sampler):
        if use_mass_sampler:
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
        # Here we can look at the masses directly, so we never need the sampler
        ob2_samples = get_transformed(high_mass_training, target_dist=datasets.validationset, oversample=1,
                                      use_mass_sampler=False)
        save_samples(ob2_samples, 'SB2_to_OB2_samples')
        auc_ob2 = run_classifiers(ob2_samples, datasets.validationset.data, sv_dir, nm + 'OB2_vs_TSB2',
                                  sup_title=f'T(SB2) vs OB2', load=0, **classifier_args)
        auc_dict['SB2/OB2'] = auc_ob2

        # Validation set two, SB1 to one mass bin lower
        get_maps('SB1', low_mass_training, {'OB1': datasets.validationset_lm})
        # AUC for OB1 vs T(SB1)
        print('SB1 from OB1')
        ob1_samples = get_transformed(low_mass_training, target_dist=datasets.validationset_lm, oversample=1,
                                      use_mass_sampler=False)
        # if plot:
        #     title = f'SB1 to OB1' if not cathode else f'Sampled from OB1'
        #     getFeaturePlot(datasets.validationset_lm, ob1_samples, nm, sv_dir, title, datasets.signalset.feature_nms,
        #                    low_mass_training, n_sample_for_plot=n_sample_for_plot, summary_writer=summary_writer)
        save_samples(ob1_samples, 'SB1_to_OB1_samples')
        auc_ob1 = run_classifiers(ob1_samples, datasets.validationset_lm.data, sv_dir, nm + 'OB1_vs_TSB1',
                                  sup_title=f'T(SB1) vs OB1', load=0, **classifier_args)
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
        auc_sb2 = run_classifiers(sb2_samples, datasets.signalset.data, sv_dir, nm + 'SB2', sup_title=f'T(SB2) vs SR',
                                  load=load, **classifier_args)
        print('SB1 from signal set')
        auc_sb1 = run_classifiers(sb1_samples, datasets.signalset.data, sv_dir, nm + 'SB1', sup_title=f'T(SB1) vs SR',
                                  load=load, **classifier_args)

        auc_dict['SB1/SR'] = auc_sb1
        auc_dict['SB2/SR'] = auc_sb2

    if (not light_job) or (light_job == 3):
        # Get the AUC of the ROC for a classifier trained to separate interpolated samples from data
        print('With anomalies injected')
        rates_sr_vs_transformed = {}
        # rates_sr_qcd_vs_anomalies = {'Supervised': auc_super_info[1]}
        rates_sr_qcd_vs_anomalies = {}

        auc_info = run_classifiers(samples, datasets.signalset.data, sv_dir, nm + f'Anomalies_no_eps',
                                   anomaly_data=signal_anomalies.data.to(device),
                                   sup_title=f'QCD in SR', load=load, return_rates=True,
                                   **classifier_args)
        auc_anomalies = auc_info[0]
        print(f'AUC separation {auc_anomalies}')
        auc_dict['SB12/SR'] = auc_anomalies
        rates_sr_vs_transformed[f'0.0'] = auc_info[3]
        rates_sr_qcd_vs_anomalies[f'0.0'] = auc_info[2]
        with open(f'{sv_dir}/counts.pkl', 'wb') as f:
            pickle.dump(auc_info[4], f)

        plot_rates_dict(sv_dir, rates_sr_qcd_vs_anomalies, 'SR QCD vs SR Anomalies')
        plot_rates_dict(sv_dir, rates_sr_vs_transformed, 'T(SB12) vs SR')

        with open(f'{sv_dir}/rates.pkl', 'wb') as f:
            pickle.dump([rates_sr_qcd_vs_anomalies, rates_sr_vs_transformed], f)

    if summary_writer_passed:
        # summary_writer.add_hparams(vars(args), dict(auc_dict, **rates_sr_qcd_vs_anomalies))
        summary_writer.add_hparams(vars(args), auc_dict)

    with open(f'{sv_dir}/aucs.pkl', 'wb') as f:
        pickle.dump(auc_dict, f)
