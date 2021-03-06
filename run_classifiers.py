import argparse
import glob
import json
import os
import pdb
import pickle
import random

import pandas as pd
import numpy as np

import torch

import matplotlib.pyplot as plt

from data.data_loaders import get_data, load_curtains_pd
from utils.DRE import run_classifiers
from utils.io import get_top_dir, register_experiment
from utils.plotting import plot_rates_dict, hist_features
from utils.torch_utils import shuffle_tensor


def parse_args():
    parser = argparse.ArgumentParser()

    # Saving
    parser.add_argument('-d', '--outputdir', type=str, default='bump_200_alt_final',
                        help='Choose the base output directory')
    parser.add_argument('-n', '--outputname', type=str, default='bump_200_alt_final',
                        help='Set the output name directory')
    parser.add_argument('--load', type=int, default=2, help='Load a model?')

    # Multiple runs
    parser.add_argument('--shift_seed', type=int, default=0,
                        help='Add this number to the fixed seed.')

    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='curtains', help='The dataset to train on.')
    # parser.add_argument("--bins", type=str, default='2900,3100,3200,3800,3900,4100')
    # parser.add_argument("--bins", type=str, default='2900,3100,3300,3700,3900,4100')
    parser.add_argument("--bins", type=str, default='3000,3200,3400,3600,3800,4000')
    parser.add_argument("--feature_type", type=int, default=3)
    parser.add_argument("--doping", type=int, default=667,
                        help='Raw number of signal events to be added into the entire bg spectra.')
    # parser.add_argument("--doping", type=int, default=1800,
    #                     help='Raw number of signal events to be added into the entire bg spectra.')
    parser.add_argument("--split_data", type=int, default=2,
                        help='2 for idealised classifier, 3 for supervised.')
    parser.add_argument("--match_idealised", type=int, default=1,
                        help='If set to 1 only provide as many signal samples as there are in the idealised training'
                             'to the supervised training.')

    # parser.add_argument("--data_directory", type=str,
    #                     default='/home/users/k/kleins/MLproject/CURTAINS/images/ot_fig7_200_OT_fig7_200_7',
    #                     help='The directory within which to search for data.')
    parser.add_argument("--data_directory", type=str,
                        default='/home/users/k/kleins/MLproject/CURTAINS/images/cathode_test_cathode_test_f_0',
                        help='The directory within which to search for data.')
    parser.add_argument("--data_file", type=str, default='SB2_to_SR_samples.npy',
                        help='The file to load and train against within data_directory.')
    parser.add_argument("--nx_oversample", type=int, default=-1,
                        help='The number of times to oversample relative to the number of events in the signal region.')
    parser.add_argument("--nx_bg_template", type=int, default=-1,
                        help='The number of bg template samples to take.')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=128, help='Size of batch for training.')
    parser.add_argument('--nepochs', type=int, default=2, help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=0.001, help='Classifier learning rate.')
    parser.add_argument('--wd', type=float, default=0.1, help='Weight Decay, set to None for ADAM.')
    parser.add_argument('--drp', type=float, default=0.0, help='Dropout to apply.')
    parser.add_argument('--width', type=int, default=32, help='Width to use for the classifier.')
    parser.add_argument('--depth', type=int, default=3, help='Depth of classifier to use.')
    parser.add_argument('--batch_norm', type=int, default=0, help='Apply batch norm?')
    parser.add_argument('--layer_norm', type=int, default=0, help='Apply layer norm?')
    parser.add_argument('--use_scheduler', type=int, default=1, help='Use cosine annealing of the learning rate?')
    parser.add_argument('--run_cathode_classifier', type=int, default=0, help='Use cathode classifier?')
    parser.add_argument('--n_run', type=int, default=1, help='Number of classifiers to train.')

    # Classifier settings
    parser.add_argument('--false_signal', type=int, default=0, help='Add random noise samples to the signal set?')
    parser.add_argument('--use_weight', type=int, default=1, help='Apply weights to the data?')
    parser.add_argument('--beta_add_noise', type=float, default=0.3,
                        help='The value of epsilon to use in the 1-e training.')
    parser.add_argument('--cf_activ', type=str, default='relu',
                        help='The value of epsilon to use in the 1-e training.')
    parser.add_argument('--cf_norm', type=int, default=1,
                        help='2 for normalization and 1 for standardization.')

    return parser.parse_args()


def reset_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def test_classifier():
    args = parse_args()

    # If a data directory is passed, load the log and set the args appropriately.
    # This allows samples to be loaded and a classifier to be trained against them.
    if (args.data_directory != 'none') and (args.split_data == 1):
        print(args.data_directory)
        exp_info = glob.glob(os.path.join(args.data_directory, '*.json'))[0]
        with open(exp_info, "r") as file_name:
            json_dict = json.load(file_name)
        exp_dict = json.loads(json_dict)
        args.doping = exp_dict['doping']
        args.bins = exp_dict['bins']
        args.feature_type = exp_dict['feature_type']
        sb1_samples = np.load(os.path.join(args.data_directory, 'SB1_to_SR_samples.npy'))
        sb2_samples = np.load(os.path.join(args.data_directory, 'SB2_to_SR_samples.npy'))
        bg_template = np.concatenate((sb1_samples, sb2_samples))
        if args.nx_bg_template > 0:
            np.random.shuffle(bg_template)
            bg_template = bg_template[:args.nx_bg_template]
        print(len(bg_template))
        args.outputdir = args.data_directory.split("/")[-1]

    seed = 1638128 + args.shift_seed
    reset_seed(seed)

    top_dir = get_top_dir()
    sv_dir = top_dir + f'/images/{args.outputdir}_{args.outputname}/'
    if not os.path.exists(sv_dir):
        os.makedirs(sv_dir, exist_ok=True)
    nm = args.outputname
    register_experiment(top_dir, f'{args.outputdir}_{args.outputname}/{args.outputname}', args)

    curtains_bins = args.bins.split(",")
    curtains_bins = [int(b) for b in curtains_bins]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the data and dope appropriately
    sm = load_curtains_pd(feature_type=args.feature_type)
    ad = load_curtains_pd(sm='WZ_allhad_pT', feature_type=args.feature_type)
    bg_truth_labels = None

    sm = sm.sample(frac=1).dropna()
    ad = ad.sample(frac=1).dropna()

    # Bin the data
    def mx_data(data, bins):
        context_df = data['mjj']
        mx = (context_df >= bins[0]) & (context_df < bins[1])
        return data.loc[mx], data.loc[~mx]

    if args.split_data == 1:
        if args.match_idealised:
            args.doping = int(args.doping / 2)
        # Select the data
        ad_extra = ad.iloc[args.doping:].to_numpy()
        ad = ad.iloc[:args.doping]
        # ad = ad.sample(frac=1, random_state=seed)
        # Bin the data
        sr_bin = [curtains_bins[2], curtains_bins[3]]
        sm, sm_out = mx_data(sm, sr_bin)
        if args.match_idealised:
            sm = sm.iloc[:int(len(sm) / 2)]
        ad, ad_out = mx_data(ad, sr_bin)
        data_to_dope = pd.concat((sm, ad)).to_numpy()
        undoped_data = bg_template[~np.isnan(bg_template).any(axis=1)]
        if args.nx_oversample > 0:
            mx_inds = np.random.permutation(np.arange(0, len(undoped_data)))
            undoped_data = undoped_data[mx_inds]
            undoped_data = undoped_data[:int(args.nx_oversample * len(data_to_dope))]
        bg_truth_labels = torch.cat((
            -torch.ones(len(undoped_data)),
            torch.zeros(len(sm)),
            torch.ones(len(ad))
        ))

    elif args.split_data == 2:
        bg_frac = 0.5
        # Idealised anomaly detection
        # Split the anomalies into the same fraction as the signal region QCD will be split
        ad_extra = ad.iloc[int(args.doping * bg_frac):]
        ad = ad.iloc[:int(args.doping * bg_frac)]
        ad = ad.sample(frac=1, random_state=seed)

        sr_bin = [curtains_bins[2], curtains_bins[3]]
        sm, sm_out = mx_data(sm, sr_bin)
        ad, ad_out = mx_data(ad, sr_bin)
        ad_extra, ad_extra_out = mx_data(ad_extra, sr_bin)
        ad_extra = ad_extra.to_numpy()

        ndata = int(len(sm) * bg_frac)
        print(len(ad) / ndata * 100)
        print(len(ad))
        print(ndata)
        mx_ind = np.random.permutation(np.arange(0, ndata + len(ad)))
        data_to_dope = pd.concat((sm.iloc[:ndata], ad)).to_numpy()[mx_ind]
        bg_truth = torch.cat((torch.zeros(len(sm.iloc[:ndata])),
                              torch.ones(len(ad))))[mx_ind]
        # undoped_data = pd.concat((sm.iloc[ndata:], ad_bg))
        undoped_data = sm.iloc[ndata:].to_numpy()
        # This is ordered from undoped data to data to dope
        bg_truth_labels = torch.cat((
            torch.zeros(len(undoped_data)),
            bg_truth
        ))
    else:
        # Supervised classifier
        sr_bin = [curtains_bins[2], curtains_bins[3]]

        if args.match_idealised:
            ad_extra = ad.iloc[int(args.doping / 2):]
            ad = ad.iloc[:int(args.doping / 2)]

        sm, _ = mx_data(sm, sr_bin)
        ad, _ = mx_data(ad, sr_bin)

        if args.match_idealised:
            ad_extra, _ = mx_data(ad_extra, sr_bin)
        else:
            ntake = int(3 * ad.shape[0] / 4)
            ad_extra = ad.iloc[ntake:]
            ad = ad.iloc[:ntake]

        data_to_dope = ad.to_numpy()
        undoped_data = sm.to_numpy()
        ad_extra = ad_extra.to_numpy()

        # bg_truth_labels = torch.cat((
        #     torch.zeros(len(undoped_data)),
        #     torch.zeros(len(data_to_dope))
        # ))

    dtype = torch.float32

    data_to_dope = torch.tensor(data_to_dope).type(dtype)
    undoped_data = torch.tensor(undoped_data).type(dtype)
    signal_anomalies = torch.tensor(ad_extra).type(dtype)
    pure_noise = False

    n_feature = data_to_dope.shape[1] - 1
    labels = ['to dope', 'doped']

    # Plot some distributions
    fig, axes = plt.subplots(n_feature, 1, figsize=(7, 5 * n_feature))
    hist_features(data_to_dope, undoped_data, n_feature, axes, labels=labels)
    # hist_features(signal_anomalies, data_to_dope, n_feature, axes, labels=labels)
    os.makedirs(f'images/{args.outputdir}', exist_ok=True)
    fig.savefig(os.path.join(sv_dir, 'features.png'))

    rates_sr_vs_transformed = {}
    rates_sr_qcd_vs_anomalies = {}
    counts = []
    for i in range(args.n_run):
        torch.manual_seed(seed + i)
        np.random.seed(seed + i)
        random.seed(seed + i)

        run_dir = f'{sv_dir}run_{i}/'
        auc_info = run_classifiers(undoped_data, data_to_dope, run_dir, nm + f'Anomalies_no_eps',
                                   anomaly_data=signal_anomalies.data.to(device),
                                   sup_title=f'Idealised anomaly detector.', load=args.load, return_rates=True,
                                   false_signal=args.false_signal, batch_size=args.batch_size, nepochs=args.nepochs, lr=args.lr,
                                   wd=args.wd, drp=args.drp, width=args.width, depth=args.depth, batch_norm=args.batch_norm,
                                   layer_norm=args.layer_norm, use_scheduler=args.use_scheduler, use_weights=args.use_weight,
                                   beta_add_noise=args.beta_add_noise, pure_noise=pure_noise, bg_truth_labels=bg_truth_labels,
                                   run_cathode_classifier=args.run_cathode_classifier, n_run=args.n_run, cf_activ=args.cf_activ,
                                   normalize=args.cf_norm)

        rates_sr_vs_transformed[f'{i}'] = auc_info[3]
        sr_qcd_rates = auc_info[5]
        sr_qcd_rates.pop('random', None)
        for rate in sr_qcd_rates:
            rates_sr_qcd_vs_anomalies[f'{i}_{rate}'] = sr_qcd_rates[rate]
        counts += [auc_info[4]]

    with open(f'{sv_dir}/counts.pkl', 'wb') as f:
        pickle.dump(counts, f)

    with open(f'{sv_dir}/rates.pkl', 'wb') as f:
        pickle.dump([rates_sr_qcd_vs_anomalies, rates_sr_vs_transformed], f)

    plot_rates_dict(sv_dir, rates_sr_qcd_vs_anomalies, 'SR QCD vs SR Anomalies')
    plot_rates_dict(sv_dir, rates_sr_vs_transformed, 'T(SB12) vs SR')


if __name__ == '__main__':
    test_classifier()
