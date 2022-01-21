import argparse
import os
import pdb
import pickle
import random

import pandas as pd
import numpy as np

import torch

import matplotlib.pyplot as plt

from data.data_loaders import get_data, load_curtains_pd
from utils.DRE import get_auc
from utils.io import get_top_dir, register_experiment
from utils.plotting import plot_rates_dict, hist_features
from utils.torch_utils import shuffle_tensor


def parse_args():
    parser = argparse.ArgumentParser()

    # Saving
    parser.add_argument('-d', '--outputdir', type=str, default='classifier_local',
                        help='Choose the base output directory')
    parser.add_argument('-n', '--outputname', type=str, default='local',
                        help='Set the output name directory')
    parser.add_argument('--load', type=int, default=1, help='Load a model?')

    # Multiple runs
    parser.add_argument('--shift_seed', type=int, default=0,
                        help='Add this number to the fixed seed.')

    # Classifier set up

    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='curtains', help='The dataset to train on.')
    parser.add_argument("--bins", type=str, default='2900,3100,3300,3700,3900,4100')
    parser.add_argument("--feature_type", type=int, default=3)
    parser.add_argument("--split_data", type=int, default=2,
                        help='2 for idealised classifier, 3 for supervised.')
    parser.add_argument("--doping", type=int, default=1000,
                        help='Raw number of signal events to be added into the entire bg spectra.')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=1000, help='Size of batch for training.')
    parser.add_argument('--nepochs', type=int, default=10, help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=0.001, help='Classifier learning rate.')
    parser.add_argument('--wd', type=float, default=0.1, help='Weight Decay, set to None for ADAM.')
    parser.add_argument('--drp', type=float, default=0.0, help='Dropout to apply.')
    parser.add_argument('--width', type=int, default=64, help='Width to use for the classifier.')
    parser.add_argument('--depth', type=int, default=3, help='Depth of classifier to use.')
    parser.add_argument('--batch_norm', type=int, default=0, help='Apply batch norm?')
    parser.add_argument('--layer_norm', type=int, default=0, help='Apply layer norm?')
    parser.add_argument('--use_scheduler', type=int, default=1, help='Use cosine annealing of the learning rate?')

    # Classifier settings
    parser.add_argument('--false_signal', type=int, default=0, help='Add random noise samples to the signal set?')
    parser.add_argument('--use_weight', type=int, default=1, help='Apply weights to the data?')
    parser.add_argument('--beta_add_noise', type=float, default=0.1,
                        help='The value of epsilon to use in the 1-e training.')

    return parser.parse_args()


def test_classifier():
    args = parse_args()

    seed = 42 + args.shift_seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    top_dir = get_top_dir()
    sv_dir = top_dir + f'/images/{args.outputdir}_{args.outputname}/'
    if not os.path.exists(sv_dir):
        os.makedirs(sv_dir, exist_ok=True)
    nm = args.outputname
    register_experiment(top_dir, f'{args.outputdir}_{args.outputname}/{args.outputname}', args)

    curtains_bins = args.bins.split(",")
    curtains_bins = [int(b) for b in curtains_bins]
    args.bins = curtains_bins

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    bg_truth_labels = None

    if args.split_data < 2:
        datasets, signal_anomalies = get_data(args.dataset, sv_dir + nm, bins=args.bins, doping=args.doping,
                                              feature_type=args.feature_type)
        training_data = shuffle_tensor(datasets.signalset.data)
    else:
        betas_to_scan = [0.0]
        # Load the data and dope appropriately
        sm = load_curtains_pd(feature_type=args.feature_type)
        sm = sm.sample(frac=1)
        ad = load_curtains_pd(sm='WZ_allhad_pT', feature_type=args.feature_type)
        ad = ad.sample(frac=1)

        # Bin the data
        def mx_data(data, bins):
            context_df = data['mjj']
            mx = (context_df >= bins[0]) & (context_df < bins[1])
            return data.loc[mx], data.loc[~mx]

        if args.split_data == 2:
            # Idealised anomaly detection
            # Split the anomalies into two so that the doping fraction is the same (SR will be split into two)
            ad_extra = ad.iloc[int(args.doping / 2):]
            ad = ad.iloc[:int(args.doping / 2)]

            sr_bin = [args.bins[2], args.bins[3]]
            sm, sm_out = mx_data(sm, sr_bin)
            ad, ad_out = mx_data(ad, sr_bin)
            ad_extra, ad_extra_out = mx_data(ad_extra, sr_bin)

            # Need to figure out how much data there is in the SBs to figure out how much signal to add to the signal
            # template
            sbs = [[args.bins[1], args.bins[2]], [args.bins[3], args.bins[4]]]
            fracs = []
            for bns in sbs:
                sm_d, _ = mx_data(sm_out, bns)
                ad_d, _ = mx_data(ad_out, bns)
                fracs += [len(ad_d) / len(sm_d)]
            frac = np.mean(fracs)
            n_to_bg = int(frac * len(sm) / 2)
            ad_bg = ad_extra.iloc[:n_to_bg]
            ad_extra = ad_extra.iloc[n_to_bg:]

            ndata = int(len(sm) / 2)
            data_to_dope = pd.concat((sm.iloc[:ndata], ad)).dropna()
            # undoped_data = pd.concat((sm.iloc[ndata:], ad_bg)).dropna()
            undoped_data = sm.iloc[ndata:].dropna()
            # This is ordered from undoped data to data to dope
            bg_truth_labels = torch.cat((
                torch.ones(len(sm.iloc[:ndata].dropna())),
                torch.zeros(len(ad.dropna())),
                torch.ones(len(undoped_data))
            ))
        else:
            # Supervised classifier
            sr_bin = [args.bins[2], args.bins[3]]
            sm, _ = mx_data(sm, sr_bin)
            ad, _ = mx_data(ad, sr_bin)
            ntake = int(3 * ad.shape[0] / 4)
            ad_extra = ad.iloc[ntake:]
            ad = ad.iloc[:ntake]
            ad_extra, _ = mx_data(ad_extra, sr_bin)
            data_to_dope = ad.dropna()
            undoped_data = sm.dropna()
            ad_extra = ad_extra.dropna()

        dtype = torch.float32
        names = undoped_data.keys()

        data_to_dope = torch.tensor(data_to_dope.to_numpy()).type(dtype)
        undoped_data = torch.tensor(undoped_data.to_numpy()).type(dtype)
        signal_anomalies = torch.tensor(ad_extra.to_numpy()).type(dtype)
        pure_noise = False

        n_feature = data_to_dope.shape[1] - 1
        labels = ['to dope', 'doped']

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(n_feature, 1, figsize=(7, 5 * n_feature))
        # hist_features(data_to_dope, undoped_data, n_feature, axes, labels=labels, axs_nms=list(names))
        hist_features(signal_anomalies, data_to_dope, n_feature, axes, labels=labels, axs_nms=list(names))
        os.makedirs(f'images/{args.outputdir}', exist_ok=True)
        fig.savefig(os.path.join(sv_dir, 'features.png'))

    if args.split_data == 1:
        betas_to_scan = [0.0, 0.5, 1, 5, 15]
        data_to_dope, undoped_data = torch.split(training_data, int(len(training_data) / 2))
        pure_noise = False
        pdb.set_trace()
    elif args.split_data == 0:
        betas_to_scan = [0]
        args.false_signal = 0
        undoped_data = training_data
        l1 = undoped_data.min()
        l2 = undoped_data.max()
        data_to_dope = (l1 - l2) * torch.rand_like(undoped_data) + l2
        pure_noise = True

    rates_sr_vs_transformed = {}
    rates_sr_qcd_vs_anomalies = {}
    for beta in betas_to_scan:
        auc_info = get_auc(undoped_data, data_to_dope, sv_dir, nm + f'{beta}%Anomalies',
                           anomaly_data=signal_anomalies.data.to(device),
                           sup_title=f'QCD in SR doped with {beta:.3f}% anomalies', load=args.load, return_rates=True,
                           false_signal=args.false_signal, batch_size=args.batch_size, nepochs=args.nepochs, lr=args.lr,
                           wd=args.wd, drp=args.drp, width=args.width, depth=args.depth, batch_norm=args.batch_norm,
                           layer_norm=args.layer_norm, use_scheduler=args.use_scheduler, use_weights=args.use_weight,
                           beta_add_noise=args.beta_add_noise, pure_noise=pure_noise, bg_truth_labels=bg_truth_labels)

        rates_sr_vs_transformed[f'{beta}'] = auc_info[3]
        rates_sr_qcd_vs_anomalies[f'{beta}'] = auc_info[2]

    plot_rates_dict(sv_dir, rates_sr_qcd_vs_anomalies, 'SR QCD vs SR Anomalies')
    plot_rates_dict(sv_dir, rates_sr_vs_transformed, 'T(SB12) vs SR')

    with open(f'{sv_dir}/rates.pkl', 'wb') as f:
        pickle.dump([rates_sr_qcd_vs_anomalies, rates_sr_vs_transformed], f)


if __name__ == '__main__':
    test_classifier()
