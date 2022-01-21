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
from data.physics_datasets import Curtains, ClassifierData, minimum_validation_loss_models
from utils.CATHODE_classifier import train_n_models, preds_from_models, full_single_evaluation
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

    # rates_sr_vs_transformed = {}
    # rates_sr_qcd_vs_anomalies = {}
    # for beta in betas_to_scan:
    #     auc_info = get_auc(undoped_data, data_to_dope, sv_dir, nm + f'{beta}%Anomalies',
    #                        anomaly_data=signal_anomalies.data.to(device),
    #                        sup_title=f'QCD in SR doped with {beta:.3f}% anomalies', load=args.load, return_rates=True,
    #                        false_signal=args.false_signal, batch_size=args.batch_size, nepochs=args.nepochs, lr=args.lr,
    #                        wd=args.wd, drp=args.drp, width=args.width, depth=args.depth, batch_norm=args.batch_norm,
    #                        layer_norm=args.layer_norm, use_scheduler=args.use_scheduler, use_weights=args.use_weight,
    #                        beta_add_noise=args.beta_add_noise, pure_noise=pure_noise, bg_truth_labels=bg_truth_labels)
    #
    #     rates_sr_vs_transformed[f'{beta}'] = auc_info[3]
    #     rates_sr_qcd_vs_anomalies[f'{beta}'] = auc_info[2]
    #
    # plot_rates_dict(sv_dir, rates_sr_qcd_vs_anomalies, 'SR QCD vs SR Anomalies')
    # plot_rates_dict(sv_dir, rates_sr_vs_transformed, 'T(SB12) vs SR')
    #
    # with open(f'{sv_dir}/rates.pkl', 'wb') as f:
    #     pickle.dump([rates_sr_qcd_vs_anomalies, rates_sr_vs_transformed], f)

    stack = torch.cat((undoped_data, data_to_dope, signal_anomalies), 0)
    stack = ClassifierData(stack, 1)
    stack.preprocess()
    undoped_data = stack.data[:len(undoped_data)]
    data_to_dope = stack.data[len(undoped_data):len(undoped_data) + len(data_to_dope)]
    signal_anomalies = stack.data[len(undoped_data) + len(data_to_dope):]

    # First thing you need to do is put the mass first
    feature_filter = [-1, 0, 1, 2, 3]
    X_train = torch.cat((undoped_data[:, feature_filter], data_to_dope[:, feature_filter]), 0)
    # Then we are going to treat the undoped data as the samples/bg template
    y_train = torch.cat((torch.zeros(len(undoped_data)), torch.ones(len(data_to_dope))), 0)
    bg_labls = 1 - bg_truth_labels
    X_train = torch.cat((X_train, y_train.view(-1, 1), bg_labls.view(-1, 1)), 1)
    X_train[:, 0] /= 1000

    # Append additional signal
    # lbs = torch.ones(len(signal_anomalies)).view(-1, 1)
    lbs = torch.ones(len(signal_anomalies)).view(-1, 1)
    add_anomalies = torch.cat((signal_anomalies[:, feature_filter],
                               lbs, lbs), 1)
    X_test = torch.cat((X_train[y_train == 0], add_anomalies), 0).numpy()
    y_test = torch.cat((torch.ones(sum(y_train == 0)).view(-1, 1), lbs)).numpy()[:, 0]
    X_test[:, -2] = y_test

    X_train = X_train.numpy()
    y_train = y_train.numpy()

    # our_data = X_train
    # our_labels = y_train
    # our_test = X_test
    #
    # # Load the Cathode data and manually build the datasets
    # data_dir = 'data/downloads/separated_data/'
    # # def process_cathode_data():
    # inner_train = np.load(os.path.join(data_dir, 'innerdata_train.npy')).astype(np.float32)
    # bg = inner_train[inner_train[:, -1] == 0]
    # signal = inner_train[inner_train[:, -1] == 1]
    # ntake = int(len(bg) / 2)
    # doped = np.concatenate((bg[:ntake], signal))
    # bg = bg[ntake:]
    # X_train = np.concatenate((bg, doped))
    # y_train = np.concatenate((np.zeros(len(bg)), np.ones(len(doped)))).astype(np.float32)
    # X_train = np.insert(X_train, -1, y_train, axis=1)
    #
    # X_test = np.load(os.path.join(data_dir, 'innerdata_test.npy')).astype(np.float32)
    # y_test = np.ones(len(X_test)).astype(np.float32)
    # X_test = np.insert(X_test, -1, y_test, axis=1)
    #
    # Xtr = ClassifierData(torch.tensor(X_train[:, 1:-2], dtype=torch.float32), 1)
    # Xte = ClassifierData(torch.tensor(X_test[:, 1:-2], dtype=torch.float32), 1)
    # Xtr.preprocess()
    # Xte.preprocess(Xtr.get_preprocess_info())
    # X_train[:, 1:-2] = Xtr.data.numpy()
    # X_test[:, 1:-2] = Xte.data.numpy()
    #
    # import matplotlib.pyplot as plt
    # nbins = 50
    # density = True
    #
    # data_to = X_test
    # labels_to = X_test[:, -1]
    # data_two = our_test
    # labels_two = our_test[:, -1]
    #
    # n_features = X_train.shape[1]
    # fig, ax = plt.subplots(2, n_features, figsize=(5 * n_features, 14))
    # for i in range(n_features):
    #     data = data_to[:, i]
    #     data_t = data_two[:, i]
    #     max_ent = data.max().item()
    #     min_ent = data.min().item()
    #     bins = np.linspace(min_ent, max_ent, num=nbins)
    #     ax[0, i].hist(data[labels_to == 0], label='Correct', alpha=0.5, density=density, bins=bins,
    #                   histtype='step')
    #     # Plot samples drawn from the model
    #     ax[0, i].hist(data_t[labels_two == 0], label='Ours', alpha=0.5, density=density, bins=bins, histtype='step')
    #
    #     ax[1, i].hist(data[labels_to == 1], label='Correct', alpha=0.5, density=density, bins=bins,
    #                   histtype='step')
    #     # Plot samples drawn from the model
    #     ax[1, i].hist(data_t[labels_two == 1], label='Ours', alpha=0.5, density=density, bins=bins, histtype='step')
    #     if i == n_features - 2:
    #         x = 1
    # handles, labels = ax[1, i].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.9, 0.89), frameon=False)
    # fig.savefig(f'{sv_dir}/feature_comparisons.png')

    # TODO: a useful function for loading and plotting saved SIC curves
    # plt.figure()
    # with open(f'{sv_dir}rates.pkl', 'rb') as f:
    #     rates = pickle.load(f)[0]['0.0']
    # fpr, tpr = rates
    # fpr_nz = fpr[fpr != 0.]
    # tpr_nz = tpr[fpr != 0.]
    # plt.plot(tpr_nz, tpr_nz / (fpr_nz ** 0.5))
    # plt.savefig(f'{sv_dir}/new_sic.png')

    # # This will load the CATHODE data directly
    # data_dir = sv_dir
    # X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    # X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    # y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    # y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

    import matplotlib.pyplot as plt
    nbins = 50
    data_to = X_train
    labels_to = y_train
    data_two = X_test
    labels_two = y_test
    n_features = X_train.shape[1]
    fig, ax = plt.subplots(2, n_features, figsize=(5 * n_features, 14))
    for i in range(n_features):
        data = data_to[:, i]
        data_t = data_two[:, i]
        max_ent = data.max().item()
        min_ent = data.min().item()
        bins = np.linspace(min_ent, max_ent, num=nbins)
        ax[0, i].hist(data[labels_to == 0], label='label 0', alpha=0.5, density=False, bins=bins,
                      histtype='step')
        # Plot samples drawn from the model
        ax[0, i].hist(data[labels_to == 1], label='label 0', alpha=0.5, density=False, bins=bins, histtype='step')

        ax[1, i].hist(data_t[labels_two == 0], label='label 0', alpha=0.5, density=False, bins=bins,
                      histtype='step')
        # Plot samples drawn from the model
        ax[1, i].hist(data_t[labels_two == 1], label='label 0', alpha=0.5, density=False, bins=bins, histtype='step')
    fig.savefig(f'{sv_dir}/test.png')

    loss_matris, val_loss_matris = train_n_models(
        1, 'utils/classifier.yml', args.nepochs, X_train, y_train, X_test, y_test,
        batch_size=args.batch_size,
        supervised=False, verbose=False,
        savedir=sv_dir, save_model=f'{sv_dir}model')

    model_paths = minimum_validation_loss_models(sv_dir, n_epochs=10)
    _ = preds_from_models(model_paths, X_test, sv_dir)

    _ = full_single_evaluation(sv_dir, X_test, n_ensemble_epochs=10, sic_range=(0, 20),
                               savefig=os.path.join(sv_dir, 'result_SIC'))


if __name__ == '__main__':
    test_classifier()
