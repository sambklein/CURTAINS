import torch

from utils.plotting import hist_features
from utils.torch_utils import sample_data
from .physics_datasets import JetsDataset, WrappingCurtains, Curtains, CurtainsTrainSet

import os
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt

# Taken from https://github.com/bayesiains/nsf/blob/master/data/base.py
from utils.io import get_top_dir, on_cluster


def load_jets(sm='QCD', split=0.1, normalize=True, dtype='float32'):
    """
    This will return a training and a test dataset for the JETS dataset generated for RODEM.
    :param sm: The type of the jet, QCD, WZ or tt
    :param split: The fraction of the events to take as a test set
    :return:
    """

    dir = '/srv/beegfs/scratch/groups/rodem/AnomalyDetection/HEP/jets/'

    with h5py.File(dir + "{}dijet100k.h5".format(sm), 'r') as readfile:
        # with h5py.File(dir + "output.h5", 'r') as readfile:
        lo_obs = np.array(readfile["objects/jets/lo_obs"][:], dtype=dtype)
        nlo_obs = np.array(readfile["objects/jets/nlo_obs"][:], dtype=dtype)
        lo_const = np.array(readfile["objects/jets/lo_constituents"][:], dtype=dtype)
        nlo_const = np.array(readfile["objects/jets/nlo_constituents"][:], dtype=dtype)

    if split == 0:
        return JetsDataset(lo_obs, nlo_obs, lo_const, nlo_const)
    nevents = lo_obs.shape[0]
    nsample = int(split * nevents)
    indices = np.random.choice(nevents, size=nsample, replace=False)
    objects = (lo_obs, nlo_obs, lo_const, nlo_const)
    trainset = JetsDataset(*[np.delete(array, indices, 0) for array in objects])
    testset = JetsDataset(*[array[indices, :] for array in objects], scale=[trainset.max_vals, trainset.min_vals])

    if normalize:
        trainset.normalize()
        testset.normalize()
        print(trainset.data.max())
        print(testset.data.max())

    return trainset, testset


# def load_curtains_pd():
#     # TODO: make this a wrapper for loading generic pandas files
#     data_dir = '/srv/beegfs/scratch/groups/dpnc/atlas/AnomalousJets'
#     slim_dir = get_top_dir() + '/data/slims'
#     filename = 'final_jj_1MEvents_substructure.h5'
#     data_file = data_dir + '/' + filename
#     slim_file = slim_dir + '/' + filename
#     # If you aren't on the cluster load a local slim version for testing
#     if on_cluster():
#         df = pd.read_hdf(data_file)
#         # If you are on the cluster and the slim file doesn't exist, make it
#         if not os.path.isfile(slim_file):
#             df_sv = df.take(list(range(5000)))
#             os.makedirs(slim_dir)
#             df_sv.to_csv(slim_file, index=False)
#     else:
#         df = pd.read_csv(slim_file)
#     return df.dropna()

# TODO: this is stupid code duplication from the pytorch-utils repo in the anomaly tools.
def fill_array(to_fill, obj, dtype):
    arr = np.array(obj, dtype=dtype)
    to_fill[:len(arr)] = arr


def load_curtains_pd(sm='QCDjj_pT', dtype='float32'):
    if on_cluster():

        directory = '/srv/beegfs/scratch/groups/rodem/anomalous_jets/data/'
        nchunks = 6 if sm[:3] == 'QCD' else 5
        lo_obs = np.empty((nchunks, 190000, 11))
        nlo_obs = np.empty((nchunks, 190000, 11))
        for i in range(nchunks):
            with h5py.File(directory + f"20210430_{sm}_450_1200_nevents_1M/merged_selected_{i}.h5", 'r') as readfile:
                fill_array(lo_obs[i], readfile["objects/jets/jet1_obs"][:], dtype)
                fill_array(nlo_obs[i], readfile["objects/jets/jet2_obs"][:], dtype)

        low_level_names = ['pt', 'eta', 'phi', 'mass', 'tau1', 'tau2', 'tau3', 'd12', 'd23', 'ECF2', 'ECF3']
        lo_obs = np.vstack(lo_obs)
        mx = lo_obs[:, 0] != 0
        df = pd.DataFrame(np.hstack((lo_obs[mx], np.vstack(nlo_obs)[mx])),
                          columns=low_level_names + ['nlo_' + nm for nm in low_level_names])

        slim_file = get_top_dir() + f'/data/slims/{sm}.csv'
        if not os.path.isfile(slim_file):
            if not os.path.isfile(slim_file):
                df_sv = df.take(list(range(10000)))
                os.makedirs(get_top_dir() + '/data/slims/', exist_ok=True)
                df_sv.to_csv(slim_file, index=False)

        return df

    else:
        slim_dir = get_top_dir() + '/data/slims'
        filename = f'{sm}.csv'
        slim_file = slim_dir + '/' + filename
        return pd.read_csv(slim_file)


def load_curtains():
    df = load_curtains_pd()
    return Curtains(df)


def get_bin(process, bin, trainset=None, normalize=True):
    df = load_curtains_pd(sm=process)
    context_feature = df['mass']
    df = df.loc[(context_feature < bin[1]) & (context_feature > bin[0])]
    if trainset is None:
        return df
    else:
        data = Curtains(df, norm=[trainset.max_vals, trainset.min_vals])
        if normalize:
            data.normalize()
        return data


def dope_dataframe(undoped, anomaly_data, doping):
    n = int(len(undoped) * doping)
    if len(anomaly_data) < n:
        raise Exception('Not enough anomalies in this region for this level of doping.')
    anomaly_data = anomaly_data.sample(frac=1)
    mixing_anomalies = anomaly_data.iloc[:n]
    anomaly_data = anomaly_data.iloc[n:]
    df = pd.concat((mixing_anomalies, undoped), 0)
    df = df.sample(frac=1)
    return anomaly_data, df


def mask_dataframe(df, context_feature, bins, indx, doping=None, anomaly_data=None):

    def mx_data(data):
        context_df = data[context_feature]
        mx = (context_df > bins[indx[0]]) & (context_df < bins[indx[1]])
        return data.loc[mx]

    undoped_df = mx_data(df)
    anomaly_data = mx_data(anomaly_data)

    if doping is not None:
        remaining_anomalies, df = dope_dataframe(undoped_df, anomaly_data, doping)
    else:
        remaining_anomalies = anomaly_data
        df = undoped_df

    return remaining_anomalies, df


def get_data(dataset, sv_nm, bins=None, normalize=True, mix_qs=False, flow=False,
             anomaly_process='WZ_allhad_pT', doping=None):
    # Using bins and quantiles to separate semantics between separating base on self defined mass bins and quantiles
    if dataset == 'curtains':
        df = load_curtains_pd()
    else:
        raise NotImplementedError('The loader of this dataset has not been implemented yet.')

    if bins:
        # Split the data into different datasets based on the binning
        context_feature = 'mass'

        anomaly_data = load_curtains_pd(sm=anomaly_process)

        lm_anomalies, lm = mask_dataframe(df, context_feature, bins, [1, 2], doping, anomaly_data)
        hm_anomalies, hm = mask_dataframe(df, context_feature, bins, [3, 4], doping, anomaly_data)
        ob1_anomalies, ob1 = mask_dataframe(df, context_feature, bins, [0, 1], doping, anomaly_data)
        ob2_anomalies, ob2 = mask_dataframe(df, context_feature, bins, [4, 5], doping, anomaly_data)
        signal_anomalies, signal = mask_dataframe(df, context_feature, bins, [2, 3], doping, anomaly_data)

        lm = Curtains(lm)
        hm = Curtains(hm)

        # Take a look at the input features prior to scaling
        nfeatures = len(lm.feature_nms) - 1
        fig, axs = plt.subplots(1, nfeatures, figsize=(5 * nfeatures + 2, 5))
        hist_features(lm, hm, nfeatures, axs, axs_nms=lm.feature_nms, labels=['SB1', 'SB2'], legend=False)
        print(sv_nm + '_inital_features.png')
        for i in range(nfeatures):
            axs[i].vlines(lm[:, i].max().item(), 0, 0.5, label='SB1', colors='r')
            axs[i].vlines(hm[:, i].max().item(), 0, 0.5, label='SB2', colors='b')
            axs[i].vlines(lm[:, i].min().item(), 0, 0.5, colors='r')
            axs[i].vlines(hm[:, i].min().item(), 0, 0.5, colors='b')
        # (lm[:, 1] > hm[:, 1].max()).sum(), 3 variables for our outlier!! TODO: implement this counting
        handles, labels = axs[i].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        fig.savefig(sv_nm + '_inital_features.png')

        training_data = CurtainsTrainSet(lm, hm, mix_qs=mix_qs, stack=flow)

        # Set the normalization factors for the other datasets
        scale = training_data.set_and_get_norm_facts()
        validation_data_lm = Curtains(ob1, norm=scale)
        validation_data = Curtains(ob2, norm=scale)
        signal_data = Curtains(signal, norm=scale)
        signal_anomalies = Curtains(signal_anomalies, norm=scale)

        drape = WrappingCurtains(training_data, signal_data, validation_data, validation_data_lm, bins)

    else:
        return Curtains(df)

    if normalize:
        drape.normalize()
        signal_anomalies.normalize()

    return drape, signal_anomalies


def main():
    # import argparse
    # import matplotlib.pyplot as plt
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--download', type=int, default=0,
    #                     help='Choose the base output directory')
    #
    # args = parser.parse_args()
    #
    # if args.download:
    #     load_hepmass(mass='1000')
    #     load_hepmass(mass='all')
    #     load_hepmass(mass='not1000')
    #
    # data_train, data_test = load_hepmass(mass='1000', slim=True)
    #
    # fig, axs_ = plt.subplots(9, 3, figsize=(5 * 3 + 2, 5 * 9 + 2))
    # axs = fig.axes
    # for i, data in enumerate(data_train.data.t()):
    #     axs[i].hist(data.numpy())
    # fig.savefig(get_top_dir() + '/images/hepmass_features.png')

    return 0


if __name__ == '__main__':
    main()
