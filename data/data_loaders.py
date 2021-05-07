import torch

from .physics_datasets import JetsDataset, WrappingCurtains, Curtains, CurtainsTrainSet

import os
import pandas as pd
import numpy as np
import h5py

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
    directory = '/srv/beegfs/scratch/groups/rodem/anomalous_jets/data/'
    nchunks = 6
    lo_obs = np.empty((nchunks, 180000, 11))
    nlo_obs = np.empty((nchunks, 180000, 11))
    for i in range(nchunks):
        with h5py.File(directory + f"20210430_{sm}_450_1200_nevents_1M/merged_selected_{i}.h5", 'r') as readfile:
            fill_array(lo_obs[i], readfile["objects/jets/jet1_obs"][:], dtype)
            fill_array(nlo_obs[i], readfile["objects/jets/jet2_obs"][:], dtype)

    low_level_names = ['pt', 'eta', 'phi', 'mass', 'tau1', 'tau2', 'tau3', 'd12', 'd23', 'ECF2', 'ECF3']
    lo_obs = np.vstack(lo_obs)
    mx = lo_obs[:, 0] != 0
    df = pd.DataFrame(np.hstack((lo_obs[mx], np.vstack(nlo_obs)[mx])),
                        columns=low_level_names + ['nlo_' + nm for nm in low_level_names])
    df = df.dropna()
    return df


def load_curtains():
    df = load_curtains_pd()
    return Curtains(df)


def get_data(dataset, bins=None, quantiles=None, normalize=True, mix_qs=False, flow=False):
    # Using bins and quantiles to separate semantics between separating base on self defined mass bins and quantiles
    if dataset == 'curtains':
        df = load_curtains_pd()
    else:
        raise NotImplementedError('The loader of this dataset has not been implemented yet.')

    dset = Curtains(df)
    features = dset.data

    if bins:
        # Split the data into different datasets based on the binning
        context_feature = df['mass']

        lm = Curtains(df.loc[(context_feature < bins[2]) & (context_feature > bins[1])])
        hm = Curtains(df.loc[(context_feature < bins[4]) & (context_feature > bins[3])])
        training_data = CurtainsTrainSet(lm, hm, mix_qs=mix_qs, stack=flow)

        # Set the normalization factors for the other datasets
        scale = training_data.set_and_get_norm_facts()
        validation_data = Curtains(df.loc[(context_feature > bins[4]) & (context_feature < bins[5])], norm=scale)
        # TODO: we also have this validation set now, it needs to be included
        # validation_data = Curtains(df.loc[(context_feature > bins[0]) & (context_feature < bins[1])], norm=scale)
        signal_data = Curtains(df.loc[(context_feature < bins[3]) & (context_feature > bins[2])], norm=scale)

        drape = WrappingCurtains(training_data, signal_data, validation_data, bins)

    elif quantiles:
        # Split the data into different datasets based on the binning
        # TODO: need to make get_quantiles accept lists as well as ints to have more validation regions
        def get_quantile(ind, norm=None):
            return Curtains(df.loc[dset.get_quantile_mask(quantiles[ind])], norm=norm)

        lm = get_quantile(0)
        hm = get_quantile(2)
        training_data = CurtainsTrainSet(lm, hm, mix_qs=mix_qs, stack=flow)
        # Set the normalization factors for the other datasets
        scale = training_data.set_and_get_norm_facts()
        validation_data = get_quantile(3, norm=scale)
        signal_data = get_quantile(1, norm=scale)
        drape = WrappingCurtains(training_data, signal_data, validation_data, bins)
    else:
        drape = Curtains(df)

    if normalize:
        drape.normalize()

    return drape


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
