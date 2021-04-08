import torch

from data.physics_datasets import HepmassDataset, JetsDataset, WrappingCurtains, Curtains
from data.plane import GaussianDataset
from data.plane import CrescentDataset
from data.plane import CrescentCubedDataset
from data.plane import SineWaveDataset
from data.plane import AbsDataset
from data.plane import SignDataset
from data.plane import FourCircles
from data.plane import DiamondDataset
from data.plane import TwoSpiralsDataset
from data.plane import CheckerboardDataset
from data.plane import CornersDataset
from data.plane import EightGaussiansDataset

from data.hyper_plane import HyperCheckerboardDataset, SparseHyperCheckerboardDataset

import os
import pandas as pd
import numpy as np

# Taken from https://github.com/bayesiains/nsf/blob/master/data/base.py
from utils.io import get_top_dir


def load_plane_dataset(name, num_points, flip_axes=False, scale=True, npad=0, dim=None):
    """Loads and returns a plane dataset.
    Args:
        name: string, the name of the dataset.
        num_points: int, the number of points the dataset should have,
        flip_axes: bool, flip x and y axes if True.
    Returns:
        A Dataset object, the requested dataset.
    Raises:
         ValueError: If `name` an unknown dataset.
    """

    try:
        if dim:
            dataset = {
                'hypercheckerboard': HyperCheckerboardDataset,
                'hypersparsecheckerboard': SparseHyperCheckerboardDataset

            }[name](num_points=num_points, dim=dim, flip_axes=flip_axes)
        else:
            dataset = {
                'gaussian': GaussianDataset,
                'crescent': CrescentDataset,
                'crescent_cubed': CrescentCubedDataset,
                'sine_wave': SineWaveDataset,
                'abs': AbsDataset,
                'sign': SignDataset,
                'four_circles': FourCircles,
                'diamond': DiamondDataset,
                'two_spirals': TwoSpiralsDataset,
                'checkerboard': CheckerboardDataset,
                'corners': CornersDataset,
                'eightgauss': EightGaussiansDataset,
                'hypercheckerboard': HyperCheckerboardDataset

            }[name](num_points=num_points, flip_axes=flip_axes)
        if scale:
            # Scale data to be between zero and one
            # dataset.data = 2 * (dataset.data - dataset.data.min()) / (dataset.data.max() - dataset.data.min()) - 1
            dataset.data = (dataset.data + 4) / 4 - 1
        if npad > 0:
            padder = torch.distributions.uniform.Uniform(torch.zeros(npad), torch.ones(npad), validate_args=None)
            pads = padder.sample([num_points])
            dataset.data = torch.cat((dataset.data, pads), 1)
        return dataset

    except KeyError:
        raise ValueError('Unknown dataset: {}'.format(name))


def load_hepmass(mass='1000', slim=False):
    # Pass mass as a string because of 'all' case
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00347/'
    dir = get_top_dir()

    def get_data(type):
        filename = dir + '/data/downloads/{}_{}.csv'.format(mass, type)
        if not os.path.isfile(filename):
            print('Downloading {} data for mass {}'.format(type, mass))
            df = pd.read_csv(url + '{}_{}.csv.gz'.format(mass, type), compression='gzip', header=0, sep=',')
            df.to_csv(filename, index=False)
        else:
            if slim:
                # For debugging use a slimmed down dataset
                slim_file = dir + '/data/downloads/{}_{}_slim.csv'.format(mass, type)
                if not os.path.isfile(slim_file):
                    df = pd.read_csv(filename)
                    df = df.take(list(range(1000)))
                    df.to_csv(slim_file, index=False)
                else:
                    df = pd.read_csv(slim_file)
            else:
                df = pd.read_csv(filename)
        # Drop the labels TODO: why are these not binary?
        df = df.drop('# label', axis=1)
        # Drop any NaN rows
        df = df.dropna()
        # # Take only the low level features
        # df = df[df.columns[:21]]
        data = HepmassDataset(df)
        return data

    return get_data('train'), get_data('test')


def load_jets(sm='QCD', split=0.1, normalize=True, dtype='float32'):
    """
    This will return a training and a test dataset for the JETS dataset generated for RODEM.
    :param sm: The type of the jet, QCD, WZ or tt
    :param split: The fraction of the events to take as a test set
    :return:
    """
    import h5py as h5py

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


def load_curtains():
    nfeatures = 5

    df = pd.read_hdf('/srv/beegfs/scratch/groups/dpnc/atlas/AnomalousJets/final_jj_1MEvents_substructure.h5')
    data = np.zeros((df.shape[0], nfeatures + 1))

    # TODO: this is really just a useful utility
    def rm_nan_features(array):
        mx = ~np.any(np.isnan(data), 1)
        return array[mx]

    # The last data feature is always the context, this could/should be handled by the class
    data[:, 0] = df['tau3s'] / df['taus']
    data[:, 1] = df['tau3s'] / df['tau2s']
    data[:, 2] = df['Qws']
    data[:, 3] = df['d34s']
    data[:, 5] = df['m']

    data = rm_nan_features(data)

    return Curtains(data)


def get_data(dataset, bins, normalize=True):
    if dataset == 'curtains':
        data = load_curtains()
    else:
        raise NotImplementedError('The loader of this dataset has not been implemented yet.')

    if normalize:
        data.normalize()

    # Split the data into different datasets based on the binning
    context_feature = data[:, -1]
    # data = data[:, :-1]
    validation_data = data[(context_feature < bins[0]) | (context_feature > bins[-1])]
    signal_data = data[(context_feature < bins[2]) & (context_feature > bins[1])]
    training_data = data[((context_feature < bins[1]) & (context_feature > bins[0])) | (
            (context_feature < bins[-1]) & (context_feature > bins[1]))]
    return WrappingCurtains(training_data, signal_data, validation_data, bins)


# A class for generating data for plane datasets.
class data_handler():
    def __init__(self, nsample, batch_size, latent_dim, dataset, device):
        self.nsample = nsample
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.dataset = dataset
        self.device = device
        self.dim = None if not dataset[:5] == 'hyper' else self.latent_dim
        self.bounded = load_plane_dataset(self.dataset, 1, dim=self.dim).bounded
        self.scale = 1.
        self.update_data()
        self.nsteps = int(self.nsample / self.batch_size)
        # TODO: pass a steps valid parameter to define this properly
        self.nval = int(self.nsample / 10)
        self.nsteps_val = int(self.nval / self.batch_size)

    def update_data(self):
        trainset = load_plane_dataset(self.dataset, self.nsample, dim=self.dim)
        self.data = trainset.data.to(self.device).view(-1, self.batch_size, self.latent_dim) * self.scale

    def update_validation(self):
        trainset = load_plane_dataset(self.dataset, int(self.nsample / 10), dim=self.dim)
        self.valid = trainset.data.to(self.device).view(-1, self.batch_size, self.latent_dim) * self.scale

    def get_data(self, i):
        # On the start of each epoch generate new samples, and then for each proceeding epoch iterate through the data
        if i == 0:
            self.update_data()
        return self.data[i]

    def get_val_data(self, i):
        # On the start of each epoch generate new samples, and then for each proceeding epoch iterate through the data
        if i == 0:
            self.update_validation()
        return self.valid[i]


def main():
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('--download', type=int, default=0,
                        help='Choose the base output directory')

    args = parser.parse_args()

    if args.download:
        load_hepmass(mass='1000')
        load_hepmass(mass='all')
        load_hepmass(mass='not1000')

    data_train, data_test = load_hepmass(mass='1000', slim=True)

    fig, axs_ = plt.subplots(9, 3, figsize=(5 * 3 + 2, 5 * 9 + 2))
    axs = fig.axes
    for i, data in enumerate(data_train.data.t()):
        axs[i].hist(data.numpy())
    fig.savefig(get_top_dir() + '/images/hepmass_features.png')

    return 0


if __name__ == '__main__':
    main()
