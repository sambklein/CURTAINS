# Check to see if samples from within are quantile are nor further away from each other than samples in any other quantile

import numpy as np
import torch
from utils.io import get_top_dir
import matplotlib.pyplot as plt

from utils.hyperparams import get_measure

from data.physics_datasets import Curtains
from data.data_loaders import load_curtains_pd


# TODO: all of the methods here should be a part of the data class
def get_quantile(df, quantile):
    # Returns a numpy array of the training features, plus the context feature on the end
    features = Curtains.get_features(df)[:, :-1]
    mx = df['mass_q{}'.format(quantile)]
    return features[np.array(mx, dtype='bool')]


def get_average_distance(data1, sample_size, ncheck, measure, data_o=None):
    dist = np.zeros(ncheck)
    # The measure expects tensor inputs
    data1 = torch.tensor(data1)
    for i in range(ncheck):
        # Get indicies to draw random samples from data
        indices = np.random.permutation(len(data1))
        # Get sample one
        s1 = data1[indices[:sample_size]]
        # If data2 is passed then we are comparing between different quantiles
        if not data_o is None:
            indices = np.random.permutation(len(data_o))
            data2 = torch.tensor(data_o)
        # If data2 is not passed then we are comparing within a quantile, and we do not want to have overlapping sets
        else:
            indices = indices[sample_size:]
            data2 = data1
        # Get sample two
        s2 = data2[indices[:sample_size]]
        # Calculate the distance between sample one and sample two
        # Pass batches as (number of batches, batch_size, dimension)
        dist[i] = measure(s1, s2).item()
    return np.mean(dist), np.std(dist)


def main():
    # Returns a pandas array with all the data, NaNs dropped
    data = load_curtains_pd()
    # Returns a measure that can be called on two sets of samples to calculate the named distance
    distance = 'sinkhorn'
    measure = get_measure(distance)

    # The size of the batches to check distances between
    sample_size = 100
    # The number of times to calculate the distances between samples of size sample_size
    ncheck = 100

    # Get the average distance within a given quantile
    nquantiles = 10
    av_dist_in = np.zeros(nquantiles)
    var_dist_in = np.zeros(nquantiles)
    for i in range(nquantiles):
        q = get_quantile(data, i)
        av_dist_in[i], var_dist_in[i] = get_average_distance(q, sample_size, ncheck, measure)

    # Get the average distance between adjacent quantiles
    av_dist_inter = np.zeros(nquantiles - 1)
    var_dist_inter = np.zeros(nquantiles - 1)
    for i in range(nquantiles - 1):
        q = get_quantile(data, i)
        q1 = get_quantile(data, i + 1)
        av_dist_inter[i], var_dist_inter[i] = get_average_distance(q, sample_size, ncheck, measure, q1)

    sv_dir = get_top_dir() + '/images/'
    ax = plt.subplot(111)
    x = list(range(nquantiles))
    ax.errorbar(x[:-1], av_dist_inter, fmt='x', yerr=var_dist_inter, label='inter')
    ax.errorbar(x, av_dist_in, fmt='x', yerr=var_dist_in, label='within')
    ax.set_xlabel('Quantile')
    ax.set_ylabel('Distance')
    ax.legend()
    plt.title(distance)
    plt.xticks(x)
    plt.tight_layout()
    plt.savefig(sv_dir + 'dist_measure_validation_{}.png'.format(sample_size))


if __name__ == '__main__':
    main()
