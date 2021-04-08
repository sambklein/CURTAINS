# Check to see if samples from within are quantile are nor further away from each other than samples in any other quantile

import numpy as np
from utils.hyperparams import get_measure

from data.data_loaders import load_curtains_pd, get_features


# TODO: all of the methods here should be a part of the data class
def get_quantile(df, quantile):
    # Returns a numpy array of the training features, plus the context feature on the end
    features = get_features(df)
    mx = df['mass_q{}'.format(quantile)]
    return features[np.array(mx, dtype='bool')]


def random_choice(data, nsamples):
    # Get indicies to draw random samples from data
    indices = np.random.permutaion(len(data))
    return data[indices[:nsamples]]


def get_average_distance(data1, data2, sample_size, ncheck, measure):
    # TODO: remove this for loop
    dist = np.zeros(ncheck)
    for i in range(ncheck):
        s1 = random_choice(data1, sample_size)
        s2 = random_choice(data2, sample_size)
        dist[i] = measure(s1, s2)

    return dist


def main():
    # Returns a pandas array with all the data, NaNs dropped
    data = load_curtains_pd()
    # Returns a measure that can be called on two sets of samples to calculate the named distance
    measure = get_measure('sinkhorn')

    # The size of the batches to check distances between
    sample_size = 100
    # The number of times to calculate the distances between samples of size sample_size
    ncheck = 100

    # Get the average distance within a given quantile
    nquantiles = 10
    av_dist_in = np.zeros(nquantiles)
    for i in range(nquantiles):
        q = get_quantile(data, i)
        av_dist_in[i] = get_average_distance(q, q, sample_size, ncheck, measure)

    # Get the average distance between adjacent quantiles
    av_dist_inter = np.zeros(nquantiles - 1)
    for i in range(nquantiles - 1):
        q = get_quantile(data, i)
        q1 = get_quantile(data, i + 1)
        av_dist_inter[i] = get_average_distance(q, q1, sample_size, ncheck, measure)


if __name__ == '__main__':
    main()
