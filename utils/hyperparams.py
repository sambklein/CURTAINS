import torch
import torch.nn as nn
import torch.distributions as torch_distributions
import nflows
from geomloss import SamplesLoss
from torch.nn import functional as F

from data.data_loaders import load_plane_dataset, load_hepmass
from data.hyper_plane import SparseHyperCheckerboardDataset, HyperCheckerboardDataset
from utils.io import on_cluster


def my_relu(x):
    return F.relu6(x * 6)


activations = {
    'none': nn.Identity(),
    'relu': F.relu,
    'elu': F.elu,
    'leaky_relu': F.leaky_relu,
    'my_relu': my_relu,
    'sigmoid': F.sigmoid,
    'tanh': torch.tanh
}

recon_losses = {
    'none': nn.Identity(),
    'mse': nn.MSELoss(),
    'bce': nn.BCELoss(),
    'sinkhorn': SamplesLoss('sinkhorn', scaling=0.5, blur=0.01, p=1)
}

# A wrapper for nflows distributions that will discard values outside of bound, assumes out of bounds values are unlikely
# Bound is a single value outside of which no samples are drawn
# TODO: this should be made a proper class in the nflows package.
from nflows.distributions.base import Distribution


class rejection_sampler(Distribution):
    def __init__(self, dist, bound=None):
        super().__init__()
        self.sampler = dist
        self.bound = bound

    def sample_with_rejection(self, num, context=None):
        # TODO: need to fix this with context
        # sample = self.sampler.sample(num + 1000, context=context)
        # if context:
        #     sample = sample[torch.all((-self.bound < sample) & (sample < self.bound), -1)]
        # else:
        #     sample = sample[torch.all((-self.bound < sample) & (sample < self.bound), -1)]
        #     sample = sample[:num]
        sample = self.sampler.sample(num, context=context)
        return sample

    def _sample(self, num, context):
        if self.bound:
            # TODO: this should be a while loop or something, for now it isn't important
            sample = self.sample_with_rejection(num, context)
        else:
            sample = self.sampler.sample(num, context)
        return sample

    def _log_prob(self, inputs, context):
        return self.sampler._log_prob(inputs, context)

    def _mean(self, context):
        return self.sampler._mean(context)


def nflows_dists(nm, inp_dim, shift=False, bound=None):
    try:
        tshift = 0
        bshift = 0
        if shift:
            tshift = shift
            bshift = shift - 1
        return {
            'uniform': nflows.distributions.uniform.BoxUniform(torch.zeros(inp_dim) - tshift,
                                                               torch.ones(inp_dim) + bshift),
            'normal': rejection_sampler(nflows.distributions.StandardNormal([inp_dim]), bound)
            # 'normal': nflows.distributions.StandardNormal([inp_dim])
        }[nm]

    except KeyError:
        raise ValueError('Unknown nflows base distribution: {}'.format(nm))


class indp_gaussians():
    def __init__(self, dim):
        self.latent_dim = dim
        self.sampler = torch.distributions.normal.Normal(torch.tensor(0.0), torch.tensor(1.0))

    def sample(self, ndata):
        samples = []
        for _ in range(self.latent_dim):
            samples += [self.sampler.sample(ndata)]
        return torch.stack(samples)


def torch_dists(nm, latent_dim):
    try:
        return {
            'uniform': torch_distributions.uniform.Uniform(torch.zeros(latent_dim) - 1, torch.ones(latent_dim),
                                                           validate_args=None),
            'normal': torch_distributions.MultivariateNormal(torch.zeros(latent_dim), torch.eye(latent_dim)),
            'indp_gauss': indp_gaussians(latent_dim)
        }[nm]

    except KeyError:
        raise ValueError('Unknown torch base distribution: {}'.format(nm))


def get_measure(name):
    if name == 'None' or name == 'none':
        def dist(x, y):
            return torch.tensor(0)

    if name == 'sinkhorn':
        dist = SamplesLoss('sinkhorn', scaling=0.7, blur=0.01)

    if name == 'sinkhorn1':
        dist = SamplesLoss('sinkhorn', scaling=0.5, blur=0.01, p=1)

    if name == 'mmd':
        dist = SamplesLoss('gaussian')

    def dist_measure(x, y):
        return dist(x, y)

    return dist_measure


class sampler():
    def __init__(self, name):
        self.name = name

    def sample(self, ndata):
        data_obj = load_plane_dataset(self.name, ndata[0])
        return data_obj.data


class hepmass_sudo_sampler():
    """
    This class is a wrapper for the HEPMASS dataset when training with generator semantics.
    """

    def __init__(self, mass='1000'):
        trainset, testset = load_hepmass(mass, slim=(not on_cluster()))
        self.trainset = trainset[:, -5:]
        self.testset = testset[:, -5:]

    def sample(self, ndata):
        # For the current single use case of this we only want the high level features.
        return self.trainset

    def sample_valid(self, ndata):
        # For the current single use case of this we only want the high level features.
        return self.testset


def get_dist(name, dim):
    try:
        dist = torch_dists(name, dim)
    except:
        if name == 'checkerboard':
            dist = HyperCheckerboardDataset(int(1e3), dim)
        elif name == 'sparse_checkerboard':
            dist = SparseHyperCheckerboardDataset(int(1e3), dim)
        elif name == 'hepmass':
            dist = hepmass_sudo_sampler()
        else:
            dist = sampler(name)
            # TODO why is this here?
            dist.sample([8])
    return dist
