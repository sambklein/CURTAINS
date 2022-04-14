import torch
import torch.nn as nn
import torch.distributions as torch_distributions
import nflows
from geomloss import SamplesLoss
from torch.nn import functional as F

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
    'tanh': torch.tanh,
    'selu': nn.SELU()
}

recon_losses = {
    'none': nn.Identity(),
    'mse': nn.MSELoss(),
    'bce': nn.BCELoss(),
    'sinkhorn': SamplesLoss('sinkhorn', scaling=0.5, blur=0.01, p=1)
}

from nflows.distributions.base import Distribution


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


def get_sinkhorn(sinkhorn_dist):
    # This will remove the entropic bias from the OT distance.
    def sinkhorn_divergence(x, y):
        return sinkhorn_dist(x, y) / sinkhorn_dist(y, y)

    return sinkhorn_divergence


def get_measure(name, **kwargs):
    if name.casefold() == 'none':
        def dist(x, y):
            return torch.tensor(0)

    if name == 'sinkhorn':
        dist = SamplesLoss('sinkhorn', scaling=0.7, blur=0.01)
        # dist = get_sinkhorn(dist)

    if name == 'sinkhorn1':
        dist = SamplesLoss('sinkhorn', scaling=0.5, blur=0.01, p=1)

    if name == 'sinkhorn_slow':
        dist = SamplesLoss('sinkhorn', scaling=0.9, blur=0.05)
        # dist = get_sinkhorn(dist)

    if name == 'sinkhorn_slower':
        dist = SamplesLoss('sinkhorn', scaling=0.95, blur=0.05)
        # dist = get_sinkhorn(dist)

    if name == 'sinkhorn_slowest':
        dist = SamplesLoss('sinkhorn', scaling=0.99, blur=0.05)
        # dist = get_sinkhorn(dist)

    if name == 'energy':
        dist = SamplesLoss('energy', blur=0.05)

    if name.casefold() == 'mmd':
        dist = SamplesLoss('gaussian')

    if name.casefold() == 'mse':
        dist = torch.nn.MSELoss()

    if name.casefold() == 'huber':
        if 'beta' in kwargs:
            beta = kwargs['beta']
        else:
            beta = 1.0
        dist = torch.nn.SmoothL1Loss(beta=beta)  # Can update this once we update pytorch.

    def dist_measure(x, y):
        return dist(x, y)

    return dist_measure
