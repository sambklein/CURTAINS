# A standard inn model
import numpy as np

import torch
import torch.optim as optim

from nflows import flows

from tensorboardX import SummaryWriter

import sys
from pathlib import Path

from utils.hyperparams import get_dist
from utils.plotting import plot_slice
from utils.training import fit_generator

sys.path.append(str(Path('.').absolute().parent))

# from utils.load_mnist import load_mnist
from models.flow_models import flow_builder
from models.nn.flows import spline_flow

from utils import hyperparams
from utils.post_process import post_process_plane, get_ood
from utils.io import get_top_dir
from utils.MC_estimators import get_kl_and_error

from data.data_loaders import load_plane_dataset, data_handler

# Ideally this would be implemented as follows
# Currently only runs an autoencoder with no distribution matching
import argparse

parser = argparse.ArgumentParser()

# Dataset parameters
parser.add_argument('--dataset', type=str, default='hypersparsecheckerboard', help='The dataset to train on.')
# Currently this is not implemented, but it is a useful feature.
parser.add_argument('-d', type=str, default='INN_test', help='Directory to save contents into.')
parser.add_argument('--ndata', type=int, default=int(1e5), help='The number of data points to generate.')
parser.add_argument('--latent_dim', type=int, default=5, help='The data dimension.')

## Hyper parameters
parser.add_argument('--batch_size', type=int, default=100, help='Size of batch for training.')
parser.add_argument('--epochs', type=int, default=2,
                    help='The number of epochs to train for.')
parser.add_argument('--base_dist', type=str, default='uniform',
                    help='A string to index the corresponding nflows distribution.')
parser.add_argument('--nstack', type=int, default='3',
                    help='The number of spline transformations to stack in the inn.')
parser.add_argument('--nblocks', type=int, default='3',
                    help='The number of layers in the networks in each spline transformation.')
parser.add_argument('--nodes', type=int, default='20',
                    help='The number of nodes in each of the neural spline layers.')
parser.add_argument('--activ', type=str, default='relu',
                    help='The activation function to use in the networks in the neural spline.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='The learning rate.')
parser.add_argument('--reduce_lr_plat', type=int, default=0,
                    help='Whether to apply the reduce learning rate on plateau scheduler.')
parser.add_argument('--gclip', type=int, default=None,
                    help='The value to clip the gradient by.')
parser.add_argument('--nbins', type=int, default=10,
                    help='The number of bins to use in each spline transformation.')

## Names for saving
parser.add_argument('-n', type=str, default='test_implicit', help='The name with which to tag saved outputs.')
parser.add_argument('--get_kl', type=int, default=0, help='Integer whether to calculate the KL divergence or not.')
parser.add_argument('--get_ood', type=int, default=0,
                    help='Integer whether to calculate the fraction of OOD samples or not.')

## reproducibility
parser.add_argument('--seed', type=int, default=1638128,
                    help='Random seed for PyTorch and NumPy.')

## KL estimate
parser.add_argument('--nrun', type=int, default=2,
                    help='The number of MC KL estimates to calculate.')
parser.add_argument('--ncalc', type=int, default=int(1e5),
                   git  help='The number of samples to pass through the encoder per sample.')
parser.add_argument('--n_test', type=int, default=10,
                    help='The number of times to calculate ncalc samples.')

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Hyper params from passed args
bsize = args.batch_size
n_epochs = args.epochs
exp_name = args.n

sv_dir = get_top_dir()
log_dir = sv_dir + '/logs/' + exp_name
writer = SummaryWriter(log_dir=log_dir)

inp_dim = args.latent_dim

# Set all tensors to be created on gpu, this must be done after dataset creation
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device('cpu')
print(device)

# Make dataset
# ndata = args.ndata
# dim = None if not args.dataset == 'hypercheckerboard' else args.latent_dim
# trainset = load_plane_dataset(args.dataset, ndata, dim=dim)
trainset = data_handler(args.ndata, bsize, inp_dim, args.dataset, device)

# Set up base transformation
# If using a normal distribution you have to allow for the possibility of samples coming from outside of the tail bound
bdist_shift = None
if args.base_dist == 'uniform':
    tail_bound = 1.
    if trainset.bounded:
        tails = None
    else:
        bdist_shift = tail_bound
        tails = 'linear'
if args.base_dist == 'normal':
    tail_bound = 4.
    tails = None
    # Scale the data to be at the tail bounds
    # trainset.data *= tail_bound
    trainset.scale = tail_bound
transformation = spline_flow(inp_dim, args.nodes, num_blocks=args.nblocks, nstack=args.nstack, tail_bound=tail_bound,
                             tails=tails, activation=hyperparams.activations[args.activ], num_bins=args.nbins)
base_dist = hyperparams.nflows_dists(args.base_dist, inp_dim, shift=bdist_shift, bound=tail_bound)
flow = flows.Flow(transformation, base_dist)

# Build model
flow_model = flow_builder(flow, base_dist, device, exp_name, dir=args.d)

# Define optimizers and learning rate schedulers
optimizer = optim.Adam(flow.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.ndata / bsize * n_epochs, 0)

# Reduce lr on plateau at end of epochs
if args.reduce_lr_plat:
    reduce_lr_inn = [optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')]
else:
    reduce_lr_inn = None

# Fit the model
fit_generator(flow_model, optimizer, trainset, n_epochs, bsize, writer, schedulers=scheduler,
              schedulers_epoch_end=reduce_lr_inn, gclip=args.gclip)
# flow_model.load(sv_dir + '/experiments/data/saved_models/model_INN_6D_big_1')

# Generate test data and preprocess etc
# TODO: generate testset with data handler instance to ensure correct scaling etc - important for transformation to noise
if args.latent_dim == 2:
    dim = None if not args.dataset == 'hypercheckerboard' else args.latent_dim
    testset = load_plane_dataset(args.dataset, int(1e5), dim=dim)
    if args.base_dist == 'normal':
        testset.data *= tail_bound
    bnd = tail_bound + 0.5
    post_process_plane(flow_model, testset, invertible=True, implicit=False, sup_title=args.dataset + ' INN',
                       bounds=[-bnd, bnd])

# With this you sample (n_calc * n_test) number of samples and calculate the kl divergence, This is repeated nrun times
if args.get_kl:
    nrun = args.nrun
    n_calc = args.ncalc
    n_test = args.n_test

    # TODO: grid hyper params need to be passed
    flow_model.flow.eval()
    kl_div_info = get_kl_and_error(get_dist(args.dataset, 2), get_dist(args.base_dist, 2), flow_model.encode, n_calc,
                                   nrun, n_test, device)

    print('Estimated KL divergence of {} with variance {}'.format(kl_div_info[0], kl_div_info[1]))

    with open(sv_dir + '/images' + '/' + flow_model.dir + '/score_{}.npy'.format(exp_name), 'wb') as f:
        np.save(f, kl_div_info)

if args.get_ood:
    # TODO: make nsamples something that can be passed
    nsamples = int(1e5)
    nrun = args.nrun

    bound = 4
    nbins = 50

    percent_ood, percent_oob, counts = get_ood(flow_model, nsamples, nrun, bound, nbins, max_it=1)

    nm = sv_dir + '/images' + '/' + flow_model.dir + '/slice_{}'.format(exp_name + '_') + '{}.png'''
    plot_slice(counts, nm.format('pred'))

    with open(sv_dir + '/images' + '/' + flow_model.dir + '/ood_{}.npy'.format(exp_name), 'wb') as f:
        np.save(f, percent_ood)
        np.save(f, percent_oob)
