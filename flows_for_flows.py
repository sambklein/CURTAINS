# A standard inn model
import numpy as np

import torch
import torch.optim as optim

from nflows import flows

from tensorboardX import SummaryWriter

import sys
from pathlib import Path

from utils.training import fit

sys.path.append(str(Path('.').absolute().parent))

from models.flow_models import flow_for_flow, contextual_flow, flow_builder
from models.nn.flows import spline_flow

from utils import hyperparams
from utils.post_process import post_process_flows_for_flows, post_process_curtains
from utils.io import get_top_dir

from data.data_loaders import get_data

import argparse

parser = argparse.ArgumentParser()

## Dataset parameters
parser.add_argument('--dataset', type=str, default='curtains', help='The dataset to train on.')
parser.add_argument('--resonant_feature', type=str, default='mass', help='The resonant feature to use for binning.')

## Binning parameters
parser.add_argument("--bins", nargs="*", type=float, default=[-0.71, -0.7, -0.4, -0.35])
parser.add_argument("--quantiles", nargs="*", type=float, default=[0, 1, 2, 3])

## Names for saving
parser.add_argument('-n', type=str, default='NSF_flows_for_flows', help='The name with which to tag saved outputs.')
parser.add_argument('-d', type=str, default='NSF_CURT', help='Directory to save contents into.')

## Hyper parameters
parser.add_argument('--batch_size', type=int, default=100, help='Size of batch for training.')
parser.add_argument('--epochs', type=int, default=200,
                    help='The number of epochs to train for.')
parser.add_argument('--base_dist', type=str, default='uniform',
                    help='A string to index the corresponding nflows distribution.')
parser.add_argument('--nstack', type=int, default=5,
                    help='The number of spline transformations to stack in the inn.')
parser.add_argument('--nblocks', type=int, default=3,
                    help='The number of layers in the networks in each spline transformation.')
parser.add_argument('--nodes', type=int, default=64,
                    help='The number of nodes in each of the neural spline layers.')
parser.add_argument('--activ', type=str, default='relu',
                    help='The activation function to use in the networks in the neural spline.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='The learning rate.')
parser.add_argument('--reduce_lr_plat', type=int, default=0,
                    help='Whether to apply the reduce learning rate on plateau scheduler.')
parser.add_argument('--gclip', type=int, default=None,
                    help='The value to clip the gradient by.')
parser.add_argument('--nbins', type=int, default=20,
                    help='The number of bins to use in each spline transformation.')

## reproducibility
parser.add_argument('--seed', type=int, default=1638128,
                    help='Random seed for PyTorch and NumPy.')

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

# Make datasets
datasets = get_data(args.dataset, quantiles=args.quantiles)
ndata = datasets.ndata
inp_dim = datasets.nfeatures
print('There are {} training examples, {} validation examples and {} signal examples.'.format(
    datasets.trainset.data.shape[0], datasets.validationset.data.shape[0], datasets.signalset.data.shape[0]))

# Set all tensors to be created on gpu, this must be done after dataset creation
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device('cpu')
print(device)

# Set up base transformation
bdist_shift = None
if args.base_dist == 'uniform':
    tail_bound = 1.
    tails = None
    # tails = 'linear'
if args.base_dist == 'normal':
    tail_bound = 4.
    tails = 'linear'
    # Scale the data to be at the tail bounds
    datasets.scale = tail_bound
    datasets.scale_data()


################# Build the base distribution and fit it ###############################################################

dist_to_data_transform = spline_flow(inp_dim, args.nodes, num_blocks=args.nblocks, nstack=args.nstack,
                                     tail_bound=tail_bound, tails=tails, activation=hyperparams.activations[args.activ],
                                     num_bins=args.nbins, context_features=None)
base_dist = hyperparams.nflows_dists(args.base_dist, inp_dim, shift=bdist_shift, bound=tail_bound)

dist_to_data_flow = flows.Flow(dist_to_data_transform, base_dist)

# bath_tub = contextual_flow(dist_to_data_flow, base_dist, device, exp_name + '_base', dir=args.d)
bath_tub = flow_builder(dist_to_data_flow, base_dist, device, exp_name + '_base', dir=args.d)

# Define optimizers and learning rate schedulers
optimizer_base_dist = optim.Adam(dist_to_data_flow.parameters(), lr=args.lr)
scheduler_base_dist = optim.lr_scheduler.CosineAnnealingLR(optimizer_base_dist, ndata / bsize * n_epochs, 0)

# Fit the model
fit(bath_tub, [optimizer_base_dist], datasets.trainset, n_epochs, bsize, writer,
    schedulers=[scheduler_base_dist], gclip=args.gclip)
# Bath tub shouldn't change after this
dist_to_data_flow.train(False)
# This is model specific, and is more for checking that we actually learn something.
post_process_flows_for_flows(bath_tub, datasets, sup_title='NSF')

################# Build the transformer ################################################################################
mass_to_mass_transform = spline_flow(inp_dim, args.nodes, num_blocks=args.nblocks, nstack=args.nstack,
                                     tail_bound=tail_bound, tails=tails, activation=hyperparams.activations[args.activ],
                                     num_bins=args.nbins, context_features=1)

mass_to_mass_flow = flows.Flow(mass_to_mass_transform, dist_to_data_flow)

# Build model
shower_curtain = flow_for_flow(mass_to_mass_flow, dist_to_data_flow, base_dist, device, exp_name, dir=args.d)

optimizer_transformer = optim.Adam(mass_to_mass_flow.parameters(), lr=args.lr)
scheduler_transformer = optim.lr_scheduler.CosineAnnealingLR(optimizer_transformer, ndata / bsize * n_epochs, 0)

# Fit the model
fit(shower_curtain, [optimizer_transformer], datasets.trainset, n_epochs, bsize, writer,
    schedulers=[scheduler_transformer], gclip=args.gclip)

# fit(shower_curtain, [optimizer_base_dist, optimizer_transformer], datasets.trainset, n_epochs, bsize, writer,
#     schedulers=[scheduler_base_dist, scheduler_transformer], schedulers_epoch_end=reduce_lr_inn, gclip=args.gclip)

################# Post process #########################################################################################
# Generate test data and preprocess etc
post_process_curtains(shower_curtain, datasets, sup_title='NSF')
# # This is model specific, and is more for checking that we actually learn something.
# post_process_flows_for_flows(shower_curtain, datasets, sup_title='NSF')