# TODO: all that changes between this and ANODE is the data loader and the model that you load, should be called with one script
# A standard inn model
import numpy as np

import torch
import torch.optim as optim

from nflows import flows

from tensorboardX import SummaryWriter
from utils.hyperparams import get_measure

from utils.training import fit

from models.flow_models import curtains_transformer
from models.nn.flows import spline_flow

from utils import hyperparams
from utils.post_process import post_process_curtains
from utils.io import get_top_dir

from data.data_loaders import get_data

import argparse

parser = argparse.ArgumentParser()

## Dataset parameters
parser.add_argument('--dataset', type=str, default='curtains', help='The dataset to train on.')
parser.add_argument('--resonant_feature', type=str, default='mass', help='The resonant feature to use for binning.')

## Binning parameters
parser.add_argument("--quantiles", nargs="*", type=float, default=[0, 2, 1, 3])

## Names for saving
parser.add_argument('-n', type=str, default='Transformer', help='The name with which to tag saved outputs.')
parser.add_argument('-d', type=str, default='NSF_CURT', help='Directory to save contents into.')

## Hyper parameters
parser.add_argument('--batch_size', type=int, default=10, help='Size of batch for training.')
parser.add_argument('--epochs', type=int, default=50,
                    help='The number of epochs to train for.')
parser.add_argument('--base_dist', type=str, default='normal',
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
# TODO: make a cl arg
distance = 'sinkhorn'

measure = get_measure(distance)

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

# TODO: this base distribution is used nowhere and should be removed, however, what about the tails...
# TODO sometimes the validset has samples outside of the tail bound - but hopefully not many
# Set up base transformation
bdist_shift = None
if args.base_dist == 'uniform':
    tail_bound = 1.
    tails = None
if args.base_dist == 'normal':
    tail_bound = 4.
    tails = 'linear'
    # Scale the data to be at the tail bounds
    datasets.scale = tail_bound
    datasets.scale_data()

# TODO: this is an autoregressive transform at present - may be fast enough?
transformation = spline_flow(inp_dim, args.nodes, num_blocks=args.nblocks, nstack=args.nstack, tail_bound=tail_bound,
                             tails=tails, activation=hyperparams.activations[args.activ], num_bins=args.nbins,
                             context_features=2)
base_dist = hyperparams.nflows_dists(args.base_dist, inp_dim, shift=bdist_shift, bound=tail_bound)
flow = flows.Flow(transformation, base_dist)

# Build model
flow_model = curtains_transformer(flow, base_dist, device, exp_name, measure, datasets.nfeatures, dir=args.d)

# Define optimizers and learning rate schedulers
optimizer = optim.Adam(flow.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, ndata / bsize * n_epochs, 0)

# Reduce lr on plateau at end of epochs
if args.reduce_lr_plat:
    reduce_lr_inn = [optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')]
else:
    reduce_lr_inn = None

# Fit the model
fit(flow_model, optimizer, datasets.trainset, n_epochs, bsize, writer, schedulers=scheduler,
    schedulers_epoch_end=reduce_lr_inn, gclip=args.gclip, shuffle_epoch_end=True)

# Generate test data and preprocess etc
post_process_curtains(flow_model, datasets, sup_title='NSF')