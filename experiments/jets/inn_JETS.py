# An NSF model for the JETS dataset from Deb
import numpy as np

import torch
import torch.optim as optim

from nflows import flows

from tensorboardX import SummaryWriter

import sys
from pathlib import Path

from utils.training import fit

sys.path.append(str(Path('.').absolute().parent))

from models.flow_models import flow_builder
from models.nn.flows import spline_flow

from utils import hyperparams
from utils.post_process import post_process_jets
from utils.io import get_top_dir

from data.data_loaders import load_jets

import argparse

parser = argparse.ArgumentParser()

# Dataset parameters
# parser.add_argument('--dataset', type=str, default='jets', help='The dataset to train on.')
parser.add_argument('-d', type=str, default='JETS_INN', help='Directory to save contents into.')

## Hyper parameters
parser.add_argument('--batch_size', type=int, default=100, help='Size of batch for training.')
parser.add_argument('--epochs', type=int, default=10,
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

## Names for saving
parser.add_argument('-n', type=str, default='test_inn', help='The name with which to tag saved outputs.')

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

# Load the jets dataset
trainset, testset = load_jets()
inp_dim = trainset.data.shape[1]

# Set all tensors to be created on gpu, this must be done after dataset creation
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device('cpu')
print(device)

# Set up base transformation
# If using a normal distribution you have to allow for the possibility of samples coming from outside of the tail bound
bdist_shift = None
if args.base_dist == 'uniform':
    tail_bound = 1.
    tails = None
if args.base_dist == 'normal':
    tail_bound = 4.
    tails = 'linear'
    # Scale the data to be at the tail bounds, as the data is scaled by the max of the train set, remove epsilon
    trainset.data *= tail_bound
    testset.data *= tail_bound

transformation = spline_flow(inp_dim, args.nodes, num_blocks=args.nblocks, nstack=args.nstack, tail_bound=tail_bound,
                             tails=tails, activation=hyperparams.activations[args.activ], num_bins=args.nbins)
base_dist = hyperparams.nflows_dists(args.base_dist, inp_dim, shift=bdist_shift, bound=tail_bound)
flow = flows.Flow(transformation, base_dist)

# Build model
flow_model = flow_builder(flow, base_dist, device, exp_name, dir=args.d)

# Define optimizers and learning rate schedulers
optimizer = optim.Adam(flow.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, trainset.data.shape[0] / bsize * n_epochs, 0)

# Reduce lr on plateau at end of epochs
if args.reduce_lr_plat:
    reduce_lr_inn = [optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')]
else:
    reduce_lr_inn = None

# # Fit the model
fit(flow_model, optimizer, trainset, n_epochs, bsize, writer, schedulers=scheduler, schedulers_epoch_end=reduce_lr_inn,
    gclip=args.gclip)
# # flow_model.load(sv_dir + '/experiments/data/saved_models/model_INN_6D_big_1')

bnd = tail_bound + 0.5
anomaly_set = load_jets(sm='tt', split=0)
if args.base_dist == 'normal':
    # Scale the data to be at the tail bounds, as the data is scaled by the max of the train set, remove epsilon
    anomaly_set.data *= tail_bound
post_process_jets(flow_model, testset, anomaly_set=anomaly_set, sup_title='INN')
