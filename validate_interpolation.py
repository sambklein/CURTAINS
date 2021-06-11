# TODO: all that changes between this and ANODE is the data loader and the model that you load, should be called with one script
# A standard inn model
import numpy as np

import torch
import torch.optim as optim

from tensorboardX import SummaryWriter

from models.nn.networks import dense_net
from utils.hyperparams import get_measure

from utils.training import fit

from models.OT_models import curtains_transformer, tucan, delta_mass_tucan, delta_tucan, \
    delta_mass_curtains_transformer, delta_curtains_transformer
from models.nn.flows import spline_flow, coupling_inn

from utils import hyperparams
from utils.post_process import post_process_curtains
from utils.io import get_top_dir, register_experiment

from data.data_loaders import get_data, get_bin

import argparse

parser = argparse.ArgumentParser()

## Dataset parameters
parser.add_argument('--dataset', type=str, default='curtains', help='The dataset to train on.')
# TODO: not currently implemented, NOT a priority
parser.add_argument('--resonant_feature', type=str, default='mass', help='The resonant feature to use for binning.')

## Binning parameters
parser.add_argument("--quantiles", nargs="*", type=float, default=[1, 2, 3, 4])
parser.add_argument("--bins", nargs="*", type=float, default=[55, 65, 75, 85, 95, 105])

## Names for saving
parser.add_argument('-n', type=str, default='Transformer', help='The name with which to tag saved outputs.')
parser.add_argument('-d', type=str, default='NSF_CURT', help='Directory to save contents into.')

## Hyper parameters
parser.add_argument('--distance', type=str, default='mse', help='Type of dist measure to use.')
parser.add_argument('--coupling', type=int, default=1, help='One to use coupling layers, zero for autoregressive.')
parser.add_argument('--spline', type=int, default=0, help='One to use spline transformations.')
parser.add_argument('--two_way', type=int, default=1,
                    help='One to train mapping from high mass to low mass, and low mass to high mass.')
parser.add_argument('--shuffle', type=int, default=1, help='Shuffle on epoch end.')

parser.add_argument('--batch_size', type=int, default=10, help='Size of batch for training.')
parser.add_argument('--epochs', type=int, default=1,
                    help='The number of epochs to train for.')
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
parser.add_argument('--ncond', type=int, default=1,
                    help='The number of features to condition on.')

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
distance = args.distance

# measure(x, y) returns distance from x to y (N, D) for N samples in D dimensions, or (B, N, D) with a batch index
measure = get_measure(distance)

sv_dir = get_top_dir()
log_dir = sv_dir + '/logs/' + exp_name
writer = SummaryWriter(log_dir=log_dir)

# Make datasets
# If the distance measure is the sinkhorn distance then don't mix samples between quantiles
mix_qs = distance != 'sinkhorn'
# datasets = get_data(args.dataset, quantiles=args.quantiles, mix_qs=mix_qs)
datasets = get_data(args.dataset, bins=args.bins, mix_qs=mix_qs, save_mass=True)
anomaly_data = get_bin('WZ_allhad_pT', args.bins[2:4], datasets.validationset)
ndata = datasets.ndata
inp_dim = datasets.nfeatures
print('There are {} training examples, {} validation examples and {} signal examples.'.format(
    datasets.trainset.data.shape[0], datasets.validationset.data.shape[0], datasets.signalset.data.shape[0]))


# Set all tensors to be created on gpu, this must be done after dataset creation, and before the INN creation
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device('cpu')
print(device)

# Spline transformations require us to
tail_bound = 1.2  # This sets the bounds of the acceptable data for the spline transformation
tails = 'linear'  # This will ensure that any samples from outside of [-tail_bound, tail_bound] do not throw an error
# - but the transformation is more flexible if this can be set to None, as then the derivatives at the boundary are not
# fixed

if args.coupling:
    # TODO clean this up
    mx = [1] * int(np.ceil(datasets.nfeatures / 2)) + [0] * int(datasets.nfeatures - np.ceil(datasets.nfeatures / 2))


    # this has to be an nn.module that takes as first arg the input dim and second the output dim
    def maker(input_dim, output_dim):
        return dense_net(input_dim, output_dim, layers=[64, 64, 64], context_features=args.ncond)


    INN = coupling_inn(inp_dim, maker, nstack=args.nstack, tail_bound=tail_bound, tails=tails, lu=0,
                       num_bins=args.nbins, mask=mx, spline=args.spline)
else:
    INN = spline_flow(inp_dim, args.nodes, num_blocks=args.nblocks, nstack=args.nstack, tail_bound=tail_bound,
                      tails=tails, activation=hyperparams.activations[args.activ], num_bins=args.nbins,
                      context_features=args.ncond)

# Build model
if args.two_way:
    if args.ncond == 2:
        transformer = delta_mass_tucan  # tucan
    else:
        transformer = delta_tucan  # tucan
else:
    if args.ncond == 2:
        transformer = delta_mass_curtains_transformer
    else:
        transformer = delta_curtains_transformer

curtain_runner = transformer(INN, device, exp_name, measure, datasets.nfeatures, dir=args.d)

# Define optimizers and learning rate schedulers
optimizer = optim.Adam(INN.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, ndata / bsize * n_epochs, 0)

# Reduce lr on plateau at end of epochs
if args.reduce_lr_plat:
    reduce_lr_inn = [optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')]
else:
    reduce_lr_inn = None

# Switch back to the default tensor being on cpu, otherwise there are memory issues while training with the DataLoader
torch.set_default_tensor_type('torch.FloatTensor')

# Fit the model
fit(curtain_runner, optimizer, datasets.trainset, n_epochs, bsize, writer, schedulers=scheduler,
    schedulers_epoch_end=reduce_lr_inn, gclip=args.gclip, shuffle_epoch_end=args.shuffle)

# Generate test data and preprocess etc
post_process_curtains(curtain_runner, datasets, sup_title='NSF', anomaly_data=anomaly_data)

# Save options used for running
register_experiment(sv_dir, exp_name, args)
