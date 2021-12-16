# TODO: all that changes between this and ANODE is the data loader and the model that you load, should be called with one script
# A standard inn model
import os

import numpy as np

import torch
import torch.optim as optim

from tensorboardX import SummaryWriter

from models.nn.networks import dense_net
from utils.hyperparams import get_measure
from utils.sampling_utils import signalMassSampler

from utils.training import fit

from models.OT_models import curtains_transformer, tucan, delta_mass_tucan, delta_tucan, \
    delta_mass_curtains_transformer, delta_curtains_transformer
from models.nn.flows import spline_flow, coupling_inn

from utils import hyperparams
from utils.post_process import post_process_curtains
from utils.io import get_top_dir, register_experiment, get_timestamp

from data.data_loaders import get_data, get_bin

import argparse

parser = argparse.ArgumentParser()

## Dataset parameters
parser.add_argument('--dataset', type=str, default='curtains', help='The dataset to train on.')
# TODO: not currently implemented, NOT a priority
parser.add_argument('--resonant_feature', type=str, default='mass', help='The resonant feature to use for binning.')
parser.add_argument('--mix_sb', type=int, default=2, help='Mix sidebands while training?')

## Binning parameters
parser.add_argument("--quantiles", nargs="*", type=float, default=[1, 2, 3, 4])
# parser.add_argument("--bins", nargs="*", type=float, default=[55, 65, 75, 85, 95, 105])
# parser.add_argument("--bins", nargs="*", type=float, default=[1000, 1200, 1800, 1900, 2000, 2100])
# parser.add_argument("--bins", nargs="*", type=float, default=[1000, 1200, 1400, 1600, 1800, 2000])
# parser.add_argument("--bins", nargs="*", type=float, default=[2000, 2500, 3000, 3500, 4000, 4500])
# parser.add_argument("--bins", nargs="*", type=float, default=[3000, 3200, 3400, 3600, 3800, 4000])
# parser.add_argument("--bins", nargs="*", type=float, default=[2700, 3000, 3300, 3600, 3900, 4100])
# parser.add_argument("--bins", nargs="*", type=float, default=[2300, 2700, 3300, 3700, 4000, 4300])
# parser.add_argument("--bins", nargs="*", type=float, default=[2300, 2700, 3300, 3700, 4900, 5000])
# parser.add_argument("--bins", nargs="*", type=float, default=[3000, 3200, 3400, 3600, 3800, 4000])
parser.add_argument("--bins", type=str, default='3000,3200,3400,3600,3800,4000')
# parser.add_argument("--bins", nargs="*", type=float, default=[2900, 3100, 3300, 3500, 3800, 4000])
parser.add_argument("--doping", type=int, default=0,
                    help='Raw number of signal events to be added into the entire bg spectra.')
parser.add_argument("--feature_type", type=int, default=3)

## Names for saving
parser.add_argument('-n', type=str, default='Transformer', help='The name with which to tag saved outputs.')
parser.add_argument('-d', type=str, default='NSF_CURT', help='Directory to save contents into.')
parser.add_argument('--load', type=int, default=1, help='Whether or not to load a model.')
parser.add_argument('--model_name', type=str, default=None, help='Saved name of model to load.')
parser.add_argument('--load_classifiers', type=int, default=0, help='Whether or not to load a model.')
parser.add_argument('--log_dir', type=str, default='no_scan', help='Whether or not to load a model.')

## Hyper parameters
parser.add_argument('--distance', type=str, default='energy', help='Type of dist measure to use.')
parser.add_argument('--coupling', type=int, default=1, help='One to use coupling layers, zero for autoregressive.')
parser.add_argument('--spline', type=int, default=0, help='One to use spline transformations.')
parser.add_argument('--two_way', type=int, default=1,
                    help='One to train mapping from high mass to low mass, and low mass to high mass.')
parser.add_argument('--shuffle', type=int, default=1, help='Shuffle on epoch end.')
parser.add_argument('--coupling_width', type=int, default=64,
                    help='Width of network used to learn transformer parameters.')
parser.add_argument('--coupling_depth', type=int, default=3,
                    help='Depth of network used to learn transformer parameters.')

parser.add_argument('--batch_size', type=int, default=100, help='Size of batch for training.')
parser.add_argument('--epochs', type=int, default=10,
                    help='The number of epochs to train for.')
parser.add_argument('--nstack', type=int, default=4,
                    help='The number of spline transformations to stack in the inn.')
parser.add_argument('--nblocks', type=int, default=2,
                    help='The number of layers in the networks in each spline transformation.')
parser.add_argument('--nodes', type=int, default=20,
                    help='The number of nodes in each of the neural spline layers.')
parser.add_argument('--activ', type=str, default='relu',
                    help='The activation function to use in the networks in the neural spline.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='The learning rate.')
parser.add_argument('--reduce_lr_plat', type=int, default=0,
                    help='Whether to apply the reduce learning rate on plateau scheduler.')
parser.add_argument('--gclip', type=int, default=5,
                    help='The value to clip the gradient by.')
parser.add_argument('--nbins', type=int, default=10,
                    help='The number of bins to use in each spline transformation.')
parser.add_argument('--ncond', type=int, default=1,
                    help='The number of features to condition on.')
parser.add_argument('--load_best', type=int, default=0, help='Load the model that has the best validation score.')
parser.add_argument('--det_beta', type=float, default=0.1, help='Factor to multiply determinant by in the loss.')
parser.add_argument('--sample_m_train', type=int, default=0, help='Use mass sampler during training?')
parser.add_argument('--oversample', type=int, default=4,
                    help='How many times do we want to sample a point from the target distribution to transform to?')

## Classifier training
parser.add_argument('--beta_add_noise', type=float, default=0.,
                    help='The value of epsilon to use in the 1-e training.')
parser.add_argument('--classifier_epochs', type=int, default=1,
                    help='The value of epsilon to use in the 1-e training.')
parser.add_argument('--use_mass_sampler', type=int, default=0, help='Whether or not to sample the mass.')

## Plotting
parser.add_argument('--n_sample', type=int, default=1000,
                    help='The number of features to use when calculating contours in the feature plots.')
parser.add_argument('--light', type=int, default=3,
                    help='We do not always want to plot everything and calculate all of the ROC plots.')
parser.add_argument('--plot', type=int, default=1, help='Plot all feature dists?')

## reproducibility
parser.add_argument('--seed', type=int, default=1638128,
                    help='Random seed for PyTorch and NumPy.')

args = parser.parse_args()
# args.d += '_' + args.n

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
image_dir = sv_dir + f'/images/{args.d}/'
if not os.path.exists(image_dir):
    os.makedirs(image_dir, exist_ok=True)
log_dir = f'{sv_dir}/logs/{args.log_dir}/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)
timestamp = get_timestamp()
writer = SummaryWriter(log_dir=f'{log_dir}/{exp_name}_{timestamp}')

# Save options used for running
register_experiment(sv_dir, f'{args.d}/{exp_name}', args)

curtains_bins = args.bins.split(",")
curtains_bins = [int(b) for b in curtains_bins]
args.bins = curtains_bins
# Make datasets
# If the distance measure is the sinkhorn distance then don't mix samples between quantiles
# mix_qs = distance[:8] != 'sinkhorn'
mix_qs = args.mix_sb
datasets, signal_anomalies = get_data(args.dataset, image_dir + exp_name, bins=args.bins, mix_qs=mix_qs,
                                      doping=args.doping, feature_type=args.feature_type)
ndata = datasets.ndata
inp_dim = datasets.nfeatures
print('There are {} training examples, {} validation examples, {} signal examples and {} anomaly samples.'.format(
    datasets.trainset.data.shape[0], datasets.validationset.data.shape[0], datasets.signalset.data.shape[0],
    signal_anomalies.data.shape[0]))

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
    n_mask = int(np.ceil(datasets.nfeatures / 2))
    mx = [1] * n_mask + [0] * int(datasets.nfeatures - n_mask)


    # this has to be an nn.module that takes as first arg the input dim and second the output dim
    def maker(input_dim, output_dim):
        return dense_net(input_dim, output_dim, layers=[args.coupling_width] * args.coupling_depth,
                         context_features=args.ncond)


    INN = coupling_inn(inp_dim, maker, nstack=args.nstack, tail_bound=tail_bound, tails=tails, lu=0,
                       num_bins=args.nbins, mask=mx, spline=args.spline, curtains_transformer=True)
else:
    INN = spline_flow(inp_dim, args.nodes, num_blocks=args.nblocks, nstack=args.nstack, tail_bound=tail_bound,
                      tails=tails, activation=hyperparams.activations[args.activ], num_bins=args.nbins,
                      context_features=args.ncond, curtains_transformer=True)

# Build model
if args.two_way == 1:
    if args.ncond == 2:
        transformer = delta_mass_tucan  # tucan
    else:
        transformer = delta_tucan  # tucan
else:
    if args.ncond == 2:
        transformer = delta_mass_curtains_transformer
    else:
        transformer = delta_curtains_transformer

if args.sample_m_train:
    m1 = datasets.trainset.data1[:, -1]
    m2 = datasets.trainset.data2[:, -1]
    masses = torch.cat((m1, m2))
    edge1 = datasets.mass_bins[2].item()
    edge2 = datasets.mass_bins[3].item()
    low_mass_training = datasets.trainset.data1
    mass_sampler = signalMassSampler(masses, edge1, edge2, plt_sv_dir=sv_dir,
                                     scaler=low_mass_training.unnorm_mass, unscaler=low_mass_training.norm_mass)
else:
    mass_sampler = None

curtain_runner = transformer(INN, device, exp_name, measure, datasets.nfeatures, dir=args.d, det_beta=args.det_beta,
                             mass_sampler=mass_sampler)

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
if args.load:
    if args.model_name is not None:
        nm = args.model_name
    else:
        nm = exp_name
    path = get_top_dir() + f'/data/saved_models/model_{nm}'
    curtain_runner.load(path)
else:
    fit(curtain_runner, optimizer, datasets.trainset, n_epochs, bsize, writer, schedulers=scheduler,
        schedulers_epoch_end=reduce_lr_inn, gclip=args.gclip, shuffle_epoch_end=args.shuffle, load_best=args.load_best)

# TODO: check the LR in particular!!
classifier_args = {'false_signal': 2, 'batch_size': 1000, 'nepochs': args.classifier_epochs,
                   'lr': 0.001, 'balance': 0, 'pure_noise': 0, 'beta_add_noise': args.beta_add_noise}

# Generate test data and preprocess etc
post_process_curtains(curtain_runner, datasets, sup_title='NSF', signal_anomalies=signal_anomalies,
                      load=args.load_classifiers, use_mass_sampler=args.use_mass_sampler,
                      n_sample_for_plot=args.n_sample, light_job=args.light, classifier_args=classifier_args,
                      plot=args.plot, mass_sampler=mass_sampler, summary_writer=writer, args=args)
