import os

import numpy as np

import torch
import torch.optim as optim

from tensorboardX import SummaryWriter

from models.cathode import Cathode

from utils.training import fit

from utils.post_process import post_process_curtains
from utils.io import get_top_dir, register_experiment

from data.data_loaders import get_data

import argparse

parser = argparse.ArgumentParser()

## Dataset parameters
parser.add_argument('--dataset', type=str, default='curtains', help='The dataset to train on.')

## Binning parameters
parser.add_argument("--quantiles", nargs="*", type=float, default=[1, 2, 3, 4])
parser.add_argument("--bins", type=str, default='2700,2710,3300,3700,4990,5000')
parser.add_argument("--doping", type=int, default=0,
                    help='Raw number of signal events to be added into the entire bg spectra.')
parser.add_argument("--feature_type", type=int, default=13)

## Names for saving
parser.add_argument('-n', type=str, default='cathode', help='The name with which to tag saved outputs.')
parser.add_argument('-d', type=str, default='NSF_CATHODE', help='Directory to save contents into.')
parser.add_argument('--load', type=int, default=1, help='Whether or not to load a model.')
parser.add_argument('--cathode_load', type=int, default=0, help='Load SR samples directly.')
parser.add_argument('--model_name', type=str, default=None, help='Saved name of model to load.')
parser.add_argument('--load_classifiers', type=int, default=0, help='Whether or not to load a model.')
parser.add_argument('--use_mass_sampler', type=int, default=1, help='Whether or not to sample the mass.')
parser.add_argument('--log_dir', type=str, default='no_scan', help='Whether or not to load a model.')

## Hyper parameters
parser.add_argument('--coupling', type=int, default=1, help='One to use coupling layers, zero for autoregressive.')
parser.add_argument('--spline', type=int, default=0, help='One to use spline transformations.')
parser.add_argument('--base_dist', type=str, default='normal',
                    help='A string to index the corresponding nflows distribution.')
parser.add_argument('--shuffle', type=int, default=1, help='Shuffle on epoch end.')
parser.add_argument('--coupling_width', type=int, default=64,
                    help='Width of network used to learn transformer parameters.')
parser.add_argument('--coupling_depth', type=int, default=3,
                    help='Depth of network used to learn transformer parameters.')

parser.add_argument('--batch_size', type=int, default=100, help='Size of batch for training.')
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
parser.add_argument('--load_best', type=int, default=0, help='Load the model that has the best validation score.')
parser.add_argument('--det_beta', type=float, default=0.1, help='Factor to multiply determinant by in the loss.')
parser.add_argument('--sample_m_train', type=int, default=0, help='Use mass sampler during training?')
parser.add_argument('--oversample', type=int, default=4,
                    help='How many times do we want to sample a point from the target distribution to transform to?')

## Classifier training
parser.add_argument('--beta_add_noise', type=float, default=0.01,
                    help='The value of epsilon to use in the 1-e training.')
parser.add_argument('--classifier_epochs', type=int, default=0,
                    help='The value of epsilon to use in the 1-e training.')
parser.add_argument('--c_nruns', type=int, default=1, help='Number of classifiers to run.')

## Redundat args for matching to Curtains
parser.add_argument('--distance', type=str, default='sinkhorn_slow', help='Type of dist measure to use.')
parser.add_argument('--ncond', type=int, default=1,
                    help='The number of features to condition on.')
parser.add_argument('--two_way', type=int, default=1,
                    help='One to train mapping from high mass to low mass, and low mass to high mass.')
parser.add_argument('--mix_sb', type=int, default=0, help='Mix sidebands while training?')

## Plotting
parser.add_argument('--n_sample', type=int, default=1000,
                    help='The number of features to use when calculating contours in the feature plots.')
parser.add_argument('--light', type=int, default=3,
                    help='We do not always want to plot everything and calculate all of the ROC plots.')
parser.add_argument('--plot', type=int, default=0, help='Plot all feature dists?')

## reproducibility
parser.add_argument('--seed', type=int, default=1638128,
                    help='Random seed for PyTorch and NumPy.')

args = parser.parse_args()
args.d += '_' + args.n

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Hyper params from passed args
bsize = args.batch_size
n_epochs = args.epochs
exp_name = args.n

sv_dir = get_top_dir()
image_dir = sv_dir + f'/images/{args.d}/'
if not os.path.exists(image_dir):
    os.makedirs(image_dir, exist_ok=True)
log_dir = sv_dir + '/logs/' + exp_name
writer = SummaryWriter(log_dir=log_dir)

# Save options used for running
register_experiment(sv_dir, f'{args.d}/{exp_name}', args)

# Make datasets
curtains_bins = args.bins.split(",")
curtains_bins = [int(b) for b in curtains_bins]
args.bins = curtains_bins
mix_qs = True
datasets, signal_anomalies = get_data(args.dataset, image_dir + exp_name, bins=args.bins, doping=args.doping,
                                      mix_qs=mix_qs, feature_type=args.feature_type)
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

# Build model
# cathode = contextual_flow(flow, base_dist, device, exp_name, dir=args.d)
cathode = Cathode(inp_dim, exp_name, device, dir=args.d)

# Define optimizers and learning rate schedulers
optimizer = cathode.optimizer
# optimizer = optim.Adam(cathode.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, ndata / bsize * n_epochs, 0)
# scheduler = None

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
    cathode.load(path)
    cathode.eval()
else:
    fit(cathode, optimizer, datasets.trainset, n_epochs, bsize, writer, schedulers=scheduler,
        schedulers_epoch_end=reduce_lr_inn, gclip=args.gclip, shuffle_epoch_end=args.shuffle, load_best=args.load_best)

classifier_args = {'false_signal': 0, 'batch_size': 128, 'nepochs': args.classifier_epochs,
                   'lr': 0.001, 'pure_noise': 0, 'beta_add_noise': args.beta_add_noise, 'drp': 0.0,
                   'normalize': True, 'data_unscaler': datasets.signalset.unnormalize, 'width': 32,
                   'use_scheduler': True}

# Generate test data and preprocess etc
post_process_curtains(cathode, datasets, sup_title='NSF', signal_anomalies=signal_anomalies,
                      load=args.load_classifiers, use_mass_sampler=args.use_mass_sampler,
                      n_sample_for_plot=args.n_sample, light_job=args.light, classifier_args=classifier_args,
                      plot=args.plot, cathode=True, summary_writer=writer, args=args, cathode_load=args.cathode_load)
