# A standard inn model
import os

import numpy as np

import torch
import torch.optim as optim

from tensorboardX import SummaryWriter

from models.nn.networks import dense_net
from utils.DRE import get_auc
from utils.hyperparams import get_measure
from utils.plotting import plot_rates_dict

from utils.training import fit

from models.OT_models import curtains_transformer, tucan, delta_mass_tucan, delta_tucan, \
    delta_mass_curtains_transformer, delta_curtains_transformer
from models.nn.flows import spline_flow, coupling_inn

from utils import hyperparams
from utils.post_process import post_process_curtains
from utils.io import get_top_dir, register_experiment

from data.data_loaders import get_koala_data, get_bin

import argparse

parser = argparse.ArgumentParser()

## Dataset parameters
parser.add_argument('--dataset', type=str, default='curtains', help='The dataset to train on.')

## Binning parameters
parser.add_argument("--quantiles", nargs="*", type=float, default=[1, 2, 3, 4])
parser.add_argument("--bins", nargs="*", type=float, default=[55, 65, 75, 85, 95, 105])
parser.add_argument("--doping", type=float, default=0.)

## Names for saving
parser.add_argument('-n', type=str, default='test', help='The name with which to tag saved outputs.')
parser.add_argument('-d', type=str, default='KOALA', help='Directory to save contents into.')
parser.add_argument('--load', type=int, default=0, help='Whether or not to load a model.')
parser.add_argument('--model_name', type=str, default=None, help='Saved name of model to load.')

## Hyper parameters
parser.add_argument('--batch_size', type=int, default=10, help='Size of batch for training.')
parser.add_argument('--epochs', type=int, default=1,
                    help='The number of epochs to train for.')
parser.add_argument('--activ', type=str, default='relu',
                    help='The activation function to use in the networks in the neural spline.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='The learning rate.')
parser.add_argument('--reduce_lr_plat', type=int, default=0,
                    help='Whether to apply the reduce learning rate on plateau scheduler.')
parser.add_argument('--gclip', type=int, default=None,
                    help='The value to clip the gradient by.')
parser.add_argument('--load_best', type=int, default=0, help='Load the model that has the best validation score.')

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
image_dir = sv_dir + f'/images/{args.d}/'
if not os.path.exists(image_dir):
    os.makedirs(image_dir, exist_ok=True)
log_dir = sv_dir + '/logs/' + exp_name
writer = SummaryWriter(log_dir=log_dir)

# Make datasets
background, signal, signal_region_anomalies = get_koala_data(bins=args.bins, doping=args.doping)
ndata, inp_dim = background.shape
print('There are {} training examples, {} signal examples and {} anomaly samples.'.format(
    len(background), len(signal), len(signal_region_anomalies)))

# # Set all tensors to be created on gpu, this must be done after dataset creation, and before the INN creation
# if torch.cuda.is_available():
#     device = torch.device('cuda')
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')
# else:
#     device = torch.device('cpu')
# print(device)

rates_sr_vs_transformed = {}
rates_sr_qcd_vs_anomalies = {}
# TODO: to be fair to curtains, when normalizing, you should use the thresholding technique you are using for Curtains proper.
for beta in [0.5, 1, 5, 10]:
    auc_info = get_auc(background, signal, image_dir, f'{args.n}_{beta}', anomaly_data=signal_region_anomalies,
                       thresholds=[0, 0.5, 0.8, 0.95], beta=beta / 100,
                       sup_title=f'QCD in SR doped with {beta:.3f}% anomalies',
                       return_rates=True, plot_mass_dists=False, normalize=True)
    auc_anomalies = auc_info[0]
    rates_sr_vs_transformed[f'{beta}'] = auc_info[1]
    rates_sr_qcd_vs_anomalies[f'{beta}'] = auc_info[2]

plot_rates_dict(image_dir, rates_sr_qcd_vs_anomalies, 'SR QCD vs SR Anomalies')
plot_rates_dict(image_dir, rates_sr_vs_transformed, 'T(SB12) vs SR')

# # Define optimizers and learning rate schedulers
# optimizer = optim.Adam(INN.parameters(), lr=args.lr)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, ndata / bsize * n_epochs, 0)
#
# # Reduce lr on plateau at end of epochs
# if args.reduce_lr_plat:
#     reduce_lr_inn = [optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')]
# else:
#     reduce_lr_inn = None
#
# # Switch back to the default tensor being on cpu, otherwise there are memory issues while training with the DataLoader
# torch.set_default_tensor_type('torch.FloatTensor')
#
# # Fit the model
# if args.load:
#     if args.model_name is not None:
#         nm = args.model_name
#     else:
#         nm = exp_name
#     path = get_top_dir() + f'/data/saved_models/model_{nm}'
#     curtain_runner.load(path)
# else:
#     fit(curtain_runner, optimizer, datasets.trainset, n_epochs, bsize, writer, schedulers=scheduler,
#         schedulers_epoch_end=reduce_lr_inn, gclip=args.gclip, shuffle_epoch_end=args.shuffle, load_best=args.load_best)
#
# # TODO: pass inputs to this dictionary as args.
# classifier_args = {'false_signal': True, 'batch_size': 10000, 'nepochs': 100, 'lr': 0.001}
#
# # Generate test data and preprocess etc
# post_process_curtains(curtain_runner, datasets, sup_title='NSF', signal_anomalies=signal_anomalies,
#                       load=args.load_classifiers, use_mass_sampler=args.use_mass_sampler,
#                       n_sample_for_plot=args.n_sample, light_job=args.light, classifier_args=classifier_args)
#
# # Save options used for running
# register_experiment(sv_dir, f'{args.d}/{exp_name}', args)
