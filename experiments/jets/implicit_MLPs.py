# An autoencoder with latent space matching applied directly to the output of the first encoder, or implicitly using
# Dense nets.

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from nflows import distributions

from tensorboardX import SummaryWriter

from models.nn.dense_nets import dense_net, stochastic_dense_encoder, dense_decoder, dense_inner, \
    stochastic_dense_inner
from utils.hyperparams import get_measure
from models.implicit_ae import implicit_autoencoder
from models.standard_ae import standard_autoencoder

from utils.training import fit
from utils import hyperparams
from utils.post_process import post_process_plane, post_process_hepmass
from utils.io import get_top_dir, on_cluster

from data.data_loaders import load_hepmass

import argparse

parser = argparse.ArgumentParser()

## Names for saving
parser.add_argument('-n', type=str, default='test_jets', help='The name with which to tag saved outputs.')
# Currently this is not implemented, but it is a useful feature.
parser.add_argument('-d', type=str, default='jets', help='Directory to save contents into.')

# Define model type to train
parser.add_argument('--ae_type', type=str, default='standard', help='The type of model to train.')

# Dataset parameters
parser.add_argument('--sm', type=str, default='QCD',
                    help='The dataset to train on, defined by the model from which the jets are generated.')

## Hyper parameters
parser.add_argument('--batch_size', type=int, default=1000, help='Size of batch for training.')
parser.add_argument('--latent_dim', type=int, default=5, help='Dimension of latent space.')
parser.add_argument('--dist_measure', type=str, default='sinkhorn',
                    help='The type of distribution matching to use.')
parser.add_argument('--epochs', type=int, default=20,
                    help='The number of epochs to train for.')
parser.add_argument('--activation', type=str, default='none',
                    help='The activation function to apply to the output of the outer encoder.')
parser.add_argument('--inner_activ', type=str, default='relu',
                    help='The activation function to apply to the layers of the inner networks.')
parser.add_argument('--npad', type=int, default=0,
                    help='The number of random numbers to pad around the data.')
parser.add_argument('--dropout', type=float, default=0,
                    help='A global parameter for controlling the dropout between layers.')
parser.add_argument('--recon_loss', type=str, default='mse',
                    help='A string that defines the type of loss to use for the outer reconstruction loss.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='The learning rate.')
parser.add_argument('--wd', type=float, default=0.01,
                    help='The weight decay parameter to use in the AdamW optimizer.')
parser.add_argument('--base_dist', type=str, default='uniform',
                    help='A string to index the corresponding torch distribution.')
parser.add_argument('--bnorm_outer', type=int, default=0,
                    help='An integer specifying whether to apply batch normalization to the outer pair of networks.')
parser.add_argument('--lnorm_outer', type=int, default=0,
                    help='An integer specifying whether to apply layer normalization to the outer pair of networks.')
parser.add_argument('--bnorm_inner', type=int, default=0,
                    help='An integer specifying whether to apply batch normalization to the inner pair of networks.')
parser.add_argument('--lnorm_inner', type=int, default=0,
                    help='An integer specifying whether to apply layer normalization to the inner pair of networks.')
parser.add_argument('--stochastic', type=int, default=0,
                    help='An integer specifying whether to use a stochastic encoder or not.')
parser.add_argument('--inner_stochastic', type=int, default=0,
                    help='An integer specifying whether to use stochastic internal networks.')
parser.add_argument('--decoder_activ', type=str, default='none',
                    help='An integer specifying the output of the decoder.')

# The beta hyperparameters
parser.add_argument('--noise_beta', type=float, default=1,
                    help='The number used to multiply the distribution matching term at output of first encoder for the '
                         'outer networks loss.')
parser.add_argument('--primary_noise_beta', type=float, default=1,
                    help='The number used to multiply the distribution matching term at output of first encoder for the '
                         'inner networks loss.')
parser.add_argument('--recon_beta', type=float, default=1,
                    help='The number used to multiply the reconstruction loss for the outer pair of networks.')
parser.add_argument('--inner_recon_beta', type=float, default=0,
                    help='The number used to multiply the inner reconstruction loss in the outer optimizer.')
parser.add_argument('--primary_inner_recon_beta', type=float, default=1,
                    help='The number used to multiply the inner reconstruction loss in the inner optimizer.')
parser.add_argument('--sample_beta', type=float, default=1,
                    help='The number used to multiply the sample distance in the outer optimizer.')
parser.add_argument('--primary_sample_beta', type=float, default=1,
                    help='The number used to multiply the sample distance in the inner optimizer.')
parser.add_argument('--recondist_beta', type=float, default=0,
                    help='The number used to multiply the reconstruction distance in the outer optimizer.')
parser.add_argument('--primary_recondist_beta', type=float, default=0,
                    help='The number used to multiply the reconstruction distance in the inner optimizer.')

parser.add_argument('--beta_sparse', type=float, default=0,
                    help='The amount of l1 regularization to apply to the output of the outer encoder.')

## reproducibility
parser.add_argument('--seed', type=int, default=1638128,
                    help='Random seed for PyTorch and NumPy.')

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

bsize = args.batch_size
n_epochs = args.epochs
# TODO: Add possibility of dim(A) != dim(Z), this has already been done in implicit_plane.py
latent_dim = args.latent_dim
out_activ = hyperparams.activations[args.activation]
inner_activ = hyperparams.activations[args.inner_activ]
betas_outer = {'outer recon': args.recon_beta, 'inner recon': args.inner_recon_beta, 'noise dist': args.noise_beta,
               'sample dist': args.sample_beta, 'recon dist': args.recondist_beta}
betas_inner = {'outer recon': 0, 'inner recon': args.primary_inner_recon_beta, 'noise dist': args.primary_noise_beta,
               'sample dist': args.primary_sample_beta, 'recon dist': args.primary_recondist_beta}
betas_standard = {'outer recon': args.recon_beta, 'noise dist': args.noise_beta}
exp_name = args.n
dir = args.d + '_'

sv_dir = get_top_dir()
log_dir = sv_dir + '/logs/' + exp_name
writer = SummaryWriter(log_dir=log_dir)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Make dataset
trainset, testset = load_hepmass(args.sm, slim=(not on_cluster()))
ndata = len(trainset)
inp_dim = trainset.dimension

layers_inner = [64, 64, 32, 16]
layers_outer = [512, 512, 256, 128, 128]
drop_p = args.dropout

# TODO: tidy up network definitions
if args.stochastic:
    encoder = stochastic_dense_encoder(inp_dim, latent_dim, layers=layers_outer, output_activ=out_activ, drp=drop_p,
                                       batch_norm=args.bnorm_outer, layer_norm=args.lnorm_outer).to(device)
else:
    encoder = dense_net(inp_dim, latent_dim, layers=layers_outer, output_activ=out_activ, drp=drop_p,
                        batch_norm=args.bnorm_outer, layer_norm=args.lnorm_outer).to(device)
decoder = dense_decoder(latent_dim, inp_dim, layers=layers_outer, drp=drop_p,
                        batch_norm=args.bnorm_outer, layer_norm=args.lnorm_outer,
                        output_activ=hyperparams.activations[args.decoder_activ]).to(device)
# TODO: dim change as above
if args.inner_stochastic:
    az = stochastic_dense_inner(latent_dim, latent_dim, layers=layers_inner, batch_norm=args.bnorm_inner,
                                layer_norm=args.lnorm_inner, inner_activ=inner_activ).to(device)
    zy = stochastic_dense_inner(latent_dim, latent_dim, layers=layers_outer, output_activ=out_activ,
                                batch_norm=args.bnorm_inner, layer_norm=args.lnorm_inner, inner_activ=inner_activ).to(
        device)
else:
    az = dense_inner(latent_dim, latent_dim, layers=layers_inner, batch_norm=args.bnorm_inner,
                     layer_norm=args.lnorm_inner, inner_activ=inner_activ).to(device)
    zy = dense_inner(latent_dim, latent_dim, layers=layers_outer, output_activ=out_activ, batch_norm=args.bnorm_inner,
                     layer_norm=args.lnorm_inner, inner_activ=inner_activ).to(device)

# Define optimizers and learning rate schedulers
max_step = args.ndata / bsize * n_epochs
optimizer_inner = optim.AdamW(list(az.parameters()) + list(zy.parameters()), lr=args.lr, weight_decay=args.wd)
scheduler_inner = optim.lr_scheduler.CosineAnnealingLR(optimizer_inner, max_step, 0)
# scheduler_inner = optim.lr_scheduler.StepLR(optimizer_inner, 10, 0.1)
optimizer_outer = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr, weight_decay=args.wd)
scheduler_outer = optim.lr_scheduler.CosineAnnealingLR(optimizer_outer, max_step, 0)

# Reduce lr on plateau at end of epochs
reduce_lr_outer = optim.lr_scheduler.ReduceLROnPlateau(optimizer_outer, 'min')
reduce_lr_inner = optim.lr_scheduler.ReduceLROnPlateau(optimizer_inner, 'min')

# Set up regularizers
last_layer_params = torch.cat([x.view(-1) for x in encoder.functions[-1].parameters()])
l1_regularization = lambda: args.beta_sparse * torch.norm(last_layer_params, 1)
reg_inner = lambda: torch.zeros(1).to(device)
reg_outer = lambda: l1_regularization()  # + other things?

# Define the Model
dist_measure = get_measure(args.dist_measure)
base_dist = hyperparams.torch_dists(args.base_dist, latent_dim)
recon_loss = hyperparams.recon_losses[args.recon_loss]

if args.ae_type == 'implicit':
    print('outer betas: ', betas_outer)
    print('inner betas: ', betas_inner)
    ae = implicit_autoencoder([encoder, decoder, az, zy], dist_measure, base_dist, betas_inner, betas_outer, device,
                              exp_name, recon_loss=recon_loss, dir=dir + args.ae_type)

    # Fit the model
    fit(ae, [optimizer_inner, optimizer_outer], trainset, n_epochs, bsize, writer,
        schedulers=[scheduler_inner, scheduler_outer], regularizers=[reg_inner, reg_outer],
        schedulers_epoch_end=[reduce_lr_outer, reduce_lr_inner])

elif args.ae_type == 'standard':
    print(betas_standard)
    ae = standard_autoencoder([encoder, decoder], dist_measure, base_dist, betas_standard, device, exp_name,
                              recon_loss=recon_loss, dir=dir + args.ae_type)

    # Fit the model
    fit(ae, optimizer_outer, trainset, n_epochs, bsize, writer, schedulers=scheduler_outer, regularizers=reg_outer,
        schedulers_epoch_end=[reduce_lr_outer])

else:
    raise NameError('Incorrect ae_type specified, pleas use "implicit" or "standard"')

post_process_hepmass(ae, testset, sup_title='OT ' + args.ae_type)