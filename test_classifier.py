import argparse
import os
import pickle

import torch

from data.data_loaders import get_data
from utils.DRE import get_auc
from utils.io import get_top_dir, register_experiment
from utils.plotting import plot_rates_dict
from utils.torch_utils import shuffle_tensor


def parse_args(): 
    parser = argparse.ArgumentParser()

    # Saving
    parser.add_argument('-d', '--outputdir', type=str, default='classifier_local',
                        help='Choose the base output directory')
    parser.add_argument('-n', '--outputname', type=str, default='local',
                        help='Set the output name directory')
    parser.add_argument('--load', type=int, default=2,
                        help='Load a model?')

    # Classifier set up

    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='curtains', help='The dataset to train on.')
    parser.add_argument("--bins", nargs="*", type=float, default=[2300, 2700, 3300, 3700, 4000, 4300])
    parser.add_argument("--feature_type", type=int, default=2)
    parser.add_argument("--split_data", type=int, default=1)
    parser.add_argument("--sb_signal_frac", type=int, default=0,
                        help='Raw number of signal events to be added into the entire bg spectra.')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=100, help='Size of batch for training.')
    parser.add_argument('--nepochs', type=int, default=1, help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Classifier learning rate.')
    parser.add_argument('--wd', type=float, default=0., help='Weight Decay, set to None for ADAM.')
    parser.add_argument('--drp', type=float, default=0.5, help='Dropout to apply.')
    parser.add_argument('--width', type=int, default=32, help='Width to use for the classifier.')
    parser.add_argument('--depth', type=int, default=3, help='Depth of classifier to use.')
    parser.add_argument('--batch_norm', type=int, default=0, help='Apply batch norm?')
    parser.add_argument('--layer_norm', type=int, default=0, help='Apply layer norm?')
    parser.add_argument('--use_scheduler', type=int, default=0, help='Use cosine annealing of the learning rate?')

    # Classifier settings
    parser.add_argument('--false_signal', type=int, default=2, help='Add random noise samples to the signal set?')
    parser.add_argument('--use_weight', type=int, default=1, help='Apply weights to the data?')
    parser.add_argument('--beta_add_noise', type=float, default=0.01,
                        help='The value of epsilon to use in the 1-e training.')

    return parser.parse_args()


def test_classifier():
    args = parse_args()

    top_dir = get_top_dir()
    sv_dir = top_dir + f'/images/{args.outputdir}_{args.outputname}/'
    if not os.path.exists(sv_dir):
        os.makedirs(sv_dir, exist_ok=True)
    nm = args.outputname
    register_experiment(top_dir, f'{args.outputdir}_{args.outputname}/{args.outputname}', args)

    datasets, signal_anomalies = get_data(args.dataset, sv_dir + nm, bins=args.bins, doping=args.sb_signal_frac,
                                          feature_type=args.feature_type) 

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    training_data = shuffle_tensor(datasets.signalset.data)

    if args.split_data:
        betas_to_scan = [0.5, 1, 5, 15]
        data_to_dope, undoped_data = torch.split(training_data, int(len(training_data) / 2))
        pure_noise = False
    else:
        betas_to_scan = [0]
        args.false_signal = 0
        undoped_data = training_data
        l1 = undoped_data.min()
        l2 = undoped_data.max()
        data_to_dope = (l1 - l2) * torch.rand_like(undoped_data) + l2
        pure_noise = True

    rates_sr_vs_transformed = {}
    rates_sr_qcd_vs_anomalies = {}
    for beta in betas_to_scan:
        auc_info = get_auc(undoped_data, data_to_dope, sv_dir, nm + f'{beta}%Anomalies',
                           anomaly_data=signal_anomalies.data.to(device),
                           beta=beta / 100,
                           sup_title=f'QCD in SR doped with {beta:.3f}% anomalies',
                           load=args.load,
                           return_rates=True,
                           false_signal=args.false_signal,
                           batch_size=args.batch_size,
                           nepochs=args.nepochs,
                           lr=args.lr,
                           wd=args.wd,
                           drp=args.drp,
                           width=args.width,
                           depth=args.depth,
                           batch_norm=args.batch_norm,
                           layer_norm=args.layer_norm,
                           use_scheduler=args.use_scheduler,
                           use_weights=args.use_weight,
                           beta_add_noise=args.beta_add_noise,
                           pure_noise=pure_noise
                           )

        rates_sr_vs_transformed[f'{beta}'] = auc_info[1]
        rates_sr_qcd_vs_anomalies[f'{beta}'] = auc_info[2]

    plot_rates_dict(sv_dir, rates_sr_qcd_vs_anomalies, 'SR QCD vs SR Anomalies')
    plot_rates_dict(sv_dir, rates_sr_vs_transformed, 'T(SB12) vs SR')

    with open(f'{sv_dir}/rates.pkl', 'wb') as f:
        pickle.dump([rates_sr_qcd_vs_anomalies, rates_sr_vs_transformed], f)


if __name__ == '__main__':
    test_classifier()
