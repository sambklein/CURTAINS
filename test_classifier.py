import argparse
import os

import torch

from data.data_loaders import get_data
from utils.DRE import get_auc
from utils.io import get_top_dir
from utils.plotting import plot_rates_dict
from utils.torch_utils import shuffle_tensor


def parse_args():
    parser = argparse.ArgumentParser()

    # Saving
    parser.add_argument('-d', '--outputdir', type=str, default='classifier_local',
                        help='Choose the base output directory')
    parser.add_argument('-n', '--outputname', type=str, default='local',
                        help='Set the output name directory')
    parser.add_argument('--load', type=int, default=0,
                        help='Load a model?')

    # Classifier set up

    # Dataset and training parameters
    parser.add_argument('--dataset', type=str, default='curtains', help='The dataset to train on.')
    parser.add_argument("--bins", nargs="*", type=float, default=[55, 65, 75, 85, 95, 105])

    return parser.parse_args()


def test_classifier():
    args = parse_args()

    top_dir = get_top_dir()
    sv_dir = top_dir + f'/images/{args.outputdir}/'
    if not os.path.exists(sv_dir):
        os.makedirs(sv_dir, exist_ok=True)
    nm = args.outputname

    datasets, signal_anomalies = get_data(args.dataset, sv_dir + nm, bins=args.bins)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    training_data = shuffle_tensor(datasets.signalset.data)

    data_to_dope, undoped_data = torch.split(training_data, int(len(training_data) / 2))

    rates_sr_vs_transformed = {}
    rates_sr_qcd_vs_anomalies = {}
    for beta in [0.5, 1, 5]:
        auc_info = get_auc(data_to_dope, undoped_data, sv_dir, nm + f'{beta}%Anomalies',
                           anomaly_data=signal_anomalies.data.to(device), beta=beta / 100,
                           sup_title=f'QCD in SR doped with {beta:.3f}% anomalies',
                           load=args.load, return_rates=True)

        rates_sr_vs_transformed[f'{beta}'] = auc_info[1]
        rates_sr_qcd_vs_anomalies[f'{beta}'] = auc_info[2]

    plot_rates_dict(sv_dir, rates_sr_qcd_vs_anomalies, 'SR QCD vs SR Anomalies')
    plot_rates_dict(sv_dir, rates_sr_vs_transformed, 'T(SB12) vs SR')


if __name__ == '__main__':
    test_classifier()
