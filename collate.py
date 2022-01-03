import argparse
import glob
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np

from utils.io import get_top_dir


def parse_args():
    parser = argparse.ArgumentParser()

    # Names for saving
    parser.add_argument('-d', type=str, default='Transformer', help='The name with which to tag saved outputs.')
    parser.add_argument('--jobs', type=str, default='0', help='Comma separated list of jobs to scan.')

    args = parser.parse_args()
    return args


def get_args(directory):
    args_file = glob.glob(f'{directory}/*.json')
    if len(args_file) > 1:
        print(args_file)
        raise Exception(f'Multiple json files in {directory}.')
    file = args_file[0]
    with open(file, "r") as file_name:
        json_dict = json.load(file_name)
    return json.loads(json_dict)


def get_property(args):
    """Return a tuple of x axis tick and legend."""
    bins = args['bins'].split(',')
    int_bins = [int(b) for b in bins]
    x = np.mean(int_bins)
    return x, str(args['doping'])


def get_max_sic():
    args = parse_args()

    sv_dir = get_top_dir()
    # directories = [f'{sv_dir}/images/build_images', f'{sv_dir}/images/build_images', f'{sv_dir}/images/build_images']
    directories = [f'bins_doping_scan_OT_{i}' for i in range(0, 44)]

    vals = {}
    for directory in directories:
        try:
            # with open(f'{sv_dir}/images/{directory}/rates.pkl', 'rb') as f:
            #     fpr, tpr = pickle.load(f)[0]['0']
            with open(f'{sv_dir}/images/{directory}/counts.pkl', 'rb') as f:
                count = np.sum(pickle.load(f), 0)[-1]
            passed = 1
        except Exception as e:
            print(e)
            passed = 0
        if passed:
            # fpr_nz = fpr[fpr != 0.]
            # tpr_nz = tpr[fpr != 0.]
            # sic = max(tpr_nz / (fpr_nz ** 0.5))
            args = get_args(f'{sv_dir}/images/{directory}')
            x, label = get_property(args)
            # if label in vals:
            #     vals[label] += [[x, sic]]
            # else:
            #     vals[label] = [[x, sic]]
            if label in vals:
                vals[label] += [[x, count]]
            else:
                vals[label] = [[x, count]]

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for label, lst in vals.items():
        xy = np.array(lst)
        ax.plot(xy[:, 0], xy[:, 1], 'x', label=label)

    ax.set_ylabel('Max significance improvement')
    ax.set_xlabel('Mean bin width')
    fig.legend()
    fig.savefig(f'{sv_dir}/images/sic_collated.png', bbox_inches='tight')


def get_counts():
    args = parse_args()

    sv_dir = get_top_dir()
    # directories = [f'build_images', f'build_images', f'build_images']
    directories = [f'bins_doping_scan_OT_{i}' for i in range(0, 44)] + \
                  [f'increase_bump_scan_OT_{i}' for i in range(0, 12)]
    thresholds = [0, 0.5, 0.8, 0.9, 0.95, 0.99]

    vals = {}
    for directory in directories:
        try:
            with open(f'{sv_dir}/images/{directory}/counts.pkl', 'rb') as f:
                counts = np.sum(pickle.load(f), 0)
            passed = 1
        except Exception as e:
            print(e)
            passed = 0
        if passed:
            args = get_args(f'{sv_dir}/images/{directory}')
            x, label = get_property(args)
            if label in vals:
                vals[label] += [np.hstack((x, counts))]
            else:
                vals[label] = [np.hstack((x, counts))]

    dopings = sorted(set(vals.keys()))
    n_dopings = len(dopings)
    # print(vals.keys())
    fig, axes = plt.subplots(n_dopings, 1, figsize=(7, 5 * n_dopings + 2))
    fig1, axes1 = plt.subplots(1, 1, figsize=(7, 5))
    for j in range(n_dopings):
        ax = axes[j]
        label = dopings[j]
        lst = vals[label]
        for i in range(len(thresholds)):
            if i > -1:
                xy = np.array(lst)
                # ax.plot(xy[:, 0], xy[:, i + 1], 'x', label=f'Cut = {thresholds[i]}')
                y = xy[:, i + 1]
                lines = {'linestyle': 'None'}
                plt.rc('lines', **lines)
                p = ax.plot(xy[:, 0], xy[:, i + 1], 'o', markersize=3)
                ax.errorbar(xy[:, 0], y, yerr=np.sqrt(y), label=f'Cut = {thresholds[i]}', fmt='',
                            color=p[0].get_color())
                axes1.plot(xy[:, 0], xy[:, i + 1], 'o', label=f'Cut = {thresholds[i]}', markersize=3)

        ax.set_ylabel('Counts')
        ax.set_xlabel('Mean SR mass')
        ax.set_title(f'{label}')
        ax.set_yscale('log')

        axes1.set_ylabel('Counts')
        axes1.set_xlabel('Mean SR mass')
        axes1.set_title(f'{label}')
        axes1.set_yscale('log')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.32, 0.89), frameon=False)
    fig.savefig(f'{sv_dir}/images/counts_collated.png', bbox_inches='tight')
    fig.clf()

    handles, labels = axes1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.32, 0.89), frameon=False)
    fig1.savefig(f'{sv_dir}/images/counts_ensemble.png', bbox_inches='tight')
    fig1.clf()


if __name__ == '__main__':
    get_counts()
    # get_max_sic()
