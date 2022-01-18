import argparse
import glob
import json
import os
import pdb
import pickle

import matplotlib.pyplot as plt
import numpy as np

from data.data_loaders import get_data, load_curtains_pd
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


class PropertiesHandler():

    def __init__(self):
        sv_dir = get_top_dir()
        self.image_dir = sv_dir + f'/images/properties_storage'
        os.makedirs(self.image_dir, exist_ok=True)
        try:
            with open(f'{self.image_dir}/counts.pkl', 'rb') as f:
                self.property_dict = pickle.load(f)
        except Exception as e:
            print(e)
            self.property_dict = {}
        # self.property_dict = {}

    def __del__(self):
        with open(f'{self.image_dir}/counts.pkl', 'wb') as f:
            pickle.dump(self.property_dict, f)

    def get_expected_count(self, args, int_bins):
        bn = np.mean(int_bins)
        try:
            expected_count = self.property_dict[bn]
        except:
            datasets, signal_anomalies = get_data(args['dataset'], self.image_dir, bins=int_bins, mix_qs=args['mix_sb'],
                                                  doping=args['doping'], feature_type=args['feature_type'])
            expected_count = datasets.signalset.data.shape[0]
            self.property_dict[bn] = expected_count
        return expected_count

    def __call__(self, args):
        """Return a tuple of x axis tick and legend."""
        bins = args['bins'].split(',')
        int_bins = [int(b) for b in bins]
        x = np.mean(int_bins)
        # Get the expected numbers
        expected_sr_count = self.get_expected_count(args, int_bins)
        thresholds = np.array([0, 0.5, 0.8, 0.9, 0.95, 0.99])
        expected = (1 - thresholds) * expected_sr_count
        return x, expected, str(args['doping'])


def get_max_sic():
    args = parse_args()

    sv_dir = get_top_dir()
    # directories = [f'{sv_dir}/images/build_images', f'{sv_dir}/images/build_images', f'{sv_dir}/images/build_images']
    directories = [f'bins_doping_scan_OT_{i}' for i in range(0, 44)]

    vals = {}
    get_property = PropertiesHandler()
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
            x, _, label = get_property(args)
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
    nfolds = 5

    # Gather saved quantities
    vals = {}
    rates = {}
    get_property = PropertiesHandler()
    for directory in directories:
        try:
            with open(f'{sv_dir}/images/{directory}/counts.pkl', 'rb') as f:
                info_dict = pickle.load(f)
                true_counts = np.sum(info_dict['counts'], 0)
                expected_counts = np.sum(info_dict['expected_counts'], 0) / 8
                # counts = true_counts / expected_counts  # + 4 * (1 - np.array(thresholds))
                counts = true_counts - expected_counts  # + 4 * (1 - np.array(thresholds))
                error = counts * (np.sqrt(expected_counts) / expected_counts + np.sqrt(true_counts) / true_counts)
                # counts = true_counts - expected_counts / 4
                rate = info_dict['pass_rates']
                signal_pass_rate = rate[:, 0].reshape(nfolds, len(thresholds)).mean(0)
                bg_pass_rate = rate[:, 1].reshape(nfolds, len(thresholds)).mean(0)
            passed = 1
        except Exception as e:
            print(e)
            passed = 0
        if passed:
            args = get_args(f'{sv_dir}/images/{directory}')
            x, expected, label = get_property(args)
            # # TODO: why does this work better than the true expected values?
            expected = counts[0] * (1 - np.array(thresholds))
            inf = true_counts / expected  # + 4 * (1 - np.array(thresholds))
            inf = true_counts / (true_counts[0] * (1 - np.array(thresholds)))
            # inf = true_counts / (expected_counts[0] * (1 - np.array(thresholds)))
            inf = counts
            rt = np.vstack((signal_pass_rate, bg_pass_rate))
            if label in vals:
                vals[label] += [np.hstack((x, inf, error))]
                rates[label] += [rt]
            else:
                vals[label] = [np.hstack((x, inf, error))]
                rates[label] = [rt]

    # Start plotting different quantities
    dopings = sorted(set(vals.keys()))
    n_dopings = len(dopings)
    n_thresh = len(thresholds)
    # print(vals.keys())
    fig, axes = plt.subplots(n_dopings, n_thresh, figsize=(7 * n_thresh, 5 * n_dopings + 2))
    fig1, axes1 = plt.subplots(1, 1, figsize=(7, 5))
    n_plots = 5
    fig2, axes2 = plt.subplots(n_dopings, n_plots, figsize=(7 * n_plots, 5 * n_dopings + 2))

    class MassSpectrum:

        def __init__(self):
            self.sm = load_curtains_pd(feature_type=3)
            self.ad = load_curtains_pd(sm='WZ_allhad_pT', feature_type=3)

        def __call__(self, num):
            """Get the counts of BG events and Anomaly events in each bin."""
            set_of_bins = [np.arange(3100, 4700, 200), np.arange(3200, 4800, 200)]
            bins = np.arange(3200, 4600, 100)
            n_bins = set_of_bins[0].shape[0] + set_of_bins[1].shape[0] - 2
            dtype = np.float32
            bg_counts = np.empty((n_bins,), dtype=dtype)
            ad_counts = np.empty((n_bins,), dtype=dtype)
            ad = self.ad.sample(frac=1).iloc[:num]
            for i, bin in enumerate(set_of_bins):
                bg_counts[i::2] = np.histogram(self.sm['mjj'], bins=bin)[0]
                ad_counts[i::2] = np.histogram(ad['mjj'], bins=bin)[0]
            return bins, bg_counts, ad_counts

    get_mass_spectrum = MassSpectrum()
    spc = 0.1
    mxv = spc * len(thresholds)
    shift = np.arange(0, mxv, spc)
    clist = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for j in range(n_dopings):
        label = dopings[j]
        lst = vals[label]
        rt = rates[label]
        for i in range(n_thresh):
            ax = axes[j, i]
            if i > -1:
                xy = np.array(lst)
                # ax.plot(xy[:, 0], xy[:, i + 1], 'x', label=f'Cut = {thresholds[i]}')
                y = xy[:, i + 1]
                norm_fact = 1  # max(y)
                # sf = shift[i]
                sf = 0
                err = xy[:, i + 1 + len(thresholds)] / norm_fact
                # err = np.sqrt(abs(y)) / norm_fact
                lines = {'linestyle': 'None'}
                plt.rc('lines', **lines)
                # p = ax.plot(xy[:, 0], y / norm_fact + sf, 'o', markersize=3)
                clr = clist[i]
                p = ax.bar(xy[:, 0], y / norm_fact + sf, width=100, color='None', edgecolor=clr)
                ax.errorbar(xy[:, 0], y / norm_fact + sf, yerr=err,
                            label=f'Cut = {thresholds[i]}', fmt='', color=clr)
                axes1.plot(xy[:, 0], y / norm_fact + sf, 'o', label=f'Cut = {thresholds[i]}', markersize=3)

                rt = np.array(rt)
                bins, bg_counts, ad_counts = get_mass_spectrum(int(label))
                mx = np.digitize(xy[:, 0], bins=bins) - 1
                axes2[j, 0].bar(bins[mx], rt[:, 0, i] * ad_counts[mx], width=100, color='None', edgecolor='r')
                axes2[j, 1].bar(bins[mx], rt[:, 1, i] * bg_counts[mx], width=100, color='None', edgecolor='b')
                axes2[j, 2].bar(bins[mx], rt[:, 0, i] * ad_counts[mx], width=100, color='None', edgecolor='r')
                axes2[j, 2].bar(bins[mx], rt[:, 1, i] * bg_counts[mx], width=100, color='None', edgecolor='b')
                axes2[j, 3].bar(bins[mx], rt[:, 0, i] * ad_counts[mx] / (rt[:, 1, i] * bg_counts[mx]), width=100,
                                label=f'Cut = {thresholds[i]}', color='None', edgecolor=clr)
                axes2[j, 4].bar(bins[mx], rt[:, 0, i] * ad_counts[mx] / np.sqrt(rt[:, 1, i] * bg_counts[mx]),
                                width=100, label=f'Cut = {thresholds[i]}', color='None', edgecolor=clr)

            if i == 0:
                ax.set_ylabel('Counts / Expected Counts')
            ax.set_xlabel('Mean SR mass')
            ax.set_title(f'{label} Anomalies, Cut = {thresholds[i]}')
            # ax.set_yscale('log')

        axes1.set_ylabel('Counts / Expected Counts')
        axes1.set_xlabel('Mean SR mass')
        axes1.set_title(f'{label}')
        # axes1.set_yscale('log')

        axes2[j, 0].set_ylabel('Events')
        [axes2[j, k].set_xlabel('Mean SR mass') for k in range(3)]
        axes2[j, 0].set_title(f'{label} Signal')
        axes2[j, 1].set_title(f'{label} Background')
        axes2[j, 2].set_title(f'{label} Both')
        axes2[j, 3].set_title(f'{label} S/B')
        axes2[j, 4].set_title(f'{label} S/sqrt(B)')
        axes2[j, 2].set_yscale('log')
        # axes2[j, 3].set_yscale('log')
        # axes2[j, 4].set_yscale('log')

    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.9, 0.89), frameon=False)
    fig.savefig(f'{sv_dir}/images/counts_collated.png', bbox_inches='tight')
    fig.clf()

    # handles, labels = axes1.get_legend_handles_labels()
    # fig1.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.9, 0.89), frameon=False)
    fig1.savefig(f'{sv_dir}/images/counts_ensemble.png', bbox_inches='tight')
    fig1.clf()

    handles, labels = axes2[-1, 4].get_legend_handles_labels()
    fig2.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.9, 0.89), frameon=False)
    fig2.savefig(f'{sv_dir}/images/sb_total.png', bbox_inches='tight')
    fig2.clf()


def get_sics():
    sv_dir = get_top_dir()
    # directories = [f'fixed_widths_cathode_OT_{i}' for i in range(0, 9)] + \
    #               [f'fixed_widths_curtains_OT_{i}' for i in range(0, 9)] + \
    #               [f'classifier_test_fixed_classifier_{i}' for i in range(0, 9)] + \
    # directories = [f'no_eps_classifier_no_eps_{i}' for i in range(0, 9)] + \
    #               [f'no_eps_no_dope_classifier_no_eps_no_dope_{i}' for i in range(0, 9)] + \
    #               [f'idealised_cathode_idealised_cathode_{i}' for i in range(0, 9)] + \
    #               [f'supervised_cathode_super_cathode_{i}' for i in range(0, 9)] + \
    #               [f'no_adam_no_adam_{i}' for i in range(0, 9)] + \
    #               [f'no_adam_no_scheduler_no_adam_ns_{i}' for i in range(0, 9)] + \
    #               [f'supervised_no_eps_supervised_no_eps_{i}' for i in range(0, 9)]
    directories = [f'test_test_imbal_{i}' for i in range(0, 9)] + \
                  [f'super_cathode_cathy_new_weights_{i}' for i in range(0, 9)] + \
                  [f'super_slim_super_{i}' for i in range(0, 9)]
    # names = ['cathode' for i in range(0, 9)] + \
    #         [f'curtains' for i in range(0, 9)] + \
    #         [f'idealized' for i in range(0, 9)] + \
    # names = [f'idealized_no_eps' for i in range(0, 9)] + \
    #         [f'idealized_no_eps_or_dope' for i in range(0, 9)] + \
    #         [f'idealised_cathode' for i in range(0, 9)] + \
    #         [f'supervised_cathode' for i in range(0, 9)] + \
    #         [f'no_adam' for i in range(0, 9)] + \
    #         [f'no_adam_no_scheduler' for i in range(0, 9)] + \
    #         [f'supervised_no_eps' for i in range(0, 9)]
    names = ['SUPER'] * 9 + ['SUPER_schedule_wd'] * 9 + ['SUPER_slim'] * 9

    # Gather saved quantities
    vals = {}
    super_saver = []
    for i, directory in enumerate(directories):
        try:
            with open(f'{sv_dir}/images/{directory}/rates.pkl', 'rb') as f:
                loaded = pickle.load(f)[0]
                # info = loaded['0.0_no_eps']
                info = loaded['0.0']
                if 'Supervised' in loaded.keys():
                    super = loaded['Supervised']
            passed = 1
        except Exception as e:
            print(e)
            passed = 0
        if passed:
            args = get_args(f'{sv_dir}/images/{directory}')
            label = f'{args["feature_type"]}__{args["bins"]}'
            store = [names[i], info]
            if label in vals:
                vals[label] += [store]
            else:
                vals[label] = [store]
            if 'Supervised' in loaded.keys():
                if label not in super_saver:
                    super_saver += [label]
                    vals[label] += [['Supervised', super]]

    # Start plotting different quantities
    runs = sorted(set(vals.keys()))
    n_runs = len(runs)
    unique_names = list(set(names)) + ['Supervised']
    n_versions = len(unique_names)
    fig, axes = plt.subplots(n_runs, 2, figsize=(14, 5 * n_runs + 2))

    # clist = sorted(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    clist = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22',
             u'#17becf']
    clrs = {unique_names[i]: clist[i] for i in range(n_versions)}
    max_sic = 20

    for j in range(n_runs):
        label = runs[j]
        lst = vals[label]
        lst += [['random', [np.linspace(0, 1, 50), np.linspace(0, 1, 50)]]]
        for values in lst:
            nm = values[0]
            fpr = values[1][0]
            tpr = values[1][1]
            ax = axes[j, 1]
            ax1 = axes[j, 0]
            # pdb.set_trace()
            fpr_mx = fpr != 0.
            fpr_nz = fpr[fpr_mx]
            tpr_nz = tpr[fpr_mx]
            if nm == 'random':
                line = '--'
                color = 'k'
            else:
                line = '-'
                color = clrs[nm]
            ax.plot(tpr_nz, tpr_nz / fpr_nz ** 0.5, linewidth=2, label=nm, linestyle=line, color=color)
            ax1.plot(tpr_nz, 1 / fpr_nz, linewidth=2, label=nm, linestyle=line, color=color)
            ax.set_ylim(0, max_sic)

        ax.set_xlabel('Signal efficiency')
        ax.set_ylabel('Significance improvement')
        ax1.set_xlabel('Signal efficiency')
        ax1.set_ylabel('Rejection (1 / false positive rate)')
        ax1.set_yscale('log')
        feature_type, bins = label.split('__')
        ax.set_title(f'Feature type{feature_type}, Bins {bins}')

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.9, 0.89), frameon=False)
    fig.savefig(f'{sv_dir}/images/sics.png', bbox_inches='tight')
    fig.clf()


if __name__ == '__main__':
    # get_counts()
    get_sics()
    # get_max_sic()
