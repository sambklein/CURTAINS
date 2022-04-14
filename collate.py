import argparse
import glob
import json
import os
import pdb
import pickle
from collections import defaultdict
from copy import deepcopy

import matplotlib.pyplot as plt
import pandas as pd
import scipy
from matplotlib.pyplot import cm
import numpy as np
from scipy.interpolate import interp1d

from data.data_loaders import get_data, load_curtains_pd
from utils.io import get_top_dir, on_cluster

red = '#A92A2A'
blue = '#2D4BDC'

cathode_color = blue

label_to_color = {'Curtains': red,
                  'Cathode': cathode_color,
                  'Idealised': '#309402',
                  'Supervised': 'k',
                  'Cathode_full': cathode_color
                  }

label_to_name = {'Curtains': r'CURTAINs',
                 'Cathode': r'CATHODE',
                 'Idealised': 'Idealised',
                 'Supervised': 'Supervised',
                 'Cathode_full': 'CATHODE (full)'}


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

    # def __del__(self):
    #     with open(f'{self.image_dir}/counts.pkl', 'wb') as f:
    #         pickle.dump(self.property_dict, f)

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
        x = np.mean(int_bins[2:4])
        # Get the expected numbers
        # TODO: sort out this function call, it's crazy expensive and only needs to be made once, also it isn't used...
        # expected_sr_count = self.get_expected_count(args, int_bins)
        expected_sr_count = 0
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
    fig.savefig(f'{sv_dir}/images/sic_collated.pdf', bbox_inches='tight')


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), frameon=False, loc='upper right')


def make_steps(axis, x, y, width, drop_bars=True, plot=True, **kwargs):
    half_width = int(width / 2)
    n_points = 500
    y = y.reshape(-1, 1) * np.ones(n_points)
    if drop_bars:
        y[:, -1] = np.nan
        linestyle = '-'
    else:
        linestyle = '--'
    y = y.flatten()
    x_to_plot = np.concatenate(
        [np.linspace(mn, mx, n_points) for mn, mx in np.vstack((x - half_width, x + half_width)).transpose()])
    if plot:
        axis.plot(x_to_plot, y, ls=linestyle, **kwargs)
        xticks = np.unique(np.hstack(((x - half_width), (x + half_width))))
        axis.set_xticks(xticks)
    else:
        return x_to_plot, y


def add_errors(axis, x, y, width, error, **kwargs):
    x_to_plot, y1 = make_steps(axis, x, y - error, width, plot=False)
    _, y2 = make_steps(axis, x, y + error, width, plot=False)
    axis.fill_between(x_to_plot, y1, y2, alpha=0.5, facecolor=None, linewidth=0.0, **kwargs)


def get_counts():
    args = parse_args()

    sv_dir = os.path.join(get_top_dir())
    add_dict = None
    no_eps = True
    rescale = 1
    add_shading = 0

    # The real hunt
    # name = 'OT_bump_two_hundred'
    # dd = 'curtains_bump'

    # # A very nice hunt with bins of size 200
    # name = 'OT_bump_centered'
    # dd = 'curtains_bump_cfinal'
    # filename = '200'
    # bin_width = 200
    # n_runs = 40
    # y_max = 500000
    # add_dict = [f'no_dope_ot_OT_no_dope_{i}' for i in range(0, 8)]

    # # A very nice hunt with bins of size 200, with the classifier retrained multiple times
    # # curtains_bump_cfinal_OT_bump_centered_0_bump_200_0
    # name = 'bump_200'
    # dd = 'curtains_bump_cfinal_OT_bump_centered'
    # filename = '200'
    # bin_width = 200
    # n_runs = 40
    # y_max = 500000
    # no_eps = False
    # add_dict = [f'{dd}_{i}_{name}_{i}' for i in range(0, n_runs)]

    # # An alternate bump hunt with bins of size 200
    # # curtains_bump_cfinal_OT_bump_centered_0_bump_200_0
    # name = 'OT_bump_two'
    # dd = 'curtains_bump_two'
    # filename = '200_alt'
    # bin_width = 100
    # n_runs = 48
    # y_max = 500000
    # no_eps = True
    # add_dict = []

    # # A very nice hunt with bins of size 200, with the classifier retrained multiple times
    # # curtains_bump_cfinal_OT_bump_centered_0_bump_200_0
    # name = 'bump_200_alt'
    # dd = 'curtains_bump_two_OT_bump_two'
    # filename = '200_alt_from_testc'
    # bin_width = 200
    # n_runs = 40
    # y_max = 500000
    # no_eps = False
    # add_dict = [f'{dd}_{i}_{name}_{i}' for i in range(0, n_runs)]

    # # Alt classifier, alt bump hunt, good looker
    # name = 'bump_200_alt_nc'
    # dd = 'curtains_bump_two_OT_bump_two'
    # filename = '200_alt_alt_from_testc'
    # bin_width = 200
    # n_runs = 48
    # y_max = 500000
    # no_eps = False
    # add_dict = [f'{dd}_{i}_{name}_{i}' for i in range(0, n_runs)]

    # # Alt classifier, alt bump hunt, good looker shaded
    # # name = 'bump_200_alt_reprod'
    # name = 'bump_200_alt_reprod_bpf'
    # dd = 'curtains_bump_two_OT_bump_two'
    # filename = '200_with_shading'
    # # filename = '200_bpf'
    # bin_width = 200
    # n_runs = 48
    # y_max = 500000
    # no_eps = False
    # add_dict = [f'{dd}_{i}_{name}_{i}' for i in range(0, n_runs)]
    # add_shading = True

    # # Cathode 200 restricted
    # name = 'bump_cathode_mco'
    # dd = 'cathode_bump_two_Cathode_bump_two'
    # # name = 'cathode_bump_no_lrs'
    # # dd = 'cathode_bump_no_lrs'
    # # name = 'bump_cathode_part'
    # # dd = 'cathode_bump_no_lrs_cathode_bump_no_lrs'
    # filename = '200_cathode'
    # # filename = '200_bpf'
    # bin_width = 200
    # n_runs = 48
    # y_max = 500000
    # no_eps = False
    # add_dict = [f'{dd}_{i}_{name}_{i}' for i in range(0, n_runs)]
    # add_shading = False

    # Cathode 200 full sidebands
    # name = 'cathode_bump_full_no_lrs'
    # dd = 'cathode_bump_full_no_lrs'
    name = 'bump_cathode_full'
    dd = 'cathode_bump_full_cathode_bump_full'
    filename = '200_cathode_full'
    bin_width = 200
    n_runs = 48
    y_max = 500000
    no_eps = False
    add_dict = [f'{dd}_{i}_{name}_{i}' for i in range(0, n_runs)]
    add_shading = True

    # # Cathode on the full bands
    # name = 'bump_cathode_full'
    # dd = 'cathode_bump_full_cathode_bump_full'
    # filename = 'cathode_full_sidebands'
    # bin_width = 200
    # n_runs = 48
    # y_max = 500000
    # no_eps = False
    # add_dict = [f'{dd}_{i}_{name}_{i}' for i in range(0, n_runs)]
    # add_shading = False

    # # Alt classifier, alt bump hunt, good looker
    # name = 'bthree'
    # dd = 'OT_bump_three_OT_bump_three'
    # filename = 'bthree'
    # bin_width = 300
    # n_runs = 48
    # y_max = 500000
    # no_eps = False
    # add_dict = [f'{dd}_{i}_{name}_{i}' for i in range(0, n_runs)]

    # # 400 GeV
    # # curtains_bump_cfinal_OT_bump_centered_0_bump_200_0
    # name = 'four_bumps'
    # dd = 'OT_bump_four_OT_bump_four'
    # filename = '400_gev'
    # bin_width = 400
    # n_runs = 24
    # y_max = 500000
    # no_eps = False
    # add_dict = [f'{dd}_{i}_{name}_{i}' for i in range(0, n_runs)]

    # # A cathode bump hunt
    # # curtains_bump_cfinal_OT_bump_centered_0_bump_200_0
    # name = 'Cathode_bump_two'
    # dd = 'cathode_bump_two'
    # filename = '200_cathode_alt'
    # bin_width = 200
    # n_runs = 48
    # y_max = 500000
    # no_eps = True
    # add_dict = []

    # # An idealised hunt
    # name = 'idealised_hunt_cfinal'
    # dd = 'idealised_hunt_cfinal'
    # filename = 'idealised_hunt_cfinal'
    # bin_width = 200
    # n_runs = 48
    # y_max = 500000
    # no_eps = False
    # # The idealised classifier has half as many signal region QCD events
    # rescale = 2

    # # An idealised hunt including 1 + eps
    # name = 'idealised_hunt_noise'
    # dd = 'idealised_hunt_noise'
    # filename = 'idealised_hunt_noise'
    # bin_width = 200
    # n_runs = 8

    # # With bins of size 100
    # name = 'OT_bump_100'
    # dd = 'curtains_bump_100_cfinal'
    # filename = '100'
    # bin_width = 100
    # n_runs = 64
    # y_max = 100000

    reload = 0
    cathode_classifier = 0
    new_width = 100
    # new_width = None

    directories = [f'{dd}_{name}_{i}' for i in range(0, n_runs)]
    if add_dict is not None:
        directories += add_dict

    # thresholds = [0, 0.5, 0.8, 0.9, 0.95, 0.99]
    thresholds = [0, 0.5, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999]
    n_thresh_to_take = 7
    nfolds = 5

    def get_mask(mass):
        """
        Given the mass values that were scanned, return those values to include in the plot.
        This is useful if there are overlapping bins in the bump hunt scan.
        """
        # return x_all / 100 % 2 == 1
        return [True] * mass.shape[0]

    if reload:
        # Gather saved quantities
        vals = defaultdict(list)
        rates = defaultdict(list)
        masses = defaultdict(list)
        get_property = PropertiesHandler()
        for i, directory in enumerate(directories):
            try:
                if cathode_classifier:
                    with open(
                            f'{sv_dir}/images/{directory}/models_{name}_{i}Anomalies/counts_cathode.pkl',
                            'rb') as f:
                        info_dict = pickle.load(f)
                    true_counts = info_dict['counts']
                    if true_counts.ndim > 1:
                        # true_counts = true_counts.mean(1)
                        true_counts = np.median(true_counts, axis=1)
                    # expected_counts = info_dict['expected_counts']
                    # expected_counts = expected_counts / 8
                    expected_counts = true_counts[0] * (1 - np.array(thresholds))
                else:
                    nm = f'{sv_dir}/images/{directory}/counts'
                    if no_eps:
                        nm += '_no_eps'
                    nm += '.pkl'
                    with open(nm, 'rb') as f:
                        # with open(f'{sv_dir}/images/{directory}/counts_no_eps.pkl', 'rb') as f:
                        info_dict = pickle.load(f)
                    if not isinstance(info_dict, list):
                        info_dict = [info_dict]
                    true_counts = np.mean([id['counts'] for id in info_dict], 0).squeeze()
                    # expected_counts = np.sum(info_dict['expected_counts'], 0) / 8
                    expected_counts = true_counts[0] * (1 - np.array(thresholds))

                counts = true_counts

                # error = counts * (np.sqrt(expected_counts) / expected_counts + np.sqrt(true_counts) / true_counts)
                error = np.zeros_like(counts)
                counts = [true_counts, expected_counts]
                rate = np.mean([id['pass_rates'] for id in info_dict], 0)
                if cathode_classifier:
                    signal_pass_rate = rate[:, 0].mean(1)
                    bg_pass_rate = rate[:, 1].mean(1)
                else:
                    rate = np.array(rate)
                    signal_pass_rate = rate[:, 0]
                    bg_pass_rate = rate[:, 1]

                passed = 1
            except Exception as e:
                print(e)
                passed = 0
            if passed:
                args = get_args(f'{sv_dir}/images/{directory}')
                x, expected, label = get_property(args)
                # if (x == 3500) and (label == '1000') and (name == 'OT_bump_100'):
                # if (x == 3500) and (label == '1000'):
                if (x == 3500):
                    print(label)
                    print(directory)
                rt = np.vstack((signal_pass_rate, bg_pass_rate))
                vals[label] += [np.hstack((x, *counts, error))]
                rates[label] += [rt]
                masses[label] += [[id['masses'] for id in info_dict]]
                if add_shading:
                    masses[f'{label}_bgl'] += [[id['masses_labels'] for id in info_dict]]
                # masses[label] += [0]

            with open(f'{sv_dir}/images/rates_info_{filename}.pkl', 'wb') as f:
                pickle.dump(rates, f)
            with open(f'{sv_dir}/images/vals_info_{filename}.pkl', 'wb') as f:
                pickle.dump(vals, f)
            with open(f'{sv_dir}/images/masses_info_{filename}.pkl', 'wb') as f:
                pickle.dump(masses, f)
    else:
        with open(f'{sv_dir}/images/rates_info_{filename}.pkl', 'rb') as f:
            rates = pickle.load(f)
        with open(f'{sv_dir}/images/vals_info_{filename}.pkl', 'rb') as f:
            vals = pickle.load(f)
        with open(f'{sv_dir}/images/masses_info_{filename}.pkl', 'rb') as f:
            masses = pickle.load(f)

    # if not on_cluster():
    #     if len(rates['8000'][0]) == 16:
    #         for key in rates.keys():
    #             for i, arr in enumerate(rates[key]):
    #                 rates[key][i] = np.concatenate((arr[:8].mean(1)[np.newaxis, :], arr[8:].mean(1)[np.newaxis, :]), 0)

    # Start plotting different quantities
    dopings = sorted(set(vals.keys()))
    n_dopings = len(dopings)
    n_thresh = len(thresholds)
    fig, axes = plt.subplots(n_dopings, 1, figsize=(7, 5 * n_dopings + 2))
    fig1, axes1 = plt.subplots(1, 1, figsize=(7, 5))
    n_plots = 5
    fig2, axes2 = plt.subplots(n_dopings, n_plots, figsize=(7 * n_plots, 5 * n_dopings + 2))

    if n_dopings == 1:
        axes2 = np.array([axes2])
        axes = np.array([axes])

    class MassSpectrum:

        def __init__(self):
            self.sm = load_curtains_pd(feature_type=3).dropna()
            self.ad = load_curtains_pd(sm='WZ_allhad_pT', feature_type=3).dropna()

        def __call__(self, num):
            """Get the counts of BG events and Anomaly events in each bin."""
            # # TODO: what the fuck
            # if name in ['OT_bump_centered', 'bump_200', 'OT_bump_two',
            #             'bump_200_alt', 'bump_200_alt_nc', 'bump_200_alt_final']:
            #     bins = np.unique(np.hstack([[i + 400, i + 600] for i in range(2600, 4200, 200)]))
            # else:
            #     bins = np.unique(np.hstack([[i + 400, i + 500] for i in range(2600, 3800, 100)]))
            bins = np.unique(np.hstack([[i + 400, i + 600] for i in range(2600, 4400, 200)]))

            ad = self.ad.sample(frac=1).iloc[:num]
            bg_counts = np.histogram(self.sm['mjj'], bins=bins)[0]
            ad_counts = np.histogram(ad['mjj'], bins=bins)[0]
            return bins, bg_counts, ad_counts

    get_mass_spectrum = MassSpectrum()
    spc = 0.1
    mxv = spc * len(thresholds)
    clist = plt.rcParams['axes.prop_cycle'].by_key()['color']
    mass = get_mass_spectrum.ad['mjj']

    signicance_dict = {}
    half_width = int(bin_width / 2)

    for j in range(n_dopings):
        label = dopings[j]
        lst = vals[label]
        lst_masses = masses[label]
        masses_lbls = masses[f'{label}_bgl']
        rt = rates[label]
        significance = np.zeros(n_thresh_to_take)
        for i in range(n_thresh_to_take):
            # ax = axes[j, i]
            ax = axes[j]
            if i > -1:
                xy = np.array(lst)
                x_all = xy[:, 0]
                mx = get_mask(x_all)
                x = xy[mx, 0]
                y = xy[mx, i + 1]
                expected = xy[mx, i + 1 + len(thresholds)]
                x_e = xy[mx, 0]

                if new_width is not None:
                    # mass_y = np.concatenate([m[j][i].numpy() for m in lst_masses for j in range(5)])
                    n_models = len(lst_masses[0])
                    this_layer = []
                    for bin in lst_masses:
                        for sub_bin in bin:
                            # this_layer += [np.concatenate([m[i].numpy() for m in sub_bin])]
                            this_layer += [np.concatenate([m[i] for m in sub_bin])]
                    if add_shading:
                        this_layer_lbls = []
                        for bin in masses_lbls:
                            for sub_bin in bin:
                                # this_layer += [np.concatenate([m[i].numpy() for m in sub_bin])]
                                this_layer_lbls += [np.concatenate([m[i] for m in sub_bin])]
                        mass_l = np.concatenate(this_layer_lbls)
                    mass_y = np.concatenate(this_layer)
                    # Define new bin centers
                    min_mass = min(x_all) - half_width
                    max_mass = max(x_all) + half_width
                    new_bins = np.arange(min_mass, max_mass, new_width)
                    bin_width = new_width
                    y, _ = np.histogram(mass_y, bins=new_bins)
                    y = y / n_models
                    y = y * rescale
                    if add_shading:
                        ll, _, _ = scipy.stats.binned_statistic(mass_y, mass_l, bins=new_bins, statistic='sum')
                        ll = ll / n_models
                        ll = ll * rescale

                    x = np.convolve(new_bins, np.ones(2), 'valid') / 2
                    x_e = x
                    if i == 0:
                        raw_top = y
                    expected = raw_top * (1 - thresholds[i])

                # make_steps(ax, x, y, bin_width, color='r', label='Measured')
                if add_shading:
                    n_bg = y - ll
                    ax.plot(x, n_bg, marker='o', color='b', label='Observed BG', linestyle="None", markersize=3)
                    ax.plot(x, n_bg + ll, marker='o', color='r', label='Observed Signal', linestyle="None",
                            markersize=3)
                else:
                    ax.plot(x, y, marker='o', color='r', label='Observed', linestyle="None", markersize=3)
                make_steps(ax, x_e, expected, bin_width, color='k', label='Expected', drop_bars=False)
                if i == 0:
                    signal_bins = np.sort(np.unique(np.concatenate((x - bin_width / 2, x + bin_width / 2))))
                    ax.hist(mass.iloc[:int(label)], bins=signal_bins, alpha=0.2, color='y', label=f'{label} signal')
                    handles, labels = ax.get_legend_handles_labels()
                    # fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.9, 0.89), frameon=False)
                    ax.legend(handles, labels, frameon=False, loc='upper right')
                counts_error = np.sqrt(y)
                # x_error = bin_width / 2
                # ax.errorbar(x, y, yerr=counts_error, xerr=x_error, color='r', linestyle="None")
                if not add_shading:
                    ax.errorbar(x, y, yerr=counts_error, color='r', linestyle="None")
                # add_errors(ax, x, y, bin_width, counts_error, color='r')
                # ax.errorbar(x, y, yerr=counts_error, color='r', linestyle="None")
                if i == 0:
                    top_line = np.sqrt(expected)
                error_in_expected = top_line * (1 - thresholds[i])
                add_errors(ax, x_e, expected, bin_width, error_in_expected, color='k')
                axes1.plot(x, y, 'o', label=f'Cut = {thresholds[i]}', markersize=3)

                # rt = np.array(rt)
                # bins, bg_counts, ad_counts = get_mass_spectrum(int(label))
                # clr = clist[i]
                # fact = int(label) / ad_counts.sum()
                # ad_counts = fact * ad_counts
                # mx = np.digitize(xy[:, 0], bins=bins) - 1
                # total_signal = (rt[:, 0, i] * ad_counts[mx])
                # # total_bg = (rt[:, 1, i] * bg_counts[mx]).sum()
                # total_bg = (bg_counts[mx] * (1 - thresholds[i]))
                # significance[i] = np.sqrt(
                #     2 * ((total_signal + total_bg) * np.log(1 + total_signal / total_bg) - total_signal).sum())

                # axes2[j, 0].bar(bins[mx], rt[:, 0, i] * ad_counts[mx], width=bin_width, color='None', edgecolor='r')
                # axes2[j, 1].bar(bins[mx], rt[:, 1, i] * bg_counts[mx], width=bin_width, color='None', edgecolor='b')
                # axes2[j, 2].bar(bins[mx], rt[:, 0, i] * ad_counts[mx], width=bin_width, color='None', edgecolor='r')
                # axes2[j, 2].bar(bins[mx], rt[:, 1, i] * bg_counts[mx], width=bin_width, color='None', edgecolor='b')
                # axes2[j, 3].bar(bins[mx], rt[:, 0, i] * ad_counts[mx] / (rt[:, 1, i] * bg_counts[mx]), width=bin_width,
                #                 label=f'Cut = {thresholds[i]}', color='None', edgecolor=clr)
                # axes2[j, 4].bar(bins[mx], rt[:, 0, i] * ad_counts[mx] / np.sqrt(rt[:, 1, i] * bg_counts[mx]),
                #                 width=100, label=f'Cut = {thresholds[i]}', color='None', edgecolor=clr)

            if i == 0:
                ax.set_ylabel('Events / bin')
            ax.set_xlabel(r'$m_{JJ}$ [GeV]')
            # plt.rcParams['axes.titlepad'] = -14
            # ax.set_title(f'{label} injected signal', y=1.0, pad=-14)
            ax.set_yscale('log')
            ax.set_ylim([1, y_max])

        signicance_dict[label] = significance
        axes1.set_ylabel('Counts')
        axes1.set_xlabel(r'$m_{JJ}$ [GeV]')
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

    pd.DataFrame.from_dict(signicance_dict).to_csv(f'{sv_dir}/images/rates_dict_{name}.csv')
    fig.savefig(f'{sv_dir}/images/counts_collated.pdf', bbox_inches='tight')
    fig.clf()

    # # handles, labels = axes1.get_legend_handles_labels()
    # # fig1.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.9, 0.89), frameon=False)
    # fig1.savefig(f'{sv_dir}/images/counts_ensemble.pdf', bbox_inches='tight')
    # fig1.clf()
    #
    # handles, labels = axes2[-1, 4].get_legend_handles_labels()
    # fig2.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.9, 0.89), frameon=False)
    # fig2.savefig(f'{sv_dir}/images/sb_total.pdf', bbox_inches='tight')
    # fig2.clf()


def plot_counts_():
    # filenames = ['200_alt_alt_from_testc', '200_cathode_alt', 'idealised_hunt_cfinal']
    # names = ['bump_200_alt_nc', 'Cathode_bump_two', 'idealised_hunt_cfinal']
    # bin_widths = [200, 200, 200]

    # filenames = ['200_alt_alt_from_testc', '200_cathode', '200_cathode_full']
    # figure_name = 'all_models'
    filenames = ['200_alt_alt_from_testc']
    figure_name = 'curtains_only'

    names = ['', '', '']
    bin_widths = [200, 200, 200]
    model_names = ['Curtains', 'Cathode', 'Cathode_full']
    measured_colors = [label_to_color[n] for n in model_names]
    alphas = [1, 1, 1]
    markers = ['o', 'o', 'x']
    # dopings = [['0'], ['500', '667', '1000', '8000']]
    dopings = [['500', '667', '1000', '8000']]
    # dopings = ['0', '333']
    n_columns = [1, 2]
    y_max = 500000 if len(filenames) < 3 else 1200000
    # y_max = 1200000
    plot_counts(filenames, names, bin_widths, measured_colors, model_names, alphas, dopings, n_columns, markers,
                new_width=100, y_max=y_max, add_tags=True, figure_name=figure_name)

    # filenames = ['200_cathode', '200_cathode_full']
    # figure_name = 'cathode_only'
    #
    # names = ['', '', '']
    # bin_widths = [200, 200, 200]
    # model_names = ['Cathode', 'Cathode_full']
    # measured_colors = [label_to_color[n] for n in model_names]
    # alphas = [1, 1, 1]
    # markers = ['o', 'x']
    # dopings = [['0'], ['500', '667', '1000', '8000']]
    # # dopings = ['0', '333']
    # n_columns = [1, 2]
    # y_max = 500000 if len(filenames) < 3 else 1200000
    # # y_max = 1200000
    # plot_counts(filenames, names, bin_widths, measured_colors, model_names, alphas, dopings, n_columns, markers,
    #             new_width=100, y_max=y_max, add_tags=True, figure_name=figure_name)

    # filenames = ['200_alt_alt_from_testc', '200_cathode_full']
    # figure_name = 'curtains_cathode_full'
    #
    # names = ['', '', '']
    # bin_widths = [200, 200, 200]
    # model_names = ['Curtains', 'Cathode_full']
    # measured_colors = [label_to_color[n] for n in model_names]
    # alphas = [1, 1, 1]
    # markers = ['o', 'x']
    # dopings = [['0'], ['500', '667', '1000', '8000']]
    # # dopings = ['0', '333']
    # n_columns = [1, 2]
    # y_max = 500000 if len(filenames) < 3 else 1200000
    # # y_max = 1200000
    # plot_counts(filenames, names, bin_widths, measured_colors, model_names, alphas, dopings, n_columns, markers,
    #             new_width=100, y_max=y_max, add_tags=True, figure_name=figure_name)
    return 0


def plot_counts(filenames, names, bin_widths, measured_colors, model_names, alphas,
                dopings_list, n_columns, markers,
                new_width=100, y_max=100000, add_tags=True, figure_name='', plot_singles=True):
    sv_dir = os.path.join(get_top_dir())

    # thresholds = [0, 0.5, 0.8, 0.9, 0.95, 0.99]
    thresholds = [0, 0.5, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999]
    tags = [100, 50, 20, 10, 5, 1, 0.1, 0.01]
    ticks_spacing = bin_widths[0]
    n_thresh_to_take = 7
    # Start plotting different quantities
    # dopings = ['0', '333', '500', '667', '1000', '8000']
    if not isinstance(dopings_list[0], list):
        dopings_list = [dopings_list]

    class MassSpectrum:

        def __init__(self):
            self.sm = load_curtains_pd(feature_type=3).dropna()
            self.ad = load_curtains_pd(sm='WZ_allhad_pT', feature_type=3).dropna()

        def __call__(self, num):
            """Get the counts of BG events and Anomaly events in each bin."""
            # # TODO: what the fuck
            # if name in ['OT_bump_centered', 'bump_200', 'OT_bump_two',
            #             'bump_200_alt', 'bump_200_alt_nc', 'bump_200_alt_final']:
            #     bins = np.unique(np.hstack([[i + 400, i + 600] for i in range(2600, 4400, 200)]))
            # else:
            #     bins = np.unique(np.hstack([[i + 400, i + 500] for i in range(2600, 3800, 100)]))
            bins = np.unique(np.hstack([[i + 400, i + 600] for i in range(2600, 4400, 200)]))

            ad = self.ad.sample(frac=1).iloc[:num]
            bg_counts = np.histogram(self.sm['mjj'], bins=bins)[0]
            ad_counts = np.histogram(ad['mjj'], bins=bins)[0]
            return bins, bg_counts, ad_counts

    get_mass_spectrum = MassSpectrum()

    for n_plot, (dopings, n_c) in enumerate(zip(dopings_list, n_columns)):
        n_dopings = len(dopings)
        # fig, axes = plt.subplots(n_dopings, 1, figsize=(7, 5 * n_dopings + 2))
        n_rows = int(np.ceil(n_dopings / n_c))
        if plot_singles:
            fig = []
            axes = []
            for _ in range(n_rows):
                f, a = plt.subplots(1, 1, figsize=(7, 5))
                fig += [f]
                axes += [a]

        else:
            fig, ax_ = plt.subplots(n_rows, n_c, figsize=(7 * n_c, 5 * n_rows))
            axes = fig.axes
        first = 0
        fcond = len(filenames) - 1

        drop_indx = [1, 3, 5]
        # TODO: install tex in the container properly
        # plt.rcParams['text.usetex'] = True

        for filename, name, bin_width, measured_color, model_name, alph, marker in zip(filenames, names, bin_widths,
                                                                                       measured_colors,
                                                                                       model_names, alphas, markers):

            with open(f'{sv_dir}/images/rates_info_{filename}.pkl', 'rb') as f:
                rates = pickle.load(f)
            with open(f'{sv_dir}/images/vals_info_{filename}.pkl', 'rb') as f:
                vals = pickle.load(f)
            with open(f'{sv_dir}/images/masses_info_{filename}.pkl', 'rb') as f:
                masses = pickle.load(f)

            def get_mask(mass):
                """
                Given the mass values that were scanned, return those values to include in the plot.
                This is useful if there are overlapping bins in the bump hunt scan.
                """
                # return x_all / 100 % 2 == 1
                return [True] * mass.shape[0]

            mass = get_mass_spectrum.ad['mjj']

            signicance_dict = {}
            half_width = int(bin_width / 2)

            for j in range(n_dopings):
                label = dopings[j]
                lst = vals[label]
                lst_masses = masses[label]
                rt = rates[label]
                significance = np.zeros(n_thresh_to_take)
                for i in range(n_thresh_to_take):
                    if i not in drop_indx:
                        # ax = axes[j, i]
                        ax = axes[j]
                        if i > -1:
                            xy = np.array(lst)
                            x_all = xy[:, 0]
                            mx = get_mask(x_all)
                            x = xy[mx, 0]
                            y = xy[mx, i + 1]
                            expected = xy[mx, i + 1 + len(thresholds)]
                            x_e = xy[mx, 0]

                            if new_width is not None:
                                # mass_y = np.concatenate([m[j][i].numpy() for m in lst_masses for j in range(5)])
                                n_models = len(lst_masses[0])
                                this_layer = []
                                for bin in lst_masses:
                                    for sub_bin in bin:
                                        # this_layer += [np.concatenate([m[i].numpy() for m in sub_bin])]
                                        this_layer += [np.concatenate([m[i] for m in sub_bin])]
                                mass_y = np.concatenate(this_layer)
                                # Define new bin centers
                                min_mass = min(x_all) - half_width
                                max_mass = max(x_all) + half_width + 1
                                new_bins = np.arange(min_mass, max_mass, new_width)
                                bin_width = new_width
                                y, _ = np.histogram(mass_y, bins=new_bins)
                                y = y / n_models
                                x = np.convolve(new_bins, np.ones(2), 'valid') / 2
                                x_e = x
                                if i == 0:
                                    raw_top = y
                                expected = raw_top * (1 - thresholds[i])

                            # make_steps(ax, x, y, bin_width, color='r', label='Measured')
                            ax.plot(x, y, marker=marker, color=measured_color, label=label_to_name[model_name],
                                    linestyle="None",
                                    markersize=3, alpha=alph)
                            if first == fcond:
                                make_steps(ax, x_e, expected, bin_width, color='k', label='Expected', drop_bars=False)
                                if add_tags:
                                    tag = tags[i]
                                    height = expected[0]
                                    x_placement = x_e[0] - bin_width + 35
                                    ax.text(x_placement, height, f'{tag}%', horizontalalignment='right',
                                            verticalalignment='center')
                            if (i == 0) and (first == fcond):
                                signal_bins = np.sort(np.unique(np.concatenate((x - bin_width / 2, x + bin_width / 2))))
                                ax.hist(mass.iloc[:int(label)], bins=signal_bins, alpha=0.2, color='y',
                                        label=f'{label} signal')
                                # handles, labels = ax.get_legend_handles_labels()
                                # # fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.9, 0.89), frameon=False)
                                # ax.legend(handles, labels, frameon=False, loc='upper right')
                                legend_without_duplicate_labels(ax)
                            counts_error = np.sqrt(y)
                            # x_error = bin_width / 2
                            # ax.errorbar(x, y, yerr=counts_error, xerr=x_error, color=measured_color, linestyle="None")
                            ax.errorbar(x, y, yerr=counts_error, color=measured_color, linestyle="None", alpha=alph)
                            # add_errors(ax, x, y, bin_width, counts_error, color=measured_color)
                            # ax.errorbar(x, y, yerr=counts_error, color=measured_color, linestyle="None")
                            if i == 0:
                                top_line = np.sqrt(expected)
                            error_in_expected = top_line * (1 - thresholds[i])
                            # if first == fcond:
                            #     add_errors(ax, x_e, expected, bin_width, error_in_expected, color='k')

                            # rt = np.array(rt)
                            bins, bg_counts, ad_counts = get_mass_spectrum(int(label))
                            fact = int(label) / ad_counts.sum()
                            ad_counts = fact * ad_counts
                            mx = np.digitize(xy[:, 0], bins=bins) - 1
                            ad_rate = np.array([r[0].reshape(-1, 8).mean(0) for r in rt]).mean(0)[i]
                            total_signal = (ad_rate * ad_counts[mx])
                            bg_rate = np.array([r[1].reshape(-1, 8).mean(0) for r in rt]).mean(0)[i]
                            total_bg = (bg_rate * bg_counts[mx]).sum()
                            # total_bg = (bg_counts[mx] * (1 - thresholds[i]))
                            significance[i] = np.sqrt(
                                2 * ((total_signal + total_bg) * np.log(
                                    1 + total_signal / total_bg) - total_signal).sum())

                        if i == 0:
                            ax.set_ylabel('Events / bin', loc='top', fontsize=12)
                        ax.set_xlabel(r'$m_{JJ}$ [GeV]', loc='right', fontsize=12)
                        # plt.rcParams['axes.titlepad'] = -14
                        # ax.set_title(f'{label} injected signal', y=1.0, pad=-14)
                        ax.set_yscale('log')
                        ax.set_ylim([1, y_max])
                        ax.set_xticks(np.arange(3000, 4700, 100))
                        ax.set_xlim([2830, 4600])

                signicance_dict[label] = significance
            first += 1

        for ax in axes:
            [l.set_visible(False) for (i, l) in enumerate(ax.xaxis.get_ticklabels()) if i % 2 != 0]
        pd.DataFrame.from_dict(signicance_dict).to_csv(f'{sv_dir}/images/rates_dict_{name}.csv')
        if plot_singles:
            [f.savefig(f'{sv_dir}/images/{figure_name}_{n_plot}_{i}.pdf', bbox_inches='tight') for i,f in enumerate(fig)]
        else:
            fig.savefig(f'{sv_dir}/images/counts_superimposed_{figure_name}_{n_plot}.pdf', bbox_inches='tight')
        # fig.clf()


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
                  [f'super_slim_super_{i}' for i in range(0, 9)] + \
                  [f'ideal_nobells_ideal_{i}' for i in range(0, 9)]
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
    names = ['SUPER'] * 9 + ['SUPER_av_no_wd'] * 9 + ['SUPER_slim_wd'] * 9 + ['ideal'] * 9

    # Classifier test settings
    # directories = [f'test_clas_classifier_settings_{i}' for i in range(0, 64)]
    # names = [f'{i}' for i in range(0, 64)]
    # directories = [f'test_clas_no_fs_classifier_settings_{i}' for i in range(0, 32)]
    # names = [f'{i}' for i in range(0, 32)]
    # directories = [f'super_slim_super_{i}' for i in range(0, 32)]
    # names = ['SUPER_test'] * 32
    directories = [f'test_seeds_f_test_seeds_f_{i}' for i in range(0, 10)]
    names = ['seed'] * 32

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
    fig, axes = plt.subplots(n_runs, 2, figsize=(14, 5 * n_runs + 2), squeeze=False)

    # clist = sorted(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    # clist = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22',
    #          u'#17becf']
    clist = cm.rainbow(np.linspace(0, 1, n_versions))
    clrs = {unique_names[i]: clist[i] for i in range(n_versions)}
    max_sic = 20

    for j in range(n_runs):
        label = runs[j]
        lst = vals[label]
        lst += [['Random', [np.linspace(0, 1, 50), np.linspace(0, 1, 50)]]]
        for values in lst:
            nm = values[0]
            fpr = values[1][0]
            tpr = values[1][1]
            ax = axes[j, 1]
            ax1 = axes[j, 0]
            fpr_mx = fpr != 0.
            fpr_nz = fpr[fpr_mx]
            tpr_nz = tpr[fpr_mx]
            if nm == 'Random':
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
    fig.savefig(f'{sv_dir}/images/sics.pdf', bbox_inches='tight')
    fig.clf()


def figs_six_and_seven():
    sv_dir = get_top_dir()
    fig_6_doping = '1000'
    add_dashes = False

    # directories = [f'ot_fig7_OT_fig7_{i}' for i in range(0, 8)] + \
    #               [f'cathode_fig7_CATHODE_fig7_{i}' for i in range(0, 8)] + \
    #               ['idealised_class_cath_idealised_class_cath_0'] + \
    #               ['super_class_cath_super_class_cath_0'] + \
    #               ['cathode_match_CATHODE_match_0']
    #               # ['cathode_match_CATHODE_match_0'] 0 = all sideband data 100 epochs, 4 = restircted but 100 epochs
    # names = ['Curtains'] * 8 + ['Cathode'] * 8 + ['Idealised'] + ['Supervised'] + ['CATHODE_full']
    # This is Curtains and Cathode as you will actually use them
    # This has CATHODE trained only for 100 epochs and taking the last model, not the best
    # directories = [f'ot_fig7_OT_fig7_{i}' for i in range(0, 8)] + \
    #               [f'cathode_match_CATHODE_match_4'] + \
    #               ['idealised_class_feature_idealised_class_feature_0'] + \
    #               ['super_class_cath_super_class_cath_0']
    # names = ['Curtains'] * 8 + ['Cathode'] + ['Idealised'] + ['Supervised']
    # filename = 'fig_6_7'

    # # This has some 1 + eps mixed in
    # directories = [f'ot_fig7_200_OT_fig7_200_{i}_C_F7_{i}' for i in range(0, 8)] + \
    #               [f'ot_fig7_200_OT_fig7_200_{i}_C_F7_{i + 8}' for i in range(0, 8)]
    # # These names are reall Curtains and Curtains 1 + ep
    # names = ['Cathode'] * 8 + ['Curtains'] * 8
    # filename = 'fig_6_7_joint'

    # # This is restricted sidebands, with all of the data
    # directories = [f'ot_fig7_200_OT_fig7_200_{i}_C_F7_{i}' for i in range(0, 8)] + \
    #               [f'super_class_200_super_class_200_{i}' for i in range(0, 12)] + \
    #               [f'ideal_class_200_ideal_class_200_{i}' for i in range(0, 12)] + \
    #               [f'ot_fig7_200_hd_OT_fig7_200_hd_{i}_fig_6_hd_{i}' for i in range(0, 4)] + \
    #               [f'CATHODE_match_200_hd_full_bins_CATHODE_match_200_hd_full_bins_{i}_bump_search_cathode_{i}' for i in
    #                range(72)]

    # # This is restricted sidebands, everything trained with same raw counts as idealised
    # directories = [f'ot_fig7_200_OT_fig7_200_{i}_C_F7_match_idealised_{i}' for i in range(0, 8)] + \
    #               [f'super_class_200_super_class_200_{i}' for i in range(0, 12)] + \
    #               [f'ideal_class_200_ideal_class_200_{i}' for i in range(0, 12)] + \
    #               [f'ot_fig7_200_hd_OT_fig7_200_hd_{i}_fig_6_hd_match_idealised_{i}' for i in range(0, 4)] + \
    #               [
    #                   f'CATHODE_match_200_hd_full_bins_CATHODE_match_200_hd_full_bins_{i}_bump_search_cathode_match_idealised_{i}'
    #                   for i in range(72)]

    # # This is unrestricted sidebands, everything trained with same raw counts as idealised and no oversampling
    # directories = [f'ot_fig7_200_OT_fig7_200_{i}_C_F7_match_idealised_true_{i}' for i in range(0, 8)] + \
    #               [f'super_class_200_super_class_200_{i}' for i in range(0, 12)] + \
    #               [f'ideal_class_200_ideal_class_200_{i}' for i in range(0, 12)] + \
    #               [f'ot_fig7_200_hd_OT_fig7_200_hd_{i}_fig_6_hd_match_idealised_true_{i}' for i in range(0, 4)] + \
    #               [
    #                   f'CATHODE_match_200_hd_full_bins_CATHODE_match_200_hd_full_bins_{i}_bump_search_cathode_match_idealised_true_{i}'
    #                   for i in range(72)]
    # names = ['Curtains'] * 8 + ['Supervised'] * 12 + ['Idealised'] * 12 + ['Curtains'] * 4 + ['Cathode'] * 72

    # This is 400 GeV windows
    directories = [f'ot_fig7_400_OT_fig7_400_{i}_fig_499_OT_true_{i}' for i in range(0, 12)] + \
                  [f'super_class_400_super_class_400_{i}' for i in range(0, 12)] + \
                  [f'ideal_class_400_ideal_class_400_{i}' for i in range(0, 12)] + \
                  [
                      f'CATHODE_match_200_hd_full_bins_CATHODE_match_200_hd_full_bins_{i}_bump_search_cathode_match_idealised_true_{i}'
                      for i in range(72)]
    names = ['Curtains'] * 12 + ['Supervised'] * 12 + ['Idealised'] * 12 + ['Cathode'] * 72
    # filename = 'fig_6_7_400'
    filename = 'fig_6_7_400_cathode_versions_less'
    fig_6_doping = '3000'

    # # This is 400 GeV windows additional statistics
    # cathode_jobs = [24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46]
    # directories = [f'ot_fig7_400_OT_fig7_400_{i}_fig_499_OT_more_{i}' for i in range(0, 12)] + \
    #               [f'super_class_400_more_super_class_400_more_{i}' for i in range(0, 12)] + \
    #               [f'ideal_class_400_ideal_class_400_{i}' for i in range(0, 12)] + \
    #               [
    #                   f'CATHODE_match_200_hd_full_bins_CATHODE_match_200_hd_full_bins_{i}_cathode_idealised_more_{j}'
    #                   for i, j in zip(cathode_jobs, range(len(cathode_jobs)))]
    # names = ['Curtains'] * 12 + ['Supervised'] * 12 + ['Idealised'] * 12 + ['Cathode'] * len(cathode_jobs)
    # # filename = 'fig_6_7_400'
    # filename = 'fig_6_7_400_cathode_versions'
    # fig_6_doping = '3000'

    # # This is the Cathode comparisons plot, including full and restricted sidebands
    # directories = [f'ot_fig7_400_OT_fig7_400_{i}_fig_499_OT_true_{i}' for i in range(0, 12)] + \
    #               [f'cathode_test_cathode_test_f_{i}_bs_c_check_proper_{i}' for i in range(2)]
    # names = ['Curtains'] * 12 + ['Cathode_full', 'Cathode']
    # # filename = 'fig_6_7_400'
    # filename = 'fig_6_cathode'
    # fig_6_doping = '3000'
    # add_dashes = False

    # # This is the Cathode comparisons plot, including full and restricted sidebands, with more stats
    # directories = [f'ot_fig7_400_OT_fig7_400_{i}_fig_499_OT_more_{i}' for i in range(0, 12)] + \
    #               [f'cathode_test_cathode_test_f_{i}_bs_c_more_{i}' for i in range(2)]
    # names = ['Curtains'] * 12 + ['Cathode_full', 'Cathode']
    # # filename = 'fig_6_7_400'
    # filename = 'fig_6_cathode_more'
    # fig_6_doping = '3000'
    # add_dashes = False

    # This is full sidebands, TODO 1 here will give you the same features as CATHODE used
    # directories = ['curtains_match_200_CURTAINS_match_200_0'] + \
    #               ['cathode_match_200_CATHODE_match_200_0'] + \
    #               [f'super_class_200_super_class_200_{i}' for i in range(0, 8)] + \
    #               [f'ideal_class_200_ideal_class_200_{i}' for i in range(0, 8)]
    # names = ['Curtains'] * 4 + ['Cathode'] + ['Supervised'] * 8 + ['Idealised'] * 8
    # filename = 'fig_6_7_unrestricted'

    # # This is Curtains and cathode trained on the full sidebands
    # directories = ['curtains_match_CURTAINS_match_0'] + \
    #               [f'cathode_match_CATHODE_match_0'] + \
    #               ['idealised_class_cath_idealised_class_cath_0'] + \
    #               ['super_class_cath_super_class_cath_0']
    # names = ['Curtains'] + ['Cathode'] + ['Ideal ised'] + ['Supervised']
    # filename = 'fig_6_7_alt'

    # # This is full sidebands,
    # directories = [f'ot_fig7_200_OT_fig7_200_{i}_C_F7_{i}' for i in range(0, 8)] + \
    #               ['classifier_local_local', 'classifier_local_two_local_t']
    # names = ['Curtains'] * 8 + ['Supervised', 'Idealised']
    # filename = 'local'

    reload = 0
    plot_individual_lines = False
    quart = 0
    x_axis_rejection = 1
    # max_ax_six_1 = 12007
    max_ax_six_1 = 24268.4
    # max_ax_six_1 = 12134.4
    min_ax_six_1 = 12134.4

    if reload:
        # Gather saved quantities
        vals = defaultdict(list)
        for i, directory in enumerate(directories):
            try:
                # with open(f'{sv_dir}/images/{directory}/tpr_cathode.pkl', 'rb') as f:
                #     tpr_l = pickle.load(f)
                # with open(f'{sv_dir}/images/{directory}/fpr_cathode.npy', 'rb') as f:
                #     fpr_l = pickle.load(f)
                with open(f'{sv_dir}/images/{directory}/rates.pkl', 'rb') as f:
                    the_dict = pickle.load(f)[0]
                # Separate out the fpr and tpr from the dictionary
                the_dict.pop('Random', None)
                fpr_l, tpr_l = [], []
                for f, t in the_dict.values():
                    fpr_l += [f]
                    tpr_l += [t]
                passed = 1
            except Exception as e:
                print(e)
                passed = 0

            if passed:
                args = get_args(f'{sv_dir}/images/{directory}')
                # TODO: a better way of handling this, this is here just to catch CATHODE jobs on the full widths
                # if (args['bins'] in ['2300,3200,3400,3600,3800,4000', '3000,3200,3400,3600,3800,4000']) and \
                #         (args['feature_type'] == 3):
                what_youre_after = 0
                if (args['feature_type'] == 3) and (
                        args['bins'] in ['2900,3100,3300,3700,3900,4100',
                                         '3000,3200,3300,3700,3800,4000',
                                         '2700,3100,3300,3700,3900,4300']):
                    what_youre_after = 1
                # what_youre_after = args['feature_type'] == 3
                # if names[i] in ['Curtains', 'Cathode']:
                #     if not args['bins'] == '2700,2710,3300,3700,4990,5000':
                #         what_youre_after = 0
                if what_youre_after:
                    label = f'{args["doping"]}'
                    print(directory)
                    vals[label] += [[names[i], tpr_l, fpr_l]]

            with open(f'{sv_dir}/images/{filename}.pkl', 'wb') as f:
                pickle.dump(vals, f)
    else:
        with open(f'{sv_dir}/images/{filename}.pkl', 'rb') as f:
            vals = pickle.load(f)

    # Plot fig 6
    fig_six, ax_six = plt.subplots(1, 2, figsize=(14, 5))
    # We make this figure for this doping level
    data = deepcopy(vals[fig_6_doping])
    data += [['Random', [np.linspace(0, 1, 100000)], [np.linspace(0, 1, 100000)]]]
    max_sic = 20
    clrs = label_to_color
    for lst in data:
        label = lst[0]
        fpr_list = lst[2]
        tpr_list = lst[1]
        data = defaultdict(list)
        for tpr, fpr in zip(tpr_list, fpr_list):
            fpr_mx = fpr != 0.
            fpr_nz = fpr[fpr_mx]
            tpr_nz = tpr[fpr_mx]
            if label == 'Random':
                line = '--'
                color = 'k'
                alpha = 0.8
            else:
                line = '-'
                color = clrs[label]
                alpha = 0.1
            sic = tpr_nz / fpr_nz ** 0.5
            data['sic'] += [tpr_nz / fpr_nz ** 0.5]
            data['tpr'] += [tpr_nz]
            rejection = 1 / fpr_nz
            data['rejection'] += [rejection]
            if x_axis_rejection:
                x_axis = rejection
            else:
                x_axis = tpr_nz
            # Non-unique x-axis values must be masked out to avoid issues with the interpolation
            _, indx = np.unique(x_axis, return_index=True)
            data['interp_sic'] += [interp1d(x_axis[indx], sic[indx], fill_value="extrapolate")]
            data['interp_rejection'] += [interp1d(tpr_nz, rejection, fill_value="extrapolate")]
            if plot_individual_lines:
                ax_six[1].plot(x_axis, sic, linewidth=2, linestyle=line, color=color, alpha=alpha)
                ax_six[0].plot(tpr_nz, rejection, linewidth=2, linestyle=line, color=color, alpha=alpha)
            # ax_six[1].plot(fpr_nz, sic, linewidth=2, label=label, linestyle=line, color=color, alpha=alpha)
            # ax_six[1].set_xscale('log')
            # ax_six[0].plot(tpr_nz, data['rejection'][-1], linewidth=2, label=label, linestyle=line, color=color,
            #                alpha=alpha)
        if label != 'Random':

            def make_axis(name):
                mtp = np.concatenate(data[name])
                min_tpr, max_tpr = mtp.min(), mtp.max()
                return np.linspace(min_tpr, max_tpr, 1000)

            ax_six_0 = make_axis('tpr')

            if x_axis_rejection:
                # ax_six_1 = make_axis('rejection')
                # ax_six_1 = np.linspace(1, max_ax_six_1, 1000)
                if label in ['Cathode', 'Curtains']:
                    ax_six_1 = np.linspace(1, min_ax_six_1, 1000)
                else:
                    ax_six_1 = np.linspace(1, max_ax_six_1, 1000)
                ax_six[1].set_xscale('log')
            else:
                ax_six_1 = ax_six_0

            host = []
            reggie = []
            for sic_func, rej_func in zip(data['interp_sic'], data['interp_rejection']):
                host += [sic_func(ax_six_1)]
                reggie += [rej_func(ax_six_0)]
            host = np.vstack(host)
            reggie = np.vstack(reggie)

            if quart:
                mean_sic = np.median(host, 0)
                mean_rej = np.median(reggie, 0)
            else:
                mean_sic = host.mean(0)
                mean_rej = reggie.mean(0)

            print(max(mean_rej))
            if add_dashes:
                if label == 'Cathode':
                    if np.max(mean_sic) > 11:
                        line = '--'
                if label == 'Curtains':
                    if mean_sic[300] > 13:
                        line = '--'
            is_full = label.split('_')[-1] == 'full'
            if is_full:
                line = '--'

            ax_six[1].plot(ax_six_1, mean_sic, linewidth=2, label=label_to_name[label], linestyle=line, color=color)
            ax_six[0].plot(ax_six_0, mean_rej, linewidth=2, label=label_to_name[label], linestyle=line, color=color)
            if (not plot_individual_lines) and (not add_dashes):
                def add_band(data, x_axis, mean, ax):
                    if quart:
                        lb = np.quantile(data, 0.16, axis=0)
                        ub = np.quantile(data, 0.84, axis=0)
                    else:
                        # error = data.std(0)
                        # lb = mean - error
                        # ub = mean + error
                        lb = np.quantile(data, 0.16, axis=0)
                        ub = np.quantile(data, 0.84, axis=0)
                    if is_full:
                        ax.fill_between(x_axis, lb, ub, alpha=alpha, linewidth=2, linestyle=line, color=color,
                                        hatch="///////")
                    else:
                        ax.fill_between(x_axis, lb, ub, alpha=alpha, linewidth=2, linestyle=line, color=color)

                add_band(host, ax_six_1, mean_sic, ax_six[1])
                add_band(reggie, ax_six_0, mean_rej, ax_six[0])
        else:
            ax_six[1].plot(x_axis, sic, linewidth=2, linestyle=line, color=color, alpha=alpha, label='Random')
            mnx = rejection < max_ax_six_1
            ax_six[0].plot(tpr_nz[mnx], rejection[mnx], linewidth=2, linestyle=line, color=color,
                           alpha=alpha, label='Random')

        ax_six[1].set_ylim(0, max_sic)
        ax_six[1].set_xlim(1, min_ax_six_1)

    if x_axis_rejection:
        ax_six[1].set_xlabel('Rejection')
    else:
        ax_six[1].set_xlabel('Signal efficiency')
    ax_six[1].set_ylabel('Significance improvement')
    ax_six[0].set_xlabel('Signal efficiency')
    ax_six[0].set_ylabel('Rejection (1 / false positive rate)')
    ax_six[0].set_yscale('log')
    # ax_six[1].set_xscale('log')

    handles, labels = ax_six[0].get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    # fig_six.legend(*zip(*unique), loc='upper left', bbox_to_anchor=(0.9, 0.89), frameon=False)
    ax_six[0].legend(*zip(*unique), loc='upper right', frameon=False)
    ax_six[1].legend(*zip(*unique), loc='upper left', frameon=False)
    fig_six.savefig(f'{sv_dir}/images/figure_six.pdf', bbox_inches='tight')
    fig_six.clf()
    print('Fig 6 done.')

    # Plot fig 7
    runs = sorted(set(vals.keys()))
    n_runs = len(runs)
    unique_names = list(set(names))
    n_versions = len(unique_names)

    # clist = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f',
    #          u'#bcbd22', u'#17becf']
    # clrs = {unique_names[i]: clist[i] for i in range(n_versions)}

    # # TODO: these numbers are wrong, and ostensibly for the s/b fraction
    # doping_dict = {'8000': 4.80, '6000': 3.60, '4000': 2.40, '3000': 1.80, '1000': 0.60, '667': 0.40, '500': 0.30,
    #                '333': 0.20, '250': 0.15, '100': 0.10, '50': 0.05, '0': 0}
    # These counts are for the raw counts
    n_1000 = 369
    n_bg = 60000
    doping_dict = {'8000': int(n_1000 * 8), '6000': int(n_1000 * 6), '4000': int(n_1000 * 4),
                   '3000': int(n_1000 * 3), '1000': 369, '667': int(n_1000 * 0.66666666), '500': int(n_1000 * 0.5),
                   '333': int(n_1000 * 0.33333333), '250': int(n_1000 * 0.25),
                   '100': int(n_1000 * 0.1), '50': int(n_1000 * 0.05), '0': 0}

    central_line = defaultdict(lambda: defaultdict(list))
    for j in range(n_runs):
        label = runs[j]
        lst = vals[label]
        for values in lst:
            nm = values[0]
            tpr_list = values[1]
            fpr_list = values[2]
            sics_store = defaultdict(list)
            for tpr, fpr in zip(tpr_list, fpr_list):
                # Need to fit each of these, find the mean, then take the max, that gives the max sic at a fixed value.
                fpr_mx = fpr != 0.
                fpr_nz = fpr[fpr_mx]
                tpr_nz = tpr[fpr_mx]
                sic = tpr_nz / fpr_nz ** 0.5
                _, indx = np.unique(tpr_nz, return_index=True)
                sics_store['interp'] += [interp1d(tpr_nz[indx], sic[indx], fill_value="extrapolate")]
                sics_store['tpr'] += [tpr_nz[indx]]

            tpr_fit = np.linspace(0.0, 1.0, 1000)
            s = []
            for func in sics_store['interp']:
                s += [func(tpr_fit).reshape(1, -1)]
            sics = np.concatenate(s, 0)
            mean_sic = np.mean(sics, 0)
            max_ind = np.argmax(mean_sic)
            ub = np.quantile(sics, 0.84, 0)
            lb = np.quantile(sics, 0.16, 0)

            central_line[f'{nm}']['mean'] += [mean_sic[max_ind]]
            central_line[f'{nm}']['ub'] += [ub[max_ind]]
            central_line[f'{nm}']['lb'] += [lb[max_ind]]
            central_line[f'{nm}']['x'] += [doping_dict[label]]
            # if (label == '3000') and (nm == 'Curtains'):
            #     ff, aa = plt.subplots()
            #     aa.plot(tpr_fit, mean_sic)
            #     ff.savefig('test.pdf')

    alpha = 0.1

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for model_nm, model in central_line.items():
        sorted_inds = np.argsort(model['x'])
        for key in model.keys():
            model[key] = np.array(model[key])[sorted_inds]
        ax.plot(model['x'], model['mean'], linewidth=2, label=label_to_name[model_nm], color=clrs[model_nm])
        ax.fill_between(model['x'], model['lb'], model['ub'], alpha=alpha, linewidth=2,
                        color=clrs[model_nm])

    ax.set_xlabel(r'Number signal events', labelpad=10)
    ax.set_ylabel(r'Maximum significance improvement')
    ax.set_xscale('log')
    ldd = list(doping_dict.values())
    ax.set_xticks(ldd)
    ax.set_xticklabels(ldd)
    # ax.set_xlim(1800, 90)
    ax.set_xlim(int(n_1000 * 8), int(n_1000 * 0.5))
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    ax.set_yticks([1, 5, 10, 15, 20, 25])
    ax.set_ylim(0.95, 30)

    def count_to_percent(x):
        return x / n_bg * 100

    def percent_to_count(x):
        return x / 100 * n_bg

    secax = ax.secondary_xaxis('bottom', functions=(count_to_percent, percent_to_count))
    # secax.set_xlabel('S / B %')
    secax.set_xlabel(r'${}^{S}\!/_{B} \%$', fontsize=12)
    # secax.set_xlabel('{}^{S}\!/_{B}')
    ticks = count_to_percent(np.array(list(doping_dict.values())))
    secax.set_xticks(ticks)
    ticks = [f'{t:.2f}' for t in ticks]
    secax.set_xticklabels(ticks)
    secax.minorticks_off()

    # ax.set_yscale('log')
    plt.minorticks_off()
    # fig.legend(frameon=False, borderaxespad=4)
    ax.legend(frameon=False, loc='upper right')
    # fig.savefig(f'{sv_dir}/images/fig_7.pdf', bbox_inches='tight')
    fig.savefig(f'{sv_dir}/images/fig_7.pdf', bbox_inches='tight')
    fig.clf()


def figs_oversampling():
    sv_dir = get_top_dir()
    fig_6_doping = '1000'
    add_dashes = False

    # This is the Curtains oversampling
    directories = []
    c = 0
    for idf in [7, 8, 11]:
        directories += [f'ot_fig7_400_OT_fig7_400_{idf}_fig_400_os_{i}' for i in range(c, c + 9)]
        c += 9
    directories += [f'ideal_class_400_ideal_class_400_{i}' for i in range(0, 12)]
    names = [None] * 27 + ['Idealised'] * 12
    filename = 'oversampling_tests'
    fig_6_doping = '3000'
    os_to_take = [1, 2, 3, 4, 9]
    os_to_take = [f'{o * 60000}' for o in os_to_take] + ['Idealised']

    reload = 0
    plot_individual_lines = False
    quart = 0
    x_axis_rejection = 1
    # max_ax_six_1 = 12007
    max_ax_six_1 = 24268.4
    # max_ax_six_1 = 12134.4
    min_ax_six_1 = 12134.4

    if reload:
        # Gather saved quantities
        vals = defaultdict(list)
        for i, directory in enumerate(directories):
            try:
                # with open(f'{sv_dir}/images/{directory}/tpr_cathode.pkl', 'rb') as f:
                #     tpr_l = pickle.load(f)
                # with open(f'{sv_dir}/images/{directory}/fpr_cathode.npy', 'rb') as f:
                #     fpr_l = pickle.load(f)
                with open(f'{sv_dir}/images/{directory}/rates.pkl', 'rb') as f:
                    the_dict = pickle.load(f)[0]
                # Separate out the fpr and tpr from the dictionary
                the_dict.pop('Random', None)
                fpr_l, tpr_l = [], []
                for f, t in the_dict.values():
                    fpr_l += [f]
                    tpr_l += [t]
                passed = 1
            except Exception as e:
                print(e)
                passed = 0

            if passed:
                args = get_args(f'{sv_dir}/images/{directory}')
                label = f'{args["doping"]}'
                if names[i] is not None:
                    oversample = names[i]
                else:
                    oversample = f'{args["nx_bg_template"]}'
                vals[label] += [[oversample, tpr_l, fpr_l]]

            with open(f'{sv_dir}/images/{filename}.pkl', 'wb') as f:
                pickle.dump(vals, f)
    else:
        with open(f'{sv_dir}/images/{filename}.pkl', 'rb') as f:
            vals = pickle.load(f)

    # Plot fig 6
    fig_six, ax_six = plt.subplots(1, 2, figsize=(14, 5))
    # We make this figure for this doping level
    data = deepcopy(vals[fig_6_doping])
    data += [['Random', [np.linspace(0, 1, 100000)], [np.linspace(0, 1, 100000)]]]
    max_sic = 20
    viridis = cm.get_cmap('viridis', 9)
    clrs = {f'{i * 60000}': viridis(i / 9) for i in range(1, 10)}
    clrs['Idealised'] = label_to_color['Idealised']
    for lst in data:
        label = lst[0]
        if label in os_to_take:
            fpr_list = lst[2]
            tpr_list = lst[1]
            data = defaultdict(list)
            for tpr, fpr in zip(tpr_list, fpr_list):
                fpr_mx = fpr != 0.
                fpr_nz = fpr[fpr_mx]
                tpr_nz = tpr[fpr_mx]
                if label == 'Random':
                    line = '--'
                    color = 'k'
                    alpha = 0.8
                else:
                    line = '-'
                    color = clrs[label]
                    alpha = 0.1
                sic = tpr_nz / fpr_nz ** 0.5
                data['sic'] += [tpr_nz / fpr_nz ** 0.5]
                data['tpr'] += [tpr_nz]
                rejection = 1 / fpr_nz
                data['rejection'] += [rejection]
                if x_axis_rejection:
                    x_axis = rejection
                else:
                    x_axis = tpr_nz
                # Non-unique x-axis values must be masked out to avoid issues with the interpolation
                _, indx = np.unique(x_axis, return_index=True)
                data['interp_sic'] += [interp1d(x_axis[indx], sic[indx], fill_value="extrapolate")]
                data['interp_rejection'] += [interp1d(tpr_nz, rejection, fill_value="extrapolate")]
                if plot_individual_lines:
                    ax_six[1].plot(x_axis, sic, linewidth=2, linestyle=line, color=color, alpha=alpha)
                    ax_six[0].plot(tpr_nz, rejection, linewidth=2, linestyle=line, color=color, alpha=alpha)
                # ax_six[1].plot(fpr_nz, sic, linewidth=2, label=label, linestyle=line, color=color, alpha=alpha)
                # ax_six[1].set_xscale('log')
                # ax_six[0].plot(tpr_nz, data['rejection'][-1], linewidth=2, label=label, linestyle=line, color=color,
                #                alpha=alpha)
            if label != 'Random':

                def make_axis(name):
                    mtp = np.concatenate(data[name])
                    min_tpr, max_tpr = mtp.min(), mtp.max()
                    return np.linspace(min_tpr, max_tpr, 1000)

                ax_six_0 = make_axis('tpr')

                if x_axis_rejection:
                    # ax_six_1 = make_axis('rejection')
                    # ax_six_1 = np.linspace(1, max_ax_six_1, 1000)
                    if label in ['Cathode', 'Curtains']:
                        ax_six_1 = np.linspace(1, min_ax_six_1, 1000)
                    else:
                        ax_six_1 = np.linspace(1, max_ax_six_1, 1000)
                    ax_six[1].set_xscale('log')
                else:
                    ax_six_1 = ax_six_0

                host = []
                reggie = []
                for sic_func, rej_func in zip(data['interp_sic'], data['interp_rejection']):
                    host += [sic_func(ax_six_1)]
                    reggie += [rej_func(ax_six_0)]
                host = np.vstack(host)
                reggie = np.vstack(reggie)

                if quart:
                    mean_sic = np.median(host, 0)
                    mean_rej = np.median(reggie, 0)
                else:
                    mean_sic = host.mean(0)
                    mean_rej = reggie.mean(0)

                print(max(mean_rej))
                if add_dashes:
                    if label == 'Cathode':
                        if np.max(mean_sic) > 11:
                            line = '--'
                    if label == 'Curtains':
                        if mean_sic[300] > 13:
                            line = '--'
                is_full = label.split('_')[-1] == 'full'
                if is_full:
                    line = '--'

                ax_six[1].plot(ax_six_1, mean_sic, linewidth=2, label=label, linestyle=line, color=color)
                ax_six[0].plot(ax_six_0, mean_rej, linewidth=2, label=label, linestyle=line, color=color)
                if (not plot_individual_lines) and (not add_dashes):
                    def add_band(data, x_axis, mean, ax):
                        if quart:
                            lb = np.quantile(data, 0.16, axis=0)
                            ub = np.quantile(data, 0.84, axis=0)
                        else:
                            # error = data.std(0)
                            # lb = mean - error
                            # ub = mean + error
                            lb = np.quantile(data, 0.16, axis=0)
                            ub = np.quantile(data, 0.84, axis=0)
                        if is_full:
                            ax.fill_between(x_axis, lb, ub, alpha=alpha, linewidth=2, linestyle=line, color=color,
                                            hatch="///////")
                        else:
                            ax.fill_between(x_axis, lb, ub, alpha=alpha, linewidth=2, linestyle=line, color=color)

                    add_band(host, ax_six_1, mean_sic, ax_six[1])
                    add_band(reggie, ax_six_0, mean_rej, ax_six[0])
            else:
                ax_six[1].plot(x_axis, sic, linewidth=2, linestyle=line, color=color, alpha=alpha, label='Random')
                mnx = rejection < max_ax_six_1
                ax_six[0].plot(tpr_nz[mnx], rejection[mnx], linewidth=2, linestyle=line, color=color,
                               alpha=alpha, label='Random')

            ax_six[1].set_ylim(0, max_sic)
            ax_six[1].set_xlim(1, min_ax_six_1)

    if x_axis_rejection:
        ax_six[1].set_xlabel('Rejection')
    else:
        ax_six[1].set_xlabel('Signal efficiency')
    ax_six[1].set_ylabel('Significance improvement')
    ax_six[0].set_xlabel('Signal efficiency')
    ax_six[0].set_ylabel('Rejection (1 / false positive rate)')
    ax_six[0].set_yscale('log')
    # ax_six[1].set_xscale('log')

    handles, labels = ax_six[0].get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    # fig_six.legend(*zip(*unique), loc='upper left', bbox_to_anchor=(0.9, 0.89), frameon=False)
    ax_six[0].legend(*zip(*unique), loc='upper right', frameon=False)
    ax_six[1].legend(*zip(*unique), loc='upper left', frameon=False)
    fig_six.savefig(f'{sv_dir}/images/oversampling.pdf', bbox_inches='tight')
    fig_six.clf()
    print('Fig 6 done.')


if __name__ == '__main__':
    # get_counts()
    # figs_six_and_seven()
    # figs_oversampling()
    # get_sics()
    # get_max_sic()
    plot_counts_()
