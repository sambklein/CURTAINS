# Some plotting functions
import colorsys

import numpy
import numpy as np
import torch
from matplotlib import pyplot as plt  # , pyplot  # , pyplot

# import matplotlib.patches as mpatches
# from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.metrics import auc

from utils.io import get_top_dir
from utils.torch_utils import tensor2numpy, shuffle_tensor
from scipy import stats


def get_bins(data, nbins=20):
    max_ent = data.max().item()
    min_ent = data.min().item()
    return np.linspace(min_ent, max_ent, num=nbins)


def get_mask(x, bound):
    return np.logical_and(x > bound[0], x < bound[1])


def apply_bound(data, bound):
    mask = np.logical_and(get_mask(data[:, 0], bound), get_mask(data[:, 1], bound))
    return data[mask, 0], data[mask, 1]


def plot2Dhist(data, ax, bins=50, bounds=None):
    if bounds:
        x, y = apply_bound(data, bounds)
    else:
        x = data[:, 0]
        y = data[:, 1]
    count, xbins, ybins = np.histogram2d(x, y, bins=bins)
    count[count == 0] = np.nan
    ax.imshow(count.T,
              origin='lower', aspect='auto',
              extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()],
              )


def add_error_hist(ax, data, bins, color, error_bars=False, normalised=True, label='', norm=None):
    y, binEdges = np.histogram(data, bins=bins)
    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    width = 0.05
    norm_passed = norm is None
    n_fact = np.sum(y) if norm_passed else norm
    menStd = np.sqrt(y)
    if normalised or norm_passed:
        y = y / n_fact
        menStd = menStd / n_fact
    if error_bars:
        ax.errorbar(bincenters, y, yerr=menStd, color=color, fmt='.', label=label)
    else:
        ax.bar(bincenters, menStd, width=width, edgecolor=color, lw=0, fc=(0, 0, 0, 0),
               bottom=y, hatch='\\\\\\\\\\', label=label)
        ax.bar(bincenters, -menStd, width=width, edgecolor=color, lw=0, fc=(0, 0, 0, 0),
               bottom=y, hatch='\\\\\\\\\\', label=label)


def add_hist(ax, data, bin, color, label):
    ax.hist(data, bins=bin, density=False, histtype='step', color=color, label=label, weights=get_weights(data))


def get_weights(data):
    return np.ones_like(data) / len(data)


def add_off_diagonal(axes, i, j, data, color):
    bini = get_bins(data[:, i])
    binj = get_bins(data[:, j])
    f1 = tensor2numpy(data[:, i])
    f2 = tensor2numpy(data[:, j])
    axes[i, j].hist2d(f1, f2, bins=[bini, binj], density=True, cmap=color)
    axes[i, j].set_xlim([-1, 1.1])
    axes[i, j].set_ylim([-1, 1.2])
    # Pearson correlation
    # coef = np.corrcoef(f1, f2)[0, 1]
    # Spearman correlation between features
    coef, pval = stats.spearmanr(f1, f2)
    axes[i, j].annotate(f'SPR {coef:.2f}', xy=(0.95, 0.95), xycoords='axes fraction', ha='right', va='top', size=6)


def kde_plot(x, y, axes, levels=5):
    sns.kdeplot(tensor2numpy(x), y=tensor2numpy(y), ax=axes, alpha=0.4, levels=levels, color='red', fill=True)


def add_contour(axes, i, j, data, sampled):
    sns.kdeplot(tensor2numpy(data[:, j]), y=tensor2numpy(data[:, i]), ax=axes[i, j], alpha=0.4, levels=3,
                color='red', fill=True)
    sns.kdeplot(tensor2numpy(sampled[:, j]), y=tensor2numpy(sampled[:, i]), ax=axes[i, j], alpha=0.4, levels=3,
                color='blue', fill=True)
    axes[i, j].set_xlim([-1.2, 1.2])
    axes[i, j].set_ylim([-1.2, 1.2])
    # axes[i, j].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    # if j - i > 1:
    #     axes[i, j].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
    #                            labelbottom=False, labelleft=False)


def getFeaturePlot(model, original, sampled, lm_sample, nm, savedir, region, feature_names, nbins=20, contour=True,
                   n_sample_for_plot=-1):
    if n_sample_for_plot > 0:
        original = shuffle_tensor(original)
        sampled = shuffle_tensor(sampled)

    nfeatures = len(feature_names) - 1
    fig, axes = plt.subplots(nfeatures, nfeatures, figsize=(2 * nfeatures + 2, 2 * nfeatures + 1),
                             gridspec_kw={'wspace': 0.03, 'hspace': 0.03})
    for i in range(nfeatures):
        if (i != 0) or (not contour):
            axes[i, 0].set_ylabel(feature_names[i])
        else:
            axes[0, 0].set_ylabel('Normalised Entries', horizontalalignment='right', y=1.0)
        for j in range(nfeatures):
            if not contour:
                axes[0, j].set_title(feature_names[j])
            else:
                axes[-1, j].set_xlabel(feature_names[j])
                if i != nfeatures - 1:
                    axes[i, j].tick_params(axis='x', which='both', direction='in', labelbottom=False)

                axes[i, j].set_yticks([-1, 0, 1])
                if i == j == 0:
                    axes[i, j].tick_params(axis='y', colors='w')
                elif j > 0:
                    axes[i, j].tick_params(axis='y', which='both', direction='in', labelleft=False)

            if i == j:
                og = original[:, i]
                bin = get_bins(og[(og > -1.2) & (og < 1.2)], nbins=nbins)
                add_hist(axes[i, j], model.get_numpy(original[:, i]), bin, 'red', 'Original')
                add_hist(axes[i, j], model.get_numpy(sampled[:, i]), bin, 'blue', 'Transformed')
                add_error_hist(axes[i, j], model.get_numpy(original[:, i]), bins=bin, color='red')
                add_error_hist(axes[i, j], model.get_numpy(sampled[:, i]), bins=bin, color='blue')
                data = model.get_numpy(lm_sample[:, i])
                axes[i, j].hist(data, density=False, bins=bin, histtype='step',
                                color='black', linestyle='dashed', label='Input Sample', weights=get_weights(data))
                axes[i, j].set_xlim([-1.2, 1.2])

            if contour:
                if i > j:
                    add_contour(axes, i, j, original[:n_sample_for_plot], sampled[:n_sample_for_plot])
                elif i < j:
                    axes[i, j].set_visible(False)
            else:
                if i < j:
                    add_off_diagonal(axes, i, j, original, 'Reds')

                if i > j:
                    add_off_diagonal(axes, i, j, sampled, 'Blues')

    fig.suptitle(region)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.32, 0.89), frameon=False)
    plt.savefig(savedir + '/featurespread_{}_{}_{}.png'.format(region, nm, 'transformed_data'), bbox_inches="tight")
    plt.clf()


def getCrossFeaturePlot(model, original, sampled, nm, savedir, mass, feature_names):
    nfeatures = len(feature_names) - 1
    fig, axes = plt.subplots(nfeatures, nfeatures, figsize=(2 * nfeatures + 2, 2 * nfeatures - 1))
    for i in range(nfeatures):
        axes[i, 0].set_ylabel(feature_names[i])
        for j in range(nfeatures):
            axes[0, j].set_title(feature_names[j] + '_trans')

            if i <= j:
                bini = get_bins(original[:, i])
                binj = get_bins(sampled[:, j])
                axes[i, j].hist2d(model.get_numpy(original[:, i]), model.get_numpy(sampled[:, j]), bins=[bini, binj],
                                  density=True, cmap='Reds')
    fig.suptitle(f"Mass: {mass + 1}")
    fig.tight_layout()
    plt.savefig(savedir + '/feature_correlations_{}_{}_{}.png'.format(mass, nm, 'transformed_data'))


def get_counts(data, to_slice, bound=4, nbins=50):
    bin_edges = np.linspace(-bound, bound, nbins + 1)
    # Apply a slice to the data
    mask = torch.all((to_slice > 0) & (to_slice < 2), 1)
    data = data[mask.type(torch.bool)].cpu().numpy()
    return np.histogram2d(data[:, 0], data[:, 1], bins=bin_edges)[0]


def hist_features(originals, sample, data_dim, axs, axs_nms=None, labels=None, legend=True):
    if labels is None:
        labels = ['original', 'samples']
    for i in range(data_dim):
        bins = get_bins(originals[:, i])
        axs[i].hist(tensor2numpy(originals[:, i]), label=labels[0], alpha=0.5, density=True, bins=bins,
                    histtype='step')
        # Plot samples drawn from the model
        axs[i].hist(tensor2numpy(sample[:, i]), label=labels[1], alpha=0.5, density=True, bins=bins, histtype='step')
        if axs_nms:
            axs[i].set_title(axs_nms[i])
        else:
            axs[i].set_title('Feature {}'.format(i))
        if legend:
            axs[i].legend()


def hist_features_single(originals, feature_nms, axs, bins, label='data'):
    data_dim = len(feature_nms) - 1
    # if not isinstance(bins, list):
    #     bins = [bins] * data_dim
    for i in range(data_dim):
        axs[i].hist(tensor2numpy(originals[:, i]), label=label, alpha=0.5, density=True, bins=bins[i],
                    histtype='step')
        axs[i].set_title(feature_nms[i])


def plot_single_feature_mass_diagnostic(model, samples, generating_data, feature_names, sv_dir, title, nm):
    generating_mass = generating_data[:, -1]
    nfeatures = samples.shape[1]
    fig, ax = plt.subplots(1, nfeatures, figsize=(5 * nfeatures + 2, 5))
    binx = get_bins(generating_mass)
    # TODO: this is SLOW, need ot write something that can do histograms in batches and accumulate
    for i in range(nfeatures):
        biny = get_bins(samples[:, i])
        ax[i].hist2d(model.get_numpy(generating_mass), model.get_numpy(samples[:, i]), alpha=0.5, density=True,
                     bins=[binx, biny])
        ax[i].set_ylabel(feature_names[i])
        ax[i].set_xlabel('Target Mass')
    fig.suptitle(title)
    fig.savefig(sv_dir + f'/features_mass_diagnostic_{nm}')


def plot_rates_dict(sv_dir, rates_dict, title):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot([0, 1], [0, 1], 'k--')
    for nm, rates in rates_dict.items():
        fpr, tpr = rates
        c_auc = f' AUC: {auc(fpr, tpr):.2f}'
        if 'Supervised' in nm:
            label = nm + c_auc
        else:
            label = f'Doping {nm}%' + c_auc
        ax.plot(fpr, tpr, linewidth=2, label=label)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title(title)
    ax.set_aspect('equal')
    # fig.tight_layout(rect=[0, 0, 0.9, 1])
    lgd = fig.legend(bbox_to_anchor=(0.76, 0.94), loc='upper left')
    fig.savefig(sv_dir + f'/{title}_roc.png', bbox_extra_artists=(lgd,), bbox_inches='tight')


def get_windows_plot(bgspectra, anomalyspectra, woi, windows, sv_dir):
    '''
        args: bgspectra: The background masses: np array.
              anomalyspectra: The anomaly masses: np array.
              woi: Window of Interest: Tuple of masses - These are the mass bounds of this plot.
              windows: SB1, SB2, SW bins. Pass the args.bins here.
              sv_dir = where you want to save the plot.
    '''
    fig, ax = plt.subplots()
    anomalyspectra = anomalyspectra.values
    bgspectra = bgspectra.values

    sigAnomaly = anomalyspectra[np.where(np.logical_and(anomalyspectra > windows[2], anomalyspectra < windows[3]))]

    bgcount, bins, _ = ax.hist(bgspectra, bins=np.arange(woi[0], woi[1], 2), label='QCD', histtype='step')
    # count, _ , _ =ax.hist(anomalyspectra, bins=bins, label='Signal', histtype='step')
    # ax.hist(sigAnomaly, bins=bins, label='Signal', histtype='step', color='red')
    ax.hist(anomalyspectra, bins=bins, histtype='step', color='red')
    ax.axvspan(windows[1], windows[2], ymin=0., ymax=1.5 * np.max(bgcount), alpha=0.1, color='green',
               label='Side bands')
    ax.axvspan(windows[2], windows[3], ymin=0., ymax=1.5 * np.max(bgcount), alpha=0.1, color='red',
               label='Signal Window')
    ax.axvspan(windows[3], windows[4], ymin=0., ymax=1.5 * np.max(bgcount), alpha=0.1, color='green')
    ax.vlines([windows[1], windows[2], windows[3], windows[4]], ymin=0, ymax=1.5 * np.max(bgcount), ls='dashed',
              color='black')
    plt.legend(frameon=False)
    ax.set_xlabel("Mass (Gev)")
    ax.set_ylabel("Count")
    ax.set_ylim(0, 1.5 * np.max(bgcount))

    fig.savefig(f'{sv_dir}_windows.png')


def getInputTransformedHist(model, input, transformed, nm, savedir, region, feature_names):
    s1 = input.data.shape[0]
    s2 = transformed.data.shape[0]
    print(f"Region: {region}, Input {input.data.shape[0]}, Transformed {transformed.data.shape[0]}\n")
    nsamp = min(s1, s2)
    input = input[:nsamp]
    transformed = transformed[:nsamp]

    nfeatures = len(feature_names) - 1

    fig, axes = plt.subplots(1, nfeatures, figsize=(8 * nfeatures + 3, 8), sharex=True, sharey=True)
    for i in range(nfeatures):
        axes[i].set_xlabel(feature_names[i])
        axes[i].hist2d(model.get_numpy(input[:, i]), model.get_numpy(transformed[:, i]),
                       bins=[np.linspace(-1, 1, 60), np.linspace(-1, 1, 60)], alpha=0.6)
        axes[i].set_xlim(-1.0, 1.0)
        axes[i].set_ylim(-1.0, 1.0)

    fig.suptitle(region)
    plt.tight_layout()
    plt.savefig(savedir + '/InputvTransformedHist_{}.png'.format(region))
    plt.clf()


def plot_delta_mass(deltas, true_deltas=None, name=None, alpha=0.5, bins=50):
    deltas = tensor2numpy(deltas)
    fig = plt.figure(figsize=(8, 5))
    plt.hist(deltas, bins=bins, alpha=alpha, label='Training')
    if true_deltas is not None:
        true_deltas = tensor2numpy(true_deltas)
        plt.hist(true_deltas, bins=bins, alpha=alpha, label='True')
        plt.legend()
    if name is None:
        name = f'{get_top_dir()}/images/delta_mass.png'
    plt.savefig(name)
    fig.clf()
