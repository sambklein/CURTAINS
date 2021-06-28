# Some plotting functions
import colorsys

import os
import numpy as np
import torch
from matplotlib import pyplot as plt  # , pyplot  # , pyplot

# import matplotlib.patches as mpatches
# from sklearn.manifold import TSNE
# import seaborn as sns
from utils.torch_utils import torch2numpy


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


# # From Johnny
# def projectiontionLS_2D(dim1, dim2, latent_space, *args, **kwargs):
#     '''Plot a two dimension latent space projection with marginals showing each dimension.
#     Can overlay multiple different datasets by passing more than one latent_space argument.
#     Inputs:
#         dim1: First LS dimension to plot on x axis
#         dim2: Second LS dimension to plot on y axis
#         latent_space (latent_space2, latent_space3...): the data to plot
#     Optional:
#         xrange: specify xrange in form [xmin,xmax]
#         yrange: specify xrange in form [ymin,ymax]
#         labels: labels in form ['ls1','ls2','ls3'] to put in legend
#         Additional options will be passed to the JointGrid __init__ function
#     Returns:
#         seaborn JointGrid object
#     '''
#     if 'xrange' in kwargs:
#         xrange = kwargs.get('xrange')
#     else:
#         xrange = (np.floor(np.quantile(latent_space[:, dim1], 0.02)), np.ceil(np.quantile(latent_space[:, dim1], 0.98)))
#     if 'yrange' in kwargs:
#         yrange = kwargs.get('yrange')
#     else:
#         yrange = (np.floor(np.quantile(latent_space[:, dim2], 0.02)), np.ceil(np.quantile(latent_space[:, dim2], 0.98)))
#     labels = [None] * (1 + len(args))
#     if 'labels' in kwargs:
#         labels = kwargs.get('labels')
#     kwargs.pop('xrange', None)
#     kwargs.pop('yrange', None)
#     kwargs.pop('labels', None)
#     g = sns.JointGrid(latent_space[:, dim1], latent_space[:, dim2], xlim=xrange, ylim=yrange, **kwargs)
#     # for label in [0,1]:
#     sns.kdeplot(latent_space[:, dim1], ax=g.ax_marg_x, legend=False, shade=True, alpha=0.3, label=labels[0])
#     sns.kdeplot(latent_space[:, dim2], ax=g.ax_marg_y, vertical=True, legend=False, shade=True, alpha=0.3,
#                 label=labels[0])
#     sns.kdeplot(latent_space[:, dim1], latent_space[:, dim2], ax=g.ax_joint, shade=True, shade_lowest=False, bw=0.2,
#                 alpha=1, label=labels[0])
#     i = 1
#     for ls in args:
#         sns.kdeplot(ls[:, dim1], ax=g.ax_marg_x, legend=False, shade=True, alpha=0.3, label=labels[i])
#         sns.kdeplot(ls[:, dim2], ax=g.ax_marg_y, vertical=True, legend=False, shade=True, alpha=0.3, label=labels[i])
#         sns.kdeplot(ls[:, dim1], ls[:, dim2], ax=g.ax_joint, shade=True, shade_lowest=False, bw=0.2, alpha=0.4,
#                     label=labels[i])
#         i += 1
#     g.ax_joint.spines['right'].set_visible(True)
#     g.ax_joint.spines['top'].set_visible(True)
#     g.set_axis_labels('LS Dim. {}'.format(dim1), 'LS Dim. {}'.format(dim2))
#     if labels[0] is not None:
#         g.ax_joint.legend()
#     return g

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
    f1 = torch2numpy(data[:, i])
    f2 = torch2numpy(data[:, j])
    axes[i, j].hist2d(f1, f2, bins=[bini, binj], density=True, cmap=color)
    coef = np.corrcoef(f1, f2)
    axes[i, j].set_xlim([-1, 1.1])
    axes[i, j].set_ylim([-1, 1.2])
    axes[i, j].annotate(f'PC {coef[0, 1]:.2f}', xy=(0.95, 0.95), xycoords='axes fraction', ha='right', va='top', size=6)
    # bbox=dict(boxstyle='round', fc='w'), size=6)


def getFeaturePlot(model, original, sampled, lm_sample, nm, savedir, region, feature_names, nbins=20):
    nfeatures = len(feature_names) - 1
    fig, axes = plt.subplots(nfeatures, nfeatures, figsize=(2 * nfeatures + 2, 2 * nfeatures - 1))
    sigcolour = ['red', 'blue']
    for i in range(nfeatures):
        axes[i, 0].set_ylabel(feature_names[i])
        for j in range(nfeatures):
            axes[0, j].set_title(feature_names[j])

            if i == j:
                bin = get_bins(original[:, i], nbins=nbins)
                add_hist(axes[i, j], model.get_numpy(original[:, i]), bin, 'red', 'Original')
                add_hist(axes[i, j], model.get_numpy(sampled[:, i]), bin, 'blue', 'Transformed')
                add_error_hist(axes[i, j], model.get_numpy(original[:, i]), bins=bin, color='red')
                add_error_hist(axes[i, j], model.get_numpy(sampled[:, i]), bins=bin, color='blue')
                data = model.get_numpy(lm_sample[:, i])
                axes[i, j].hist(data, density=False, bins=bin, histtype='step',
                                color='black', linestyle='dashed', label='Input Sample', weights=get_weights(data))

            if i < j:
                add_off_diagonal(axes, i, j, original, 'Reds')

            if i > j:
                add_off_diagonal(axes, i, j, sampled, 'Blues')

    # fig.legend(signal_handle, signal_labels, bbox_to_anchor=(1.001, 0.99), frameon=False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    fig.suptitle(region)
    fig.tight_layout()
    plt.savefig(savedir + '/featurespread_{}_{}_{}.png'.format(region, nm, 'transformed_data'))


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


def hist_features(originals, sample, model, data_dim, axs, axs_nms=None):
    for i in range(data_dim):
        bins = get_bins(originals[:, i])
        axs[i].hist(model.get_numpy(originals[:, i]), label='original', alpha=0.5, density=True, bins=bins,
                    histtype='step')
        # Plot samples drawn from the model
        axs[i].hist(model.get_numpy(sample[:, i]), label='samples', alpha=0.5, density=True, bins=bins, histtype='step')
        if axs_nms:
            axs[i].set_title(axs_nms[i])
        else:
            axs[i].set_title('Feature {}'.format(i))
        axs[i].legend()


def hist_features_single(originals, model, feature_nms, axs, bins, label='data'):
    data_dim = len(feature_nms) - 1
    for i in range(data_dim):
        axs[i].hist(model.get_numpy(originals[:, i]), label=label, alpha=0.5, density=True, bins=bins[i],
                    histtype='step')
        axs[i].set_title(feature_nms[i])
        # axs[i].legend()


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
        if 'Supervised' in nm:
            label = nm
        else:
            label = f'Doping {nm}%'
        ax.plot(fpr, tpr, linewidth=2, label=label)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title(title)
    fig.tight_layout(rect=[0, 0, 0.75, 1])
    fig.legend(bbox_to_anchor=(0.76, 0.94), loc='upper left')
    fig.savefig(sv_dir + f'/{title}_roc.png')


def get_anomaly_mass_plot(masses, sv_dir, name, title):
    
    fig, ax = plt.subplots()
    ax.hist(masses)
    ax.set_xlabel("Mass")
    plt.title(title)
    fig.savefig(f'{sv_dir}_mass_distribution_{name}.png')


def get_windows_plot(bgspectra, anomalyspectra, woi, windows, sv_dir):

    '''
        args: bgspectra: The background masses: np array.
              anomalyspectra: The anomaly masses: np array.
              woi: Window of Interest: Tuple of masses - These are the mass bounds of this plot.
              windows: SB1, SB2, SW bins. Pass the args.bins here.
              sv_dir = where you want to save the plot.
    '''
    
    #TODO: Eventually need to replace this when we dope the whole spectra?
    anomalyspectra = np.random.choice(anomalyspectra, 50000)
    fig, ax = plt.subplots()
    
    bgcount, bins, _ = ax.hist(bgspectra, bins=np.arange(woi[0], woi[1], 5), label='QCD', histtype='step')
    count, _ , _ =ax.hist(anomalyspectra, bins=bins, label='Signal', histtype='step')
    ax.axvspan(windows[1], windows[2], ymin=0., ymax=1.5*np.max(bgcount), alpha=0.1, color='green', label='Side bands')
    ax.axvspan(windows[2], windows[3], ymin=0., ymax=1.5*np.max(bgcount), alpha=0.1, color='red', label='Signal Window')
    ax.axvspan(windows[3], windows[4], ymin=0., ymax=1.5*np.max(bgcount), alpha=0.1, color='green')
    ax.vlines([windows[1], windows[2], windows[3], windows[4]], ymin=0, ymax=1.5*np.max(bgcount), ls='dashed', color='black')
    plt.legend(frameon=False)
    ax.set_xlabel("Mass (Gev)")
    ax.set_ylabel("Count")
    ax.set_ylim(0, 1.5*np.max(bgcount))
    
    if not os.path.exists(sv_dir):
        os.makedirs(sv_dir)
    
    fig.savefig(f'{sv_dir}/windows.png')