# Some plotting functions
import colorsys
import numpy as np
import torch
from matplotlib import pyplot as plt, pyplot
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
import seaborn as sns


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


# From Johnny
def projectiontionLS_2D(dim1, dim2, latent_space, *args, **kwargs):
    '''Plot a two dimension latent space projection with marginals showing each dimension.
    Can overlay multiple different datasets by passing more than one latent_space argument.
    Inputs:
        dim1: First LS dimension to plot on x axis
        dim2: Second LS dimension to plot on y axis
        latent_space (latent_space2, latent_space3...): the data to plot
    Optional:
        xrange: specify xrange in form [xmin,xmax]
        yrange: specify xrange in form [ymin,ymax]
        labels: labels in form ['ls1','ls2','ls3'] to put in legend
        Additional options will be passed to the JointGrid __init__ function
    Returns:
        seaborn JointGrid object
    '''
    if 'xrange' in kwargs:
        xrange = kwargs.get('xrange')
    else:
        xrange = (np.floor(np.quantile(latent_space[:, dim1], 0.02)), np.ceil(np.quantile(latent_space[:, dim1], 0.98)))
    if 'yrange' in kwargs:
        yrange = kwargs.get('yrange')
    else:
        yrange = (np.floor(np.quantile(latent_space[:, dim2], 0.02)), np.ceil(np.quantile(latent_space[:, dim2], 0.98)))
    labels = [None] * (1 + len(args))
    if 'labels' in kwargs:
        labels = kwargs.get('labels')
    kwargs.pop('xrange', None)
    kwargs.pop('yrange', None)
    kwargs.pop('labels', None)
    g = sns.JointGrid(latent_space[:, dim1], latent_space[:, dim2], xlim=xrange, ylim=yrange, **kwargs)
    # for label in [0,1]:
    sns.kdeplot(latent_space[:, dim1], ax=g.ax_marg_x, legend=False, shade=True, alpha=0.3, label=labels[0])
    sns.kdeplot(latent_space[:, dim2], ax=g.ax_marg_y, vertical=True, legend=False, shade=True, alpha=0.3,
                label=labels[0])
    sns.kdeplot(latent_space[:, dim1], latent_space[:, dim2], ax=g.ax_joint, shade=True, shade_lowest=False, bw=0.2,
                alpha=1, label=labels[0])
    i = 1
    for ls in args:
        sns.kdeplot(ls[:, dim1], ax=g.ax_marg_x, legend=False, shade=True, alpha=0.3, label=labels[i])
        sns.kdeplot(ls[:, dim2], ax=g.ax_marg_y, vertical=True, legend=False, shade=True, alpha=0.3, label=labels[i])
        sns.kdeplot(ls[:, dim1], ls[:, dim2], ax=g.ax_joint, shade=True, shade_lowest=False, bw=0.2, alpha=0.4,
                    label=labels[i])
        i += 1
    g.ax_joint.spines['right'].set_visible(True)
    g.ax_joint.spines['top'].set_visible(True)
    g.set_axis_labels('LS Dim. {}'.format(dim1), 'LS Dim. {}'.format(dim2))
    if labels[0] is not None:
        g.ax_joint.legend()
    return g


def getFeaturePlot(model, original, sampled, nm, savedir, region, feature_names):
    nfeatures = len(feature_names) - 1
    fig, axes = plt.subplots(nfeatures, nfeatures, figsize=(2 * nfeatures + 2, 2 * nfeatures - 1))
    sigcolour = ['red', 'blue']
    signal_handle = [mpatches.Patch(color=colors) for colors in sigcolour]
    signal_labels = ["Original", "Sampled"]
    for i in range(nfeatures):
        axes[i, 0].set_ylabel(feature_names[i])
        for j in range(nfeatures):
            axes[0, j].set_title(feature_names[j])

            if i == j:
                bin = get_bins(original[:, i])
                _, bins, _ = axes[i, j].hist(model.get_numpy(original[:, i]), bins=bin, density=True, histtype='step',
                                             color='red')
                axes[i, j].hist(model.get_numpy(sampled[:, i]), density=True, bins=bins, histtype='step', color='blue')

            if i < j:
                bini = get_bins(original[:, i])
                binj = get_bins(original[:, j])
                axes[i, j].hist2d(model.get_numpy(original[:, i]), model.get_numpy(original[:, j]), bins=[bini, binj],
                                  density=True, cmap='Reds')

            if i > j:
                bini = get_bins(sampled[:, i])
                binj = get_bins(sampled[:, j])
                axes[i, j].hist2d(model.get_numpy(sampled[:, j]), model.get_numpy(sampled[:, i]), bins=[binj, bini],
                                  density=True, cmap="Blues")

    fig.legend(signal_handle, signal_labels, bbox_to_anchor=(1.001, 0.99), frameon=False)
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


def hist_features(originals, sample, model, data_dim, axs):
    for i in range(data_dim):
        bins = get_bins(originals[:, i])
        axs[i].hist(model.get_numpy(originals[:, i]), label='original', alpha=0.5, density=True, bins=bins,
                    histtype='step')
        # Plot samples drawn from the model
        axs[i].hist(model.get_numpy(sample[:, i]), label='samples', alpha=0.5, density=True, bins=bins, histtype='step')
        axs[i].set_title('Feature {}'.format(i))
        axs[i].legend()


def hist_features_single(originals, model, feature_nms, axs, bins, label='data'):
    data_dim = len(feature_nms) - 1
    for i in range(data_dim):
        axs[i].hist(model.get_numpy(originals[:, i]), label=label, alpha=0.5, density=True, bins=bins[i],
                    histtype='step')
        axs[i].set_title(feature_nms[i])
        # axs[i].legend()


def plot_single_feature_mass_diagnostic(model, samples, generating_data, feature_names, sv_dir, region, title):
    generating_mass = torch.cat([generating_data[:, -1]] * int(samples.shape[0] / generating_data.shape[0]))
    nfeatures = samples.shape[1]
    fig, ax = plt.subplots(1, nfeatures, figsize=(5 * nfeatures + 2, 5))
    binx = get_bins(generating_mass)
    for i in range(nfeatures):
        biny = get_bins(samples[:, i])
        ax[i].hist2d(model.get_numpy(generating_mass), model.get_numpy(samples[:, i]), alpha=0.5, density=True,
                     bins=[binx, biny])
        ax[i].set_ylabel(feature_names[i])
        ax[i].set_xlabel('Gen Mass')
    fig.suptitle(title)
    fig.savefig(sv_dir + f'/features_mass_diagnostic_{region}')