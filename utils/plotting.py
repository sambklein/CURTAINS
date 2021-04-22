# Some plotting functions
import colorsys
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns


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


def get_colors(num_colors):
    colors = []
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i / 360.
        lightness = (50 + np.random.rand() * 10) / 100.
        saturation = (90 + np.random.rand() * 10) / 100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors


def plot_embedding(embed, y_test, clrs, ax=None, title='', dims=(8, 8)):
    if not ax:
        plt.figure(figsize=dims)
        ax = plt
    else:
        ax.set_title(title)
    for i in range(9):
        mx = y_test == i
        ax.plot(embed[:, 0][mx], embed[:, 1][mx], 'x', color=clrs[i], alpha=0.3)


def hist_latents(inp, title='', bins=20):
    fig, ax = plt.subplots(1, inp.shape[1], figsize=(20, 5))
    fig.suptitle(title, fontsize=16)
    for i in range(inp.shape[1]):
        ax[i].hist(inp[:, i], bins=bins)


def plot_latents(encoder, data, title=None):
    enc_ims = encoder(data)
    if enc_ims.shape[1] > 2:
        x_embeddor = TSNE(n_components=2)
        X_emb = x_embeddor.fit_transform(enc_ims)
    else:
        X_emb = enc_ims

    clrs = get_colors(9)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plot_embedding(X_emb, clrs, ax=ax, title='Latent Space')
    if title:
        plt.savefig(title)


def plot_samples_mnist(samples, title='', subtitles=None, subfnt=50):
    samples = np.array(samples)
    nsamples = len(samples)
    fig, ax = plt.subplots(1, nsamples, figsize=(10 * nsamples, 10))
    for i, sample in enumerate(samples):
        sample = sample.reshape(28, 28)
        ax[i].imshow(sample)
        if subtitles:
            ax[i].set_title(subtitles[i], fontsize=subfnt)
    fig.suptitle(title, fontsize=100)
    return fig


def plot_slice(counts, nm, bound=4):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    counts[counts == 0] = np.nan
    ax.imshow(counts.T,
              origin='lower', aspect='auto',
              extent=[-bound, bound, -bound, bound],
              )
    fig.savefig(nm)


def plot_coloured(data, to_mask, ax, name, base_dist):
    try:
        to_mask = to_mask.detach().cpu().numpy()
    except Exception as e:
        print(e)
        pass

    marker_size = 2
    alpha = 0.1
    x = data[:, 0]
    y = data[:, 1]
    mx_low_prob = (x ** 2 + y ** 2) ** (0.5) > 3
    mx_checkers = ((y / 2).floor() % 2).type(torch.bool)

    def scatter(mx, color, alpha):
        # This must also be cast to a numpy, otherwise the masked array is also a tensor
        mx = mx.detach().cpu().numpy()
        ax.scatter(to_mask[mx, 0], to_mask[mx, 1], s=marker_size, color=color, alpha=alpha)

    def scatter_mask(id, color):
        if base_dist != 'checkerboard':
            mx = id & ~mx_low_prob
        else:
            mx = id & ~mx_checkers
        # ax.scatter(to_mask[mx, 0], to_mask[mx, 1], s=marker_size, color=color, alpha=alpha)
        scatter(mx, color, alpha)
        if base_dist != 'checkerboard':
            mx = id & mx_low_prob
            # ax.scatter(to_mask[mx, 0], to_mask[mx, 1], s=2, color='grey', alpha=1)
            scatter(mx, 'grey', 1)
        else:
            mx = id & mx_checkers
            # ax.scatter(to_mask[mx, 0], to_mask[mx, 1], s=marker_size, color='dark' + color, alpha=alpha)
            scatter(mx, 'dark' + color, alpha)

    id = torch.logical_and(x < 0, y < 0)
    scatter_mask(id, 'red')

    id = torch.logical_and(x > 0, y < 0)
    scatter_mask(id, 'green')

    id = torch.logical_and(x < 0, y > 0)
    scatter_mask(id, 'blue')

    id = torch.logical_and(x > 0, y > 0)
    scatter_mask(id, 'slategrey')

    bound = 4.5
    ax.set_xlim([-bound, bound])
    ax.set_ylim([-bound, bound])
    ax.set_title(name)


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
