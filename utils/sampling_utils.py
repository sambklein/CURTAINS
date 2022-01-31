import matplotlib.pyplot as plt
import numpy as np
import zfit

from .fitter_utils import linearPDF, dijet_ATLAS_fit


# TODO: This is only for the linear PDF. Need to handle this arbitrarily.
# TODO: This is only for the first band. No way of knowing the best initial parameters OTF.
# TODO: Add GP mechanism here. But need to get this out, or we would never get on with it.

# Functions that do the actual thing.
def zfitDataHandler(mass):
    '''
    Returns the edges of the signal mass window, given the sideband masses.
    Does not need it's own doc string. But I am on a roll here.

    ARGS: mass : float, numpy array : The masses in the sidebands.
    '''

    rawcounts, bins = np.histogram(mass, bins=80)
    bincenters = (bins[:-1] + bins[1:]) / 2.0
    edge1, edge2 = bincenters[rawcounts == 0][0], bincenters[rawcounts == 0][-1]

    return edge1, edge2


def fitPlot(model, data, edges, plt_sv_dir, **kwargs):
    '''
    Returns the plot of the fitted histogram and the fitted pdf overlayed.

    ARGS:
          1. model : zfit model : The model object used to fit the data.
          2. data : zfit core data : The zfit data, i.e. after you do data = zfit.Data(...)
          3. edges : float, tuple : The min and max value for the masses to be fit.
    KWARGS:
          1. bins : int : Number of bins for histogramming the mass distribution. Defaults to 80.
          2. samples : int : Number of sample mass points within the range to evaluate the pdf.
                             Defaults to 10000.
    '''

    if 'bins' in kwargs:
        bins = kwargs.get('bins')
    else:
        bins = 80

    if 'samples' in kwargs:
        samples = kwargs.get('samples')
    else:
        samples = 10000

    root_s = model.root_s
    lower, higher = edges
    data_np = zfit.run(data.value()[:, 0])

    fig, ax = plt.subplots()

    counts, bin_edges = np.histogram(data_np, bins, range=(lower, higher), density=True)
    bincenters = (bin_edges[1:] + bin_edges[:-1]) / 2
    x = np.linspace(lower, higher, samples)
    y = zfit.run(model.pdf(x))

    ax.hist(data_np, bins=bins, range=(lower, higher), density=True, label='Background', zorder=1, color='limegreen');
    #     ax.bar(bincenters, counts, label='Background', zorder=1, color='limegreen');
    ax.plot(x, y, ls='--', color='firebrick', label='Fit', zorder=2)

    ax.set_xlabel("Mass (GeV)")
    ax.set_ylabel("A.U.")
    ax.legend(frameon=False)

    plt.tight_layout()

    plt.savefig(f'{plt_sv_dir}/massrange_{edges[0] * root_s:3.2f}-{edges[1] * root_s:3.2f}_fit.png')
    print(f'{plt_sv_dir}/massrange_{edges[0] * root_s:3.2f}-{edges[1] * root_s:3.2f}_fit.png')
    plt.clf()


# def signalMassSampler(masses, edge1, edge2, getStatus=False, plt_sv_dir=None, scaler=None,
#                       unscaler=None):
#     '''
#     Takes in the masses from sidebands, fits (unbinned) a pdf, and returns samples from
#     the fitted pdf. Additionally returns the result of the fit.
#
#     ARGS:
#     1. masses : torch array : The masses to be fit
#     2. edge1, edge2 : float: boundaries of signal region.
#     3. getStatus : bool : Will return the result of fit if set to True.
#                           Contains param values, and fit convergence details.
#     4. plt_sv_dir : str : directory where the fit plot will be saved.
#     5. scaler : function : function to convert all masses back to GeV.
#     6. unscaler : function : normalise all masses from GeV.
#     '''
#
#     if len(masses) <= 0:
#         raise ValueError("No masses found to fit. Please pass a valid mass array.")
#
#     if scaler is not None:
#         masses = scaler(masses)
#         edge1 = scaler(edge1)
#         edge2 = scaler(edge2)
#     masses = masses.detach().cpu().numpy()
#
#     root_s = 13000.  # The ATLAS di jet functions expect values in mass/sqrtS.
#
#     masses = masses / root_s
#
#     sideband1 = zfit.Space('massRange', limits=(masses.min(), edge1 / root_s))
#     sideband2 = zfit.Space('massRange', limits=(edge2 / root_s, masses.max()))
#     massRange = sideband1 + sideband2
#
#     # parameters for the dijet function - move this outside of the function, if the
#     # script complains about repeated parameters. Shouldn't be an issue.
#     p1 = zfit.Parameter('p1', 100.)
#     p2 = zfit.Parameter('p2', 10.)
#     p3 = zfit.Parameter('p3', 0.1)
#
#     data = zfit.Data.from_numpy(obs=massRange, array=masses)
#     model = dijet_ATLAS_fit(obs=massRange, p1=p1, p2=p2, p3=p3, root_s=root_s, scaler=scaler, unscaler=unscaler)
#
#     nll = zfit.loss.UnbinnedNLL(model=model, data=data)
#     minimizer = zfit.minimize.Minuit()
#     result = minimizer.minimize(nll)
#
#     if plt_sv_dir is not None:
#         fitPlot(model, data, (masses.min(), masses.max()), plt_sv_dir)
#
#     if getStatus:
#         return model, result
#
#     else:
#         return model




def signalMassSampler(masses, edge1, edge2, getStatus=False, plt_sv_dir=None, scaler=None,
                      unscaler=None):
    '''
    Takes in the masses from sidebands, fits (unbinned) a pdf, and returns samples from
    the fitted pdf. Additionally returns the result of the fit.

    ARGS:
    1. masses : torch array : The masses to be fit
    2. edge1, edge2 : float: boundaries of signal region.
    3. getStatus : bool : Will return the result of fit if set to True.
                          Contains param values, and fit convergence details.
    4. plt_sv_dir : str : directory where the fit plot will be saved.
    5. scaler : function : function to convert all masses back to GeV.
    6. unscaler : function : normalise all masses from GeV.
    '''

    if len(masses) <= 0:
        raise ValueError("No masses found to fit. Please pass a valid mass array.")

    if scaler is not None:
        masses = scaler(masses)
        edge1 = scaler(edge1)
        edge2 = scaler(edge2)
    masses = masses.detach().cpu().numpy()

    root_s = 13000.  # The ATLAS di jet functions expect values in mass/sqrtS.

    masses = masses / root_s

    sideband1 = zfit.Space('massRange', limits=(masses.min(), edge1 / root_s))
    sideband2 = zfit.Space('massRange', limits=(edge2 / root_s, masses.max()))
    massRange = sideband1 + sideband2

    # parameters for the dijet function - move this outside of the function, if the
    # script complains about repeated parameters. Shouldn't be an issue.
    p1 = zfit.Parameter('p1', 100.)
    p2 = zfit.Parameter('p2', 10.)
    p3 = zfit.Parameter('p3', 0.1)

    data = zfit.Data.from_numpy(obs=massRange, array=masses)
    model = dijet_ATLAS_fit(obs=massRange, p1=p1, p2=p2, p3=p3, root_s=root_s, scaler=scaler, unscaler=unscaler)

    nll = zfit.loss.UnbinnedNLL(model=model, data=data)
    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(nll)

    if plt_sv_dir is not None:
        fitPlot(model, data, (masses.min(), masses.max()), plt_sv_dir)

    if getStatus:
        return model, result

    else:
        return model
