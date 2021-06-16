import matplotlib.pyplot as plt
import numpy as np
import zfit

from .fitter_utils import linearPDF

# TODO: This is only for the linear PDF. Need to handle this arbitrarily.
# TODO: This is only for the first band. No way of knowing the best initial parameters OTF.
# TODO: Add GP mechanism here. But need to get this out, or we would never get on with it.

# Define parameters, space, instantiate pdf.

f_slope = zfit.Parameter('Slope', -33, -50, 50)
f_intercept = zfit.Parameter('Intercept', 5900, 100, 6000)


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

    lower, higher = edges
    data_np = zfit.run(data.value()[:, 0])

    fig, ax = plt.subplots()

    counts, bin_edges = np.histogram(data_np, bins, range=(lower, higher), density=True)
    bincenters = (bin_edges[1:] + bin_edges[:-1]) / 2
    x = np.linspace(lower, higher, samples)
    y = zfit.run(model.pdf(x))

    ax.bar(bincenters, counts, label='Background', zorder=1, color='limegreen');
    ax.plot(x, y, ls='--', color='firebrick', label='Fit', zorder=2)

    ax.set_xlabel("Mass (GeV)")
    ax.set_ylabel("A.U.")
    ax.legend(frameon=False)

    plt.tight_layout()

    plt.savefig(f'{plt_sv_dir}/massrange_{edges[0]}_{edges[1]}_fit.png')
    print(f'{plt_sv_dir}/massrange_{edges[0]}_{edges[1]}_fit.png')
    plt.clf()


def signalMassSampler(masses, edge1, edge2, fitter='conventional', getStatus=False, plt_sv_dir=None, scaler=None,
                      unscaler=None):
    '''
    Takes in the masses from sidebands, fits (unbinned) a pdf, and returns samples from
    the fitted pdf. Additionally returns the result of the fit.

    ARGS:
    1. path : str : The masses to be fit
    2. size : int : Number of desired samples in the signal window.
    3. fitter : str : Conventional (zFit backend) or Gaussian Process Regression (not implemented yet)
    4. getStatus : bool : Will return the result of fit if set to True. Contains param values, and fit convergence details.
    5. getFitPlot : bool : Will save the fit plot if set to True.
    '''

    if fitter.casefold() != 'conventional':
        raise NotImplementedError(
            f"{fitter} does not exist. Only conventional fitting with zfit backend is implemented.")
    if len(masses) <= 0:
        raise ValueError("No masses found to fit. Please pass a valid mass array.")

    if scaler is not None:
        masses = scaler(masses)
        edge1 = scaler(edge1)
        edge2 = scaler(edge2)
    masses = masses.detach().cpu().numpy()

    sideband1 = zfit.Space('massRange', limits=(masses.min(), edge1))
    sideband2 = zfit.Space('massRange', limits=(edge2, masses.max()))
    massRange = sideband1 + sideband2

    data = zfit.Data.from_numpy(obs=massRange, array=masses)

    model = linearPDF(obs=massRange, slope=f_slope, intercept=f_intercept, scaler=scaler, unscaler=unscaler)

    if plt_sv_dir is not None:
        fitPlot(model, data, (masses.min(), masses.max()), plt_sv_dir)

    nll = zfit.loss.UnbinnedNLL(model=model, data=data)
    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(nll)

    if getStatus:
        return model, (result, result.hesse())

    else:
        return model
