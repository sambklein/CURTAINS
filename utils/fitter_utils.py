import zfit
from zfit import z
import math
import torch


class atlas_bg_PDF(zfit.pdf.ZPDF):
    """Implements the ATLAS background fitting profile in zfit"""

    _PARAMS = ['p1', 'p2', 'p3', 'p4', 'p5']

    def _unnormalized_pdf(self, x):  # implement function
        data = z.unstack_x(x)
        p1 = self.params['p1']
        p2 = self.params['p2']
        p3 = self.params['p3']
        p4 = self.params['p4']
        p5 = self.params['p5']

        return p1 * ((1 - data) ** p2) * data ** (p3 + p4 * math.log(data) + p5 * math.log(data) * math.log(data))


class linearPDF(zfit.pdf.ZPDF):
    """Implements the trivial linear fitting profile in zfit"""

    _PARAMS = ['slope', 'intercept']

    def __init__(self, *args, scaler=None, unscaler=None, **kwargs):
        super(linearPDF, self).__init__(*args, **kwargs)
        self.scaler = scaler
        self.unscaler = unscaler

    def _unnormalized_pdf(self, x):  # implement function
        data = z.unstack_x(x)
        slope = self.params['slope']
        intercept = self.params['intercept']

        return slope * data + intercept

    def sample(self, *args, limits=None, **kwargs):
        if limits is not None:
            limits = (self.scaler(limit) for limit in limits)
        sample = super().sample(*args, limits=limits, **kwargs)
        if self.unscaler is not None:
            sample = self.unscaler(torch.tensor(sample.numpy(), dtype=torch.float32))
        return sample


class expPDF(zfit.pdf.ZPDF):
    """Implements the trivial linear fitting profile in zfit"""

    _PARAMS = ['scale', 'amp']

    def _unnormalized_pdf(self, x):  # implement function
        data = z.unstack_x(x)
        scale = self.params['scale']
        amp = self.params['amp']
        return amp * z.exp(-data * scale)
