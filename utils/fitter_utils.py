import zfit
from zfit import z
import math
import torch

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
        if self.scaler is not None:
            sample = self.unscaler(torch.tensor(sample.numpy(), dtype=torch.float32))
        return sample 


#ATLAS dijet functions
class dijet_ATLAS_fit(zfit.pdf.ZPDF):

    '''add a doc string here'''

    _PARAMS = ['p1', 'p2', 'p3']
    
    def __init__(self, *args, scaler=None, unscaler=None, root_s=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaler = scaler
        self.unscaler = unscaler
        self.root_s = root_s

    def _unnormalized_pdf(self, x):

        data = z.unstack_x(x)
        p1 = self.params['p1']
        p2 = self.params['p2']
        p3 = self.params['p3']

        return p1*z.pow(1-data, p2)*z.pow(data, p3)


    def sample(self, *args, limits=None, **kwargs):
        if limits is not None:
            limits = (self.scaler(limit) for limit in limits)
            limits = (limit/self.root_s for limit in limits)
        sample = super().sample(*args, limits=limits, **kwargs)
        sample = sample*self.root_s
        if self.scaler is not None:
            sample = self.unscaler(torch.tensor(sample.numpy(), dtype=torch.float32))
        return sample
