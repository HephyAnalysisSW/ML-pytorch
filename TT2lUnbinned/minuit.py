import numpy as np
from iminuit import Minuit
from iminuit.util import describe
from typing import Annotated

class LeastSquares:
    """
    Generic least-squares cost function with error.
    """
    # 1 for LSQ, 0.5 for NLL: 
    # https://iminuit.readthedocs.io/en/stable/reference.html#iminuit.Minuit.errordef
    errordef = Minuit.LEAST_SQUARES  # for Minuit to compute errors correctly
    
    def __init__(self, model, x, y, err):
        self.model = model  # model predicts y for given x
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.err = np.asarray(err)
        pars = describe(model, annotations=True)
        model_args = iter(pars)
        next(model_args)
        self._parameters = {k: pars[k] for k in model_args}

    def __call__(self, *par):  # we must accept a variable number of model parameters
        ym = self.model(self.x, *par)
        return np.sum((self.y - ym) ** 2 / self.err ** 2)

    @property
    def ndata(self):
        return len(self.x)

def line(x, a: float, b: Annotated[float, 0:]):
    return a + b * x

rng = np.random.default_rng(1)
x_data = np.arange(1, 6, dtype=float)
y_err = np.ones_like(x_data)
y_data = line(x_data, 1, 2) + rng.normal(0, y_err)

lsq = LeastSquares(line, x_data, y_data, y_err)

## this fails
#try:
#    m = Minuit(lsq, a=0, b=0)
#    m.errordef=Minuit.LEAST_SQUARES
#except:
#    import traceback
#    traceback.print_exc()

describe(line, annotations=True)

# we inject that into the lsq object with the special attribute
# `_parameters` that iminuit recognizes
pars = describe(line, annotations=True)
model_args = iter(pars)
next(model_args)  # we skip the first argument which is not a model parameter
lsq._parameters = {k: pars[k] for k in model_args}

# now we get the right answer
describe(lsq, annotations=True)

m = Minuit(lsq, a=0, b=1)
m.migrad()
