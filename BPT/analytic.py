import ROOT
import numpy as np
import random

from math import sin, cos, sqrt, pi, exp
''' Analytic toy
'''

import ROOT
import csv
import os
import array

parameters         = ['nu1']
combinations       = [('nu1',), ('nu1', 'nu1'),] #('nu1', 'nu1', 'nu1'), ('nu1', 'nu1', 'nu1', 'nu1')]
tex                = {"nu1":"#nu_{1}"}
base_points        = [ [0.], [.5] , [1.], [1.5] ]
nominal_base_point = (0.,)

default_parameters = {  }
default_parameters.update( {var:0. for var in parameters} )

def make_parameters(**kwargs):
    result = { key:val for key, val in default_parameters.items() }
    for key, val in kwargs.items():
        if not key in parameters:
            raise RuntimeError ("Parameter not known.")
        else:
            result[key] = float(val)
    return result

random_parameters = make_parameters(**{v:random.random() for v in parameters} )
sm         = make_parameters()

feature_names =  ['x']

from scipy.stats.sampling import NumericalInversePolynomial

class PDF:
    def __init__( self, parameters):
        self.parameters = parameters
        self.rng = NumericalInversePolynomial(self, domain=(-pi, pi), random_state=np.random.default_rng())

    # for external use
    @staticmethod
    def generic_pdf( x, parameters):
        return np.exp(parameters[0]*np.sin(x)) 

    def pdf(self, x: float) -> float:
        # note that the normalization constant isn't required
        return self.generic_pdf( x, self.parameters )

    def dpdf(self, x: float) -> float:
        return self.parameters[0]*np.cos(x)*np.exp( self.parameters[0]*np.sin(x) )

    def getFeatures( self, N_events_requested ):
        return np.array( self.rng.rvs(N_events_requested) ).reshape( -1, 1)

nominal_PDF = PDF(nominal_base_point)

def getEvents( N_events_requested, weighted=True):

    if weighted:
        x  = nominal_PDF.getFeatures( N_events_requested )
        res = {tuple(bp):{'weights':nominal_PDF.generic_pdf( x[:,0], bp)} for bp in base_points}
        #res = {tuple(bp):{'weights':np.exp(1.+bp[0]*x)*np.ones((len(x),))} for bp in base_points}
        #res = {tuple(bp):{'weights':np.exp( (.2*bp[0]-0.3*bp[0]**2+0.0*bp[0]**3))*np.ones((len(x),) )} for bp in base_points}
        res[nominal_base_point]['features'] = x
    else:
        res = {tuple(bp):{'features': PDF(bp).getFeatures( N_events_requested )} for bp in base_points}

    return res

plot_options = {
    'x': {'binning':[20,-pi,pi],      'tex':"x",},
    }

plot_points = [
    {'color':ROOT.kBlack,       'point':sm, 'tex':"SM"},
    {'color':ROOT.kMagenta+2,   'point':make_parameters(nu1=-2),'tex':"#nu_{1} = -2"},
    {'color':ROOT.kMagenta-4,   'point':make_parameters(nu1=+2), 'tex':"#nu_{1} = +2"},
    {'color':ROOT.kBlue+2,      'point':make_parameters(nu1=-1),  'tex':"#nu_{1} = -1"},
    {'color':ROOT.kBlue-4,      'point':make_parameters(nu1=+1),  'tex':"#nu_{1} = +1"},
    {'color':ROOT.kGreen+2,     'point':make_parameters(nu1=-0.5),'tex':"#nu_{1} =-.5"},
    {'color':ROOT.kGreen-4,     'point':make_parameters(nu1=0.5), 'tex':"#nu_{1} =+.5"},
]

bpt_cfg = {
    "n_trees" : 100,
    "learning_rate" : 0.2, 
    "loss" : "CrossEntropy", 
    "learn_global_param": False,
    "min_size": 50,
}
