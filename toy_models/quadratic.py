import ROOT
import numpy as np
import random

from math import sin, cos, sqrt, pi
''' Analytic toy
'''

import ROOT
import csv
import os
import array

# EFT settings, parameters, defaults
wilson_coefficients    = ['theta1']
tex                    = {"theta1":"#theta_{1}"}

default_eft_parameters = { 'Lambda':1000. }
default_eft_parameters.update( {var:0. for var in wilson_coefficients} )

first_derivatives = [('theta1',)]
second_derivatives= [('theta1','theta1')]
derivatives       = [tuple()] + first_derivatives + second_derivatives

def make_eft(**kwargs):
    result = { key:val for key, val in default_eft_parameters.items() }
    for key, val in kwargs.items():
        if not key in wilson_coefficients+["Lambda"]:
            raise RuntimeError ("Wilson coefficient not known.")
        else:
            result[key] = float(val)
    return result

random_eft = make_eft(**{v:random.random() for v in wilson_coefficients} )
sm         = make_eft()

feature_names =  ['x']

def getEvents(N_events_requested):

    #x = 0.5*np.ones(N_events_requested) 
    x = np.random.rand(N_events_requested) 

    return np.transpose(np.array( [x] ))

def getWeights(features, eft=default_eft_parameters):

    #dsigma/dx = (1+theta x)^2 = 1+2*theta*x + theta**2*x**2

    weights = { tuple():     np.ones(len(features)),
               #('theta1',): 2*np.ones(len(features)),
               ('theta1',): 2*features[:,0], 
               #('theta1','theta1'): 2*np.ones(len(features)),
               ('theta1','theta1'): 2*features[:,0]**2,
    }
    #for key in list(weights.keys()):
    #    if key==(): continue
    #    weights[key][features[:,0]<0.5]=0.

    return weights

plot_options = {
    'x': {'binning':[1,0,1],      'tex':"x",},
    }

plot_points = [
    {'color':ROOT.kBlack,       'point':sm, 'tex':"SM"},
    {'color':ROOT.kMagenta+2,   'point':make_eft(theta1=-2),'tex':"#theta_{1} = -2"},
    {'color':ROOT.kMagenta-4,   'point':make_eft(theta1=+2), 'tex':"#theta_{1} = +2"},
    {'color':ROOT.kBlue+2,      'point':make_eft(theta1=-1),  'tex':"#theta_{1} = -1"},
    {'color':ROOT.kBlue-4,      'point':make_eft(theta1=+1),  'tex':"#theta_{1} = +1"},
    {'color':ROOT.kGreen+2,     'point':make_eft(theta1=-0.5),'tex':"#theta_{1} =-.5"},
    {'color':ROOT.kGreen-4,     'point':make_eft(theta1=0.5), 'tex':"#theta_{1} =+.5"},
]

multi_bit_cfg = {'n_trees': 250,
                 'max_depth': 4,
                 'learning_rate': 0.20,
                 'min_size': 15 }
