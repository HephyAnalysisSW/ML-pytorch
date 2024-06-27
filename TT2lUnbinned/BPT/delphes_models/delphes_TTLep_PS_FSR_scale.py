import pickle
import random
import ROOT
from math import pi, log
import numpy as np
import os
if __name__=="__main__":
    import sys
    sys.path.append('..')

from tools.DataGenerator import DataGenerator
from tools.WeightInfo    import WeightInfo

from defaults import selection, feature_names

systematics = ["scale_%i"%i for i in [0,1,3,5,6,7,8]]
wilson_coefficients = systematics

observers = []

training_files =  ["TTLep/TTLep.root"]

from defaults import selection, feature_names, data_locations

scale_vars = [
    #"fsrRedHi", 
    #"fsrRedLo", 
    "fsrDefHi", 
    "fsrDefLo", 
    #"fsrConHi", 
    #"fsrConLo", 
     "muR1p0_muF1p0", 
    ]

data_generator  =  DataGenerator(
    input_files = [os.path.join( data_locations["RunII"], training_file) for training_file in training_files ],
        n_split = 1,
        splitting_strategy = "files",
        selection = selection,
        branches  = feature_names + ["weight1fb"] + scale_vars)

def set_era(era):
    data_generator.read_files( [os.path.join(data_locations[era], training_file) for training_file in training_files ] )

base_point_index = {
    #"fsrRedHi" :(log(0.707),),
    #"fsrRedLo" :(log(1.414),),
              
    "fsrDefHi" :(log(0.5),),
    "fsrDefLo" :(log(2.0),),
              
    #"fsrConHi" :(log(0.25),),
    #"fsrConLo" :(log(4.0),),

    "muR1p0_muF1p0": (0., ),
}

base_point_index.update ({val:key for key, val in base_point_index.items()})

base_points        = [ base_point_index[var] for var in scale_vars ]
parameters         = ['fsr']
combinations       = [('fsr',), ('fsr','fsr')]
tex                = {"fsr":"fsr.-scale"}
nominal_base_point = base_point_index["muR1p0_muF1p0"]

default_parameters = {  }
default_parameters.update( {var:0. for var in parameters} )

def getEvents( N_events_requested, systematic = None):

    index = -1
    res = {tuple(bp):{} for bp in base_points}

    res[tuple(nominal_base_point)]['features']  = data_generator.scalar_branches( data_generator[index], feature_names )[:N_events_requested]
    weights  = data_generator.scalar_branches( data_generator[index], scale_vars)[:N_events_requested]

    for var in scale_vars:
        res[base_point_index[var]]['weights'] = weights[:,scale_vars.index(var)]

    return res

tex.update( { 
    "muR1p0_muF1p0":"nominal",
    "isrRedHi" :"ISR 0.707",
    "fsrRedHi" :"FSR 0.707",
    "isrRedLo" :"ISR 1.414",
    "fsrRedLo" :"FSR 1.414",
              
    "isrDefHi" :"ISR 0.5",
    "fsrDefHi" :"FSR 0.5",
    "isrDefLo" :"ISR 2.0",
    "fsrDefLo" :"FSR 2.0",
              
    "isrConHi" :"ISR 0.25",
    "fsrConHi" :"FSR 0.25",
    "isrConLo" :"ISR 2.0",
    "fsrConLo" :"FSR 2.0",
    })

shape_user_range = {'log':(0.7, 1.4), 'lin':(0.7, 1.4)}

from plot_options import plot_options

bpt_cfg = {
    "n_trees" : 300,
    "learning_rate" : 0.2,
    "loss" : "CrossEntropy",
    "learn_global_param": False,
    "min_size": 1000,
}

