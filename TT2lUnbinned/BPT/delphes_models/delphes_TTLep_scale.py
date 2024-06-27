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
    "muR0p5_muF0p5", "muR0p5_muF2p0", "muR0p5_muF1p0", "muR2p0_muF0p5", 
    "muR2p0_muF2p0", "muR2p0_muF1p0", "muR1p0_muF0p5", "muR1p0_muF2p0", "muR1p0_muF1p0", 
    "isrRedHi", "fsrRedHi", "isrRedLo", "fsrRedLo", "isrDefHi", "fsrDefHi", "fsrDefLo", 
    "isrConHi", "isrConLo", "fsrConLo", "isrRedHi", "fsrRedHi", "isrRedLo", "fsrRedLo", 
    "isrDefHi", "fsrDefHi", "isrDefLo", "fsrDefLo", "isrConHi", "fsrConHi", "isrConLo", "fsrConLo", 
    ]

data_generator  =  DataGenerator(
    input_files = [os.path.join( data_locations["RunII"], training_file) for training_file in training_files ],
        n_split = 1,
        splitting_strategy = "files",
        selection = selection,
        branches  = feature_names + ["weight1fb"] + scale_vars)

def set_era(era):
    data_generator.read_files( [os.path.join(data_locations[era], training_file) for training_file in training_files ] )

systematic         = "scale001"
base_points        = [ [0.], [1.] ]
parameters         = [ 'nu' ]
combinations       = [ ('nu',), ] #('nu', 'nu'),] #('nu', 'nu', 'nu'), ('nu', 'nu', 'nu', 'nu')]
tex                = { "nu": "#nu" }
nominal_base_point = (0.,)

default_parameters = {  }
default_parameters.update( {var:0. for var in parameters} )

def getEvents( N_events_requested, systematic = systematic):

    index = -1
    res = {tuple(bp):{} for bp in base_points}
    
    res[tuple(nominal_base_point)]['features']  = data_generator.scalar_branches( data_generator[index], feature_names )[:N_events_requested]
    weights  = data_generator.scalar_branches( data_generator[index], ["muR1p0_muF1p0", systematic] )[:N_events_requested]
    res[tuple(nominal_base_point)]['weights'] = weights[:,0] 
    res[( 1.0,)]['weights']                   = weights[:,1] 

    return res 

tex.update( { 
    "muR0p5_muF0p5":"#mu_{R}=0.5 #mu_{F}=0.5",
    "muR0p5_muF2p0":"#mu_{R}=0.5 #mu_{F}=2.0",
    "muR0p5_muF1p0":"#mu_{R}=0.5 #mu_{F}=1.0",
    "muR2p0_muF0p5":"#mu_{R}=2.0 #mu_{F}=0.5",
    "muR2p0_muF2p0":"#mu_{R}=2.0 #mu_{F}=2.0",
    "muR2p0_muF1p0":"#mu_{R}=2.0 #mu_{F}=1.0",
    "muR1p0_muF0p5":"#mu_{R}=1.0 #mu_{F}=0.5",
    "muR1p0_muF2p0":"#mu_{R}=1.0 #mu_{F}=2.0",
    "muR1p0_muF1p0":"#mu_{R}=1.0 #mu_{F}=1.0",
    })

shape_user_range = {'log':(0.8, 1.2), 'lin':(0.2, 2)}

from plot_options import plot_options

bpt_cfg = {
    "n_trees" : 300,
    "learning_rate" : 0.2,
    "loss" : "CrossEntropy",
    "learn_global_param": False,
    "min_size": 1000,
}

