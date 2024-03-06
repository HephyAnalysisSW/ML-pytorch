import pickle
import random
import ROOT
from math import pi
import numpy as np
if __name__=="__main__":
    import sys
    sys.path.append('..')

from tools.DataGenerator import DataGenerator
from tools.WeightInfo    import WeightInfo

from defaults import selection, feature_names

systematics = ["scale_%i"%i for i in [0,1,3,5,6,7,8]]
wilson_coefficients = systematics

observers = []

data_generator  =  DataGenerator(
    input_files = ["/eos/vbc/group/cms/robert.schoefbeck/TT2lUnbinned/training-ntuples/MVA-training/EFT_tr-minDLmass20-dilepL-offZ1-njet3p-btag2p-ht500/TTLep/TTLep.root"],
        n_split = 1,
        splitting_strategy = "files",
        selection = selection,
        branches  = feature_names + ["scale_Weight", "overflow_counter", "weight"] ) 

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
    scale_Weight = data_generator.vector_branch( data_generator[index], "scale_Weight", padding_target=102)[:N_events_requested] 
    res[tuple(nominal_base_point)]['weights'] = scale_Weight[:,4] 
    res[( 1.0,)]['weights']                   = scale_Weight[:,int( systematic.replace('scale_','')) ] 

    return res 

tex.update( { 
    "scale_0":"Ren. Down, Fact. Down",
    "scale_1":"Ren. Down, Fact. Nom",
    "scale_3":"Ren. Nom, Fact. Down",
    "scale_5":"Ren. Nom, Fact. Up" ,
    "scale_7":"Ren. Up, Fact. Nom" ,
    "scale_8":"Ren. Up, Fact. Up",  },
    )

shape_user_range = {'log':(0.8, 1.2), 'lin':(0.2, 2)}

from plot_options import plot_options

bpt_cfg = {
    "n_trees" : 300,
    "learning_rate" : 0.2,
    "loss" : "CrossEntropy",
    "learn_global_param": False,
    "min_size": 50,
}

