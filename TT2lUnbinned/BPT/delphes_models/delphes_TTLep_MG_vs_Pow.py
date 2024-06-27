import pickle
import random
import ROOT
from math import pi
import numpy as np
import os
if __name__=="__main__":
    import sys
    sys.path.append('..')

from tools.DataGenerator import DataGenerator
from tools.WeightInfo    import WeightInfo

from defaults import selection, data_locations, feature_names


data_generator_Powheg  =  DataGenerator(
    input_files = [os.path.join(data_locations["RunII"], training_file) for training_file in ["TTLep/TTLep.root"]],
        n_split = 1,
        splitting_strategy = "files",
        selection = selection,
        branches  = feature_names + ["weight1fb"] ) 

data_generator_MG  =  DataGenerator(
    input_files = [os.path.join(data_locations["RunII"], training_file) for training_file in ["TT01j2lCAOldRef_Mtt500_ext/TT01j2lCAOldRef_Mtt500_ext.root"] ],
        n_split = 1,
        splitting_strategy = "files",
        selection = selection,
        branches  = feature_names + ["weight1fb", "p_C"]) 

def set_era(era):
    data_generator_Powheg.read_files( [os.path.join(data_locations[era], training_file) for training_file in ["TTLep/TTLep.root"] ] )
    data_generator_MG.read_files(     [os.path.join(data_locations[era], training_file) for training_file in ["TT01j2lCAOldRef_Mtt500_ext/TT01j2lCAOldRef_Mtt500_ext.root"] ] )


base_points        = [  [0.], [1.], ]
parameters         = ['gPowheg']
combinations       = [('gPowheg',), ]
tex                = {"gPowheg":"g_{pow}"}
nominal_base_point = (0.,)

default_parameters = {  }
default_parameters.update( {var:0. for var in parameters} )

def getEvents( N_events_requested, systematic=None):

    index = -1
    res = {tuple(bp):{} for bp in base_points}
    
    res[(0.,) ]['features'] = data_generator_MG.scalar_branches( data_generator_MG[index], feature_names )[:N_events_requested]
    coeffs                  = data_generator_MG.vector_branch( data_generator_MG[index], 'p_C', padding_target=1)[:N_events_requested]
    res[(0.,) ]['weights']  = coeffs[:,0] 

    res[(+1.,)]['features'] = data_generator_Powheg.scalar_branches( data_generator_Powheg[index], feature_names )[:N_events_requested]
    res[(+1.,)]['weights']  = data_generator_Powheg.scalar_branches( data_generator_Powheg[index], ["weight1fb"] )[:N_events_requested][:,0]
    # same normalization!
    res[(+1.,)]['weights'] *= (res[(0.,) ]['weights'].sum()/res[(+1.,)]['weights'].sum())
    return res 

shape_user_range = {'log':(10**-3, 3), 'lin':(0, 10)}

from plot_options import plot_options

bpt_cfg = {
    "n_trees" : 300,
    "learning_rate" : 0.2,
    "loss" : "CrossEntropy",
    "learn_global_param": False,
    "min_size": 30,
}
