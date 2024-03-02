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

data_generator_central  =  DataGenerator(
    input_files = [ "/eos/vbc/group/cms/robert.schoefbeck/tt-jec/training-ntuples-3/MVA-training/tt_hdamp_trg-dilepVL-minDLmass20-offZ1/TTLep/*.root"
    ],
        n_split = 1,
        splitting_strategy = "files",
        selection = selection,
        branches  = feature_names + ["weight", "overflow_counter"] ) 

data_generator_hUp  =  DataGenerator(
    input_files = [ "/eos/vbc/group/cms/robert.schoefbeck/tt-jec/training-ntuples-3/MVA-training/tt_hdamp_trg-dilepVL-minDLmass20-offZ1/TTLep_hUp/*.root"
    ],
        n_split = 1,
        splitting_strategy = "files",
        selection = selection,
        branches  = feature_names + ["weight", "overflow_counter"]) 

data_generator_hDown  =  DataGenerator(
    input_files = ["/eos/vbc/group/cms/robert.schoefbeck/tt-jec/training-ntuples-3/MVA-training/tt_hdamp_trg-dilepVL-minDLmass20-offZ1/TTLep_hDown/*.root"
    ],
        n_split = 1,
        splitting_strategy = "files",
        selection = selection,
        branches  = feature_names + ["weight", "overflow_counter"]) 

base_points        = [  [-1.],  [0.], [1.], ]
parameters         = ['hdamp']
combinations       = [('hdamp',), ]#('hdamp', 'hdamp'),] #('hdamp', 'hdamp', 'hdamp'), ('hdamp', 'hdamp', 'hdamp', 'hdamp')]
tex                = {"hdamp":"h_{damp}"}
nominal_base_point = (0.,)

default_parameters = {  }
default_parameters.update( {var:0. for var in parameters} )

def getEvents( N_events_requested, systematic = "hdamp"):

    index = -1
    res = {tuple(bp):{} for bp in base_points}
    
    res[(0.,) ]['features'] = data_generator_central.scalar_branches( data_generator_central[index], feature_names )[:N_events_requested]
    res[(-1.,)]['features'] = data_generator_hUp.scalar_branches( data_generator_hUp[index], feature_names )[:N_events_requested]
    res[(+1.,)]['features'] = data_generator_hDown.scalar_branches( data_generator_hDown[index], feature_names )[:N_events_requested]
    res[(0.,) ]['weights']  = data_generator_central.scalar_branches( data_generator_central[index], ["weight"] )[:N_events_requested][:,0]
    res[(-1.,)]['weights']  = data_generator_hUp.scalar_branches( data_generator_hUp[index], ["weight"] )[:N_events_requested][:,0]
    res[(+1.,)]['weights']  = data_generator_hDown.scalar_branches( data_generator_hDown[index], ["weight"] )[:N_events_requested][:,0]

    return res 

tex = {"hf":"HF", "lf":"LF", "cferr1":"cferr1", "cferr2":"cferr2", "lfstats1":"lfstats1", "lfstats2":"lfstats2", "hfstats1":"hfstats1", "hfstats2":"hfstats2"}

from plot_options import plot_options

bpt_cfg = {
    "n_trees" : 300,
    "learning_rate" : 0.2,
    "loss" : "CrossEntropy",
    "learn_global_param": False,
    "min_size": 50,
}
