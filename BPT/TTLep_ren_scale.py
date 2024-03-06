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
from math import log 

from defaults import selection, feature_names

observers = []

data_generator  =  DataGenerator(
    input_files = ["/eos/vbc/group/cms/robert.schoefbeck/TT2lUnbinned/training-ntuples/MVA-training/EFT_tr-minDLmass20-dilepL-offZ1-njet3p-btag2p-ht500/TTLep/TTLep.root"],
        n_split = 1,
        splitting_strategy = "files",
        selection = selection,
        branches  = feature_names + ["scale_Weight", "weight", "overflow_counter"] ) 

base_point_index = {
#    0 : (log(0.5), log(0.5)),
    1 : (log(0.5), ),
#    2 : (log(0.5), log(2.0)),
#    3 : (log(1.0), log(0.5)),
    4 : (log(1.0), ),
#    5 : (log(1.0), log(2.0)),
#    6 : (log(2.0), log(0.5)),
    7 : (log(2.0), ),
#    8 : (log(2.0), log(2.0)),
}
base_point_index.update ({val:key for key, val in base_point_index.items()})

base_points        = [ base_point_index[i] for i in [1,4,7] ]
parameters         = ['ren']
combinations       = [('ren',), ('ren', 'ren')]
tex                = {"ren":"ren.-scale"}
nominal_base_point = base_point_index[4]

default_parameters = {  }
default_parameters.update( {var:0. for var in parameters} )

def getEvents( N_events_requested, systematic = None):

    index = -1
    res = {tuple(bp):{} for bp in base_points}
    
    res[tuple(nominal_base_point)]['features']  = data_generator.scalar_branches( data_generator[index], feature_names )[:N_events_requested]
    weights  = data_generator.vector_branch( data_generator[index], "scale_Weight" )[:N_events_requested] 

    for i in [1,4,7]:
        res[base_point_index[i]]['weights'] = weights[:,i] 

    return res 

from plot_options import plot_options

shape_user_range = {'log':(0.8, 1.2), 'lin':(0.8, 1.4)}

bpt_cfg = {
    "n_trees" : 300,
    "learning_rate" : 0.2,
    "loss" : "CrossEntropy",
    "learn_global_param": False,
    "min_size": 50,
}

