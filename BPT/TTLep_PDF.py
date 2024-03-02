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

systematics = ["PDF%03i"%i for i in range(1,101)]
wilson_coefficients = systematics

observers = []

data_generator  =  DataGenerator(
    input_files = ["/eos/vbc/group/cms/robert.schoefbeck/TT2lUnbinned/training-ntuples/MVA-training/EFT_tr-minDLmass20-dilepL-offZ1-njet3p-btag2p-ht500/TTLep/TTLep.root"],
        n_split = 1,
        splitting_strategy = "files",
        selection = selection,
        branches  = feature_names + ["PDF_Weight", "overflow_counter", "weight"] ) 

systematic         = "PDF001"
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
    PDF_Weight = data_generator.vector_branch( data_generator[index], "PDF_Weight", padding_target=102)[:N_events_requested] 
    res[tuple(nominal_base_point)]['weights'] = PDF_Weight[:,0] 
    res[( 1.0,)]['weights']                   = PDF_Weight[:,int( systematic.replace('PDF','')) ] 

    return res 

tex = {var:var for var in systematics}

shape_user_range = {'log':(0.998, 1.002)}

from plot_options import plot_options

bpt_cfg = {
    "n_trees" : 300,
    "learning_rate" : 0.2,
    "loss" : "CrossEntropy",
    "learn_global_param": False,
    "min_size": 50,
}

