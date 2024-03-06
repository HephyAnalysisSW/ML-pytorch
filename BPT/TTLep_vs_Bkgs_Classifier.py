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

data_generator_TTLep  =  DataGenerator(
    input_files = [ "/eos/vbc/group/cms/robert.schoefbeck/TT2lUnbinned/training-ntuples/MVA-training/EFT_tr-minDLmass20-dilepL-offZ1-njet3p-btag2p-ht500/TTLep/TTLep.root"
    ],
        n_split = 1,
        splitting_strategy = "files",
        selection = selection,
        branches  = feature_names + ["weight", "overflow_counter"] ) 

data_generator_DY  =  DataGenerator(
    input_files = [ "/eos/vbc/group/cms/robert.schoefbeck/TT2lUnbinned/training-ntuples/MVA-training/EFT_tr-minDLmass20-dilepL-offZ1-njet3p-btag2p-ht500/DY/*.root"
    ],
        n_split = 1,
        splitting_strategy = "files",
        selection = selection,
        branches  = feature_names + ["weight", "overflow_counter"]) 

data_generator_others  =  DataGenerator(
    input_files = ["/eos/vbc/group/cms/robert.schoefbeck/TT2lUnbinned/training-ntuples/MVA-training/EFT_tr-minDLmass20-dilepL-offZ1-njet3p-btag2p-ht500/TTH/*.root",
                   "/eos/vbc/group/cms/robert.schoefbeck/TT2lUnbinned/training-ntuples/MVA-training/EFT_tr-minDLmass20-dilepL-offZ1-njet3p-btag2p-ht500/TTW/*.root",
                   "/eos/vbc/group/cms/robert.schoefbeck/TT2lUnbinned/training-ntuples/MVA-training/EFT_tr-minDLmass20-dilepL-offZ1-njet3p-btag2p-ht500/TTZ/*.root",
                   "/eos/vbc/group/cms/robert.schoefbeck/TT2lUnbinned/training-ntuples/MVA-training/EFT_tr-minDLmass20-dilepL-offZ1-njet3p-btag2p-ht500/ST/*.root",
                   "/eos/vbc/group/cms/robert.schoefbeck/TT2lUnbinned/training-ntuples/MVA-training/EFT_tr-minDLmass20-dilepL-offZ1-njet3p-btag2p-ht500/DiBoson/*.root",
    ],
        n_split = 1,
        splitting_strategy = "files",
        selection = selection,
        branches  = feature_names + ["weight", "overflow_counter"]) 

systematics = ["DY", "others"]

base_points        = [  [0.], [1.], ]
parameters         = ['nu']
combinations       = [('nu',),] #('hdamp', 'hdamp', 'hdamp'), ('hdamp', 'hdamp', 'hdamp', 'hdamp')]
tex                = {"nu":"#nu"}
nominal_base_point = (0.,)

default_parameters = {  }
default_parameters.update( {var:0. for var in parameters} )

def getEvents( N_events_requested, systematic = "DY"):

    index = -1
    res = {tuple(bp):{} for bp in base_points}
   
    features_nom = data_generator_TTLep.scalar_branches( data_generator_TTLep[index], feature_names )[:N_events_requested]
    weights_nom  = data_generator_TTLep.scalar_branches( data_generator_TTLep[index], ["weight"] )[:N_events_requested][:,0]
    len_nom      = len(features_nom)
    if systematic == "DY":
        features_alt = data_generator_DY.scalar_branches( data_generator_DY[index], feature_names )[:N_events_requested]
        weights_alt  = data_generator_DY.scalar_branches( data_generator_DY[index], ["weight"] )[:N_events_requested][:,0]
    elif systematic == "others":
        features_alt = data_generator_others.scalar_branches( data_generator_others[index], feature_names )[:N_events_requested]
        weights_alt  = data_generator_others.scalar_branches( data_generator_others[index], ["weight"] )[:N_events_requested][:,0]
    else:
        raise RuntimeError("Unknown variation %s"%systematic)
    len_alt      = len(features_alt)
 
    features = np.concatenate( [features_nom, features_alt] , axis=0)
 
    res[(0.,) ]['features'] = features 
    res[(+1.,)]['features'] = features 
    res[(0.,) ]['weights']  = np.concatenate( [ weights_nom, np.zeros(len_alt)] ) 
    res[(+1.,)]['weights']  = np.concatenate( [ np.zeros(len_nom), weights_alt] ) 

    return res 

tex = {"DY":"DY", "others":"others"}

shape_user_range = {'log':(0.8, 1.2), 'lin':(0.2, 2)}

from plot_options import plot_options

bpt_cfg = {
    "n_trees" : 300,
    "learning_rate" : 0.2,
    "loss" : "CrossEntropy",
    "learn_global_param": False,
    "min_size": 50,
}
