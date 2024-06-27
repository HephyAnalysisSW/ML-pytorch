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

from defaults_paper import selection


feature_names = [

            "tr_ttbar_pt",
            "tr_ttbar_mass",

            "tr_top_pt",
            "tr_topBar_pt",
            "tr_top_eta",
            "tr_topBar_eta",


            "recoLep01_pt",
            "recoLep01_mass",

            #"tr_ttbar_dEta",
            #"tr_ttbar_dAbsEta",
            #"recoLep_dEta",
            #"recoLep_dAbsEta",

            #"tr_cos_phi", 
            "tr_cos_phi_lab", "tr_abs_delta_phi_ll_lab",

            "nBTag",
            "nJetGood", #change name when ntuple is updated
#            "ht"
]



data_generator_central  =  DataGenerator(
    input_files = [ "/eos/vbc/group/cms/robert.schoefbeck/TT2lUnbinned/training-ntuples-v5/MVA-training/EFT_for_paper_tr-minDLmass20-dilepM-offZ1-njet2p-mtt750/TTLep/*.root"
    ],
        n_split = 1,
        splitting_strategy = "files",
        selection = selection,
        branches  = feature_names + ["weight", "overflow_counter_v1"] ) 

data_generator_hUp  =  DataGenerator(
    input_files = [ "/eos/vbc/group/cms/robert.schoefbeck/TT2lUnbinned/training-ntuples-v5/MVA-training/EFT_for_paper_tr-minDLmass20-dilepM-offZ1-njet2p-mtt750/TTLep_pow_hUp/*.root"
    ],
        n_split = 1,
        splitting_strategy = "files",
        selection = selection,
        branches  = feature_names + ["weight", "overflow_counter_v1"]) 

data_generator_hDown  =  DataGenerator(
    input_files = ["/eos/vbc/group/cms/robert.schoefbeck/TT2lUnbinned/training-ntuples-v5/MVA-training/EFT_for_paper_tr-minDLmass20-dilepM-offZ1-njet2p-mtt750/TTLep_pow_hDown/*.root"
    ],
        n_split = 1,
        splitting_strategy = "files",
        selection = selection,
        branches  = feature_names + ["weight", "overflow_counter_v1"]) 

def set_era( era ):
    print ("Warning. Do nothing to set era to %s"%era)

base_points        = [  [-1.],  [0.], [1.], ]
parameters         = ['hdamp']
combinations       = [('hdamp',), ('hdamp', 'hdamp'),] #('hdamp', 'hdamp', 'hdamp'), ('hdamp', 'hdamp', 'hdamp', 'hdamp')]
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

    scale                   = 0.5*( res[(-1.,)]['weights'].sum() +  res[(+1.,)]['weights'].sum())/res[(0.,) ]['weights'].sum()

    res[(0.,) ]['weights'] *= scale

    return res 

tex = {"hf":"HF", "lf":"LF", "cferr1":"cferr1", "cferr2":"cferr2", "lfstats1":"lfstats1", "lfstats2":"lfstats2", "hfstats1":"hfstats1", "hfstats2":"hfstats2"}

shape_user_range = {'log':(0.8, 1.2), 'lin':(0.8, 1.4)}

from plot_options import plot_options

bpt_cfg = {
    "n_trees" : 300,
    "learning_rate" : 0.2,
    "loss" : "CrossEntropy",
    "learn_global_param": False,
    "min_size": 50,
}
