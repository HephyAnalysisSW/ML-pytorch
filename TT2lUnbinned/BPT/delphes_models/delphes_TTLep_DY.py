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

from defaults import selection, data_locations

feature_names = [
        "tr_cosThetaPlus_n",
        "tr_cosThetaMinus_n",
        "tr_cosThetaPlus_r",
        "tr_cosThetaMinus_r",
        "tr_cosThetaPlus_r_star",
        "tr_cosThetaMinus_r_star",
        "tr_xi_nn",
        "tr_xi_rr",
        "tr_xi_nr_plus",
        "tr_xi_nr_minus",
        "tr_xi_rk_plus",
        "tr_xi_rk_minus",
        "tr_xi_nk_plus",
        "tr_xi_nk_minus",

        "tr_xi_r_star_k",
        "tr_xi_k_r_star",

        "tr_ttbar_dAbsEta",

        "tr_ttbar_mass",
        "recoLep0_pt",
        "recoLep1_pt",
        "jet0_pt",
        "jet1_pt",
        #"jet2_pt",
        "nrecoJet",
        "nBTag",
]


data_generator_TTLep  =  DataGenerator(
    input_files = [os.path.join(data_locations["RunII"], training_file) for training_file in ["TTLep/TTLep.root"]],
        n_split = 1,
        splitting_strategy = "files",
        selection = selection,
        branches  = feature_names + ["weight1fb"] ) 

data_generator_DY  =  DataGenerator(
    input_files = [os.path.join(data_locations["RunII"], training_file) for training_file in ["DYJetsToLL_M50_HT_fakeB2/DYJetsToLL_M50_HT_fakeB2.root"] ],
        n_split = 1,
        splitting_strategy = "files",
        selection = selection,
        branches  = feature_names + ["weight1fb"]) 

def set_era(era):
    data_generator_TTLep.read_files( [os.path.join(data_locations[era], training_file) for training_file in ["TTLep/TTLep.root"] ] )
    data_generator_DY.read_files( [os.path.join(data_locations[era], training_file) for training_file in ["DYJetsToLL_M50_HT_fakeB2/DYJetsToLL_M50_HT_fakeB2.root"] ] )


base_points        = [  [0.], [1.], ]
parameters         = ['gDY']
combinations       = [('gDY',), ]#('hdamp', 'hdamp'),] #('hdamp', 'hdamp', 'hdamp'), ('hdamp', 'hdamp', 'hdamp', 'hdamp')]
tex                = {"gDY":"g_{DY}"}
nominal_base_point = (0.,)

default_parameters = {  }
default_parameters.update( {var:0. for var in parameters} )

def getEvents( N_events_requested, systematic=None):

    index = -1
    res = {tuple(bp):{} for bp in base_points}
    
    res[(0.,) ]['features'] = data_generator_TTLep.scalar_branches( data_generator_TTLep[index], feature_names )[:N_events_requested]
    res[(0.,) ]['weights']  = data_generator_TTLep.scalar_branches( data_generator_TTLep[index], ["weight1fb"] )[:N_events_requested][:,0]
    res[(+1.,)]['features'] = data_generator_DY.scalar_branches( data_generator_DY[index], feature_names )[:N_events_requested]
    res[(+1.,)]['weights']  = 20*data_generator_DY.scalar_branches( data_generator_DY[index], ["weight1fb"] )[:N_events_requested][:,0]

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
