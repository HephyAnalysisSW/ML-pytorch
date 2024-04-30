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

systematics = ["leptonSF"]
weight_branches = ["reweightLeptonSFUp", "reweightLeptonSFDown", "reweightLeptonSF" ]

observers = []

training_files = {
    'RunII':            ['TTLep_RunII/TTLep_RunII.root'],
    'Summer16_preVFP':  ['TTLep_UL2016_preVFP/TTLep_UL2016_preVFP.root'],
    'Summer16':         ['TTLep_UL2016/TTLep_UL2016.root'],
    'Fall17':           ['TTLep_UL2017/TTLep_UL2017.root'],
    'Autumn18':         ['TTLep_UL2018/TTLep_UL2018.root'],
}

from defaults_paper import selection, feature_names, data_location

data_generator  =  DataGenerator(
    input_files = [os.path.join( data_location, training_file) for training_file in training_files["RunII"] ],
        n_split = 1,
        splitting_strategy = "files",
        selection = selection,
        branches  = feature_names + ["weight"] + weight_branches )

def set_era(era):
    data_generator.read_files( [os.path.join(data_location, training_file) for training_file in training_files[era] ] )

systematic         = "leptonSF"
base_points        = [  [-1.],  [0.], [1.], ]
parameters         = ['nu']
combinations       = [('nu',), ('nu', 'nu'),] #('nu', 'nu', 'nu'), ('nu', 'nu', 'nu', 'nu')]
tex                = {"nu":"#nu"}
nominal_base_point = (0.,)

default_parameters = {  }
default_parameters.update( {var:0. for var in parameters} )

def getEvents( N_events_requested, systematic = None):

    index = -1
    res = {tuple(bp):{} for bp in base_points}
    
    res[tuple(nominal_base_point)]['features']  = data_generator.scalar_branches( data_generator[index], feature_names )[:N_events_requested]
    coeffs      = data_generator.scalar_branches( data_generator[index], weight_branches)[:N_events_requested] 
    res[tuple(nominal_base_point)]['weights'] = coeffs[:,weight_branches.index("reweightLeptonSF")]
    res[(-1.0,)]['weights'] = coeffs[:,weight_branches.index("reweightLeptonSFDown")]
    res[(1.0,)]['weights']  = coeffs[:,weight_branches.index("reweightLeptonSFUp")]

    return res 

tex = {"leptonSF":"SF(l)"}

shape_user_range = {'log':(0.97, 1.03), 'lin':(0.97, 1.03)}

from plot_options import plot_options

bpt_cfg = {
    "n_trees" : 300,
    "learning_rate" : 0.2,
    "loss" : "CrossEntropy",
    "learn_global_param": False,
    "min_size": 500,
}

