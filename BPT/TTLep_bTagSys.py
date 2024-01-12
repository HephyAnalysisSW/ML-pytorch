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

selection = lambda ar: (ar.ht>500) 

feature_names = [   
     "nJetGood" ,
     "nBTag" ,
     "met"      ,
     "ht"       ,

     "jet0_pt"  ,
     "jet0_eta" ,
     "jet1_pt"  ,
     "jet1_eta" ,
     "jet2_pt"  ,
     "jet2_eta" ,
     "jet3_pt"  ,
     "jet3_eta" ,
     "jet4_pt"  ,
     "jet4_eta" ,
    ]

systematics = ["hf", "lf",  "cferr1", "cferr2", "lfstats1", "lfstats2", "hfstats1", "hfstats2"]
wilson_coefficients = systematics
weight_branches = ["reweightBTagSF_central"] + ["reweightBTagSF_%s_%s"%(ud, sys) for ud in ["up","down"] for sys in systematics ]

observers = []

data_generator  =  DataGenerator(
    input_files = ["/eos/vbc/group/cms/robert.schoefbeck/tt-jec/training-ntuples-3/MVA-training/tt_jec_trg-dilepVL-minDLmass20-offZ1/TTLep_nominal/TTLep_nominal.root"],
        n_split = 1,
        splitting_strategy = "files",
        selection = selection,
        branches  = feature_names + weight_branches  ) 

systematic         = "hf"
base_points        = [  [-1.],  [0.], [1.], ]
parameters         = ['nu']
combinations       = [('nu',), ('nu', 'nu'),] #('nu', 'nu', 'nu'), ('nu', 'nu', 'nu', 'nu')]
tex                = {"nu":"#nu"}
nominal_base_point = (0.,)

default_parameters = {  }
default_parameters.update( {var:0. for var in parameters} )

def getEvents( N_events_requested, systematic = systematic):

    index = -1
    res = {tuple(bp):{} for bp in base_points}
    
    res[tuple(nominal_base_point)]['features']  = data_generator.scalar_branches( data_generator[index], feature_names )[:N_events_requested]
    coeffs      = data_generator.scalar_branches( data_generator[index], weight_branches)[:N_events_requested] 
    res[tuple(nominal_base_point)]['weights'] = coeffs[:,weight_branches.index("reweightBTagSF_central")]
    res[(-1.0,)]['weights'] = coeffs[:,weight_branches.index("reweightBTagSF_down_%s"%(systematic))]
    res[( 1.0,)]['weights'] = coeffs[:,weight_branches.index("reweightBTagSF_up_%s"%(systematic))]

    return res 

tex = {"hf":"HF", "lf":"LF", "cferr1":"cferr1", "cferr2":"cferr2", "lfstats1":"lfstats1", "lfstats2":"lfstats2", "hfstats1":"hfstats1", "hfstats2":"hfstats2"}

plot_options =  {

     "nJetGood":    {'binning':[10,0,10],       'tex':"N_{jet}"},
     "nBTag":       {'binning':[5,0,5],       'tex':"N_{b-tag}"},
     "met":         {'binning': [20,0,500],     'tex':"E_{T}^{miss}"},
     "ht":          {'binning': [20,500,2500],  'tex':"H_{T}"},
     "jet0_pt":     {'binning': [20,0,500], 'tex':"p_{T}(j_{0})"},
     "jet0_eta":    {'binning': [20,-3,3],  'tex':"#eta(j_{0})"},
     "jet1_pt":     {'binning': [20,0,500], 'tex':"p_{T}(j_{1})"},
     "jet1_eta":    {'binning': [20,-3,3],  'tex':"#eta(j_{1})"},
     "jet2_pt":     {'binning': [20,0,500], 'tex':"p_{T}(j_{2})"},
     "jet2_eta":    {'binning': [20,-3,3],  'tex':"#eta(j_{2})"},
     "jet3_pt":     {'binning': [20,0,500], 'tex':"p_{T}(j_{3})"},
     "jet3_eta":    {'binning': [20,-3,3],  'tex':"#eta(j_{3})"},
     "jet4_pt":     {'binning': [20,0,500], 'tex':"p_{T}(j_{4})"},
     "jet4_eta":    {'binning': [20,-3,3],  'tex':"#eta(j_{4})"},
}

bpt_cfg = {
    "n_trees" : 300,
    "learning_rate" : 0.2,
    "loss" : "CrossEntropy",
    "learn_global_param": False,
    "min_size": 50,
}

