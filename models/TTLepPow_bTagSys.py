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

default_sys_parameters = {p:0 for p in systematics}
def make_sys(**kwargs):
    result = { key:val for key, val in default_sys_parameters.items() }
    for key, val in kwargs.items():
        if not key in systematics:
            raise RuntimeError ("Systematic not known.")
        else:
            result[key] = float(val)
    return result

nominal = make_sys()

def make_combinations( sys=systematics ):
    combinations = [()]
    for s in  sys:
        if s in systematics:
           combinations.append((s,)) 
    for s1 in  sys:
        for s2 in  sys:
            if s1 in systematics and s2 in systematics and s2>=s1:
               combinations.append((s1,s2))
 
    return combinations

def getEvents( nTraining, return_observers = False ):

    index = -1
    _coeffs      = data_generator.scalar_branches( data_generator[index], weight_branches)[:nTraining] 
    features     = data_generator.scalar_branches( data_generator[index], feature_names )[:nTraining]

    combinations = make_combinations( systematics )
    coeffs = {}
    coeffs[()] = _coeffs[:,weight_branches.index("reweightBTagSF_central")]
    for comb in combinations[1:]:
        if len(comb)==1:
            coeffs[comb] = 0.5*(_coeffs[:,weight_branches.index("reweightBTagSF_up_%s"%(comb[0]))] - _coeffs[:,weight_branches.index("reweightBTagSF_down_%s"%(comb[0]))])
        elif len(comb)==2 and comb[0]!=comb[1]: 
            coeffs[comb] = np.zeros_like(coeffs[()])
        elif len(comb)==2:
            coeffs[comb] = 0.5*(2*_coeffs[:,weight_branches.index("reweightBTagSF_central")] - _coeffs[:,weight_branches.index("reweightBTagSF_up_%s"%(comb[0]))] - _coeffs[:,weight_branches.index("reweightBTagSF_down_%s"%(comb[0]))])

    if return_observers:
        observers_ = data_generator.scalar_branches( data_generator[index], observers )[:nTraining]
        return features, coeffs, observers_
    else: 
        return features, coeffs 

tex = {"hf":"HF", "lf":"LF", "cferr1":"cferr1", "cferr2":"cferr2", "lfstats1":"lfstats1", "lfstats2":"lfstats2", "hfstats1":"hfstats1", "hfstats2":"hfstats2"}

plot_points = [
    {'color':ROOT.kBlack,       'point':nominal, 'tex':"nominal"},

    {'color':ROOT.kMagenta-4,   'point':make_sys(hf=+1), 'tex':"HF up"},
    {'color':ROOT.kMagenta-6,   'point':make_sys(hf=-1), 'tex':"HF down"},
    {'color':ROOT.kGreen-4,     'point':make_sys(lf=+1), 'tex':"LF up"},
    {'color':ROOT.kGreen+2,     'point':make_sys(lf=-1), 'tex':"LF down"},
    {'color':ROOT.kBlue-4,      'point':make_sys(cferr1=+1), 'tex':"cferr1 up"},
    {'color':ROOT.kBlue+2,      'point':make_sys(cferr1=-1), 'tex':"cferr1 down"},
    {'color':ROOT.kRed-4,       'point':make_sys(cferr2=+1), 'tex':"cferr2 up"},
    {'color':ROOT.kRed+2,       'point':make_sys(cferr2=-1), 'tex':"cferr2 down"},
    {'color':ROOT.kOrange-4,    'point':make_sys(lfstats1=+1), 'tex':"lfstats1 up"},
    {'color':ROOT.kOrange+2,    'point':make_sys(lfstats1=-1), 'tex':"lfstats1 down"},
    {'color':ROOT.kCyan-4,      'point':make_sys(lfstats2=+1), 'tex':"lfstats2 up"},
    {'color':ROOT.kCyan+2,      'point':make_sys(lfstats2=-1), 'tex':"lfstats2 down"},
    {'color':ROOT.kYellow-4,    'point':make_sys(hfstats1=+1), 'tex':"hfstats1 up"},
    {'color':ROOT.kYellow+2,    'point':make_sys(hfstats1=-1), 'tex':"hfstats1 down"},
    {'color':ROOT.kPink,        'point':make_sys(hfstats2=+1), 'tex':"hfstats2 up"},   
    {'color':ROOT.kPink,        'point':make_sys(hfstats2=-1), 'tex':"hfstats2 down"}, 
]

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

multi_bit_cfg = {'n_trees': 300,
                 'max_depth': 4,
                 'learning_rate': 0.20,
                 'min_size': 25 }
