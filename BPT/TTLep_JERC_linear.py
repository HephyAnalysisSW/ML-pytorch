import os
import numpy as np
if __name__=="__main__":
    import sys
    sys.path.append('..')
import ROOT

from tools.DataGenerator import DataGenerator as _DataGenerator

from defaults import selection, feature_names

encoding      = { 0.5 :("0p5", "Up"), 1.0 :("1p0", "Up"), 1.5 :("1p5", "Up"), 2.0 :("2p0", "Up"), -0.5 :("0p5", "Down"), -1.0 :("1p0", "Down"), -1.5 :("1p5", "Down"), -2.0 :("2p0", "Down")}


#systematics   = ["jesTotal", "jesAbsoluteMPFBias", "jesAbsoluteScale", "jesAbsoluteStat", "jesRelativeBal", "jesRelativeFSR", "jesRelativeJEREC1", "jesRelativeJEREC2", "jesRelativeJERHF", "jesRelativePtBB", "jesRelativePtEC1", "jesRelativePtEC2", "jesRelativePtHF", "jesRelativeStatEC", "jesRelativeStatFSR", "jesRelativeStatHF", "jesPileDataMC", "jesPilePtBB", "jesPilePtEC1", "jesPilePtEC2", "jesPilePtHF", "jesPilePtRef", "jesFlavorQCD", "jesFragmentation", "jesSinglePionECAL", "jesSinglePionHCAL", "jesTimePtEta"]

input_dir = "/eos/vbc/group/cms/robert.schoefbeck/tt-jec/training-ntuples-3/MVA-training/tt_jec_trg-dilepVL-minDLmass20-offZ1/"
redirector = "root://eos.grid.vbc.ac.at/"

def _getEvents( systematic = "jesTotal", level = 0, n_split=1, maxN=None):

    if maxN is not None and maxN<1:
        maxN=None

    if systematic is None:
        systematic = "jesTotal"

    if level == 0.:
        directory = "TTLep_nominal"
    else:
        directory = "TTLep_%s_{systematic}%s".format(systematic=systematic)%encoding[level]

    print( "Loading", os.path.join( input_dir, directory) )

    generator = _DataGenerator(
        input_files = [os.path.join( input_dir, directory,  "*.root")],
            n_split = n_split,
            splitting_strategy = "files",
            selection   = selection, #getSelection( systematic=systematic, level=level),
            branches = feature_names + ["weight", "overflow_counter"],
            redirector=redirector)

    return generator.scalar_branches( generator[-1], feature_names )[:maxN]

systematic         = "jesTotal"
base_points        = [  [-1.], [-0.5], [0.], [.5] , [1.], ]
parameters         = ['nu']
combinations       = [('nu',), ]#('nu', 'nu'),] #('nu', 'nu', 'nu'), ('nu', 'nu', 'nu', 'nu')]
tex                = {"nu":"#nu"}
nominal_base_point = (0.,)

default_parameters = {  }
default_parameters.update( {var:0. for var in parameters} )

def getEvents( N_events_requested, systematic = systematic):
    return { tuple(bp):{'features':np.nan_to_num( _getEvents( systematic = systematic, level=bp[0], maxN=N_events_requested)) } for bp in base_points }

def make_parameters(**kwargs):
    result = { key:val for key, val in default_parameters.items() }
    for key, val in kwargs.items():
        if not key in parameters:
            raise RuntimeError ("Parameter not known.")
        else:
            result[key] = float(val)
    return result

sm         = make_parameters()

plot_points = [
    {'color':ROOT.kBlack,       'point':sm, 'tex':"SM"},
    {'color':ROOT.kMagenta+1.5, 'point':make_parameters(nu=-1.5),'tex':"#nu = -1.5"},
    {'color':ROOT.kMagenta-4,   'point':make_parameters(nu=+1.5), 'tex':"#nu = +1.5"},
    {'color':ROOT.kBlue+2,      'point':make_parameters(nu=-1),  'tex':"#nu = -1"},
    {'color':ROOT.kBlue-4,      'point':make_parameters(nu=+1),  'tex':"#nu = +1"},
    {'color':ROOT.kGreen+2,     'point':make_parameters(nu=-0.5),'tex':"#nu =-.5"},
    {'color':ROOT.kGreen-4,     'point':make_parameters(nu=0.5), 'tex':"#nu =+.5"},
]

from plot_options import plot_options

bpt_cfg = {
    "n_trees" : 300,
    "learning_rate" : 0.2,
    "loss" : "CrossEntropy",
    "learn_global_param": False,
    "min_size": 50,
}
