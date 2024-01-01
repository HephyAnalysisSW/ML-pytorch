import os
import numpy as np
if __name__=="__main__":
    import sys
    sys.path.append('..')
import ROOT

from tools.DataGenerator import DataGenerator as _DataGenerator

#feature_names = [ "ht" ]
feature_names = [ "nJetGood", "ht", "jet0_pt"]#, "jet1_pt", "jet2_pt", "jet3_pt", "jet0_eta", "jet1_eta", "jet2_eta", "jet3_eta" ]

encoding      = { 0.5 :("0p5", "Up"), 1.0 :("1p0", "Up"), 1.5 :("1p5", "Up"), 2.0 :("2p0", "Up"), -0.5 :("0p5", "Down"), -1.0 :("1p0", "Down"), -1.5 :("1p5", "Down"), -2.0 :("2p0", "Down")}

selection = lambda ar:ar.ht>500

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
            branches = feature_names,
            redirector=redirector)

    return generator.scalar_branches( generator[-1], feature_names )[:maxN]

systematic         = "jesTotal"
base_points        = [  [-1.], [-0.5], [0.], [.5] , [1.], ]
parameters         = ['nu']
combinations       = [('nu',), ('nu', 'nu'),] #('nu', 'nu', 'nu'), ('nu', 'nu', 'nu', 'nu')]
tex                = {"nu":"#nu"}
nominal_base_point = (0.,)

default_parameters = {  }
default_parameters.update( {var:0. for var in parameters} )

def getEvents( N_events_requested, systematic = systematic):
    return { tuple(bp):{'features':np.nan_to_num( _getEvents( systematic = systematic, level=bp[0], maxN=N_events_requested)) } for bp in base_points }

plot_options =  {
    "met" :{'binning':[20,100,500],   'logY':True,  'tex':'E_{T}^{miss}'},
    "ht"  :{'binning':[20,500,1500],  'logY':True,  'tex':'H_{T}'},
    "nJetGood"  :{'binning':[7,3,10], 'logY':True,  'tex':'N_{jet}'},
    "jet0_pt" :{'binning':[30,0,1000],'logY':True,  'tex':'p_{T}(jet 0)'},
    "jet1_pt" :{'binning':[30,0,1000],'logY':True,  'tex':'p_{T}(jet 1)'},
    "jet2_pt" :{'binning':[30,0,500], 'logY':True,  'tex':'p_{T}(jet 2)'},
    "jet3_pt" :{'binning':[30,0,500], 'logY':True,  'tex':'p_{T}(jet 3)'},
    "jet4_pt" :{'binning':[30,0,500], 'logY':True,  'tex':'p_{T}(jet 4)'},
    "jet0_eta" :{'binning':[30,-4,4],'logY':False,  'tex':'#eta(jet 0)'},
    "jet1_eta" :{'binning':[30,-4,4],'logY':False,  'tex':'#eta(jet 1)'},
    "jet2_eta" :{'binning':[30,-4,4],'logY':False,  'tex':'#eta(jet 2)'},
    "jet3_eta" :{'binning':[30,-4,4],'logY':False,  'tex':'#eta(jet 3)'},
    "jet4_eta" :{'binning':[30,-4,4],'logY':False,  'tex':'#eta(jet 4)'},
}

bpt_cfg = {
    "n_trees" : 300,
    "learning_rate" : 0.2,
    "loss" : "CrossEntropy",
    "learn_global_param": False,
    "min_size": 50,
}
