import os
import numpy as np
if __name__=="__main__":
    import sys
    sys.path.append('../../..')

from tools.DataGenerator import DataGenerator as _DataGenerator

selection = lambda ar: (ar.ht>=500) 

#feature_names = [ "met", "nJetGood", "ht", "jet0_pt", "jet1_pt", "jet3_pt", "jet2_pt", "jet4_pt", "jet0_eta", "jet1_eta", "jet2_eta", "jet3_eta", "jet4_eta" ]
feature_names = [ "met", "nJetGood", "ht", "jet0_pt", "jet1_pt", "jet2_pt", "jet3_pt"] #"jet4_pt", "jet0_eta", "jet1_eta", "jet2_eta", "jet3_eta", "jet4_eta" ]
encoding      = { 0.5 :("0p5", "Up"), 1.0 :("1p0", "Up"), 1.5 :("1p5", "Up"), 2.0 :("2p0", "Up"), -0.5 :("0p5", "Down"), -1.0 :("1p0", "Down"), -1.5 :("1p5", "Down"), -2.0 :("2p0", "Down")}

#systematics   = ["jesTotal", "jesAbsoluteMPFBias", "jesAbsoluteScale", "jesAbsoluteStat", "jesRelativeBal", "jesRelativeFSR", "jesRelativeJEREC1", "jesRelativeJEREC2", "jesRelativeJERHF", "jesRelativePtBB", "jesRelativePtEC1", "jesRelativePtEC2", "jesRelativePtHF", "jesRelativeStatEC", "jesRelativeStatFSR", "jesRelativeStatHF", "jesPileDataMC", "jesPilePtBB", "jesPilePtEC1", "jesPilePtEC2", "jesPilePtHF", "jesPilePtRef", "jesFlavorQCD", "jesFragmentation", "jesSinglePionECAL", "jesSinglePionHCAL", "jesTimePtEta"]

input_dir = "/eos/vbc/group/cms/robert.schoefbeck/tt-jec/training-ntuples/MVA-training/tt_jec_trg-dilepVL-minDLmass20-offZ1/"

def _generator( directory, n_split=1):
    print( "Loading", os.path.join( input_dir, directory) )
    return _DataGenerator(
        input_files = [os.path.join( input_dir, directory,  "*.root")],
            n_split = n_split,
            splitting_strategy = "files",
            selection   = selection,
            branches = feature_names)

def _add_truth( data, truth ):
    return np.concatenate( (np.array(data), truth*np.ones((len(data),1))), axis=1 )

class DataGenerator:

    def __init__( self, input_dir = input_dir, systematic = "jesTotal", levels = [-2., -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0], maxN=None):

        self.input_dir  = input_dir
        self.levels     = levels
        self.generators = {'nominal':_generator( "TTLep_nominal" )}
        #for systematic in systematics:
            #self.generators[systematic] = {} 
        for level in levels:
            self.generators[level] = _generator( "TTLep_%s_{systematic}%s".format(systematic=systematic)%encoding[level] )
                
        self.maxN = maxN

    def __getitem__(self, index):
        data = np.concatenate( [ _add_truth(self.generators['nominal'].scalar_branches( self.generators['nominal'][index], feature_names )[:self.maxN], 0) ]
                              +[ _add_truth(self.generators[level].scalar_branches( self.generators[level][index], feature_names )[:self.maxN],level) for level in self.levels ], axis=0)
        return data[:, :-1], data[:, -1:]
                        

plot_options =  {
    "met" :{'binning':[50,0,1500], 'tex':'E_{T}^{miss}'},
    "ht"  :{'binning':[50,0,1500], 'tex':'H_{T}'},
    "jet0_pt" :{'binning':[50,0,500], 'tex':'p_{T}(jet 0)'},
    "jet1_pt" :{'binning':[50,0,500], 'tex':'p_{T}(jet 1)'},
    "jet2_pt" :{'binning':[50,0,500], 'tex':'p_{T}(jet 2)'},
    "jet3_pt" :{'binning':[50,0,500], 'tex':'p_{T}(jet 3)'},
    "jet4_pt" :{'binning':[50,0,500], 'tex':'p_{T}(jet 4)'},
    "jet0_eta" :{'binning':[50,0,500], 'tex':'#eta(jet 0)'},
    "jet1_eta" :{'binning':[50,0,500], 'tex':'#eta(jet 1)'},
    "jet2_eta" :{'binning':[50,0,500], 'tex':'#eta(jet 2)'},
    "jet3_eta" :{'binning':[50,0,500], 'tex':'#eta(jet 3)'},
    "jet4_eta" :{'binning':[50,0,500], 'tex':'#eta(jet 4)'},
}
