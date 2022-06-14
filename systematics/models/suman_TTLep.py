import math
import numpy as np
import ROOT
ROOT.TH1.SetDefaultSumw2()
from   matplotlib import pyplot as plt
import os
from math import sqrt, log

try:
    import uproot
except:
    import uproot3 as uproot

import awkward
import numpy as np
import pandas as pd

jet_systematics = [
    "jer",
    "jesFlavorQCD",
#    "jesRelativeBal",
#    "jesHF",
#    "jesBBEC1",
#    "jesEC2",
#    "jesAbsolute",
#    "jesAbsolute_2018",
#    "jesHF_2018",
#    "jesEC2_2018",
#    "jesRelativeSample_2018",
#    "jesBBEC1_2018",
#    "jesTotal",
#    "jer0",
#    "jer1",
#    "jer2",
#    "jer3",
#    "jer4",
#    "jer5",
#    "jesAbsoluteStat",
#    "jesAbsoluteScale",
#    "jesAbsoluteSample",
#    "jesAbsoluteFlavMap",
#    "jesAbsoluteMPFBias",
#    "jesFragmentation",
#    "jesSinglePionECAL",
#    "jesSinglePionHCAL",
#    "jesTimePtEta",
#    "jesRelativeJEREC1",
#    "jesRelativeJEREC2",
#    "jesRelativeJERHF",
#    "jesRelativePtBB",
#    "jesRelativePtEC1",
#    "jesRelativePtEC2",
#    "jesRelativePtHF",
#    "jesRelativeSample",
#    "jesRelativeFSR",
#    "jesRelativeStatFSR",
#    "jesRelativeStatEC",
#    "jesRelativeStatHF",
#    "jesPileUpDataMC",
#    "jesPileUpPtRef",
#    "jesPileUpPtBB",
#    "jesPileUpPtEC1",
#    "jesPileUpPtEC2",
#    "jesPileUpPtHF",
#    "jesPileUpMuZero",
#    "jesPileUpEnvelope",
#    "jesSubTotalPileUp",
#    "jesSubTotalRelative",
#    "jesSubTotalPt",
#    "jesSubTotalScale",
#    "jesSubTotalAbsolute",
#    "jesSubTotalMC",
#    "jesTotalNoFlavor",
#    "jesTotalNoTime",
#    "jesTotalNoFlavorNoTime",
#    "jesFlavorZJet",
#    "jesFlavorPhotonJet",
#    "jesFlavorPureGluon",
#    "jesFlavorPureQuark",
#    "jesFlavorPureCharm",
#    "jesFlavorPureBottom",
#    "jesTimeRunA",
#    "jesTimeRunB",
#    "jesTimeRunC",
#    "jesTimeRunD",
#    "jesCorrelationGroupMPFInSitu",
#    "jesCorrelationGroupIntercalibration",
#    "jesCorrelationGroupbJES",
#    "jesCorrelationGroupFlavor",
#    "jesCorrelationGroupUncorrelated",
]

systematic  = "jesFlavorQCD" 
scalar_branches = ["MET_T1_pt_%s%s"%(systematic, var) for var in ["Up","Down"] ]
scalar_branches+= ["MET_T1_pt" ]

vector_branches = [ "Jet_pt_%s%s" %(systematic,var) for var in ["Up","Down"]]
vector_branches +=[ "Jet_pt_nom" ]
max_nJet = 1

features     = ["jet_pt_0"]

def getEvents( sl = slice(None,1,None) ):

    from models.suman_TTLep_trainingfiles import files as training_files
    training_files = training_files[sl]

    print ("Loading %i file(s)."%len(training_files) )

    df_file = {}
    vec_br_f = {}
    for training_file in training_files:
        with uproot.open(training_file) as upfile:
            vec_br_f[training_file]={}
            df_file[training_file]  = upfile["Events"].pandas.df(branches = scalar_branches ) 
            for name, branch in upfile["Events"].arrays(vector_branches).items():
                vec_br_f[training_file][name.decode("utf-8")] = branch.pad(max_nJet)[:,:max_nJet].fillna(0)
        print("Loaded %s"%training_file)

    df = pd.concat([df_file[trainingfile] for trainingfile in training_files])
    df = df.dropna() # removes all Events with nan -> amounts to M3 cut
    df = df.values

    vec_br = {name: awkward.concatenate( [vec_br_f[training_file][name] for training_file in training_files] ) for name in list(vec_br_f.values())[0].keys()}
    del vec_br_f


    #features_train = { 0: torch.from_numpy(np.c_[ vec_br['Jet_pt_nom'], df[:,scalar_branches.index('MET_T1_pt')] ]).float().to(device),
    #                   1: torch.from_numpy(np.c_[ vec_br['Jet_pt_%sUp'%systematic],   df[:,scalar_branches.index('MET_T1_pt_%sUp'%systematic)] ]).float().to(device),
    #                  -1: torch.from_numpy(np.c_[ vec_br['Jet_pt_%sDown'%systematic], df[:,scalar_branches.index('MET_T1_pt_%sDown'%systematic)] ]).float().to(device) 
    #                 }

    return { 0: np.c_[ vec_br['Jet_pt_nom'] ],
             1: np.c_[ vec_br['Jet_pt_%sUp'%systematic] ],
            -1: np.c_[ vec_br['Jet_pt_%sDown'%systematic] ] 
                     }


plot_options = { "jet_pt_0":{"tex": "pt(j_{0})", "binning":[20,0,500]}, 
                #{"name":"jet_pt_1","tex": "pt(j_{1})", "binning":[20,0,500]}, 
#                {"name":"jet_pt_2","tex": "pt(j_{2})", "binning":[20,0,500]}, 
#                {"name":"jet_pt_3","tex": "pt(j_{3})", "binning":[20,0,500]}, 
                #{"name":"MET","tex": "p_{T}^{miss}", "binning":[20,0,500]},
                } 
