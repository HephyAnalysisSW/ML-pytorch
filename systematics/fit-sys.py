import torch
import math
import numpy as np
import ROOT
ROOT.TH1.SetDefaultSumw2()
from   matplotlib import pyplot as plt
import os
from math import sqrt, log
from scipy.stats import ncx2
import sys
sys.path.append('..')
from Tools import tdrstyle
from Tools import syncer
from Tools import user
from Tools import helpers

#import argparse
#argParser = argparse.ArgumentParser(description = "Argument parser")
#argParser.add_argument('--name',               action='store', type=str,   default='default', help="Name of the training")
#argParser.add_argument('--output_directory',   action='store', type=str,   default='/mnt/hephy/cms/robert.schoefbeck/TMB/models/')
#argParser.add_argument('--input_directory',    action='store', type=str,   default=os.path.expandvars("/eos/vbc/user/$USER/TMB/training-ntuples-tttt-v2/MVA-training/") )
#argParser.add_argument('--small',              action='store_true', help="small?")
#
#args = argParser.parse_args()
try:
    import uproot
except:
    import uproot3 as uproot

import awkward
import numpy as np
import pandas as pd

## directories
#plot_directory   = os.path.join( user. plot_directory, 'MVA', args.name, args.config )
#output_directory = os.path.join( args.output_directory, args.name, args.config) 

# get the training variable names

# Hyperparameters

learning_rate = 5e-3
n_epoch       = 20000
plot_every    = 100
device        = 'cuda' if torch.cuda.is_available() else 'cpu'

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
max_nJet = 2

from TT_semi_lep_suman import files as training_files
#training_files = ["root://cms-xrd-global.cern.ch//store/group/phys_higgs/ec/hbb/ntuples/VHbbPostNano/UL/EFT/2018/V1/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106v1701/220510_102939/0000/tree_31.root"]

training_files = training_files[:3]
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
#vec_br_f={}
#for name, branch in upfile["Events"].arrays(vector_branches).items():
#    vec_br_f[name.decode("utf-8")] = np.column_stack(branch.pad(max_nJet)[:,:max_nJet].fillna(0)).transpose()#.reshape( len(df), len(vector_branches), max_nJet).transpose((0,2,1))



features    = [ {"name":"jet_pt_0","tex": "pt(j_{0})", "binning":[20,0,500]}, 
                {"name":"jet_pt_1","tex": "pt(j_{1})", "binning":[20,0,500]}, 
#                {"name":"jet_pt_2","tex": "pt(j_{2})", "binning":[20,0,500]}, 
#                {"name":"jet_pt_3","tex": "pt(j_{3})", "binning":[20,0,500]}, 
                {"name":"MET","tex": "p_{T}^{miss}", "binning":[20,0,500]},
                ] 

features_train = { 0: torch.from_numpy(np.c_[ vec_br['Jet_pt_nom'], df[:,scalar_branches.index('MET_T1_pt')] ]).float().to(device),
                   1: torch.from_numpy(np.c_[ vec_br['Jet_pt_%sUp'%systematic],   df[:,scalar_branches.index('MET_T1_pt_%sUp'%systematic)] ]).float().to(device),
                  -1: torch.from_numpy(np.c_[ vec_br['Jet_pt_%sDown'%systematic], df[:,scalar_branches.index('MET_T1_pt_%sDown'%systematic)] ]).float().to(device) 
                 }

n_features  = len(features)


# make standard NN 

def make_NN( hidden_layers  = [ 32, 32] ):
    model_nn = [torch.nn.BatchNorm1d(n_features), torch.nn.ReLU(), torch.nn.Linear(n_features, hidden_layers[0])]
    for i_layer, layer in enumerate(hidden_layers):

        model_nn.append(torch.nn.Linear(hidden_layers[i_layer], hidden_layers[i_layer+1] if i_layer+1<len(hidden_layers) else 1))
        if i_layer+1<len(hidden_layers):
            model_nn.append( torch.nn.ReLU() )

    return torch.nn.Sequential(*model_nn)

networks = { 'lin':make_NN() }#, 'quad':make_NN()}

for key, network in networks.items():
    print ("networks( %s ) = \n"% ", ".join(key), network)

def r_hat( features, nu, order = "quad"):
    #if order=="quad":
        #return torch.exp(networks['lin'](features)*nu + 0.5*networks['quad'](features)*nu**2)
    #else:
        return torch.exp(networks['lin'](features)*nu )
def c( features,  nu ):
    return 1./(1.+r_hat(features,  nu)) 

sigmas = [-1,0,+1]
def f_loss():
    #predictions_0  (c(predictions[0],0)**2).sum()
    loss = -float(len(features_train[0]))
    for i_sigma, sigma in enumerate(sigmas):
        if sigma==0: continue
        loss += ((c(features_train[sigma],sigma))**2).sum() + ((1.-c(features_train[0],sigma))**2).sum()
    return loss

#optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(sum([list(model.parameters()) for model in networks.values()],[]), lr=learning_rate)
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)


losses = []

tex = ROOT.TLatex()
tex.SetNDC()
tex.SetTextSize(0.04)

# variables for ploting results
for network in networks.values():
    network.train()

for epoch in range(n_epoch):

    #print ("t", pred_t.mean(), "s", pred_s.mean())

    # Compute and print loss.
    loss = f_loss()
    losses.append(loss.item())
    if epoch % 100 == 99:
        print("epoch", epoch, "loss",  loss.item())

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
    #scheduler.step()
    if (epoch % plot_every)==0:
        with torch.no_grad():
            print (loss.item())

            pred_lin_p1  = r_hat(features_train[0], 1, order="lin").squeeze().cpu().detach().numpy()
            #pred_quad_p1 = r_hat(features_train[0], 1, order="quad").squeeze().cpu().detach().numpy()
            pred_lin_m1  = r_hat(features_train[0],-1, order="lin").squeeze().cpu().detach().numpy()
            #pred_quad_m1 = r_hat(features_train[0],-1, order="quad").squeeze().cpu().detach().numpy()

            for i_feature, feature in enumerate(features):
                binning   = feature['binning'] 
                np_binning= np.linspace(binning[1], binning[2], 1+binning[0])

                #pred_0  = np.histogram(features_train[:,feature_names.index(var)], np_binning, weights=w0_train )
                h_yield       = helpers.make_TH1F(np.histogram(features_train[0][:,i_feature], np_binning ))
                h_truth_p1      = helpers.make_TH1F(np.histogram(features_train[+1][:,i_feature], np_binning ))
                h_truth_m1      = helpers.make_TH1F(np.histogram(features_train[-1][:,i_feature], np_binning ))
                h_pred_lin_p1   = helpers.make_TH1F(np.histogram(features_train[0][:,i_feature], np_binning, weights=pred_lin_p1 ))
                h_pred_lin_m1   = helpers.make_TH1F(np.histogram(features_train[0][:,i_feature], np_binning, weights=pred_lin_m1 ))
                #h_pred_quad_p1  = helpers.make_TH1F(np.histogram(features_train[0][:,i_feature], np_binning, weights=pred_quad_p1 ))
                #h_pred_quad_m1  = helpers.make_TH1F(np.histogram(features_train[0][:,i_feature], np_binning, weights=pred_quad_m1 ))

                #for h in [h_pred_lin_p1, h_pred_lin_m1, h_pred_quad_p1, h_pred_quad_m1, h_truth_p1, h_truth_m1]:
                for h in [h_pred_lin_p1, h_pred_lin_m1, h_truth_p1, h_truth_m1]:
                    h.Divide(h_yield)
                    h.SetMarkerStyle(0)
                    h.GetXaxis().SetTitle(feature['name'])

                l = ROOT.TLegend(0.3,0.7,0.9,0.95)
                l.SetNColumns(2)
                l.SetFillStyle(0)
                l.SetShadowColor(ROOT.kWhite)
                l.SetBorderSize(0)

                l.AddEntry(h_pred_lin_p1   , "r(lin,+1)" )
                l.AddEntry(h_pred_lin_m1   , "r(lin,-1)" )
                #l.AddEntry(h_pred_quad_p1   , "r(quad,+1)" )
                #l.AddEntry(h_pred_quad_m1   , "r(quad,-1)" )
                l.AddEntry(h_truth_p1   , "r(truth,+1)" )
                l.AddEntry(h_truth_m1   , "r(truth,-1)" )

                h_yield      .SetLineColor(ROOT.kGray+2)
                h_yield      .SetMarkerColor(ROOT.kGray+2)
                h_yield      .SetMarkerStyle(0)
                h_truth_p1     .SetLineColor(ROOT.kMagenta)
                h_truth_m1     .SetLineColor(ROOT.kRed)
                h_pred_lin_p1  .SetLineColor(ROOT.kBlue)
                h_pred_lin_m1  .SetLineColor(ROOT.kGreen+2)
                #h_pred_quad_p1 .SetLineColor(ROOT.kBlue)
                #h_pred_quad_m1 .SetLineColor(ROOT.kGreen+2)
                h_pred_lin_p1  .SetMarkerColor(ROOT.kBlue)
                h_pred_lin_m1  .SetMarkerColor(ROOT.kGreen+2)
                #h_pred_quad_p1 .SetMarkerColor(ROOT.kBlue)
                #h_pred_quad_m1 .SetMarkerColor(ROOT.kGreen+2)

                h_pred_lin_p1  .SetLineStyle(ROOT.kDashed)
                h_pred_lin_m1  .SetLineStyle(ROOT.kDashed)


                max_ = max( map( lambda h:h.GetMaximum(), [h_pred_lin_p1, h_pred_lin_m1] ))#, h_pred_quad_p1, h_pred_quad_m1] ))

                l.AddEntry(h_yield     , "yield" )

                lines = [
                        (0.16, 0.965, 'Epoch %5i    Loss %6.4f'%( epoch, loss ))
                        ]

                h_yield.Scale(max_/h_yield.GetMaximum())

                c1 = ROOT.TCanvas()
                h_yield   .Draw("hist")
                h_yield   .GetYaxis().SetRangeUser( 0.9, 1.1 )
                h_yield   .Draw("hist")
                h_truth_p1  .Draw("hsame")
                h_truth_m1  .Draw("hsame")
                h_pred_lin_p1  .Draw("hsame")
                h_pred_lin_m1  .Draw("hsame")
                #h_pred_quad_p1 .Draw("hsame")
                #h_pred_quad_m1 .Draw("hsame")
                l.Draw()

                drawObjects = [ tex.DrawLatex(*line) for line in lines ]
                for o in drawObjects:
                    o.Draw()

                plot_directory = os.path.join( user.plot_directory, "systematics")
                helpers.copyIndexPHP( plot_directory )
                c1.Print( os.path.join( plot_directory, "epoch_%05i_%s_%s.png"%(epoch, systematic, feature['name']) ) )
                syncer.makeRemoteGif(plot_directory, pattern="epoch_*_%s_%s.png"%( systematic, feature['name'] ), name=feature['name'])

        syncer.sync()
