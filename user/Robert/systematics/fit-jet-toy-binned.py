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

import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--name',               action='store', type=str,   default='default', help="Name of the training")
argParser.add_argument('--plot_directory',    action='store', type=str,   default=os.path.expandvars("v2") )
args = argParser.parse_args()

import awkward
import numpy as np
import pandas as pd


# Hyperparameters

learning_rate = 1e-3
n_epoch       = 10000
plot_every    = 10
device        = 'cuda' if torch.cuda.is_available() else 'cpu'
nBins         = 10
import data_models.jet_toy as model
features_train = model.getEvents(10**4,nBins=nBins)
features_train = {k:torch.from_numpy(v).float().to(device) for k, v in features_train.items() }
n_features     = len(model.features)

## make standard NN 
#def make_NN( hidden_layers  = [ 32, 32] ):
#    model_nn = [torch.nn.BatchNorm1d(n_features), torch.nn.Linear(n_features, hidden_layers[0]), torch.nn.ReLU()]
#    for i_layer, layer in enumerate(hidden_layers):
#
#        model_nn.append(torch.nn.Linear(hidden_layers[i_layer], hidden_layers[i_layer+1] if i_layer+1<len(hidden_layers) else 1))
#        if i_layer+1<len(hidden_layers):
#            model_nn.append( torch.nn.ReLU() )
#
#    return torch.nn.Sequential(*model_nn)
#
#networks = { 'lin':make_NN() }#, 'quad':make_NN()}
#
#for key, network in networks.items():
#    print ("networks( %s ) = \n"% ", ".join(key), network)

    
c = torch.randn(nBins, requires_grad=True) 

optimizer = torch.optim.Adam([c], lr=learning_rate)
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

losses = []

tex = ROOT.TLatex()
tex.SetNDC()
tex.SetTextSize(0.04)
            
c_truth = [ 1./(1.+ np.count_nonzero(features_train[1]==i_bin)/float(np.count_nonzero(features_train[0]==i_bin))) for i_bin in range(nBins) ]

for epoch in range(n_epoch):

    # Compute and print loss.
    
    loss = 0
    for sigma in [+1]:
        for i_bin in range(nBins):
            loss+= np.count_nonzero(features_train[1]==i_bin)*c[i_bin]**2 + np.count_nonzero(features_train[0]==i_bin)*(1-c[i_bin])**2

    losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
    #scheduler.step()
    print ("loss",loss.item())

    #if (epoch % plot_every)==0:
    #    with torch.no_grad():

    #        pred_lin_p1  = r_hat(features_train[0], 1, order="lin").squeeze().cpu().detach().numpy()
    #        #pred_quad_p1 = r_hat(features_train[0], 1, order="quad").squeeze().cpu().detach().numpy()
    #        pred_lin_m1  = r_hat(features_train[0],-1, order="lin").squeeze().cpu().detach().numpy()
    #        #pred_quad_m1 = r_hat(features_train[0],-1, order="quad").squeeze().cpu().detach().numpy()

    #        for i_feature, feature in enumerate(model.features):
    #            binning   = model.plot_options[feature]['binning'] 
    #            np_binning= np.linspace(binning[1], binning[2], 1+binning[0])

    #            #pred_0  = np.histogram(features_train[:,feature_names.index(var)], np_binning, weights=w0_train )
    #            h_yield       = helpers.make_TH1F(np.histogram(features_train[0][:,i_feature], np_binning ))
    #            h_truth_p1      = helpers.make_TH1F(np.histogram(features_train[+1][:,i_feature], np_binning ))
    #            h_truth_m1      = helpers.make_TH1F(np.histogram(features_train[-1][:,i_feature], np_binning ))
    #            h_pred_lin_p1   = helpers.make_TH1F(np.histogram(features_train[0][:,i_feature], np_binning, weights=pred_lin_p1 ))
    #            h_pred_lin_m1   = helpers.make_TH1F(np.histogram(features_train[0][:,i_feature], np_binning, weights=pred_lin_m1 ))
    #            #h_pred_quad_p1  = helpers.make_TH1F(np.histogram(features_train[0][:,i_feature], np_binning, weights=pred_quad_p1 ))
    #            #h_pred_quad_m1  = helpers.make_TH1F(np.histogram(features_train[0][:,i_feature], np_binning, weights=pred_quad_m1 ))

    #            #for h in [h_pred_lin_p1, h_pred_lin_m1, h_pred_quad_p1, h_pred_quad_m1, h_truth_p1, h_truth_m1]:
    #            for h in [h_pred_lin_p1, h_pred_lin_m1, h_truth_p1, h_truth_m1]:
    #                h.Divide(h_yield)
    #                h.SetMarkerStyle(0)
    #                h.GetXaxis().SetTitle(model.plot_options[feature]['tex'])

    #            l = ROOT.TLegend(0.3,0.7,0.9,0.95)
    #            l.SetNColumns(2)
    #            l.SetFillStyle(0)
    #            l.SetShadowColor(ROOT.kWhite)
    #            l.SetBorderSize(0)

    #            l.AddEntry(h_pred_lin_p1   , "r(lin,+1)" )
    #            l.AddEntry(h_pred_lin_m1   , "r(lin,-1)" )
    #            #l.AddEntry(h_pred_quad_p1   , "r(quad,+1)" )
    #            #l.AddEntry(h_pred_quad_m1   , "r(quad,-1)" )
    #            l.AddEntry(h_truth_p1   , "r(truth,+1)" )
    #            l.AddEntry(h_truth_m1   , "r(truth,-1)" )

    #            h_yield      .SetLineColor(ROOT.kGray+2)
    #            h_yield      .SetMarkerColor(ROOT.kGray+2)
    #            h_yield      .SetMarkerStyle(0)
    #            h_truth_p1     .SetLineColor(ROOT.kMagenta)
    #            h_truth_m1     .SetLineColor(ROOT.kRed)
    #            h_pred_lin_p1  .SetLineColor(ROOT.kBlue)
    #            h_pred_lin_m1  .SetLineColor(ROOT.kGreen+2)
    #            #h_pred_quad_p1 .SetLineColor(ROOT.kBlue)
    #            #h_pred_quad_m1 .SetLineColor(ROOT.kGreen+2)
    #            h_pred_lin_p1  .SetMarkerColor(ROOT.kBlue)
    #            h_pred_lin_m1  .SetMarkerColor(ROOT.kGreen+2)
    #            #h_pred_quad_p1 .SetMarkerColor(ROOT.kBlue)
    #            #h_pred_quad_m1 .SetMarkerColor(ROOT.kGreen+2)

    #            h_pred_lin_p1  .SetLineStyle(ROOT.kDashed)
    #            h_pred_lin_m1  .SetLineStyle(ROOT.kDashed)


    #            max_ = max( map( lambda h:h.GetMaximum(), [h_pred_lin_p1, h_pred_lin_m1, h_truth_p1, h_truth_m1] ))#, h_pred_quad_p1, h_pred_quad_m1] ))

    #            #l.AddEntry(h_yield     , "yield" )

    #            lines = [
    #                    (0.16, 0.965, 'Epoch %5i    Loss %6.4f'%( epoch, loss ))
    #                    ]

    #            h_yield.Scale(max_/h_yield.GetMaximum())

    #            c1 = ROOT.TCanvas()
    #            #h_yield   .Draw("hist")
    #            #h_yield   .Draw("hist")
    #            h_truth_p1  .Draw("h")
    #            h_truth_p1   .GetYaxis().SetRangeUser( 0.7, 1.55 )
    #            h_truth_m1  .Draw("hsame")
    #            h_pred_lin_p1  .Draw("hsame")
    #            h_pred_lin_m1  .Draw("hsame")
    #            #h_pred_quad_p1 .Draw("hsame")
    #            #h_pred_quad_m1 .Draw("hsame")
    #            l.Draw()

    #            drawObjects = [ tex.DrawLatex(*line) for line in lines ]
    #            for o in drawObjects:
    #                o.Draw()

    #            plot_directory = os.path.join( user.plot_directory, "systematics", args.plot_directory)
    #            helpers.copyIndexPHP( plot_directory )
    #            
    #            c1.Print( os.path.join( plot_directory, "epoch_%05i_%s.png"%(epoch, feature) ) )
    #            syncer.makeRemoteGif(plot_directory, pattern="epoch_*_%s.png"%( feature ), name=feature)

    #    syncer.sync()
