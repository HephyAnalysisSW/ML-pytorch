import torch
import math
import numpy as np
import ROOT
ROOT.TH1.SetDefaultSumw2()
from   matplotlib import pyplot as plt
import os
import array
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
argParser.add_argument('--plot_directory',    action='store', type=str,   default=os.path.expandvars("v6") )
argParser.add_argument('--plot',              action='store_true', help="small?")
argParser.add_argument('--bias',              action='store_true', help="bias?")
argParser.add_argument('--network',           nargs='*', type=int, default = [32,32],  help='Network architecture')
argParser.add_argument('--nEvents',           type=int, default = 10**6,  help='Network architecture')
args = argParser.parse_args()

import awkward
import numpy as np
import pandas as pd

args.plot_directory += "_"+"_".join( map(str, args.network) )
if args.bias:
    args.plot_directory += "_bias"
# Hyperparameters

learning_rate = 1e-3
nEpoch        = 10000
plot_every    = 10
device        = 'cuda' if torch.cuda.is_available() else 'cpu'
nBins         = 10

sigmas = [-2, -1, -0.5, 0, 0.5, 1, 2]
colors = [ ROOT.kRed, ROOT.kOrange, ROOT.kBlue, ROOT.kBlack, ROOT.kMagenta, ROOT.kGreen, ROOT.kCyan ]

import models.jet_toy as model
features_train = model.getEvents( args.nEvents, sigmas = sigmas)
features_train = {k:v[v[:,0]<150] for k, v in features_train.items()}
features_train = {k:torch.from_numpy(v).float().to(device) for k, v in features_train.items() }
n_features     = len(model.features)

features_test  = model.getEvents( args.nEvents, sigmas = sigmas)
features_test = {k:v[v[:,0]<150] for k, v in features_test.items()}
features_test = {k:torch.from_numpy(v).float().to(device) for k, v in features_test.items() }

# make standard NN 
def make_NN( hidden_layers  = args.network ):
    model_nn = [torch.nn.BatchNorm1d(n_features), torch.nn.Linear(n_features, hidden_layers[0]), torch.nn.ReLU()]
    for i_layer, layer in enumerate(hidden_layers):

        model_nn.append(torch.nn.Linear(hidden_layers[i_layer], hidden_layers[i_layer+1] if i_layer+1<len(hidden_layers) else 1))
        if i_layer+1<len(hidden_layers):
            model_nn.append( torch.nn.ReLU() )

    return torch.nn.Sequential(*model_nn)

networks = { 'lin':make_NN(), 'quad':make_NN()}

for key, network in networks.items():
    print ("networks( %s ) = \n"% key, network)

optimizer = torch.optim.Adam(sum(list([ list(network.parameters()) for network in networks.values()]),[]), lr=learning_rate)
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

losses_train = []
losses_test  = []

tex = ROOT.TLatex()
tex.SetNDC()
tex.SetTextSize(0.04)

def r_hat( features, nu):
    return torch.exp(nu*networks['lin'](features) + .5*nu**2*networks['quad'](features))

def bias( features ):
    if args.bias:
        return torch.exp((features-20)/model.alpha)
    else:
        return 1. 

def calc_loss( features_train ):
    loss = 0
    for nu in sigmas:
        if nu==0: continue
        loss += (bias(features_train[nu])*(1./(1.+r_hat(features_train[nu], nu)))**2).sum() + (bias(features_train[0])*(1-1./(1.+r_hat(features_train[0], nu)))**2).sum() 
    return loss

for epoch in range(nEpoch):

    # Compute and print loss.

    loss = calc_loss( features_train ) 

    losses_train.append( (epoch, loss.item()) )

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
    #scheduler.step()
    print ("loss",loss.item())

    if (args.plot and (epoch % plot_every)==0) or epoch+1==nEpoch:
        with torch.no_grad():
            losses_test.append( (epoch, calc_loss( features_test ).item()) )

            pred = {sigma: r_hat(features_train[0], sigma).squeeze().cpu().detach().numpy() for sigma in sigmas}

            for i_feature, feature in enumerate(model.features): 
                binning   = [25,0,150]
                np_binning= np.linspace(binning[1], binning[2], 1+binning[0])

                h_yield       = helpers.make_TH1F(np.histogram(features_train[0][:,i_feature], np_binning ))
                h_truth       = {sigma: helpers.make_TH1F(np.histogram(features_train[sigma][:,i_feature], np_binning )) for sigma in sigmas}
                h_pred        = {sigma: helpers.make_TH1F(np.histogram(features_train[0][:,i_feature], np_binning, weights=pred[sigma] )) for sigma in sigmas}

                for h in list(h_truth.values()) + list(h_pred.values()):
                    h.Divide(h_yield)
                    h.SetMarkerStyle(0)

                l = ROOT.TLegend(0.2,0.77,0.9,0.93)
                l.SetNColumns(4)
                l.SetFillStyle(0)
                l.SetShadowColor(ROOT.kWhite)
                l.SetBorderSize(0)
                for i_sigma, sigma in enumerate(sigmas):
                    l.AddEntry(h_pred[sigma]   , "#hat{r}(%3.1f)"%sigma )
                    l.AddEntry(h_truth[sigma]  , "r(%3.1f)"%sigma )
                    h_truth[sigma]   .SetMarkerColor(colors[i_sigma])
                    h_truth[sigma]     .SetLineColor(colors[i_sigma])
                    h_pred[sigma]   .SetMarkerColor(colors[i_sigma])
                    h_pred[sigma]     .SetLineColor(colors[i_sigma])
                    h_pred[sigma]  .SetLineStyle(ROOT.kDashed) 

                max_ = max( map( lambda h:h.GetMaximum(), list(h_truth.values()) + list(h_pred.values()) ))

                lines = [
                        (0.16, 0.965, 'Epoch %5i    Loss %6.4f'%( epoch, loss ))
                        ]

                h_yield.Scale(max_/h_yield.GetMaximum())
                first = True
                c1 = ROOT.TCanvas()
                for sigma in sigmas:
                    if first:
                        h_truth[sigma].Draw("h" if first else "hsame")
                        h_truth[sigma].GetYaxis().SetRangeUser( 0.7, 1.55 )
                        first = False
                    h_truth[sigma].Draw("hsame")
                    h_pred[sigma] .Draw("hsame")

                l.Draw()

                drawObjects = [ tex.DrawLatex(*line) for line in lines ]
                for o in drawObjects:
                    o.Draw()

                plot_directory = os.path.join( user.plot_directory, "systematics", args.plot_directory)
                helpers.copyIndexPHP( plot_directory )
                
                c1.Print( os.path.join( plot_directory, "epoch_%05i_%s.png"%(epoch, feature) ) )
                syncer.makeRemoteGif(plot_directory, pattern="epoch_*_%s.png"%( feature ), name=feature)

                tg_train = ROOT.TGraph( len(losses_train), *list(map( lambda a: array.array('d',a), np.array(losses_train).transpose())) )
                tg_test  = ROOT.TGraph( len(losses_test),  *list(map( lambda a: array.array('d',a), np.array(losses_test).transpose())) )
                l = ROOT.TLegend(0.2,0.77,0.9,0.93)
                l.SetNColumns(2)
                l.SetFillStyle(0)
                l.SetShadowColor(ROOT.kWhite)
                l.SetBorderSize(0)
                l.AddEntry( tg_train, "training" ) 
                l.AddEntry( tg_test,  "validation" )

                tg_train.SetLineColor(ROOT.kRed)
                tg_test.SetLineColor(ROOT.kBlue)
                tg_train.SetMarkerColor(ROOT.kRed)
                tg_test.SetMarkerColor(ROOT.kBlue)
                tg_train.SetMarkerStyle(0)
                tg_test.SetMarkerStyle(0)
                tg_train.Draw("AL") 
                tg_test.Draw("L")
                l.Draw()
                c1.Print( os.path.join( plot_directory, "loss.png" ) )
         

        syncer.sync()
