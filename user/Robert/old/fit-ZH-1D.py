import torch
import math
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
sys.path.append('..')
from tools import syncer 
from tools import user
from tools import helpers

import ROOT
from tools import tdrstyle

# Parser
import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument("--plot_directory",     action="store",      default="v3",                       help="Plot sub-directory")
argParser.add_argument("--model",              action="store",      default="ZH_Nakamura",                      help="Which Model?")
argParser.add_argument("--nEvents",            action="store",      type=int, default=300000,                   help="nEvents")
argParser.add_argument('--network',            nargs='*', type=int, default = [32,32],  help='Network architecture')

#argParser.add_argument("--device",             action="store",      default="cpu",  choices = ["cpu", "cuda"],  help="Device?")
argParser.add_argument("--bias",            action="store",      type=float, default=0,                   help="Bias weights by bias**pT ")
args = argParser.parse_args()

learning_rate = 1e-3
device        = 'cuda' if torch.cuda.is_available() else 'cpu'
n_epoch       = 10000
plot_every    = 100

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# training data
if args.model == 'ZH_Nakamura':
    import models.ZH_Nakamura as model
    model.feature_names = model.feature_names[0:6] # restrict features
    features   = model.getEvents(args.nEvents)[:,0:6]
    feature_names  = model.feature_names
    plot_options   = model.plot_options
    plot_vars      = model.feature_names

    mask       = (features[:,feature_names.index('pT')]<900) & (features[:,feature_names.index('sqrt_s_hat')]<1800) 
    features = features[mask]

    n_features = len(features[0]) 
    weights    = model.getWeights(features, model.make_eft() )
    pT=features[:,feature_names.index('pT')]
    bias_factor=bias**pT
    
    for key,value in weights.items():
        value*=bias_factor
        weights[key]=value
    
    
    WC = 'cHW'
    features_train = torch.from_numpy(features).float().to(device)
    w0_train       = torch.from_numpy(weights[()]).float().to(device)
    wp_train       = torch.from_numpy(weights[(WC,)]).float().to(device)
    wpp_train      = torch.from_numpy(weights[(WC,WC)]).float().to(device)
    
    
elif args.model == 'const':
    features_train = torch.ones(args.nEvents).unsqueeze(-1)
    n_features     = len(features_train[0]) 
    w0_train       = torch.ones(args.nEvents)
    wp_train       = torch.ones(args.nEvents)
    wpp_train      = torch.ones(args.nEvents)
    feature_names  = ['x'] 
    plot_options   = {'x':{'binning':[1,1,2]}}
    plot_vars      = ['x']

# make standard NN 
def make_NN( hidden_layers  = [32, 32, 32, 32] ):
    model_nn = [torch.nn.BatchNorm1d(n_features), torch.nn.ReLU(), torch.nn.Linear(n_features, hidden_layers[0])]
    for i_layer, layer in enumerate(hidden_layers):

        model_nn.append(torch.nn.Linear(hidden_layers[i_layer], hidden_layers[i_layer+1] if i_layer+1<len(hidden_layers) else 1))
        if i_layer+1<len(hidden_layers):
            model_nn.append( torch.nn.ReLU() )

    return torch.nn.Sequential(*model_nn)

model_t = make_NN() 
model_s = make_NN() 

print ("model_t\n", model_t)
print ("model_s\n", model_s)

# loss functional
def f_loss(w0_input, wp_input, wpp_input, t_output, s_output):
    base_points = [-1.5, -.8, -.4, -.2, .2, .4, .8, 1.5]
    loss = -0.5*w0_input.sum()
    for theta in base_points:
        fhat  = 1./(1. + ( 1. + theta*t_output)**2 + (theta*s_output)**2 )
        loss += ( w0_input*( -0.25 + (1. + wp_input/w0_input*theta +.5*wpp_input/w0_input*theta**2)*fhat**2 + (1-fhat)**2 ) ).sum()
        #FIXME -> weight ratios should be computed only once ... this is a waste
      
    return loss

#optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(list(model_t.parameters())+list(model_s.parameters()), lr=learning_rate)

losses = []

tex = ROOT.TLatex()
tex.SetNDC()
tex.SetTextSize(0.04)

# variables for ploting results
model_s.train()
model_t.train()
for epoch in range(n_epoch):
    # Forward pass: compute predicted y by passing x to the model.
    pred_t = model_t(features_train).squeeze()
    pred_s = model_s(features_train).squeeze()

    #print ("t", pred_t.mean(), "s", pred_s.mean())

    # Compute and print loss.
    loss = f_loss(w0_train, wp_train ,wpp_train, pred_t, pred_s)
    losses.append(loss.item())
    if epoch % 100 == 99:
        print("epoch", epoch, "loss",  loss.item())

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch % plot_every)==0:
        with torch.no_grad():
            print (loss.item())
            pred_t = model_t(features_train).squeeze().cpu().detach().numpy()
            pred_s = model_s(features_train).squeeze().cpu().detach().numpy()

            for var in plot_vars:
                binning     = plot_options[var]['binning']
                np_binning  = np.linspace(binning[1], binning[2], 1+binning[0])

                truth_0  = np.histogram(features_train[:,feature_names.index(var)], np_binning, weights=w0_train )
                truth_p  = np.histogram(features_train[:,feature_names.index(var)], np_binning, weights=wp_train )
                truth_pp = np.histogram(features_train[:,feature_names.index(var)], np_binning, weights=wpp_train )

                #pred_0  = np.histogram(features_train[:,feature_names.index(var)], np_binning, weights=w0_train )
                pred_p  = np.histogram(features_train[:,feature_names.index(var)], np_binning, weights=w0_train*2*pred_t )
                pred_pp = np.histogram(features_train[:,feature_names.index(var)], np_binning, weights=w0_train*2*(pred_t*pred_t+pred_s**2) )

                h_yield       = helpers.make_TH1F(truth_0)
                h_truth_p     = helpers.make_TH1F(truth_p)
                h_truth_p     .Divide(h_yield) 
                h_truth_pp    = helpers.make_TH1F(truth_pp)
                h_truth_pp    .Divide(h_yield) 

                h_pred_p      = helpers.make_TH1F(pred_p)
                h_pred_p      .Divide(h_yield) 
                h_pred_pp     = helpers.make_TH1F(pred_pp)
                h_pred_pp     .Divide(h_yield) 

                l = ROOT.TLegend(0.3,0.7,0.9,0.95)
                l.SetNColumns(2)
                l.SetFillStyle(0)
                l.SetShadowColor(ROOT.kWhite)
                l.SetBorderSize(0)

                h_yield      .SetLineColor(ROOT.kGray+2) 
                h_truth_p    .SetLineColor(ROOT.kBlue) 
                h_truth_pp   .SetLineColor(ROOT.kRed) 
                h_pred_p     .SetLineColor(ROOT.kBlue) 
                h_pred_pp    .SetLineColor(ROOT.kRed) 
                h_yield      .SetMarkerColor(ROOT.kGray+2) 
                h_truth_p    .SetMarkerColor(ROOT.kBlue) 
                h_truth_pp   .SetMarkerColor(ROOT.kRed) 
                h_pred_p     .SetMarkerColor(ROOT.kBlue) 
                h_pred_pp    .SetMarkerColor(ROOT.kRed) 
                h_yield      .SetMarkerStyle(0)
                h_truth_p    .SetMarkerStyle(0)
                h_truth_pp   .SetMarkerStyle(0)
                h_pred_p     .SetMarkerStyle(0)
                h_pred_pp    .SetMarkerStyle(0)

                l.AddEntry(h_truth_p   , "1^{st.} der (truth)" ) 
                l.AddEntry(h_truth_pp  , "2^{st.} der (truth)" ) 
                l.AddEntry(h_pred_p    , "1^{st.} der (pred)" ) 
                l.AddEntry(h_pred_pp   , "2^{st.} der (pred)" ) 
                l.AddEntry(h_yield     , "yield" ) 

                h_truth_p    .SetLineStyle(ROOT.kDashed) 
                h_truth_pp   .SetLineStyle(ROOT.kDashed)

                lines = [ 
                        (0.16, 0.965, 'Epoch %5i    Loss %6.4f'%( epoch, loss ))
                        ]

                max_ = max( map( lambda h:h.GetMaximum(), [ h_truth_p, h_truth_pp ] ))

                h_yield.Scale(max_/h_yield.GetMaximum())
                for logY in [True, False]:
                    c1 = ROOT.TCanvas()
                    h_yield   .Draw("hist")
                    h_yield   .GetYaxis().SetRangeUser(0.1 if logY else 0, 10**(1.5)*max_ if logY else 1.5*max_)
                    h_yield   .Draw("hist")
                    h_truth_p .Draw("hsame") 
                    h_truth_pp.Draw("hsame")
                    h_pred_p  .Draw("hsame") 
                    h_pred_pp .Draw("hsame")
                    c1.SetLogy(logY) 
                    l.Draw()

                    drawObjects = [ tex.DrawLatex(*line) for line in lines ]
                    for o in drawObjects:
                        o.Draw()

                    plot_directory = os.path.join( user.plot_directory, args.model, args.plot_directory, "log" if logY else "lin")
                    helpers.copyIndexPHP( plot_directory )
                    c1.Print( os.path.join( plot_directory, "epoch_%05i_%s.png"%(epoch, var) ) )
                    syncer.makeRemoteGif(plot_directory, pattern="epoch_*_%s.png"%var, name=var )
            syncer.sync()
