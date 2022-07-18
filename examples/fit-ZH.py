import torch
import math
import numpy as np
from matplotlib import pyplot as plt
import os

import ROOT

import sys
sys.path.append('..')
from Tools import tdrstyle
from Tools import syncer 
from Tools import user
from Tools import helpers


# Parser
import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument("--plot_directory",     action="store",      default="v1_3D",                       help="Plot sub-directory")
argParser.add_argument("--coefficients",       action="store",      default=['cHW', 'cHWtil', 'cHQ3'],  help="Which coefficients?")
argParser.add_argument("--nEvents",            action="store",      type=int, default=300000,           help="nEvents")
#argParser.add_argument("--device",             action="store",      default="cpu",  choices = ["cpu", "cuda"],  help="Device?")
args = argParser.parse_args()

learning_rate = 1e-3
device        = 'cuda' if torch.cuda.is_available() else 'cpu'
n_epoch       = 10000
plot_every    = 100

# training data
import ZH_Nakamura 
ZH_Nakamura.feature_names = ZH_Nakamura.feature_names[0:6] # restrict features
features   = ZH_Nakamura.getEvents(args.nEvents)[:,0:6]
feature_names  = ZH_Nakamura.feature_names
plot_options   = ZH_Nakamura.plot_options
plot_vars      = ZH_Nakamura.feature_names

mask       = (features[:,feature_names.index('pT')]<900) & (features[:,feature_names.index('sqrt_s_hat')]<1800) 
features = features[mask]

n_features = len(features[0]) 
weights    = ZH_Nakamura.getWeights(features, ZH_Nakamura.make_eft() )

WC = 'cHW'
features_train = torch.from_numpy(features).float().to(device)

#coefficients   = ('cHW', ) 
#combinations   =  [ ('cHW',), ('cHW', 'cHW')] 
coefficients   =  ( 'cHW', 'cHWtil', 'cHQ3') 
combinations   =  [ ('cHW',), ('cHWtil',), ('cHQ3',), ('cHW', 'cHW'), ('cHWtil', 'cHWtil'), ('cHQ3', 'cHQ3'), ('cHW', 'cHWtil'), ('cHQ3', 'cHW'), ('cHQ3', 'cHWtil')] 

#base_points = [ {'cHW':value} for value in [-1.5, -.8, -.4, -.2, .2, .4, .8, 1.5] ]
base_points = [ {'cHW':value1, 'cHWtil':value2} for value1 in [-1.5, -.8, -.2, 0., .2, .8, 1.5]  for value2 in [-1.5, -.8, -.2, 0, .2, .8, 1.5]]
base_points = list(filter( (lambda point: all([ coeff in args.coefficients or (not (coeff in point.keys() and point[coeff]!=0)) for coeff in point.keys()]) and any(map(bool, point.values()))), base_points)) 

coefficients = tuple(filter( lambda coeff: coeff in args.coefficients, list(coefficients))) 
combinations = tuple(filter( lambda comb: all([c in args.coefficients for c in comb]), combinations)) 

#base_points    = [ { 'cHW':-1.5 }, {'cHW':-.8}, {'cHW':-.4}, {'cHW':-.2}, {'cHW':.2}, {'cHW':.4}, {'cHW':.8}, {'cHW':1.5} ]
#base_points   += [ { 'cHWtil':-1.5 }, {'cHWtil':-.8}, {'cHWtil':-.4}, {'cHWtil':-.2}, {'cHWtil':.2}, {'cHWtil':.4}, {'cHWtil':.8}, {'cHWtil':1.5} ]
#base_points   += [ { 'cHQ3':-.15 }, {'cHQ3':-.08}, {'cHQ3':-.04}, {'cHQ3':-.02}, {'cHQ3':.02}, {'cHQ3':.04}, {'cHQ3':.08}, {'cHQ3':0.15} ]

base_points    = list(map( lambda b:ZH_Nakamura.make_eft(**b), base_points ))

# make standard NN 
def make_NN( hidden_layers  = [32, 32, 32, 32] ):
    model_nn = [torch.nn.BatchNorm1d(n_features), torch.nn.ReLU(), torch.nn.Linear(n_features, hidden_layers[0])]
    for i_layer, layer in enumerate(hidden_layers):

        model_nn.append(torch.nn.Linear(hidden_layers[i_layer], hidden_layers[i_layer+1] if i_layer+1<len(hidden_layers) else 1))
        if i_layer+1<len(hidden_layers):
            model_nn.append( torch.nn.ReLU() )

    return torch.nn.Sequential(*model_nn)

n_hat = { combination:make_NN() for combination in combinations }

for combination in combinations:
    print ("n_hat( %s ) = \n"% ", ".join(combination), n_hat[combination])

def r_hat( predictions, eft ):
    return torch.add( 
        torch.sum( torch.stack( [(1. + predictions[(c,)]*eft[c])**2 for c in coefficients ]), dim=0),
        torch.sum( torch.stack( [torch.sum( torch.stack( [ predictions[tuple(sorted((c_1,c_2)))]*eft[c_2] for c_2 in coefficients[i_c_1:] ]), dim=0)**2 for i_c_1, c_1 in enumerate(coefficients) ] ), dim=0 ) )

def make_weight_ratio( weights, **kwargs ):
    eft      = kwargs
    result = torch.ones(len(weights[()])) 
    for combination in combinations:
        if len(combination)==1:
            result += eft[combination[0]]*weights[combination]/weights[()]
        elif len(combination)==2:# add up without the factor 1/2 because off diagonals are only summed in upper triangle 
            result += (0.5 if len(set(combination))==1 else 1.)*eft[combination[0]]*eft[combination[1]]*weights[combination]/weights[()]
    return result

base_point_weight_ratios = list( map( lambda base_point: make_weight_ratio( weights, **base_point ), base_points ) )

# loss functional
def f_loss(predictions):
    loss = -0.5*weights[()].sum()
    for i_base_point, base_point in enumerate(base_points):
        #fhat  = 1./(1. + ( 1. + theta*t_output)**2 + (theta*s_output)**2 )
        fhat  = 1./(1. + r_hat(predictions, base_point) )
        loss += ( torch.tensor(weights[()])*( -0.25 + base_point_weight_ratios[i_base_point]*fhat**2 + (1-fhat)**2 ) ).sum()
    return loss

#optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(sum([list(model.parameters()) for model in n_hat.values()],[]), lr=learning_rate)

losses = []

tex = ROOT.TLatex()
tex.SetNDC()
tex.SetTextSize(0.04)

# variables for ploting results
for nn in n_hat.values():
    nn.train()

for epoch in range(n_epoch):
    # Forward pass: compute predicted y by passing x to the model.
    predictions = {combination:n_hat[combination](features_train).squeeze() for combination in combinations}

    #print ("t", pred_t.mean(), "s", pred_s.mean())

    # Compute and print loss.
    loss = f_loss(predictions)
    losses.append(loss.item())
    if epoch % 100 == 99:
        print("epoch", epoch, "loss",  loss.item())

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch % plot_every)==0:
        with torch.no_grad():
            print (loss.item())
            for comb in combinations:
                if len(comb)==1: continue

                t1, t2 = comb

                pred_t1 = n_hat[(t1,)](features_train).squeeze().cpu().detach().numpy()
                if t1 == t2:
                    pred_t2 = pred_t1
                else: 
                    pred_t2 = n_hat[(t2,)](features_train).squeeze().cpu().detach().numpy()

                pred_s  = n_hat[comb](features_train).squeeze().cpu().detach().numpy()

                for var in plot_vars:
                    binning   = plot_options[var]['binning']
                    np_binning= np.linspace(binning[1], binning[2], 1+binning[0])


                    w0_train  = weights[()]
                    truth_0   = np.histogram(features_train[:,feature_names.index(var)], np_binning, weights=w0_train )

                    wp1_train = weights[(t1,)]
                    wp2_train = weights[(t2,)]
                    truth_p1  = np.histogram(features_train[:,feature_names.index(var)], np_binning, weights=wp1_train )

                    wpp_train = weights[comb]
                    truth_pp  = np.histogram(features_train[:,feature_names.index(var)], np_binning, weights=wpp_train )


                    #pred_0  = np.histogram(features_train[:,feature_names.index(var)], np_binning, weights=w0_train )
                    pred_p1  = np.histogram(features_train[:,feature_names.index(var)], np_binning, weights=w0_train*2*pred_t1 )
                    pred_pp = np.histogram(features_train[:,feature_names.index(var)], np_binning, weights=w0_train*2*(pred_t1*pred_t2+pred_s**2) )

                    h_yield       = helpers.make_TH1F(truth_0)
                    h_truth_p1    = helpers.make_TH1F(truth_p1)
                    h_truth_p1    .Divide(h_yield) 
                    h_truth_pp    = helpers.make_TH1F(truth_pp)
                    h_truth_pp    .Divide(h_yield) 

                    h_pred_p1      = helpers.make_TH1F(pred_p1)
                    h_pred_p1      .Divide(h_yield) 
                    h_pred_pp     = helpers.make_TH1F(pred_pp)
                    h_pred_pp     .Divide(h_yield) 

                    l = ROOT.TLegend(0.3,0.7,0.9,0.95)
                    l.SetNColumns(2)
                    l.SetFillStyle(0)
                    l.SetShadowColor(ROOT.kWhite)
                    l.SetBorderSize(0)

                    h_yield      .SetLineColor(ROOT.kGray+2) 
                    h_truth_p1   .SetLineColor(ROOT.kBlue) 
                    h_truth_pp   .SetLineColor(ROOT.kRed) 
                    h_pred_p1    .SetLineColor(ROOT.kBlue) 
                    h_pred_pp    .SetLineColor(ROOT.kRed) 
                    h_yield      .SetMarkerColor(ROOT.kGray+2) 
                    h_truth_p1   .SetMarkerColor(ROOT.kBlue) 
                    h_truth_pp   .SetMarkerColor(ROOT.kRed) 
                    h_pred_p1    .SetMarkerColor(ROOT.kBlue) 
                    h_pred_pp    .SetMarkerColor(ROOT.kRed) 
                    h_yield      .SetMarkerStyle(0)
                    h_truth_p1   .SetMarkerStyle(0)
                    h_truth_pp   .SetMarkerStyle(0)
                    h_pred_p1    .SetMarkerStyle(0)
                    h_pred_pp    .SetMarkerStyle(0)
                    h_truth_p1   .SetLineStyle(ROOT.kDashed) 
                    h_truth_pp   .SetLineStyle(ROOT.kDashed)

                    max_ = max( map( lambda h:h.GetMaximum(), [ h_truth_p1, h_truth_pp ] ))

                    l.AddEntry(h_truth_p1  , "D(%s) (truth)"%( t1 ) ) 
                    l.AddEntry(h_pred_p1   , "D(%s) (pred)"%( t1 ) ) 
                    if t1!=t2:
                        truth_p2     = np.histogram(features_train[:,feature_names.index(var)], np_binning, weights=wp2_train )
                        pred_p2      = np.histogram(features_train[:,feature_names.index(var)], np_binning, weights=w0_train*2*pred_t2 )
                        h_truth_p2   = helpers.make_TH1F(truth_p2)
                        h_truth_p2   .Divide(h_yield) 
                        h_pred_p2    = helpers.make_TH1F(pred_p2)
                        h_pred_p2    .Divide(h_yield)
                        h_truth_p2   .SetLineColor(ROOT.kGreen) 
                        h_pred_p2    .SetLineColor(ROOT.kGreen) 
                        h_truth_p2   .SetMarkerColor(ROOT.kGreen) 
                        h_pred_p2    .SetMarkerColor(ROOT.kGreen) 
                        h_truth_p2   .SetMarkerStyle(0)
                        h_pred_p2    .SetMarkerStyle(0)
                        h_truth_p2   .SetLineStyle(ROOT.kDashed)

                        max_ = max( map( lambda h:h.GetMaximum(), [ h_truth_p1, h_truth_p2, h_truth_pp ] ))

                        l.AddEntry(h_truth_p2   , "D(%s) (truth)"%( t2 ) ) 
                        l.AddEntry(h_pred_p2    , "D(%s) (pred)"%( t2 ) ) 

                    l.AddEntry(h_truth_pp  , "D(%s) (truth)"%( ",".join(comb) ) ) 
                    l.AddEntry(h_pred_pp   , "D(%s) (pred)"%( ",".join(comb) ) ) 

                    l.AddEntry(h_yield     , "yield" ) 

                    lines = [ 
                            (0.16, 0.965, 'Epoch %5i    Loss %6.4f'%( epoch, loss ))
                            ]


                    h_yield.Scale(max_/h_yield.GetMaximum())
                    for logY in [True, False]:
                        c1 = ROOT.TCanvas()
                        h_yield   .Draw("hist")
                        h_yield   .GetYaxis().SetRangeUser(0.001 if logY else 0, 10**(1.5)*max_ if logY else 1.5*max_)
                        h_yield   .Draw("hist")
                        h_pred_p1  .Draw("hsame") 
                        h_truth_p1 .Draw("hsame")
                        if t1!=t2: 
                            h_pred_p2  .Draw("hsame") 
                            h_truth_p2 .Draw("hsame") 
                        h_pred_pp .Draw("hsame")
                        h_truth_pp.Draw("hsame")
                        c1.SetLogy(logY) 
                        l.Draw()

                        drawObjects = [ tex.DrawLatex(*line) for line in lines ]
                        for o in drawObjects:
                            o.Draw()

                        plot_directory = os.path.join( user.plot_directory, "ZH_Nakamura", args.plot_directory, "log" if logY else "lin")
                        helpers.copyIndexPHP( plot_directory )
                        c1.Print( os.path.join( plot_directory, "epoch_%05i_%s_%s.png"%(epoch, "_".join(comb), var) ) )
                        syncer.makeRemoteGif(plot_directory, pattern="epoch_*_%s_%s.png"%("_".join(comb), var), name=var+"_"+"_".join(comb) )

            syncer.sync()
