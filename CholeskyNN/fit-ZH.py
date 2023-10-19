import torch
import math
import numpy as np
import os

import ROOT

import sys
sys.path.append('..')
from tools import tdrstyle
from tools import syncer 
from tools import user
from tools import helpers

dir_path = os.path.dirname(os.path.realpath(__file__))
ROOT.gROOT.LoadMacro(os.path.join( dir_path, "../tools/scripts/tdrstyle.C"))
ROOT.setTDRStyle()
ROOT.TH1.AddDirectory(0)

# Parser
import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument("--plot_directory",     action="store",      default="v5_1D",                       help="Plot sub-directory")
#argParser.add_argument("--coefficients",       action="store",      default=['cHW', 'cHWtil', 'cHQ3'],  help="Which coefficients?")
argParser.add_argument("--nEvents",            action="store",      type=int, default=300000,           help="nEvents")
argParser.add_argument("--epochs",             action="store",      type=int, default=10000,           help="nEvents")
argParser.add_argument('--bias',               action='store',      default=None, nargs = "*",  help="Bias training? Example:  --bias 'pT' '10**(({}-200)/200)' ")
#argParser.add_argument("--device",             action="store",      default="cpu",  choices = ["cpu", "cuda"],  help="Device?")
args = argParser.parse_args()

learning_rate = 1e-3
device        = 'cuda' if torch.cuda.is_available() else 'cpu'
plot_every    = 100

# training data
import toy_models.ZH_Nakamura as model

features, weights = model.getEvents(args.nEvents)

feature_names  = model.feature_names[:6]
features       = features[:,:6]

mask       = (features[:,feature_names.index('pT')]<900) & (features[:,feature_names.index('sqrt_s_hat')]<1800) 
features = features[mask]
weights = {key:value[mask] for key, value in weights.items()}
n_features = len(features[0]) 

# we get the derivatives from this model, 
# not the polynomial (upper triangular) coefficients, 
# hence a factor 0.5 is needed for same-coefficient quadratic terms
for comb, value in weights.items():
    if len(comb)==2 and comb[0]==comb[1]:
        weights[comb] = 0.5*weights[comb]

if args.bias is not None:
    if len(args.bias)!=2: raise RuntimeError ("Bias is defined by <var> <function>, i.e. 'x' '10**(({}-200)/200). Got instead %r"%args.bias)
    function     = eval( 'lambda x:'+args.bias[1].replace('{}','x') ) 
    bias_weights = np.array(list(map( function, features[:, feature_names.index(args.bias[0])] )))
    bias_weights /= np.mean(bias_weights)
    weights = {k:v*bias_weights for k,v in weights.items()} 

features_train = torch.from_numpy(features).float().to(device)

coefficients   = ('cHW', ) 
combinations   = [ ('cHW',), ('cHW', 'cHW')] 
base_points    = [ {'cHW':value} for value in [-1.5, -.8, -.4, -.2, .2, .4, .8, 1.5] ]

#coefficients   =  ( 'cHW', 'cHWtil' ) 
#combinations   =  [ ('cHW',), ('cHWtil',), ('cHW', 'cHW'), ('cHWtil', 'cHWtil'), ('cHW', 'cHWtil')] 
##base_points    =  [ {'cHW':value1, 'cHWtil':value2} for value1 in [-1.5, -.8, 0., .8, 1.5]  for value2 in [-1.5, -.8,  0, .8, 1.5]]
#base_points = []
#for  value1 in [ 0,  1, 2]:
#    for  value2 in [ 0, 1, 2]:
#        if value1+value2<=2:
#            base_points.append({'cHW':value1, 'cHWtil':value2})

# make a neural network  
def make_NN( hidden_layers  = [32, 32, 32, 32] ):
    model_nn = [torch.nn.BatchNorm1d(n_features), torch.nn.ReLU(), torch.nn.Linear(n_features, hidden_layers[0])]
    for i_layer, layer in enumerate(hidden_layers):

        model_nn.append( torch.nn.Linear(   
                hidden_layers[i_layer], 
                hidden_layers[i_layer+1] if i_layer+1<len(hidden_layers) else len(combinations))
            )

        if i_layer+1<len(hidden_layers):
            model_nn.append( torch.nn.ReLU() )

    return torch.nn.Sequential(*model_nn)

n_hat = make_NN()

def r_hat( predictions, eft ):

    # "linear" term
    result = (1 + torch.sum( torch.stack( [predictions[:,i_c]*eft[c] for i_c, c in enumerate(coefficients) ]), dim=0))**2
    # purely quadratic terms
    for i_c_1, c_1 in enumerate( coefficients ):
        result += torch.sum( torch.stack( [ predictions[:, combinations.index(tuple(sorted((c_1,c_2))))]*eft[c_2] for c_2 in coefficients[i_c_1:] ]), dim=0)**2

    return result

def make_weight_ratio( weights, **kwargs ):
    ''' compute r(x,z) from weights and eft parameter point, i.e., 
        r(x,z) = 1 + theta_i weight_i + theta_i theta_j weight_{ij} 
        where weight_{ij} is considered upper triangular
    '''
    eft      = kwargs
    result = torch.ones(len(weights[()])) 
    for combination in combinations:
        if len(combination)==1:
            result += eft[combination[0]]*weights[combination]/weights[()]
        elif len(combination)==2:
            result += eft[combination[0]]*eft[combination[1]]*weights[combination]/weights[()]
    return result

# for training, compute r(x,z) for the base points
base_point_weight_ratios = list( map( lambda base_point: make_weight_ratio( weights, **base_point ), base_points ) )

# loss functional
def f_loss(predictions):
    loss = -0.5*weights[()].sum() # for numerical stability
    for i_base_point, base_point in enumerate(base_points):
        #fhat  = 1./(1. + ( 1. + theta*t_output)**2 + (theta*s_output)**2 )
        fhat  = 1./(1. + r_hat(predictions, base_point) )
        loss += ( torch.tensor(weights[()])*( -0.25 + base_point_weight_ratios[i_base_point]*fhat**2 + (1-fhat)**2 ) ).sum()
    return loss

#optimizer = torch.optim.Adam(sum([list(nn.parameters()) for nn in n_hat.values()],[]), lr=learning_rate)
optimizer = torch.optim.Adam(n_hat.parameters(), lr=learning_rate)

losses = []

tex = ROOT.TLatex()
tex.SetNDC()
tex.SetTextSize(0.04)

n_hat.train()

# colors
color = {}
i_lin, i_diag, i_mixed = 0,0,0
for i_comb, comb in enumerate(combinations):
    if len(comb)==1:
        color[comb] = ROOT.kAzure + i_lin
        i_lin+=1
    elif len(comb)==2 and len(set(comb))==1:
        color[comb] = ROOT.kRed + i_diag
        i_diag+=1
    elif len(comb)==2 and len(set(comb))==2:
        color[comb] = ROOT.kGreen + i_mixed
        i_mixed+=1

for epoch in range(args.epochs):
    # Forward pass: compute predicted y by passing x to the model.

    #predictions = {combination:n_hat[combination](features_train).squeeze() for combination in combinations}

    #print ("t", pred_t.mean(), "s", pred_s.mean())

    # Compute and print loss.
    #loss = f_loss(predictions)
    loss = f_loss(n_hat(features_train))

    losses.append(loss.item())
    if epoch % 100 == 99:
        print("epoch", epoch, "loss",  loss.item())

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    stuff = []
    if (epoch % plot_every)==0:
        with torch.no_grad():
            print (loss.item())
            predictions = n_hat(features_train)

            # numerically compute polynomial coefficients
            # r = (1 + sum_i theta_i r_i)^2 + sum_i ( sum_j>=i theta_j r_ij)^2  
            pred = {}
            for comb in combinations:
                if len(comb)==1:
                    pred[comb] = 0.5*( r_hat(predictions, model.make_eft(**{comb[0]:1})) - r_hat(predictions, model.make_eft(**{comb[0]:-1})) ) 
                elif len(comb)==2:
                    pred[comb] = 0.5*( r_hat(predictions, model.make_eft(**{comb[0]:1,comb[1]:1})) + r_hat(predictions, model.make_eft(**{comb[0]:-1,comb[1]:-1})) - 2*r_hat(predictions, model.make_eft())  )

                pred[comb] = pred[comb].squeeze().cpu().detach().numpy()

            n_pads = len(feature_names)+1
            n_col  = int(math.sqrt(n_pads))+1
            n_rows = n_pads//n_col
            if n_rows*n_col<n_pads: n_rows+=1

            for logY in [False, True]:
                c1 = ROOT.TCanvas("c1","multipads",500*n_col,500*n_rows);
                c1.Divide(n_col,n_rows)

                l = ROOT.TLegend(0.2,0.1,0.9,0.85)
                stuff.append(l)
                l.SetNColumns(2)
                l.SetFillStyle(0)
                l.SetShadowColor(ROOT.kWhite)
                l.SetBorderSize(0)

                #t1, t2 = comb
                #i_t1 = combinations.index( (t1,))
                #i_t2 = combinations.index( (t2,))
                #i_s  = combinations.index( comb )

                for i_feature, feature in enumerate(feature_names):

                    c1.cd(i_feature+1)
                    ROOT.gStyle.SetOptStat(0)

                    binning   = model.plot_options[feature]['binning']
                    np_binning= np.linspace(binning[1], binning[2], 1+binning[0])

                    sm_weight = weights[()]
                    h_yield   = helpers.make_TH1F(np.histogram(features_train[:,feature_names.index(feature)], np_binning, weights=sm_weight ))
                    h_yield      .SetLineColor(ROOT.kGray+2) 
                    stuff.append( h_yield )

                    h_truth, h_pred = {}, {}
                    for comb in combinations:
                        h_truth[comb] = helpers.make_TH1F(np.histogram(features_train[:,feature_names.index(feature)], np_binning, weights=weights[comb] ))
                        h_truth[comb].Divide( h_yield )
                        h_truth[comb].SetLineColor( color[comb] )
                        h_truth[comb].SetLineWidth(2)
                        h_truth[comb].SetMarkerColor( color[comb] )
                        h_truth[comb].SetMarkerStyle(0)
                        h_truth[comb].SetLineStyle(ROOT.kDashed)
                        stuff.append( h_truth[comb] )

                        h_pred[comb] = helpers.make_TH1F( np.histogram(features_train[:,feature_names.index(feature)], np_binning, weights=sm_weight*pred[comb] ) )
                        h_pred[comb].Divide(h_yield)
                        h_pred[comb].SetLineColor( color[comb] )
                        h_pred[comb].SetLineWidth(2)
                        h_pred[comb].SetMarkerColor( color[comb] )
                        h_pred[comb].SetMarkerStyle(0)
                        stuff.append( h_pred[comb] )

                        if i_feature==0:
                            tex_name = "%s"%(",".join( comb ))
                            l.AddEntry( h_truth[comb], "R("+tex_name+")")
                            l.AddEntry( h_pred[comb], "#hat{R}("+tex_name+")")

                    if i_feature==0:
                        l.AddEntry( h_yield, "SM yield")

                    max_ = max( map( lambda h:h.GetMaximum(), h_truth.values() ))
                    max_ = 10**(1.5)*max_ if logY else 1.5*max_
                    min_ = min( map( lambda h:h.GetMinimum(), h_truth.values() ))
                    min_ = 0.1 if logY else (1.5*min_ if min_<0 else 0.75*min_)

                    h_yield_min = h_yield.GetMinimum()
                    h_yield_max = h_yield.GetMaximum()
                    for bin_ in range(1, h_yield.GetNbinsX() ):
                        h_yield.SetBinContent( bin_, (h_yield.GetBinContent( bin_ ) - h_yield_min)/h_yield_max*(max_-min_)*0.95 + min_  )

                    #th1d_yield.Scale(max_/th1d_yield.GetMaximum())
                    h_yield.GetXaxis().SetTitle(model.plot_options[feature]['tex'])
                    h_yield   .Draw("hist")
                    ROOT.gPad.SetLogy(logY)
                    h_yield   .GetYaxis().SetRangeUser(min_, max_)
                    h_yield   .Draw("hist")
                    for h in list(h_truth.values()) + list(h_pred.values()):
                        h .Draw("hsame")

                c1.cd(len(feature_names)+1)
                l.Draw()

                lines = [(0.16, 0.965, 'Epoch %5i    Loss %6.4f'%( epoch, loss ))]
                drawObjects = [ tex.DrawLatex(*line) for line in lines ]
                for o in drawObjects:
                    o.Draw()

                plot_directory_ = os.path.join( user.plot_directory, "ZH_Nakamura", args.plot_directory, "log" if logY else "lin" )
                if not os.path.isdir(plot_directory_):
                    try:
                        os.makedirs( plot_directory_ )
                    except IOError:
                        pass
                helpers.copyIndexPHP( plot_directory_ )
                c1.Print( os.path.join( plot_directory_, "epoch_%05i.png"%( epoch) ) )
                syncer.makeRemoteGif(plot_directory_, pattern="epoch_*.png", name="epoch" )
            syncer.sync()

