#!/usr/bin/env python

# Standard imports
import ROOT
ROOT.TH1.SetDefaultSumw2()

import numpy as np
import random
import cProfile
import time
import os, sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
from math import log, exp, sin, cos, sqrt, pi
import copy
import pickle
import itertools

dir_path = os.path.dirname(os.path.realpath(__file__))
ROOT.gROOT.LoadMacro(os.path.join( dir_path, "../../tools/scripts/tdrstyle.C"))
ROOT.setTDRStyle()

from   tools import helpers
import tools.syncer as syncer

# BIT
from BIT.MultiBoostedInformationTree import MultiBoostedInformationTree

# User
import tools.user as user

# Parser
import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument("--plot_directory",     action="store",      default="",                 help="plot sub-directory")
argParser.add_argument("--model",              action="store",      default="TT2l_EFT_delphes",                 help="Which model?")
argParser.add_argument("--modelFile",          action="store",      default="data_models",                 help="Which model directory?")
argParser.add_argument("--prefix",             action="store",      default="v1.1", type=str,  help="prefix")
argParser.add_argument("--nTraining",          action="store",      default=-1,        type=int,  help="number of training events")
argParser.add_argument("--coefficients",       action="store",      default=['ctGRe', 'ctGIm', 'cQj18', 'cQj38', 'ctj8'],       nargs="*", help="Which coefficients?")
 
args, extra = argParser.parse_known_args(sys.argv[1:])

def parse_value( s ):
    try:
        r = int( s )
    except ValueError:
        try:
            r = float(s)
        except ValueError:
            r = s
    return r

extra_args = {}
key        = None
for arg in extra:
    if arg.startswith('--'):
        # previous no value? -> Interpret as flag
        #if key is not None and extra_args[key] is None:
        #    extra_args[key]=True
        key = arg.lstrip('-')
        extra_args[key] = True # without values, interpret as flag
        continue
    else:
        if type(extra_args[key])==type([]):
            extra_args[key].append( parse_value(arg) )
        else:
            extra_args[key] = [parse_value(arg)]
for key, val in extra_args.items():
    if type(val)==type([]) and len(val)==1:
        extra_args[key]=val[0]

# import the model
exec('import %s.%s as model'%(args.modelFile, args.model)) 

model.multi_bit_cfg.update( extra_args )

feature_names = model.feature_names

# directory for plots
plot_directory = os.path.join( user.plot_directory, args.plot_directory, args.model )

if not os.path.isdir(plot_directory):
    try:
        os.makedirs( plot_directory )
    except IOError:
        pass

from data_models.plot_options import plot_options

data_model = model.DataModel(
        top_kinematics      =  False, 
        lepton_kinematics   =  False, 
        asymmetry           =  False, 
        spin_correlation    =  False, 
    )

training_features, training_weights = data_model.getEvents(args.nTraining)
print ("Created data set of size %i" % len(training_features) )

# Text on the plots
def drawObjects( offset=0 ):
    tex1 = ROOT.TLatex()
    tex1.SetNDC()
    tex1.SetTextSize(0.05)
    tex1.SetTextAlign(11) # align right

    tex2 = ROOT.TLatex()
    tex2.SetNDC()
    tex2.SetTextSize(0.04)
    tex2.SetTextAlign(11) # align right

    line1 = ( 0.15+offset, 0.95, "Boosted Info Trees" )
    return [ tex1.DrawLatex(*line1) ]#, tex2.DrawLatex(*line2) ]

base_points = []
for comb in list(itertools.combinations_with_replacement(args.coefficients,1))+list(itertools.combinations_with_replacement(args.coefficients,2)):
    base_points.append( {c:comb.count(c) for c in args.coefficients} )

if args.prefix == None:
    bit_name = "multiBit_%s_%s_coeffs_%s_nTraining_%i_nTrees_%i"%(args.model, data_model.name, "_".join(args.coefficients), args.nTraining, model.multi_bit_cfg["n_trees"])
else:
    bit_name = "multiBit_%s_%s_%s_coeffs_%s_nTraining_%i_nTrees_%i"%(args.model, data_model.name, args.prefix, "_".join(args.coefficients), args.nTraining, model.multi_bit_cfg["n_trees"])

# delete coefficients we don't need (the BIT coefficients are determined from the training weight keys)
if args.coefficients is not None:
    for key in list(training_weights.keys()):
        if not all( [k in args.coefficients for k in key]):
            del training_weights[key]

filename = os.path.join(user.model_directory, bit_name)+'.pkl'
try:
    print ("Loading %s for %s"%(bit_name, filename))
    bit = MultiBoostedInformationTree.load(filename)
except (IOError, EOFError, ValueError):
    bit = None

import glob
bit_pattern = bit_name.replace(args.model,args.model+"_*resample*")
bit_bootstrap={}
for _filename in []:#glob.glob(os.path.join(user.model_directory, bit_pattern)+'.pkl'):
    int_ = int( _filename.split("resample")[1].split("_")[0].replace("resample",""))
    bit_bootstrap[int_] = MultiBoostedInformationTree.load( _filename )
    print ("Loaded bootstrap prediction", _filename)

print ("Loaded %i bootstraps"%len(bit_bootstrap) )

test_features, test_weights, test_observers = data_model.getEvents(args.nTraining, return_observers=True)
print ("Created data set of size %i" % len(test_features) )

# delete coefficients we don't need
if args.coefficients is not None:
    for key in list(test_weights.keys()):
        if not all( [k in args.coefficients for k in key]):
            del test_weights[key]

c1 = ROOT.TCanvas("c1");

l = ROOT.TLegend(0.2,0.8,0.9,0.85)
l.SetNColumns(2)
l.SetFillStyle(0)
l.SetShadowColor(ROOT.kWhite)
l.SetBorderSize(0)

for logY in [True, False]:
    plot_directory_ = os.path.join( plot_directory, "bootstrap_plots", bit_name, "log" if logY else "lin" )

# GIF animation
tex = ROOT.TLatex()
tex.SetNDC()
tex.SetTextSize(0.06)

# colors
color = {}
i_lin, i_diag, i_mixed = 0,0,0
for i_der, der in enumerate(bit.derivatives):
    if len(der)==1:
        color[der] = ROOT.kAzure + i_lin
        i_lin+=1
    elif len(der)==2 and len(set(der))==1:
        color[der] = ROOT.kRed + i_diag
        i_diag+=1
    elif len(der)==2 and len(set(der))==2:
        color[der] = ROOT.kGreen + i_mixed
        i_mixed+=1

test_predictions = bit.vectorized_predict(test_features)
bootstrap_predictions = {}
for i_key, key in enumerate(bit_bootstrap.keys()):
    print ( "Predicting %i/%i"%(i_key, len(bit_bootstrap.keys())) )  
    bootstrap_predictions[key] = bit_bootstrap[key].vectorized_predict(test_features)

    if i_key==3: break

w0 = test_weights[()]

stuff = []

# FIXME
test_weights[('ctGRe',)]/=5.
test_predictions[:,bit.derivatives.index(('ctGRe',))]/=5
bit.derivatives = list( filter( lambda d:len(d)<=1, bit.derivatives))


for observables, features, postfix in [
    ( model.observers if hasattr(model, "observers") else [], test_observers, "_observers"),
    ( model.feature_names, test_features, ""),
    ]:

    if len(observables)==0: continue

    # 1D feature plot animation
    h_w0, h_ratio_prediction, h_ratio_bootstrap_prediction, h_ratio_truth, lin_binning = {}, {}, {}, {}, {}
    for i_feature, feature in enumerate(observables):
        # root style binning
        binning     = plot_options[feature]['binning']
        # linspace binning
        lin_binning[feature] = np.linspace(binning[1], binning[2], binning[0]+1)
        #digitize feature
        binned      = np.digitize(features[:,i_feature], lin_binning[feature] )
        # for each digit, create a mask to select the corresponding event in the bin (e.g. features[mask[0]] selects features in the first bin
        mask        = np.transpose( binned.reshape(-1,1)==range(1,len(lin_binning[feature])) )

        h_w0[feature]           = np.array([  w0[m].sum() for m in mask])
        h_derivative_prediction = np.array([ (w0.reshape(-1,1)*test_predictions)[m].sum(axis=0) for m in mask])
        h_bootstrap_predictions = {key:np.array([ (w0.reshape(-1,1)*bootstrap_predictions[key])[m].sum(axis=0) for m in mask])  for key in bootstrap_predictions.keys()}
        h_derivative_truth      = np.array([ (np.transpose(np.array([(test_weights[der] if der in test_weights else test_weights[tuple(reversed(der))]) for der in bit.derivatives])))[m].sum(axis=0) for m in mask])

        h_ratio_prediction[feature] = h_derivative_prediction/h_w0[feature].reshape(-1,1)
        h_ratio_bootstrap_prediction[feature]={}
        for key in bootstrap_predictions.keys(): 
            h_ratio_bootstrap_prediction[feature][key] = h_bootstrap_predictions[key]/h_w0[feature].reshape(-1,1)
        h_ratio_truth[feature]      = h_derivative_truth/h_w0[feature].reshape(-1,1)

        #if feature=='nJetGood':
        #    assert False, ""

    n_pads = len(observables)+1
    n_col  = int(sqrt(n_pads))
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

        for i_feature, feature in enumerate(observables):

            th1d_yield       = helpers.make_TH1F( (h_w0[feature], lin_binning[feature]) )
            c1.cd(i_feature+1)
            ROOT.gStyle.SetOptStat(0)
            th1d_ratio_pred_bootstrap  = { key: {der: helpers.make_TH1F( (h_ratio_bootstrap_prediction[feature][key][:,i_der], lin_binning[feature])) for i_der, der in enumerate( bit.derivatives ) } for key in h_ratio_bootstrap_prediction[feature].keys()}
            th1d_ratio_pred  = { der: helpers.make_TH1F( (h_ratio_prediction[feature][:,i_der], lin_binning[feature])) for i_der, der in enumerate( bit.derivatives ) }
            th1d_ratio_truth = { der: helpers.make_TH1F( (h_ratio_truth[feature][:,i_der], lin_binning[feature])) for i_der, der in enumerate( bit.derivatives ) }

            #for der in th1d_ratio_pred.keys():
            #    for i_bin in range(1, th1d_ratio_pred[der].GetNbinsX()+1):
            #        sigma = np.sqrt(np.var( [ th1d_ratio_pred_bootstrap[key][der].GetBinContent(i_bin) for key in th1d_ratio_pred_bootstrap.keys()] ))
            #        th1d_ratio_pred[der].SetBinError( i_bin, sigma )
            #        #print (i_feature, feature, der, th1d_ratio_pred[der].GetBinContent(i_bin), sigma)

            stuff.append(th1d_yield)
            stuff.append(th1d_ratio_truth)
            stuff.append(th1d_ratio_pred)
            #stuff.append(th1d_ratio_pred_bootstrap)

            th1d_yield.SetLineColor(ROOT.kGray+2)
            th1d_yield.SetMarkerColor(ROOT.kGray+2)
            th1d_yield.SetMarkerStyle(0)
            th1d_yield.GetXaxis().SetTitle(plot_options[feature]['tex'])
            th1d_yield.SetTitle("")

            th1d_yield.Draw("hist")
            
            for i_der, der in enumerate(bit.derivatives):
                th1d_ratio_truth[der].SetTitle("")
                th1d_ratio_truth[der].SetLineColor(color[der])
                th1d_ratio_truth[der].SetMarkerColor(color[der])
                th1d_ratio_truth[der].SetMarkerStyle(0)
                th1d_ratio_truth[der].SetLineWidth(2)
                th1d_ratio_truth[der].SetLineStyle(ROOT.kDashed)
                th1d_ratio_truth[der].GetXaxis().SetTitle(plot_options[feature]['tex'])

                th1d_ratio_pred[der].SetTitle("")
                th1d_ratio_pred[der].SetLineColor(color[der])
                th1d_ratio_pred[der].SetMarkerColor(color[der])
                th1d_ratio_pred[der].SetMarkerStyle(0)
                th1d_ratio_pred[der].SetLineWidth(2)
                th1d_ratio_pred[der].GetXaxis().SetTitle(plot_options[feature]['tex'])

                tex_name = "%s"%(",".join( der ))

                if i_feature==0:
                    l.AddEntry( th1d_ratio_truth[der], "R("+tex_name+")")
                    l.AddEntry( th1d_ratio_pred[der],  "#hat{R}("+tex_name+")")

            if i_feature==0:
                l.AddEntry( th1d_yield, "yield (SM)")

            max_ = max( map( lambda h:h.GetMaximum(), th1d_ratio_truth.values() ))
            max_ = 10**(1.5)*max_ if logY else 1.5*max_
            min_ = min( map( lambda h:h.GetMinimum(), th1d_ratio_truth.values() ))
            min_ = 0.1 if logY else (1.5*min_ if min_<0 else 0.75*min_)

            #if max_==min_: continue

            th1d_yield_min = th1d_yield.GetMinimum()
            th1d_yield_max = th1d_yield.GetMaximum()
            for bin_ in range(1, th1d_yield.GetNbinsX()+1 ):
                try:
                    th1d_yield.SetBinContent( bin_, (th1d_yield.GetBinContent( bin_ ) - th1d_yield_min)/th1d_yield_max*(max_-min_)*0.95 + min_  )
                except ZeroDivisionError:
                    pass

            #th1d_yield.Scale(max_/th1d_yield.GetMaximum())
            th1d_yield   .Draw("hist")
            ROOT.gPad.SetLogy(logY)
            th1d_yield   .GetYaxis().SetRangeUser(min_, max_)
            th1d_yield   .Draw("hist")
            for h in list(th1d_ratio_truth.values()):
                h .Draw("hsame")
            for h in list(th1d_ratio_pred.values()):
                h .Draw("E1hsame")

        c1.cd(len(observables)+1)
        l.Draw()

        #lines = [ (0.29, 0.9, 'N_{B} =%5i'%( max_n_tree )) ]
        #drawObjects = [ tex.DrawLatex(*line) for line in lines ]
        #for o in drawObjects:
        #    o.Draw()

        plot_directory_ = os.path.join( plot_directory, "training_plots", bit_name, "log" if logY else "lin" )
        if not os.path.isdir(plot_directory_):
            try:
                os.makedirs( plot_directory_ )
            except IOError:
                pass
        helpers.copyIndexPHP( plot_directory_ )
        c1.Print( os.path.join( plot_directory_, "bootstrap.png" ) )
    syncer.sync()
