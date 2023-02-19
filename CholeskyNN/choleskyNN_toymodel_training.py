#!/usr/bin/env python

# Standard imports
import ROOT
import numpy as np
import random
import cProfile
import time
import os, sys
from math import log, exp, sin, cos, sqrt, pi
import copy
import pickle
import itertools
import array
import torch

dir_path = os.path.dirname(os.path.realpath(__file__))
ROOT.gROOT.LoadMacro(os.path.join( dir_path, "../tools/scripts/tdrstyle.C"))
ROOT.setTDRStyle()

import sys
sys.path.append('..')

from   tools import helpers
import tools.syncer as syncer

# User
import tools.user as user

# Parser
import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument("--plot_directory",     action="store",      default="choleskyNN",                  help="plot sub-directory")
argParser.add_argument("--model",              action="store",      default="ZH_Nakamura",                 help="plot sub-directory")
argParser.add_argument("--prefix",             action="store",      default=None, type=str,  help="prefix")
argParser.add_argument("--nTraining",          action="store",      default=100000,        type=int,  help="number of training events")
argParser.add_argument("--coefficients",       action="store",      default=['cHW'],       nargs="*", help="Which coefficients?")
argParser.add_argument("--n_epoch",            action="store",      default=1000,           nargs="*", type=int, help="Number of training epochs.")
#argParser.add_argument("--snapshots",         action="store",      default=None,          nargs="*", type=int, help="Certain epochs to plot? If first epoch is -1, plot only the epochs in the list.")
argParser.add_argument('--overwrite',          action='store',      default=None, choices = [None, "training", "data", "all"],  help="Overwrite output?")
argParser.add_argument('--bias',               action='store',      default=None, nargs = "*",  help="Bias training? Example:  --bias 'pT' '10**(({}-200)/200)' ")
argParser.add_argument('--debug',              action='store_true', help="Make debug plots?")
argParser.add_argument('--feature_plots',      action='store_true', help="Feature plots?")

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

import imp
f, path, desc = imp.find_module(args.model, ["../toy_models/"])
model = imp.load_module(args.model, f, path, desc)

# directory for plots
plot_directory = os.path.join( user.plot_directory, args.plot_directory, args.model )

if not os.path.isdir(plot_directory):
    try:
        os.makedirs( plot_directory )
    except IOError:
        pass

training_data_filename = os.path.join(user.data_directory, args.model, "training_%i"%args.nTraining)+'.pkl'
if args.overwrite in ["all", "data"] or not os.path.exists(training_data_filename):
    training_features = model.getEvents(args.nTraining)
    training_weights  = model.getWeights(training_features, eft=model.default_eft_parameters)
    print ("Created data set of size %i" % len(training_features) )
    if not os.path.exists(os.path.dirname(training_data_filename)):
        os.makedirs(os.path.dirname(training_data_filename))
    with open( training_data_filename, 'wb' ) as _file:
        pickle.dump( [training_features, training_weights], _file )
        print ("Written training data to", training_data_filename)
else:
    with open( training_data_filename, 'rb') as _file:
        training_features, training_weights = pickle.load( _file )
        print ("Loaded training data from", training_data_filename)

print ("nEvents: %i Weights: %s" %( len(training_features), [ k for k in training_weights.keys() if k!=tuple()] ) )

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

    line1 = ( 0.15+offset, 0.95, "choleskyNN" )
    return [ tex1.DrawLatex(*line1) ]#, tex2.DrawLatex(*line2) ]

###############
## Plot Model #
###############

## FIXME
#model.feature_names     = model.feature_names[:6]
#training_features = training_features[:,:6]

if args.feature_plots and hasattr( model, "eft_plot_points"):

    h    = {}
    h_lin= {}
    h_rw = {}
    h_rw_lin = {}
    for i_eft, eft_plot_point in enumerate(model.eft_plot_points):
        eft = eft_plot_point['eft']

        if i_eft == 0:
            eft_sm     = eft
        name = ''
        name= '_'.join( [ (wc+'_%3.2f'%eft[wc]).replace('.','p').replace('-','m') for wc in model.wilson_coefficients if wc in eft ])
        tex_name = eft_plot_point['tex'] 

        if i_eft==0: name='SM'
        h[name] = {}
        eft['name']=name
        
        for i_feature, feature in enumerate(model.feature_names):
            h[name][feature]        = ROOT.TH1F(name+'_'+feature+'_nom',    name+'_'+feature, *model.plot_options[feature]['binning'] )

        # make reweights for x-check
        reweight = copy.deepcopy(training_weights[()])
        # linear term
        for param1 in model.wilson_coefficients:
            reweight += (eft[param1]-eft_sm[param1])*training_weights[(param1,)] 
        # quadratic term
        for param1 in model.wilson_coefficients:
            if eft[param1]-eft_sm[param1] ==0: continue
            for param2 in model.wilson_coefficients:
                if eft[param2]-eft_sm[param2] ==0: continue
                reweight += .5*(eft[param1]-eft_sm[param1])*(eft[param2]-eft_sm[param2])*training_weights[tuple(sorted((param1,param2)))]

        sign_postfix = ""
        if False:
            reweight_sign = np.sign(np.sin(2*np.arccos(training_features[:,model.feature_names.index('cos_theta')]))*np.sin(2*np.arccos(training_features[:,model.feature_names.index('cos_theta_hat')])))
            reweight     *= reweight_sign
            #reweight_lin_sign = reweight_sign*reweight_lin
            sign_postfix    = " weighted with sgn(sin(2#theta)sin(2#hat{#theta}))"

        for i_feature, feature in enumerate(model.feature_names):
            binning = model.plot_options[feature]['binning']

            h[name][feature] = helpers.make_TH1F( np.histogram(training_features[:,i_feature], np.linspace(binning[1], binning[2], binning[0]+1), weights=reweight) )
            #h[name][feature].style      = styles.lineStyle( eft_plot_point['color'], width=2, dashed=False )
            h[name][feature].SetLineWidth(2)
            h[name][feature].SetLineColor( eft_plot_point['color'] )
            h[name][feature].SetMarkerStyle(0)
            h[name][feature].SetMarkerColor(eft_plot_point['color'])

            h[name][feature].legendText = tex_name

    for i_feature, feature in enumerate(model.feature_names):

        for _h in [h]:
            norm = _h[model.eft_plot_points[0]['eft']['name']][feature].Integral()
            if norm>0:
                for eft_plot_point in model.eft_plot_points:
                    _h[eft_plot_point['eft']['name']][feature].Scale(1./norm) 
        histos = [h[eft_plot_point['eft']['name']][feature] for eft_plot_point in reversed(model.eft_plot_points)]
        #plot   = Plot.fromHisto( feature+'_nom',  histos, texX=model.plot_options[feature]['tex'], texY="1/#sigma_{SM}d#sigma/d%s"%model.plot_options[feature]['tex'] )

        for logY in [True, False]:

            max_ = max( map( lambda h:h.GetMaximum(), histos ))

            c1 = ROOT.TCanvas("c1");
            l = ROOT.TLegend(0.2,0.68,0.9,0.91)
            l.SetNColumns(2)
            l.SetFillStyle(0)
            l.SetShadowColor(ROOT.kWhite)
            l.SetBorderSize(0)
            for i_histo, histo in enumerate(histos):
                histo.GetXaxis().SetTitle(model.plot_options[feature]['tex'])
                histo.GetYaxis().SetTitle("1/#sigma_{SM}d#sigma/d%s"%model.plot_options[feature]['tex'])
                if i_histo == 0:
                    histo.Draw('hist')
                    histo.GetYaxis().SetRangeUser( 0.001 if logY else 0, 10*max_ if logY else 1.3*max_)
                    histo.Draw('hist')
                else:
                    histo.Draw('histsame')
                l.AddEntry(histo, histo.legendText)
                c1.SetLogy(logY)
            l.Draw()

            plot_directory_ = os.path.join( plot_directory, "feature_plots", "nTraining_%i"%args.nTraining, "log" if logY else "lin" )
            helpers.copyIndexPHP( plot_directory_ )
            c1.Print( os.path.join( plot_directory_, feature+'.png' ))
  
    print ("Done with plots")
    syncer.sync()

base_point_grids = [ 0.01, .5, 1 ]
base_points = []
for grid_spacing in base_point_grids:
    for comb in list(itertools.combinations_with_replacement(args.coefficients,1))+list(itertools.combinations_with_replacement(args.coefficients,2)):
        base_points.append( {c:grid_spacing*comb.count(c) for c in args.coefficients} )

if args.prefix == None:
    cnn_name = "choleskyNN_%s_%s_nTraining_%i"%(args.model, "_".join(args.coefficients), args.nTraining)
else:
    cnn_name = "choleskyNN_%s_%s_%s_nTraining_%i"%(args.model, args.prefix, "_".join(args.coefficients), args.nTraining)

# delete coefficients we don't need (the training coefficients are determined from the training weight keys)
if args.coefficients is not None:
    for key in list(training_weights.keys()):
        if not all( [k in args.coefficients for k in key]):
            del training_weights[key]

from CholeskyNN.CholeskyNN import CholeskyNN

filename = os.path.join(user.model_directory, cnn_name)+'.pkl'
if not args.overwrite in ["all", "training"]:
    try:
        print ("Loading %s for %s"%(cnn_name, filename))
        cnn = CholeskyNN.load(filename)
    except IOError:
        cnn = None
else:
    cnn = None

# reweight training data according to bias
if args.bias is not None:
    if len(args.bias)!=2: raise RuntimeError ("Bias is defined by <var> <function>, i.e. 'x' '10**(({}-200)/200). Got instead %r"%args.bias)
    function     = eval( 'lambda x:'+args.bias[1].replace('{}','x') ) 
    bias_weights = np.array(list(map( function, training_features[:, model.feature_names.index(args.bias[0])] )))
    bias_weights /= np.mean(bias_weights)
    training_weights = {k:v*bias_weights for k,v in training_weights.items()} 

test_data_filename = os.path.join(user.data_directory, args.model, "test_%i"%args.nTraining)+'.pkl'
if args.overwrite in ["all", "data"] or not os.path.exists(test_data_filename):
    test_features = model.getEvents(args.nTraining)
    test_weights  = model.getWeights(test_features, eft=model.default_eft_parameters)
    print ("Created data set of size %i" % len(test_features) )
    if not os.path.exists(os.path.dirname(test_data_filename)):
        os.makedirs(os.path.dirname(test_data_filename))
    with open( test_data_filename, 'wb' ) as _file:
        pickle.dump( [test_features, test_weights], _file )
        print ("Written test data to", test_data_filename)
else:
    with open( test_data_filename, 'rb') as _file:
        test_features, test_weights = pickle.load( _file )
        print ("Loaded test data from", test_data_filename)

# Which iterations to plot
plot_epoch = list(range(0,args.n_epoch,100)) + [args.n_epoch-1] 
## Add plot epoch from command line, if provided
#if type(args.plot_epoch)==type([]):
#    if args.plot_epoch[0]<0:
#        plot_epoch+=args.plot_epoch[1:]
#    else:
#        plot_epoch = args.plot_epoch
#    plot_epoch.sort()

if cnn is None or args.overwrite in ["all", "training"]:
    time1 = time.time()
    cnn = CholeskyNN( args.coefficients, len(training_features[0]))

    cnn.train( base_points, training_weights, training_features, test_weights, test_features, monitor_epoch = None, snapshots = plot_epoch,
               learning_rate = 1e-3, n_epoch = args.n_epoch)
 
    cnn.save(filename)
    print ("Written %s"%( filename ))

    time2 = time.time()
    training_time = time2 - time1
    print ("Training time: %.2f seconds" % training_time)

# we don't want to update any weights beyond here
torch.autograd.set_grad_enabled( False )

if args.bias is not None:
    bias_weights = np.array(list(map( function, test_features[:, model.feature_names.index(args.bias[0])] )))
    bias_weights /= np.mean(bias_weights)
    test_weights = {k:v*bias_weights for k,v in test_weights.items()} 

# delete coefficients we don't need
if args.coefficients is not None:
    for key in list(test_weights.keys()):
        if not all( [k in args.coefficients for k in key]):
            del test_weights[key]

if args.debug:

    # Loss plot
    training_losses = np.array([(m['epoch'], m['training_loss']) for m in cnn.monitoring])
    test_losses     = np.array([(m['epoch'], m['test_loss']) for m in cnn.monitoring if 'test_loss' in m])

    training_tGraph = ROOT.TGraph(len(training_losses), array.array('d', training_losses[:,0]), array.array('d', training_losses[:,1]))
    test_tGraph     = ROOT.TGraph(len(test_losses), array.array('d', test_losses[:,0]), array.array('d', test_losses[:,1]))

    c1 = ROOT.TCanvas("c1");

    l = ROOT.TLegend(0.2,0.8,0.9,0.85)
    l.SetNColumns(2)
    l.SetFillStyle(0)
    l.SetShadowColor(ROOT.kWhite)
    l.SetBorderSize(0)

    training_tGraph.GetXaxis().SetTitle("N_{B}")
    training_tGraph.GetYaxis().SetTitle("Loss")
    l.AddEntry(training_tGraph, "train")
    l.AddEntry(test_tGraph, "test")

    test_tGraph.SetLineWidth(2)
    test_tGraph.SetLineColor(ROOT.kBlue+2)
    test_tGraph.SetMarkerColor(ROOT.kBlue+2)
    test_tGraph.SetMarkerStyle(0)
    training_tGraph.SetLineWidth(2)
    training_tGraph.SetLineColor(ROOT.kRed+2)
    training_tGraph.SetMarkerColor(ROOT.kRed+2)
    training_tGraph.SetMarkerStyle(0)

    training_tGraph.Draw("AL") 
    test_tGraph.Draw("Lsame")
    l.Draw()

    for logY in [True, False]:
        plot_directory_ = os.path.join( plot_directory, "training_plots", cnn_name, "log" if logY else "lin" )
        c1.Print(os.path.join(plot_directory_, "loss.png"))

    # GIF animation
    tex = ROOT.TLatex()
    tex.SetNDC()
    tex.SetTextSize(0.06)

    # n_hat -> [ [d_lin_1, ..., d_quad_N] ] according to the "combinations" 
    def dict_to_derivatives( dict_ ):
        lin  = 2*np.array([dict_[c] for c in cnn.lin_combinations])
        quad = 2*np.array([dict_[(c[0],)]*dict_[(c[1],)] + np.sum( [dict_[tuple(sorted((c2, c[0])))]*dict_[tuple(sorted((c2, c[1])))] for c2 in cnn.coefficients if (cnn.coefficients.index(c2)>=cnn.coefficients.index(c[0]) and cnn.coefficients.index(c2)>=cnn.coefficients.index(c[1])) ],axis=0)  for c in cnn.quad_combinations ])
       
        return np.concatenate( (lin, quad), axis=0).transpose() 

    for epoch in plot_epoch:
        stuff = []

        cnn.load_snapshot( cnn.snapshots[epoch] )
        test_predictions = cnn.predict(test_features)
        test_predictions = cnn.dict_to_derivatives( test_predictions ) 

        #test_predictions = np.array( [test_predictions[k].numpy() for k in cnn.combinations ] ).transpose() 

        # colors
        color = {}
        i_lin, i_diag, i_mixed = 0,0,0
        for i_der, der in enumerate(cnn.combinations):
            if len(der)==1:
                color[der] = ROOT.kAzure + i_lin
                i_lin+=1
            elif len(der)==2 and len(set(der))==1:
                color[der] = ROOT.kRed + i_diag
                i_diag+=1
            elif len(der)==2 and len(set(der))==2:
                color[der] = ROOT.kGreen + i_mixed
                i_mixed+=1

        w0 = test_weights[()]
        h_w0, h_ratio_prediction, h_ratio_truth, lin_binning = {}, {}, {}, {}
        for i_feature, feature in enumerate(model.feature_names):
            # root style binning
            binning     = model.plot_options[feature]['binning']
            # linspace binning
            lin_binning[feature] = np.linspace(binning[1], binning[2], binning[0]+1)
            #digitize feature
            binned      = np.digitize(test_features[:,i_feature], lin_binning[feature] )
            # for each digit, create a mask to select the corresponding event in the bin (e.g. test_features[mask[0]] selects features in the first bin
            mask        = np.transpose( binned.reshape(-1,1)==range(1,len(lin_binning[feature])) )

            h_w0[feature]           = np.array([  w0[m].sum() for m in mask])
            h_derivative_prediction = np.array([ (w0.reshape(-1,1)*test_predictions)[m].sum(axis=0) for m in mask])
            h_derivative_truth      = np.array([ (np.transpose(np.array([test_weights[der] for der in cnn.combinations])))[m].sum(axis=0) for m in mask])

            h_ratio_prediction[feature] = h_derivative_prediction/h_w0[feature].reshape(-1,1) 
            h_ratio_truth[feature]      = h_derivative_truth/h_w0[feature].reshape(-1,1)

        n_pads = len(model.feature_names)+1
        n_col  = min(4, n_pads)
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

            for i_feature, feature in enumerate(model.feature_names):

                th1d_yield       = helpers.make_TH1F( (h_w0[feature], lin_binning[feature]) )
                c1.cd(i_feature+1)
                ROOT.gStyle.SetOptStat(0)
                th1d_ratio_pred  = { der: helpers.make_TH1F( (h_ratio_prediction[feature][:,i_der], lin_binning[feature])) for i_der, der in enumerate( cnn.combinations ) }
                th1d_ratio_truth = { der: helpers.make_TH1F( (h_ratio_truth[feature][:,i_der], lin_binning[feature])) for i_der, der in enumerate( cnn.combinations ) }
                stuff.append(th1d_yield)
                stuff.append(th1d_ratio_truth)
                stuff.append(th1d_ratio_pred)

                th1d_yield.SetLineColor(ROOT.kGray+2)
                th1d_yield.SetMarkerColor(ROOT.kGray+2)
                th1d_yield.SetMarkerStyle(0)
                th1d_yield.GetXaxis().SetTitle(model.plot_options[feature]['tex'])
                th1d_yield.SetTitle("")

                th1d_yield.Draw("hist")

                for i_der, der in enumerate(cnn.combinations):
                    th1d_ratio_truth[der].SetTitle("")
                    th1d_ratio_truth[der].SetLineColor(color[der])
                    th1d_ratio_truth[der].SetMarkerColor(color[der])
                    th1d_ratio_truth[der].SetMarkerStyle(0)
                    th1d_ratio_truth[der].SetLineWidth(2)
                    th1d_ratio_truth[der].SetLineStyle(ROOT.kDashed)
                    th1d_ratio_truth[der].GetXaxis().SetTitle(model.plot_options[feature]['tex'])

                    th1d_ratio_pred[der].SetTitle("")
                    th1d_ratio_pred[der].SetLineColor(color[der])
                    th1d_ratio_pred[der].SetMarkerColor(color[der])
                    th1d_ratio_pred[der].SetMarkerStyle(0)
                    th1d_ratio_pred[der].SetLineWidth(2)
                    th1d_ratio_pred[der].GetXaxis().SetTitle(model.plot_options[feature]['tex'])

                    tex_name = "_{%s}"%(",".join([model.tex[c].lstrip("C_{")[:-1] if model.tex[c].startswith('C_') else model.tex[c] for c in der]))

                    if i_feature==0:
                        l.AddEntry( th1d_ratio_truth[der], "R"+tex_name)
                        l.AddEntry( th1d_ratio_pred[der],  "#hat{R}"+tex_name)

                if i_feature==0:
                    l.AddEntry( th1d_yield, "yield (SM)")

                max_ = max( map( lambda h:h.GetMaximum(), th1d_ratio_truth.values() ))
                min_ = min( map( lambda h:h.GetMinimum(), th1d_ratio_truth.values() ))
                th1d_yield.Scale(max_/th1d_yield.GetMaximum())
                th1d_yield   .Draw("hist")
                ROOT.gPad.SetLogy(logY)
                th1d_yield   .GetYaxis().SetRangeUser(0.1 if logY else (1.5*min_ if min_<0 else 0.75*min_), 10**(1.5)*max_ if logY else 1.5*max_)
                th1d_yield   .Draw("hist")
                for h in list(th1d_ratio_truth.values()) + list(th1d_ratio_pred.values()):
                    h .Draw("hsame")

            c1.cd(len(model.feature_names)+1)
            l.Draw()

            lines = [ (0.29, 0.9, 'Epoch =%5i'%( epoch )) ]
            drawObjects = [ tex.DrawLatex(*line) for line in lines ]
            for o in drawObjects:
                o.Draw()

            plot_directory_ = os.path.join( plot_directory, "training_plots", cnn_name, "log" if logY else "lin" )
            if not os.path.isdir(plot_directory_):
                try:
                    os.makedirs( plot_directory_ )
                except IOError:
                    pass
            helpers.copyIndexPHP( plot_directory_ )
            c1.Print( os.path.join( plot_directory_, "epoch_%05i.png"%(epoch) ) )
            syncer.makeRemoteGif(plot_directory_, pattern="epoch_*.png", name="epoch" )
        syncer.sync()
