#!/usr/bin/env python

# Standard imports
import ROOT
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
argParser.add_argument("--plot_directory",     action="store",      default="multiBIT_P3_VH",                 help="plot sub-directory")
argParser.add_argument("--prefix",             action="store",      default=None, type=str,  help="prefix")
argParser.add_argument("--model",              action="store",      default="TT2lUnbinned", type=str,  help="model?")
argParser.add_argument("--nTraining",          action="store",      default=-1,        type=int,  help="number of training events")
argParser.add_argument("--coefficients",       action="store",      default=['cHW'],       nargs="*", help="Which coefficients?")
argParser.add_argument("--plot_iterations",    action="store",      default=None,          nargs="*", type=int, help="Certain iterations to plot? If first iteration is -1, plot only list provided.")
argParser.add_argument('--overwrite',          action='store',      default=None, choices = [None, "training", "data", "all"],  help="Overwrite output?")
argParser.add_argument('--bias',               action='store',      default=None, nargs = "*",  help="Bias training? Example:  --bias 'pT' '10**(({}-200)/200) ")
argParser.add_argument('--debug',              action='store_true', help="Make debug plots?")
argParser.add_argument('--feature_plots',      action='store_true', help="Feature plots?")
argParser.add_argument('--auto_clip',          action='store',      default=None, type=float, help="Remove quantiles of the training variable?")

argParser.add_argument('--top_kinematics',     action='store_true')
argParser.add_argument('--lepton_kinematics',  action='store_true')
argParser.add_argument('--asymmetry',          action='store_true')
argParser.add_argument('--spin_correlation',   action='store_true')

argParser.add_argument('--nJobs',       action='store',         nargs='?',  type=int, default=0,                                    help="Bootstrapping total number" )
argParser.add_argument('--job',         action='store',                     type=int, default=0,                                    help="Bootstrepping iteration" )

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

exec("import data_models.%s as model"%args.model)
from data_models.plot_options import plot_options

model.multi_bit_cfg.update( extra_args )
data_model = model.DataModel(
        top_kinematics      =   args.top_kinematics, 
        lepton_kinematics   =   args.lepton_kinematics, 
        asymmetry           =   args.asymmetry, 
        spin_correlation    =   args.spin_correlation
    )

# directory for plots
plot_directory = os.path.join( user.plot_directory, args.plot_directory, args.model )
os.makedirs( plot_directory, exist_ok=True)

training_data_filename = os.path.join(user.data_directory, args.model, data_model.name, "training_%i"%args.nTraining)+'.pkl'
if args.overwrite in ["all", "data"] or not os.path.exists(training_data_filename):
    training_features, training_weights = data_model.getEvents(args.nTraining)
    print ("Created data set of size %i" % len(training_features) )
    if not os.path.exists(os.path.dirname(training_data_filename)):
        os.makedirs(os.path.dirname(training_data_filename))
    with open( training_data_filename, 'wb' ) as _file:
        pickle.dump( [training_features, training_weights], _file )
        print ("Written training data to", training_data_filename)
else:
    with open( training_data_filename, 'rb') as _file:
        training_features, training_weights = pickle.load( _file )
        print ("Loaded training data from ", training_data_filename, "with size", len(training_features))

if args.auto_clip is not None:
    len_before = len(training_features)
    training_features, training_weights = helpers.clip_quantile(training_features, args.auto_clip, training_weights )
    print ("Auto clip efficiency (training) %4.3f is %4.3f"%( args.auto_clip, len(training_features)/len_before ) )

# Resample for bootstrapping
if args.nJobs>0:
    from sklearn.utils import resample
    rs_mask = resample(range(training_features.shape[0]))
    training_features = training_features[rs_mask]
    training_weights = {key:val[rs_mask] for key, val in training_weights.items()}
    print("Bootstrapping training data for job %i/%i"%( args.job, args.nJobs) )

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

###############
## Plot Model #
###############

stuff = []
if args.feature_plots and hasattr( model, "eft_plot_points"):
    h    = {}
    #h_lin= {}
    for i_eft, eft_plot_point in enumerate(model.eft_plot_points):
        eft = eft_plot_point['eft']

        if i_eft == 0:
            eft_sm     = eft

        name = ''
        name= '_'.join( [ (wc+'_%3.2f'%eft[wc]).replace('.','p').replace('-','m') for wc in model.wilson_coefficients if wc in eft ])
        tex_name = eft_plot_point['tex'] 

        if i_eft==0: name='SM'

        h[name]     = {}
        #h_lin[name] = {}

        eft['name'] = name
        
        for i_feature, feature in enumerate(data_model.feature_names):
            h[name][feature]        = ROOT.TH1F(name+'_'+feature+'_nom',    name+'_'+feature, *plot_options[feature]['binning'] )
            #h_lin[name][feature]    = ROOT.TH1F(name+'_'+feature+'_nom_lin',name+'_'+feature+'_lin', *plot_options[feature]['binning'] )

        # make reweights for x-check
        reweight     = copy.deepcopy(training_weights[()])
        # linear term
        for param1 in model.wilson_coefficients:
            reweight += (eft[param1]-eft_sm[param1])*training_weights[(param1,)] 
        reweight_lin  = copy.deepcopy( reweight )
        # quadratic term
        for param1 in model.wilson_coefficients:
            if eft[param1]-eft_sm[param1] ==0: continue
            for param2 in model.wilson_coefficients:
                if eft[param2]-eft_sm[param2] ==0: continue
                reweight += (.5 if param1!=param2 else 1)*(eft[param1]-eft_sm[param1])*(eft[param2]-eft_sm[param2])*training_weights[tuple(sorted((param1,param2)))]

        sign_postfix = ""

        for i_feature, feature in enumerate(data_model.feature_names):
            binning = plot_options[feature]['binning']

            h[name][feature] = helpers.make_TH1F( np.histogram(training_features[:,i_feature], np.linspace(binning[1], binning[2], binning[0]+1), weights=reweight) )
            #h_lin[name][feature] = helpers.make_TH1F( np.histogram(training_features[:,i_feature], np.linspace(binning[1], binning[2], binning[0]+1), weights=reweight_lin) )

            h[name][feature].SetLineWidth(2)
            h[name][feature].SetLineColor( eft_plot_point['color'] )
            h[name][feature].SetMarkerStyle(0)
            h[name][feature].SetMarkerColor(eft_plot_point['color'])
            h[name][feature].legendText = tex_name
            #h_lin[name][feature].SetLineWidth(2)
            #h_lin[name][feature].SetLineColor( eft_plot_point['color'] )
            #h_lin[name][feature].SetMarkerStyle(0)
            #h_lin[name][feature].SetMarkerColor(eft_plot_point['color'])
            #h_lin[name][feature].legendText = tex_name+(" (lin)" if name!="SM" else "")

    for i_feature, feature in enumerate(data_model.feature_names):

        for _h in [h]:#, h_lin]:
            norm = _h[model.eft_plot_points[0]['eft']['name']][feature].Integral()
            if norm>0:
                for eft_plot_point in model.eft_plot_points:
                    _h[eft_plot_point['eft']['name']][feature].Scale(1./norm) 

        for postfix, _h in [ ("", h)]:#, ("_linEFT", h_lin)]:
            histos = [_h[eft_plot_point['eft']['name']][feature] for eft_plot_point in model.eft_plot_points]
            max_   = max( map( lambda h__:h__.GetMaximum(), histos ))

            for logY in [True, False]:

                c1 = ROOT.TCanvas("c1");
                l = ROOT.TLegend(0.2,0.68,0.9,0.91)
                l.SetNColumns(2)
                l.SetFillStyle(0)
                l.SetShadowColor(ROOT.kWhite)
                l.SetBorderSize(0)
                for i_histo, histo in enumerate(reversed(histos)):
                    histo.GetXaxis().SetTitle(plot_options[feature]['tex'])
                    histo.GetYaxis().SetTitle("1/#sigma_{SM}d#sigma/d%s"%plot_options[feature]['tex'])
                    if i_histo == 0:
                        histo.Draw('hist')
                        histo.GetYaxis().SetRangeUser( (0.001 if logY else 0), (10*max_ if logY else 1.3*max_))
                        histo.Draw('hist')
                    else:
                        histo.Draw('histsame')
                    l.AddEntry(histo, histo.legendText)
                    c1.SetLogy(logY)
                l.Draw()

                plot_directory_ = os.path.join( plot_directory, "feature_plots", "nTraining_%i"%args.nTraining, "log" if logY else "lin" )
                helpers.copyIndexPHP( plot_directory_ )
                c1.Print( os.path.join( plot_directory_, feature+postfix+'.png' ))
                c1.Print( os.path.join( plot_directory_, feature+postfix+'.pdf' ))

            # Norm all shapes to 1
            for i_histo, histo in enumerate(histos):
                norm = histo.Integral()
                if norm>0:
                    histo.Scale(1./histo.Integral())

            # Divide all shapes by the SM
            ref = histos[0].Clone()
            for i_histo, histo in enumerate(histos):
                histo.Divide(ref)

            # Now plot shape differences
            for logY in [True, False]:
                c1 = ROOT.TCanvas("c1");
                l = ROOT.TLegend(0.2,0.78,0.9,0.91)
                l.SetNColumns(2)
                l.SetFillStyle(0)
                l.SetShadowColor(ROOT.kWhite)
                l.SetBorderSize(0)

                c1.SetLogy(logY)
                for i_histo, histo in enumerate(reversed(histos)):
                    histo.GetXaxis().SetTitle(plot_options[feature]['tex'])
                    histo.GetYaxis().SetTitle("shape wrt. SM")
                    if i_histo == 0:
                        histo.Draw('hist')
                        histo.GetYaxis().SetRangeUser( (0.01 if logY else 0.9), (10 if logY else 1.2))
                        histo.Draw('hist')
                    else:
                        histo.Draw('histsame')
                    l.AddEntry(histo, histo.legendText)
                    c1.SetLogy(logY)
                l.Draw()

                plot_directory_ = os.path.join( plot_directory, "shape_plots", "nTraining_%i"%args.nTraining, "log" if logY else "lin" )
                helpers.copyIndexPHP( plot_directory_ )
                c1.Print( os.path.join( plot_directory_, feature+postfix+'.png' ))

print ("Done with plots")
syncer.sync()

postfix = ""
if args.nJobs>0:
    postfix  = "_resample%05i"%args.job

base_points = []
for comb in list(itertools.combinations_with_replacement(args.coefficients,1))+list(itertools.combinations_with_replacement(args.coefficients,2)):
    base_points.append( {c:comb.count(c) for c in args.coefficients} )
if args.prefix == None:
    bit_name = "multiBit_%s_%s_coeffs_%s_nTraining_%i_nTrees_%i"%(args.model+postfix, data_model.name, "_".join(args.coefficients), args.nTraining, model.multi_bit_cfg["n_trees"])
else:
    bit_name = "multiBit_%s_%s_%s_coeffs_%s_nTraining_%i_nTrees_%i"%(args.model+postfix, data_model.name, args.prefix, "_".join(args.coefficients), args.nTraining, model.multi_bit_cfg["n_trees"])

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

# reweight training data according to bias
if args.bias is not None:
    if len(args.bias)!=2: raise RuntimeError ("Bias is defined by <var> <function>, i.e. 'x' '10**(({}-200)/200). Got instead %r"%args.bias)
    function     = eval( 'lambda x:'+args.bias[1].replace('{}','x') ) 
    bias_weights = np.array(list(map( function, training_features[:, data_model.feature_names.index(args.bias[0])] )))
    bias_weights /= np.mean(bias_weights)
    training_weights = {k:v*bias_weights for k,v in training_weights.items()} 

if bit is None or args.overwrite in ["all", "training"]:
    time1 = time.time()
    bit = MultiBoostedInformationTree(
            training_features     = training_features,
            training_weights      = training_weights,
            base_points           = base_points,
            feature_names         = data_model.feature_names,
            **model.multi_bit_cfg
                )
    bit.boost()
    bit.save(filename)
    print ("Written %s"%( filename ))

    time2 = time.time()
    boosting_time = time2 - time1
    print ("Boosting time: %.2f seconds" % boosting_time)

test_data_filename = os.path.join(user.data_directory, args.model, data_model.name, "test_%i"%args.nTraining)+'.pkl'
if args.overwrite in ["all", "data"] or not os.path.exists(test_data_filename):
    test_features, test_weights, test_observers = data_model.getEvents(args.nTraining, return_observers=True)
    print ("Created data set of size %i" % len(test_features) )
    if not os.path.exists(os.path.dirname(test_data_filename)):
        os.makedirs(os.path.dirname(test_data_filename), exist_ok=True)
    with open( test_data_filename, 'wb' ) as _file:
        pickle.dump( [test_features, test_weights, test_observers], _file )
        print ("Written test data to", test_data_filename)
else:
    with open( test_data_filename, 'rb') as _file:
        test_features, test_weights, test_observers = pickle.load( _file )
        print ("Loaded test data from ", test_data_filename, "with size", len(test_features))

if args.auto_clip is not None:
    len_before = len(test_features)

    selected = helpers.clip_quantile(test_features, args.auto_clip, return_selection = True)
    test_features = test_features[selected]
    test_weights = {k:test_weights[k][selected] for k in test_weights.keys()}
    if test_observers.size:
        test_observers = test_observers[selected] 
    print ("Auto clip efficiency (test) %4.3f is %4.3f"%( args.auto_clip, len(test_features)/len_before ) )

#if args.bias is not None:
#    bias_weights = np.array(list(map( function, test_features[:, data_model.feature_names.index(args.bias[0])] )))
#    bias_weights /= np.mean(bias_weights)
#    test_weights = {k:v*bias_weights for k,v in test_weights.items()} 

# delete coefficients we don't need
if args.coefficients is not None:
    for key in list(test_weights.keys()):
        if not all( [k in args.coefficients for k in key]):
            del test_weights[key]

if args.debug:

#    # Loss plot
#    training_losses = helpers.make_TH1F((bit.losses(training_features, training_weights),None), ignore_binning = True)
#    test_losses     = helpers.make_TH1F((bit.losses(test_features, test_weights),None),         ignore_binning = True)
#
#    c1 = ROOT.TCanvas("c1");
#
#    l = ROOT.TLegend(0.2,0.8,0.9,0.85)
#    l.SetNColumns(2)
#    l.SetFillStyle(0)
#    l.SetShadowColor(ROOT.kWhite)
#    l.SetBorderSize(0)
#
#    training_losses.GetXaxis().SetTitle("N_{B}")
#    training_losses.GetYaxis().SetTitle("Loss")
#    l.AddEntry(training_losses, "train")
#    l.AddEntry(test_losses, "test")
#
#    test_losses.SetLineWidth(2)
#    test_losses.SetLineColor(ROOT.kRed+2)
#    test_losses.SetMarkerColor(ROOT.kRed+2)
#    test_losses.SetMarkerStyle(0)
#    training_losses.SetLineWidth(2)
#    training_losses.SetLineColor(ROOT.kRed+2)
#    training_losses.SetMarkerColor(ROOT.kRed+2)
#    training_losses.SetMarkerStyle(0)
#
#    training_losses.Draw("hist") 
#    test_losses.Draw("histsame")
#
#    for logY in [True, False]:
#        plot_directory_ = os.path.join( plot_directory, "training_plots", bit_name, "log" if logY else "lin" )
#        c1.Print(os.path.join(plot_directory_, "loss.png"))

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

    # Which iterations to plot
    plot_iterations = list(range(1,10))+list(range(10,bit.n_trees+1,10))
    # Add plot iterations from command line, if provided
    if type(args.plot_iterations)==type([]):
        if args.plot_iterations[0]<0:
            plot_iterations+=args.plot_iterations[1:]
        else:
            plot_iterations = args.plot_iterations
        plot_iterations.sort()

    for max_n_tree in plot_iterations:
        if max_n_tree==0: max_n_tree=1
        test_predictions = bit.vectorized_predict(test_features, max_n_tree = max_n_tree)

        w0 = test_weights[()]

        # 2D plots for convergence + animation
        th1d_pred = {}
        th1d_truth= {}
        for i_der, der in enumerate( bit.derivatives ):
            truth_ratio = (test_weights[der] if der in test_weights else test_weights[tuple(reversed(der))])/w0
            quantiles = np.quantile(truth_ratio, q=(0.01,1-0.01))
            if len(der)==2: #quadratic
                binning = np.linspace( min([0, quantiles[0]]), quantiles[1], 21 )
            else:
                binning = np.linspace( quantiles[0], quantiles[1], 21 )
            th1d_truth[der]= helpers.make_TH1F( np.histogram( truth_ratio, bins = binning, weights=w0) )
            th1d_pred[der] = helpers.make_TH1F( np.histogram( test_predictions[:,i_der], bins = binning, weights=w0) )
            tex_name = "%s"%(",".join( der ))
            th1d_pred[der].GetXaxis().SetTitle( tex_name + " prediction" )
            th1d_pred[der].GetYaxis().SetTitle( "Number of Events" )
            th1d_truth[der].GetXaxis().SetTitle( tex_name + " truth" )
            th1d_truth[der].GetYaxis().SetTitle( "Number of Events" )

            th1d_truth[der].SetLineColor(color[der])
            th1d_truth[der].SetMarkerColor(color[der])
            th1d_truth[der].SetMarkerStyle(0)
            th1d_truth[der].SetLineWidth(2)
            th1d_truth[der].SetLineStyle(ROOT.kDashed)
            th1d_pred[der].SetLineColor(color[der])
            th1d_pred[der].SetMarkerColor(color[der])
            th1d_pred[der].SetMarkerStyle(0)
            th1d_pred[der].SetLineWidth(2)

        for observables, features, postfix in [
            #( model.observers if hasattr(model, "observers") else [], test_observers, "_observers"),
            ( data_model.feature_names, test_features, ""),
            ]:
            # 1D feature plot animation
            h_w0, h_ratio_prediction, h_ratio_truth, lin_binning = {}, {}, {}, {}
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
                h_derivative_truth      = np.array([ (np.transpose(np.array([(test_weights[der] if der in test_weights else test_weights[tuple(reversed(der))]) for der in bit.derivatives])))[m].sum(axis=0) for m in mask])

                h_ratio_prediction[feature] = h_derivative_prediction/h_w0[feature].reshape(-1,1) 
                h_ratio_truth[feature]      = h_derivative_truth/h_w0[feature].reshape(-1,1)

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
                    th1d_ratio_pred  = { der: helpers.make_TH1F( (h_ratio_prediction[feature][:,i_der], lin_binning[feature])) for i_der, der in enumerate( bit.derivatives ) }
                    th1d_ratio_truth = { der: helpers.make_TH1F( (h_ratio_truth[feature][:,i_der], lin_binning[feature])) for i_der, der in enumerate( bit.derivatives ) }

                    stuff.append(th1d_yield)
                    stuff.append(th1d_ratio_truth)
                    stuff.append(th1d_ratio_pred)

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

                    #FIXME
                    if min_<-0.1:
                        min_= -0.1

                    th1d_yield_min = th1d_yield.GetMinimum()
                    th1d_yield_max = th1d_yield.GetMaximum()
                    for bin_ in range(1, th1d_yield.GetNbinsX()+1 ):
                        th1d_yield.SetBinContent( bin_, (th1d_yield.GetBinContent( bin_ ) - th1d_yield_min)/th1d_yield_max*(max_-min_)*0.95 + min_  )

                    #th1d_yield.Scale(max_/th1d_yield.GetMaximum())
                    th1d_yield   .Draw("hist")
                    ROOT.gPad.SetLogy(logY)
                    th1d_yield   .GetYaxis().SetRangeUser(min_, max_)
                    th1d_yield   .Draw("hist")
                    for h in list(th1d_ratio_truth.values()) + list(th1d_ratio_pred.values()):
                        h .Draw("hsame")

                c1.cd(len(observables)+1)
                l.Draw()

                lines = [ (0.29, 0.9, 'N_{B} =%5i'%( max_n_tree )) ]
                drawObjects = [ tex.DrawLatex(*line) for line in lines ]
                for o in drawObjects:
                    o.Draw()

                plot_directory_ = os.path.join( plot_directory, "training_plots", bit_name, "log" if logY else "lin" )
                os.makedirs( plot_directory_, exist_ok=True)
                helpers.copyIndexPHP( plot_directory_ )
                c1.Print( os.path.join( plot_directory_, "epoch%s_%05i.png"%(postfix, max_n_tree) ) )
                syncer.makeRemoteGif(plot_directory_, pattern="epoch%s_*.png"%postfix, name="epoch" )
            syncer.sync()
