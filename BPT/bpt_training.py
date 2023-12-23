#!/usr/bin/env python

# Standard imports
import ROOT
import numpy as np
import random
import cProfile
import time
import os, sys
sys.path.insert(0, '..')
from math import log, exp, sin, cos, sqrt, pi
import copy
import pickle
import itertools

dir_path = os.path.dirname(os.path.realpath(__file__))
ROOT.gROOT.LoadMacro(os.path.join( dir_path, "../tools/scripts/tdrstyle.C"))
ROOT.setTDRStyle()

from   tools import helpers
import tools.syncer as syncer

from BPT import BPT

# User
import tools.user as user

# Parser
import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument("--plot_directory",     action="store",      default="BPT",                 help="plot sub-directory")
argParser.add_argument("--model",              action="store",      default="analytic",                 help="Which model?")
#argParser.add_argument("--modelFile",          action="store",      default="toy_models",                 help="Which model directory?")
argParser.add_argument("--prefix",             action="store",      default=None, type=str,  help="prefix")
argParser.add_argument("--nTraining",          action="store",      default=20000,       type=int,  help="number of training events")
argParser.add_argument("--plot_iterations",    action="store",      default=None,          nargs="*", type=int, help="Certain iterations to plot? If first iteration is -1, plot only list provided.")
argParser.add_argument('--overwrite',          action='store',      default=None, choices = [None, "training", "data", "all"],  help="Overwrite output?")
argParser.add_argument('--debug',              action='store_true', help="Make debug plots?")
argParser.add_argument('--feature_plots',      action='store_true', help="Feature plots?")
#argParser.add_argument('--auto_clip',          action='store',      default=None, type=float, help="Remove quantiles of the training variable?")

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
exec('import %s as model'%args.model) 

cfg = model.bpt_cfg
cfg.update( extra_args )

feature_names = model.feature_names

# directory for plots
plot_directory = os.path.join( user.plot_directory, args.plot_directory, args.model )

if not os.path.isdir(plot_directory):
    try:
        os.makedirs( plot_directory )
    except IOError:
        pass

training_data_filename = os.path.join(user.data_directory, args.model, "training_%i"%args.nTraining)+'.pkl'
if args.overwrite in ["all", "data"] or not os.path.exists(training_data_filename):
    training_data = model.getEvents(args.nTraining)
    total_size    =  sum([len(s['features']) for s in training_data.values() if 'features' in s ])
    print ("Created data set of size %i" % total_size )
    if not os.path.exists(os.path.dirname(training_data_filename)):
        os.makedirs(os.path.dirname(training_data_filename))
    with open( training_data_filename, 'wb' ) as _file:
        pickle.dump( training_data, _file )
        print ("Written training data to", training_data_filename)
else:
    with open( training_data_filename, 'rb') as _file:
        training_data = pickle.load( _file )
        total_size    =  sum([len(s['features']) for s in training_data.values() if 'features' in s ])
        print ("Loaded training data from ", training_data_filename, "with size", total_size)

#if args.auto_clip is not None:
#    len_before = len(training_features)
#    training_features, training_weights = helpers.clip_quantile(training_features, args.auto_clip, training_weights )
#    print ("Auto clip efficiency (training) %4.3f is %4.3f"%( args.auto_clip, len(training_features)/len_before ) )

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

colors = [ ROOT.kRed + 2, ROOT.kRed -4,ROOT.kCyan +2, ROOT.kCyan -4, ROOT.kMagenta+2,  ROOT.kMagenta-4,  ROOT.kBlue+2,     ROOT.kBlue-4,     ROOT.kGreen+2,    ROOT.kGreen-4, ] 
stuff = []
if args.feature_plots and hasattr( model, "plot_points"):
    h    = {}
    for i_point, point in enumerate(model.base_points):

        name     = '_'.join( [ (model.parameters[i_param]+'_%3.2f'%param).replace('.','p').replace('-','m') for i_param, param in enumerate(point)])
        tex_name = '_'.join( [ (model.parameters[i_param]+' = %3.2f'%param) for i_param, param in enumerate(point)])
        is_nominal = tuple(point) == tuple(model.nominal_base_point)
        if is_nominal:
            nominal_index = i_point
            nominal_name  = name

        if is_nominal:
            tex_name += " (nominal)"

        h[tuple(point)] = {'name':name}
        
        for i_feature, feature in enumerate(feature_names):
            h[tuple(point)][feature]        = ROOT.TH1F(name+'_'+feature+'_nom',    name+'_'+feature, *model.plot_options[feature]['binning'] )

            binning = model.plot_options[feature]['binning']

            if "weights" in training_data[tuple(point)]:
                weights = training_data[tuple(point)]['weights']
            elif "weights" in training_data[model.nominal_base_point]:
                weights = training_data[model.nominal_base_point]['weights']            
            else:
                weights = None

            h[tuple(point)][feature] = helpers.make_TH1F( np.histogram( 
                training_data[tuple(point)]['features'][:,i_feature] if 'features' in training_data[tuple(point)] else training_data[model.nominal_base_point]['features'][:,i_feature], 
                bins=np.linspace(binning[1], binning[2], binning[0]+1), weights=weights 
                )) 

            color = colors[i_point] if not is_nominal else ROOT.kBlack
            h[tuple(point)][feature].SetLineWidth(2)
            h[tuple(point)][feature].SetLineColor( color )
            h[tuple(point)][feature].SetMarkerStyle(0)
            h[tuple(point)][feature].SetMarkerColor(color)
            h[tuple(point)][feature].legendText = tex_name

    for i_feature, feature in enumerate(feature_names):

        norm = h[tuple(model.nominal_base_point)][feature].Integral()
        if norm>0:
            for point in model.base_points:
                h[tuple(point)][feature].Scale(1./norm) 

        histos = [h[tuple(point)][feature] for point in model.base_points]
        max_   = max( map( lambda h__:h__.GetMaximum(), histos ))

        for logY in [True, False]:

            c1 = ROOT.TCanvas("c1");
            l = ROOT.TLegend(0.2,0.68,0.9,0.91)
            l.SetNColumns(2)
            l.SetFillStyle(0)
            l.SetShadowColor(ROOT.kWhite)
            l.SetBorderSize(0)
            for i_histo, histo in enumerate(reversed(histos)):
                histo.GetXaxis().SetTitle(model.plot_options[feature]['tex'])
                histo.GetYaxis().SetTitle("1/#sigma_{SM}d#sigma/d%s"%model.plot_options[feature]['tex'])
                if i_histo == 0:
                    histo.Draw('hist')
                    histo.GetYaxis().SetRangeUser( (0.01 if logY else 0), (100*max_ if logY else 2*max_))
                    histo.Draw('hist')
                else:
                    histo.Draw('histsame')
                l.AddEntry(histo, histo.legendText)
                c1.SetLogy(logY)
            l.Draw()

            plot_directory_ = os.path.join( plot_directory, "feature_plots", "nTraining_%i"%args.nTraining, "log" if logY else "lin" )
            helpers.copyIndexPHP( plot_directory_ )
            c1.Print( os.path.join( plot_directory_, feature+'.png' ))

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
            l = ROOT.TLegend(0.2,0.68,0.9,0.91)
            l.SetNColumns(2)
            l.SetFillStyle(0)
            l.SetShadowColor(ROOT.kWhite)
            l.SetBorderSize(0)

            c1.SetLogy(logY)
            for i_histo, histo in enumerate(reversed(histos)):
                histo.GetXaxis().SetTitle(model.plot_options[feature]['tex'])
                histo.GetYaxis().SetTitle("shape wrt. SM")
                if i_histo == 0:
                    histo.Draw('hist')
                    histo.GetYaxis().SetRangeUser( (0.1 if logY else 0.8), (10 if logY else 1.4))
                    histo.Draw('hist')
                else:
                    histo.Draw('histsame')
                l.AddEntry(histo, histo.legendText)
                c1.SetLogy(logY)
            l.Draw()

            plot_directory_ = os.path.join( plot_directory, "shape_plots", "nTraining_%i"%args.nTraining, "log" if logY else "lin" )
            helpers.copyIndexPHP( plot_directory_ )
            c1.Print( os.path.join( plot_directory_, feature+'.png' ))

print ("Done with plots")
syncer.sync()

if args.prefix == None:
    bpt_name = "BPT_%s_nTraining_%i_nTrees_%i"%(args.model, args.nTraining, cfg["n_trees"])
else:
    bpt_name = "BPT_%s_%s_nTraining_%i_nTrees_%i"%(args.model, args.prefix, args.nTraining, cfg["n_trees"])

filename = os.path.join(user.model_directory, 'BPT', bpt_name)+'.pkl'
try:
    print ("Loading %s for %s"%(bpt_name, filename))
    bpt = BPT.load(filename)
except (IOError, EOFError, ValueError):
    bpt = None

if bpt is None or args.overwrite in ["all", "training"]:
    time1 = time.time()
    bpt = BPT(
            training_data      = training_data,
            nominal_base_point = model.nominal_base_point,
            parameters         = model.parameters,
            combinations       = model.combinations,
            feature_names      = model.feature_names,
            **cfg,
                )
    bpt.boost()
    bpt.save(filename)
    print ("Written %s"%( filename ))

    time2 = time.time()
    boosting_time = time2 - time1
    print ("Boosting time: %.2f seconds" % boosting_time)

test_data_filename = os.path.join(user.data_directory, 'BPT', args.model, "test_%i"%args.nTraining)+'.pkl'
if args.overwrite in ["all", "data"] or not os.path.exists(test_data_filename):
    test_data = model.getEvents(args.nTraining)
    total_size    =  sum([len(s['features']) for s in test_data.values() if 'features' in s ])
    print ("Created data set of size %i" % total_size )
    if not os.path.exists(os.path.dirname(test_data_filename)):
        os.makedirs(os.path.dirname(test_data_filename))
    with open( test_data_filename, 'wb' ) as _file:
        pickle.dump( test_data, _file )
        print ("Written test data to", test_data_filename)
else:
    with open( test_data_filename, 'rb') as _file:
        test_data = pickle.load( _file )
        total_size    =  sum([len(s['features']) for s in test_data.values() if 'features' in s ])
        print ("Loaded test data from ", test_data_filename, "with size", total_size)

#if args.auto_clip is not None:
#    len_before = len(test_features)
#
#    selected = helpers.clip_quantile(test_features, args.auto_clip, return_selection = True)
#    test_features = test_features[selected]
#    test_weights = {k:test_weights[k][selected] for k in test_weights.keys()}
#    if test_observers.size:
#        test_observers = test_observers[selected] 
#    print ("Auto clip efficiency (test) %4.3f is %4.3f"%( args.auto_clip, len(test_features)/len_before ) )


if args.debug:

    # Loss plot
    training_losses = helpers.make_TH1F((bit.losses(training_features, training_weights),None), ignore_binning = True)
    test_losses     = helpers.make_TH1F((bit.losses(test_features, test_weights),None),         ignore_binning = True)

    c1 = ROOT.TCanvas("c1");

    l = ROOT.TLegend(0.2,0.8,0.9,0.85)
    l.SetNColumns(2)
    l.SetFillStyle(0)
    l.SetShadowColor(ROOT.kWhite)
    l.SetBorderSize(0)

    training_losses.GetXaxis().SetTitle("N_{B}")
    training_losses.GetYaxis().SetTitle("Loss")
    l.AddEntry(training_losses, "train")
    l.AddEntry(test_losses, "test")

    test_losses.SetLineWidth(2)
    test_losses.SetLineColor(ROOT.kRed+2)
    test_losses.SetMarkerColor(ROOT.kRed+2)
    test_losses.SetMarkerStyle(0)
    training_losses.SetLineWidth(2)
    training_losses.SetLineColor(ROOT.kRed+2)
    training_losses.SetMarkerColor(ROOT.kRed+2)
    training_losses.SetMarkerStyle(0)

    training_losses.Draw("hist") 
    test_losses.Draw("histsame")

    for logY in [True, False]:
        plot_directory_ = os.path.join( plot_directory, "training_plots", bit_name, "log" if logY else "lin" )
        c1.Print(os.path.join(plot_directory_, "loss.png"))

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
        th2d = {}
        th1d_pred = {}
        th1d_truth= {}
        for i_der, der in enumerate( bit.derivatives ):
            truth_ratio = (test_weights[der] if der in test_weights else test_weights[tuple(reversed(der))])/w0
            quantiles = np.quantile(truth_ratio, q=(0.01,1-0.01))
            if len(der)==2: #quadratic
                binning = np.linspace( min([0, quantiles[0]]), quantiles[1], 21 )
            else:
                binning = np.linspace( quantiles[0], quantiles[1], 21 )
            th2d[der]      = helpers.make_TH2F( np.histogram2d( truth_ratio, test_predictions[:,i_der], bins = [binning, binning], weights=w0) )
            th1d_truth[der]= helpers.make_TH1F( np.histogram( truth_ratio, bins = binning, weights=w0) )
            th1d_pred[der] = helpers.make_TH1F( np.histogram( test_predictions[:,i_der], bins = binning, weights=w0) )
            tex_name = "%s"%(",".join( der ))
            th2d[der].GetXaxis().SetTitle( tex_name + " truth" )
            th2d[der].GetYaxis().SetTitle( tex_name + " prediction" )
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

        n_col  = len(bit.derivatives)
        n_rows = 2
        for logZ in [False, True]:
            c1 = ROOT.TCanvas("c1","multipads",500*n_col,500*n_rows);
            c1.Divide(n_col,n_rows)

            for i_der, der in enumerate(bit.derivatives):

                c1.cd(i_der+1)
                ROOT.gStyle.SetOptStat(0)
                th2d[der].Draw("COLZ")
                ROOT.gPad.SetLogz(logZ)

            lines = [ (0.29, 0.9, 'N_{B} =%5i'%( max_n_tree )) ]
            drawObjects = [ tex.DrawLatex(*line) for line in lines ]
            for o in drawObjects:
                o.Draw()

            for i_der, der in enumerate(bit.derivatives):
                c1.cd(i_der+1+len(bit.derivatives))
                l = ROOT.TLegend(0.6,0.75,0.9,0.9)
                stuff.append(l)
                l.SetNColumns(1)
                l.SetFillStyle(0)
                l.SetShadowColor(ROOT.kWhite)
                l.SetBorderSize(0)
                tex_name = "%s"%(",".join( der ))
                l.AddEntry( th1d_truth[der], "R("+tex_name+")")
                l.AddEntry( th1d_pred[der],  "#hat{R}("+tex_name+")")
                ROOT.gStyle.SetOptStat(0)
                th1d_pred[der].Draw("hist")
                th1d_truth[der].Draw("histsame")
                ROOT.gPad.SetLogy(logZ)
                l.Draw()

            plot_directory_ = os.path.join( plot_directory, "training_plots", bit_name, "log" if logZ else "lin" )
            if not os.path.isdir(plot_directory_):
                try:
                    os.makedirs( plot_directory_ )
                except IOError:
                    pass
            helpers.copyIndexPHP( plot_directory_ )
            c1.Print( os.path.join( plot_directory_, "training_2D_epoch_%05i.png"%(max_n_tree) ) )
            syncer.makeRemoteGif(plot_directory_, pattern="training_2D_epoch_*.png", name="training_2D_epoch" )


        for observables, features, postfix in [
            ( model.observers if hasattr(model, "observers") else [], test_observers, "_observers"),
            ( model.feature_names, test_features, ""),
            ]:
            # 1D feature plot animation
            h_w0, h_ratio_prediction, h_ratio_truth, lin_binning = {}, {}, {}, {}
            for i_feature, feature in enumerate(observables):
                # root style binning
                binning     = model.plot_options[feature]['binning']
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
                    th1d_yield.GetXaxis().SetTitle(model.plot_options[feature]['tex'])
                    th1d_yield.SetTitle("")

                    th1d_yield.Draw("hist")

                    for i_der, der in enumerate(bit.derivatives):
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
                if not os.path.isdir(plot_directory_):
                    try:
                        os.makedirs( plot_directory_ )
                    except IOError:
                        pass
                helpers.copyIndexPHP( plot_directory_ )
                c1.Print( os.path.join( plot_directory_, "epoch%s_%05i.png"%(postfix, max_n_tree) ) )
                syncer.makeRemoteGif(plot_directory_, pattern="epoch%s_*.png"%postfix, name="epoch" )
            syncer.sync()
