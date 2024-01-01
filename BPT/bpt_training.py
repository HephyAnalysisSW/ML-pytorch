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

from BoostedParametricTree import BoostedParametricTree

# User
import tools.user as user

# Parser
import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument("--plot_directory",     action="store",      default="BPT",                 help="plot sub-directory")
argParser.add_argument("--model",              action="store",      default="analytic",                 help="Which model?")
#argParser.add_argument("--modelFile",          action="store",      default="toy_models",                 help="Which model directory?")
argParser.add_argument("--variation",          action="store",      default=None, type=str,  help="variation")
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
plot_directory = os.path.join( user.plot_directory, args.plot_directory, args.model+("_"+args.variation if args.variation is not None else "") )

if not os.path.isdir(plot_directory):
    try:
        os.makedirs( plot_directory )
    except IOError:
        pass

training_data_filename = os.path.join(user.data_directory, args.model+("_"+args.variation if args.variation is not None else ""), "training_%i"%args.nTraining)+'.pkl'
if args.overwrite in ["all", "data"] or not os.path.exists(training_data_filename):
    training_data = model.getEvents(args.nTraining, systematic=args.variation)
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

    line1 = ( 0.15+offset, 0.95, "Boosted Param Trees" )
    return [ tex1.DrawLatex(*line1) ]#, tex2.DrawLatex(*line2) ]

###############
## Plot Model #
###############
nominal_base_point_index = np.where(np.all(np.array(model.base_points)==np.array(model.nominal_base_point),axis=1))[0][0] 

colors = [ ROOT.kRed + 2, ROOT.kRed -4,ROOT.kCyan +2, ROOT.kCyan -4, ROOT.kMagenta+2,  ROOT.kMagenta-4,  ROOT.kBlue+2,     ROOT.kBlue-4,     ROOT.kGreen+2,    ROOT.kGreen-4, ] 
stuff = []
if args.feature_plots:
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
            #elif "weights" in training_data[model.nominal_base_point]:
            #    weights = training_data[model.nominal_base_point]['weights']            
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
        ref = histos[nominal_index].Clone()
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

if args.variation == None:
    bpt_name = "BPT_%s_nTraining_%i_nTrees_%i"%(args.model, args.nTraining, cfg["n_trees"])
else:
    bpt_name = "BPT_%s_%s_nTraining_%i_nTrees_%i"%(args.model, args.variation, args.nTraining, cfg["n_trees"])

filename = os.path.join(user.model_directory, 'BPT', bpt_name)+'.pkl'
try:
    print ("Loading %s from %s"%(bpt_name, filename))
    bpt = BoostedParametricTree.load(filename)
except (IOError, EOFError, ValueError):
    bpt = None

if bpt is None or args.overwrite in ["all", "training"]:
    time1 = time.time()
    bpt = BoostedParametricTree(
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

if args.debug:

    # GIF animation
    tex = ROOT.TLatex()
    tex.SetNDC()
    tex.SetTextSize(0.06)

    # Which iterations to plot
    plot_iterations = list(range(1,10))+list(range(10,bpt.n_trees+1,10))
    # Add plot iterations from command line, if provided
    if type(args.plot_iterations)==type([]):
        if args.plot_iterations[0]<0:
            plot_iterations+=args.plot_iterations[1:]
        else:
            plot_iterations = args.plot_iterations
        plot_iterations.sort()

    for max_n_tree in plot_iterations:
        if max_n_tree==0: max_n_tree=1

        predicted_reweights = np.exp( np.dot( bpt.vectorized_predict(training_data[model.nominal_base_point]['features'],  max_n_tree = max_n_tree), bpt.VkA.transpose() ) )

        h_truth = {}
        h_pred  = {}
        h_truth_shape = {}
        h_pred_shape  = {}
        for i_point, point in enumerate(model.base_points):

            name     = '_'.join( [ (model.parameters[i_param]+'_%3.2f'%param).replace('.','p').replace('-','m') for i_param, param in enumerate(point)])
            tex_name = '_'.join( [ (model.parameters[i_param]+' = %3.2f'%param) for i_param, param in enumerate(point)])
            is_nominal = tuple(point) == tuple(model.nominal_base_point)
            if is_nominal:
                nominal_index = i_point
                nominal_name  = name

            h_truth[tuple(point)] = {'name':name, 'tex':tex_name+" (truth)"}
            h_pred[tuple(point)]  = {'name':name, 'tex':tex_name+" (pred)"}
            h_truth_shape[tuple(point)] = {'name':name, 'tex':tex_name+" (truth)"}
            h_pred_shape[tuple(point)]  = {'name':name, 'tex':tex_name+" (pred)"}

            for i_feature, feature in enumerate(feature_names):
                h_truth[tuple(point)][feature] = ROOT.TH1F(name+'_'+feature+'_nom',    name+'_'+feature, *model.plot_options[feature]['binning'] )
                h_pred[tuple(point)][feature]  = ROOT.TH1F(name+'_'+feature+'_nom',    name+'_'+feature, *model.plot_options[feature]['binning'] )

                binning = model.plot_options[feature]['binning']
    
                if "weights" in training_data[model.nominal_base_point]:
                    nominal_weights = training_data[model.nominal_base_point]['weights']
                else:
                    nominal_weights = np.ones( len( training_data[model.nominal_base_point]['features']))

                if "weights" in training_data[tuple(point)]:
                    weights = training_data[tuple(point)]['weights']
                else:
                    weights = np.ones( len( training_data[tuple(point)]['features'])) 
                #elif "weights" in training_data[model.nominal_base_point]:
                #    weights = nominal_weights 
                #else:
                #    weights = None

                h_truth[tuple(point)][feature] = helpers.make_TH1F( np.histogram(
                    training_data[tuple(point)]['features'][:,i_feature] if 'features' in training_data[tuple(point)] else training_data[model.nominal_base_point]['features'][:,i_feature],
                    bins=np.linspace(binning[1], binning[2], binning[0]+1), weights=weights
                    ))
                h_pred[tuple(point)][feature] = helpers.make_TH1F( np.histogram(
                    training_data[model.nominal_base_point]['features'][:,i_feature],
                    bins=np.linspace(binning[1], binning[2], binning[0]+1), weights=nominal_weights*predicted_reweights[:,i_point]
                    ))

                color = colors[i_point] if not is_nominal else ROOT.kBlack
                h_truth[tuple(point)][feature].SetLineWidth(2)
                h_truth[tuple(point)][feature].SetLineColor( color )
                h_truth[tuple(point)][feature].SetLineStyle(ROOT.kDashed)
                h_truth[tuple(point)][feature].SetMarkerColor(color)
                h_truth[tuple(point)][feature].SetMarkerStyle(0)
                h_truth[tuple(point)][feature].GetXaxis().SetTitle(model.plot_options[feature]['tex'])
                h_truth[tuple(point)][feature].GetYaxis().SetTitle("1/#sigma_{SM}d#sigma/d%s"%model.plot_options[feature]['tex'])
                h_pred[tuple(point)][feature].SetLineWidth(2)
                h_pred[tuple(point)][feature].SetLineColor( color )
                h_pred[tuple(point)][feature].SetMarkerColor(color)
                h_pred[tuple(point)][feature].SetMarkerStyle(0)
                h_pred[tuple(point)][feature].GetXaxis().SetTitle(model.plot_options[feature]['tex'])
                h_pred[tuple(point)][feature].GetYaxis().SetTitle("1/#sigma_{SM}d#sigma/d%s"%model.plot_options[feature]['tex'])

                h_truth_shape[tuple(point)][feature] = h_truth[tuple(point)][feature].Clone()
                h_pred_shape[tuple(point)][feature] = h_pred[tuple(point)][feature].Clone()

        # make shape plots
        for feature in feature_names:
            for i_point, point in enumerate(model.base_points):
                h_truth_shape[tuple(point)][feature].Divide(h_truth[model.nominal_base_point][feature])
                h_pred_shape[tuple(point)][feature].Divide(h_truth[model.nominal_base_point][feature])

        n_pads = len(model.feature_names)+1
        n_col  = int(sqrt(n_pads))
        n_rows = n_pads//n_col
        if n_rows*n_col<n_pads: n_rows+=1

        for logY in [False, True]:
            c1 = ROOT.TCanvas("c1","multipads",500*n_col,500*n_rows);
            c1.Divide(n_col,n_rows)
            c2 = ROOT.TCanvas("c2","multipads",500*n_col,500*n_rows);
            c2.Divide(n_col,n_rows)

            l = ROOT.TLegend(0.2,0.1,0.9,0.85)
            stuff.append(l)
            l.SetNColumns(2)
            l.SetFillStyle(0)
            l.SetShadowColor(ROOT.kWhite)
            l.SetBorderSize(0)
            # feature plots
            for i_feature, feature in enumerate(model.feature_names):

                c1.cd(i_feature+1)
                ROOT.gStyle.SetOptStat(0)

                for i_point, point in enumerate(model.base_points):
                    h_truth[tuple(point)][feature].Draw("same") 
                    h_pred[tuple(point)][feature].Draw("same") 
                    if i_feature==0:
                        l.AddEntry(  h_truth[tuple(point)][feature],  h_truth[tuple(point)]["tex"])
                        l.AddEntry(  h_pred[tuple(point)][feature],   h_pred[tuple(point)]["tex"])

                max_ = max( map( lambda h:h.GetMaximum(), [h_truth[tuple(point)][feature] for point in model.base_points] ))
                max_ = 10**(1.5)*max_ if logY else 1.5*max_
                min_ = min( map( lambda h:h.GetMinimum(), [h_truth[tuple(point)][feature] for point in model.base_points] ))
                min_ = 0.1 if logY else (1.5*min_ if min_<0 else 0.75*min_)

                first = True
                for h in [h_pred[tuple(point)][feature] for point in model.base_points] +  [h_truth[tuple(point)][feature] for point in model.base_points]:
                    if first:
                        h.Draw("h")
                        ROOT.gPad.SetLogy(logY)
                    else:
                        h .Draw("hsame")
                    h.GetYaxis().SetRangeUser(min_, max_)
                    h .Draw("hsame")
                    first=False
                ROOT.gPad.SetLogy(logY)
            # shape plots
            for i_feature, feature in enumerate(model.feature_names):

                c2.cd(i_feature+1)
                ROOT.gStyle.SetOptStat(0)

                for i_point, point in enumerate(model.base_points):
                    h_truth_shape[tuple(point)][feature].Draw("same") 
                    h_pred_shape[tuple(point)][feature].Draw("same") 

                max_ = max( map( lambda h:h.GetMaximum(), [h_truth_shape[tuple(point)][feature] for point in model.base_points] ))
                max_ = 10**(1.5)*max_ if logY else 1+1.3*(max_-1)
                min_ = min( map( lambda h:h.GetMinimum(), [h_truth_shape[tuple(point)][feature] for point in model.base_points] ))
                min_ = 0.1 if logY else 1-1.3*(1-min_)

                first = True
                for h in [h_pred_shape[tuple(point)][feature] for point in model.base_points] +  [h_truth_shape[tuple(point)][feature] for point in model.base_points]:
                    if first:
                        h.Draw("h")
                        ROOT.gPad.SetLogy(logY)
                    else:
                        h .Draw("hsame")
                    h.GetYaxis().SetRangeUser(min_, max_)
                    h .Draw("hsame")
                    first=False

                ROOT.gPad.SetLogy(logY)

            c1.cd(len(model.feature_names)+1)
            l.Draw()

            lines = [ (0.29, 0.9, 'N_{B} =%5i'%( max_n_tree )) ]
            drawObjects = [ tex.DrawLatex(*line) for line in lines ]
            for o in drawObjects:
                o.Draw()

            c2.cd(len(model.feature_names)+1)
            l.Draw()

            lines = [ (0.29, 0.9, 'N_{B} =%5i'%( max_n_tree )) ]
            drawObjects = [ tex.DrawLatex(*line) for line in lines ]
            for o in drawObjects:
                o.Draw()

            plot_directory_ = os.path.join( plot_directory, "training_plots", bpt_name, "log" if logY else "lin" )
            if not os.path.isdir(plot_directory_):
                try:
                    os.makedirs( plot_directory_ )
                except IOError:
                    pass
            helpers.copyIndexPHP( plot_directory_ )
            c1.Print( os.path.join( plot_directory_, "epoch_%05i.png"%(max_n_tree) ) )
            syncer.makeRemoteGif(plot_directory_, pattern="epoch_*.png", name="epoch" )
            if not logY:
                c2.Print( os.path.join( plot_directory_, "epoch_shape_%05i.png"%(max_n_tree) ) )
                syncer.makeRemoteGif(plot_directory_, pattern="epoch_shape_*.png", name="epoch_shape" )

        syncer.sync()
