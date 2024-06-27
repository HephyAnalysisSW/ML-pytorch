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

from BPT.BoostedParametricTree import BoostedParametricTree

# User
import tools.user as user

# Parser
import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument("--plot_directory",     action="store",      default="BPT",                 help="plot sub-directory")
argParser.add_argument("--version",            action="store",      default="",                 help="Which version?")
argParser.add_argument("--model",              action="store",      default="analytic",                 help="Which model?")
argParser.add_argument("--modelDir",          action="store",      default="models",                 help="Which model directory?")
argParser.add_argument("--variations",          action="store",      nargs="*", default=[], type=str,  help="variation")
argParser.add_argument("--era",                action="store",      default="RunII", choices = ["RunII", "Summer16_preVFP", "Summer16", "Fall17", "Autumn18"], type=str,  help="variation")
argParser.add_argument("--nTraining",          action="store",      default=50000,       type=int,  help="number of training events")
argParser.add_argument("--max_n_tree",         action="store",      default=None,       type=int,  help="boosting iteration")
argParser.add_argument('--feature',         action='store',      default="tr_ttbar_pt", help="Which feature?")
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
exec('import %s.%s as model'%( args.modelDir, args.model)) 

print("Set model era to:", args.era)
model.set_era( args.era )

cfg = model.bpt_cfg
cfg.update( extra_args )

feature_names = model.feature_names

# directory for plots
plot_directory = os.path.join( user.plot_directory, args.plot_directory,
    args.version + ( '_max_n_tree_%i'%args.max_n_tree if args.max_n_tree is not None else ""), 
    args.era, 
    args.model + ("_"+args.variations[0] if (len(args.variations)==1 and args.variations[0] is not None) else "")
    )

if not os.path.isdir(plot_directory):
    try:
        os.makedirs( plot_directory )
    except IOError:
        pass

from propaganda_plot_options import plot_options

colors = [ 210, 62, 218, 226, 51, ROOT.kRed + 2, ROOT.kRed -4,ROOT.kCyan +2, ROOT.kCyan -4,  ROOT.kBlue+2,     ROOT.kBlue-4,     ROOT.kGreen+2,    ROOT.kGreen-4, ROOT.kOrange+6, ROOT.kOrange+3] 

#import colors
##[ ROOT.cBlue, ROOT.cYellow, ROOT.cGreen, ROOT.cRed, ROOT.cBrown, ROOT.cViolet] 
#colors = [
#    ROOT.cBlue,
#    ROOT.cOrange,
#    ROOT.cGreen,
#    ROOT.cRed,
#    ROOT.cPurple,
#    ROOT.cBrown,
#    ROOT.cPink,
#    ROOT.cGray,
#    ROOT.cOlive,
#    ROOT.cCyan,
#    ROOT.cPeach,
#    ROOT.cLavender,
#    ROOT.cLightBlue,
#    ROOT.cLightGray,
#    ROOT.cLightGreen,
#]

if len(args.variations)==0:
    args.variations = [None]

extra_hypothesis = [
#    {'nu1':0.45, 'nu2':0.45},
#    {'nu1':0.55, 'nu2':0.55}
    ]

h_truth = {}
h_pred  = {}
h_truth_shape = {}
h_pred_shape  = {}
i_color = 0
for variation in args.variations:

    h_truth[variation] = {}
    h_pred[variation]  = {}
    h_truth_shape[variation] = {}
    h_pred_shape[variation]  = {}

    training_data_filename = os.path.join(user.data_directory, args.version, args.era, args.model+("_"+variation if variation is not None else ""), "training_%i"%args.nTraining)+'.pkl'
    with open( training_data_filename, 'rb') as _file:
        training_data = pickle.load( _file )
        total_size    =  sum([len(s['features']) for s in training_data.values() if 'features' in s ])
        print ("Loaded training data from ", training_data_filename, "with size", total_size)

    postfix = ("_"+args.version if args.version != "" else "") + ("_"+args.era if args.era != "RunII" else "")
    if variation == None:
        bpt_name = "BPT_%s_nTraining_%i_nTrees_%i"%(args.model+postfix, args.nTraining, cfg["n_trees"])
    else:
        bpt_name = "BPT_%s_%s_nTraining_%i_nTrees_%i"%(args.model+postfix, variation, args.nTraining, cfg["n_trees"])

    filename = os.path.join(user.model_directory, 'BPT', bpt_name)+'.pkl'

    print ("Loading %s from %s"%(bpt_name, filename))
    bpt = BoostedParametricTree.load(filename)

    nominal_base_point_index = np.where(np.all(np.array(bpt.base_points)==np.array(bpt.nominal_base_point),axis=1))[0][0] 

    bpt_predictions = bpt.vectorized_predict(training_data[model.nominal_base_point]['features'], max_n_tree=args.max_n_tree)
    predicted_reweights = np.exp( np.dot( bpt_predictions, bpt.VkA.transpose() ) )

    i_feature = model.feature_names.index(args.feature) 

    if len(plot_options[args.model][args.feature]['binning'])==3:
        binning = plot_options[args.model][args.feature]['binning']
        _binning = np.linspace(binning[1], binning[2], binning[0]+1)
    else:
        _binning = plot_options[args.model][args.feature]['binning']

    if "weights" in training_data[model.nominal_base_point]:
        nominal_weights = training_data[model.nominal_base_point]['weights']
    else:
        nominal_weights = np.ones( len( training_data[model.nominal_base_point]['features']))

    for hypothesis in extra_hypothesis:
        point = tuple( [hypothesis[var] for var in bpt.parameters])
        name     = '_'.join( [ (model.parameters[i_param]+'_%3.2f'%param).replace('.','p').replace('-','m') for i_param, param in enumerate(point)])
        tex_postfix = "_{%s}"%model.tex[variation] if variation is not None else ""
        tex_name = '_'.join( [ (model.tex[model.parameters[i_param]]+tex_postfix+(' = %2.1f'%param).rstrip('0').rstrip('.')) for i_param, param in enumerate(point)])
        predictions = np.exp(np.sum( [ np.prod([hypothesis[var] for var in comb])*bpt_predictions[:,i_comb] for i_comb, comb in enumerate(bpt.combinations)], axis=0))
        h_pred[variation][tuple(point)] = {'name':name, 'tex':tex_name}
        h_pred_shape[variation][tuple(point)] = {'name':name, 'tex':tex_name}

        h_pred[variation][tuple(point)]["histo"] = helpers.make_TH1F( np.histogram(
            training_data[model.nominal_base_point]['features'][:,i_feature],
            bins=_binning, weights=nominal_weights*predictions
            ))

    for i_point, point in enumerate(bpt.base_points):

        name     = '_'.join( [ (model.parameters[i_param]+'_%3.2f'%param).replace('.','p').replace('-','m') for i_param, param in enumerate(point)])
        tex_postfix = "_{%s}"%model.tex[variation] if variation is not None else ""
        tex_name = '_'.join( [ (model.tex[model.parameters[i_param]]+tex_postfix+(' = %3.2f'%param).rstrip('0').rstrip('.')) for i_param, param in enumerate(point)])

        h_truth[variation][tuple(point)] = {'name':name, 'tex':tex_name}
        h_pred[variation][tuple(point)]  = {'name':name, 'tex':tex_name}
        h_truth_shape[variation][tuple(point)] = {'name':name, 'tex':tex_name}
        h_pred_shape[variation][tuple(point)]  = {'name':name, 'tex':tex_name}

        is_nominal = tuple(point) == tuple(model.nominal_base_point)
        if is_nominal:
            nominal_index = i_point
            nominal_name  = name
        h_truth[variation][tuple(point)]["is_nominal"] = is_nominal
        h_pred[variation][tuple(point)]["is_nominal"] = is_nominal

        if args.model not in plot_options:
            print("Model %s not found in propaganda plot_options! Do nothing", args.model)
            raise RuntimeError

        if args.feature not in plot_options[args.model]:
            print("Feature %s not found in propaganda plot_options! Do nothing", args.feature)
            raise RuntimeError

        #h_truth[variation][tuple(point)]["histo"] = ROOT.TH1F(name+'_'+args.feature+'_nom',    name+'_'+args.feature, *plot_options[args.model][args.feature]['binning'] )
        #h_pred[variation][tuple(point)]["histo"]  = ROOT.TH1F(name+'_'+args.feature+'_nom',    name+'_'+args.feature, *plot_options[args.model][args.feature]['binning'] )

        if "weights" in training_data[tuple(point)]:
            weights = training_data[tuple(point)]['weights']
        else:
            weights = np.ones( len( training_data[tuple(point)]['features'])) 

        h_truth[variation][tuple(point)]["histo"] = helpers.make_TH1F( np.histogram(
            training_data[tuple(point)]['features'][:,i_feature] if 'features' in training_data[tuple(point)] else training_data[model.nominal_base_point]['features'][:,i_feature],
            bins=_binning, weights=weights
            ))
        h_pred[variation][tuple(point)]["histo"] = helpers.make_TH1F( np.histogram(
            training_data[model.nominal_base_point]['features'][:,i_feature],
            bins=_binning, weights=nominal_weights*predicted_reweights[:,i_point]
            ))
        # make continous shapes
        if len(plot_options[args.model][args.feature]['binning'])!=3:
            for i_bin in range(1,  1+h_truth[variation][tuple(point)]["histo"].GetNbinsX() ):
                bin_scale = h_truth[variation][tuple(point)]["histo"].GetBinWidth(i_bin)/h_truth[variation][tuple(point)]["histo"].GetBinWidth(1)
                h_truth[variation][tuple(point)]["histo"].SetBinContent( i_bin, h_truth[variation][tuple(point)]["histo"].GetBinContent(i_bin)/bin_scale )
                h_pred[variation][tuple(point)]["histo"].SetBinContent( i_bin, h_pred[variation][tuple(point)]["histo"].GetBinContent(i_bin)/bin_scale )

        #color = colors[i_point] if not is_nominal else ROOT.kBlack
        if is_nominal:
            color = ROOT.kBlack
        else:
            color = colors[i_color]
            i_color+=1
        h_truth[variation][tuple(point)]["histo"].SetLineWidth(2)
        h_truth[variation][tuple(point)]["histo"].SetLineColor( color )
        h_truth[variation][tuple(point)]["histo"].SetLineStyle(ROOT.kDashed)
        h_truth[variation][tuple(point)]["histo"].SetMarkerColor(color)
        h_truth[variation][tuple(point)]["histo"].SetMarkerStyle(0)
        h_truth[variation][tuple(point)]["histo"].GetXaxis().SetTitle(plot_options[args.model][args.feature]['tex'])
        h_truth[variation][tuple(point)]["histo"].GetYaxis().SetTitle("Number of events")
        h_pred[variation][tuple(point)]["histo"].SetLineWidth(2)
        h_pred[variation][tuple(point)]["histo"].SetLineColor( color )
        h_pred[variation][tuple(point)]["histo"].SetMarkerColor(color)
        h_pred[variation][tuple(point)]["histo"].SetMarkerStyle(0)
        h_pred[variation][tuple(point)]["histo"].GetXaxis().SetTitle(plot_options[args.model][args.feature]['tex'])
        h_pred[variation][tuple(point)]["histo"].GetYaxis().SetTitle("Number of Events")

    for k in h_pred_shape[variation].keys():
        h_pred_shape[variation][k]["histo"] = h_pred[variation][k]["histo"].Clone()
        h_pred_shape[variation][k]["histo"].Divide(h_truth[variation][model.nominal_base_point]["histo"])

    for i_point, point in enumerate(bpt.base_points):
        h_truth_shape[variation][tuple(point)]["histo"] = h_truth[variation][tuple(point)]["histo"].Clone()
        h_truth_shape[variation][tuple(point)]["histo"].Divide(h_truth[variation][model.nominal_base_point]["histo"])

c1 = ROOT.TCanvas("c1", "", 500, 800)

ROOT.gStyle.SetOptStat(0)

default_widths = {'y_width':500, 'x_width':500, 'y_ratio_width':500}
default_widths['y_width'] += default_widths['y_ratio_width']
scaleFacRatioPad = default_widths['y_width']/float( default_widths['y_ratio_width'] )
y_border = default_widths['y_ratio_width']/float( default_widths['y_width'] )

c1.Divide(1,2,0,0)
topPad = c1.cd(1)
topPad.SetBottomMargin(0)
topPad.SetLeftMargin(0.15)
topPad.SetTopMargin(0.07)
topPad.SetRightMargin(0.05)
topPad.SetPad(topPad.GetX1(), y_border, topPad.GetX2(), topPad.GetY2())
bottomPad = c1.cd(2)
bottomPad.SetTopMargin(0)
bottomPad.SetRightMargin(0.05)
bottomPad.SetLeftMargin(0.15)
bottomPad.SetBottomMargin(scaleFacRatioPad*0.13)
bottomPad.SetPad(bottomPad.GetX1(), bottomPad.GetY1(), bottomPad.GetX2(), y_border)

topPad.cd()
l = ROOT.TLegend(*plot_options[args.model][args.feature]["legendCoordinates"])
l.SetNColumns(2)
l.SetFillStyle(0)
l.SetShadowColor(ROOT.kWhite)
l.SetBorderSize(0)

for i_point, point in enumerate(model.base_points):
    l.AddEntry(  h_pred[variation][tuple(point)]["histo"],   h_pred[variation][tuple(point)]["tex"])

for variation in args.variations:
    h_nominal_scale = [ h_pred[variation][k]["histo"] for k in h_pred[variation].keys() if h_pred[variation][k]["is_nominal"]][0].Integral()
    for k in h_pred[variation].keys():
        h_pred[variation][k]["histo"].Scale( 1./h_nominal_scale )
    h_nominal_scale = [ h_truth[variation][k]["histo"] for k in h_truth[variation].keys() if h_truth[variation][k]["is_nominal"]][0].Integral()
    for k in h_truth[variation].keys():
        h_truth[variation][k]["histo"].Scale( 1./h_nominal_scale )

first = True
for variation in args.variations:
    for h in [h_pred[variation][k]["histo"] for k in h_pred[variation].keys()] +  [h_truth[variation][tuple(point)]["histo"] for point in model.base_points]:
        if first:
            h.Draw("e1h")
            if plot_options[args.model][args.feature]["logY"]:
                ROOT.gPad.SetLogy(True)
            else:
                h.GetYaxis().SetRangeUser( 0, 1.3*h.GetMaximum() )
            h.GetYaxis().SetLabelSize(0.05);
            #h.GetXaxis().SetTitleOffset( 3.2 )
            #h.GetYaxis().SetTitleOffset( 1.6 )
        else:
            h .Draw("e1hsame")
        h .Draw("e1hsame")
        first=False

for variation in args.variations:
    for i_point, point in enumerate(model.base_points):
        if h_pred[variation][tuple(point)]["is_nominal"]:
             h_pred[variation][tuple(point)]["histo"].Draw("e1hsame")
             h_truth[variation][tuple(point)]["histo"].Draw("e1hsame")
        
l.Draw()

bottomPad.cd()
first = True
for variation in args.variations:
    for h in [h_pred_shape[variation][k]["histo"] for k in h_pred_shape[variation].keys()] +  [h_truth_shape[variation][tuple(point)]["histo"] for point in model.base_points]:
        if first:
            h.Draw("e1h")
            h.GetYaxis().SetRangeUser( *plot_options[args.model][args.feature]["shape_y_range"] )

            h.GetYaxis().SetTitle("Ratio");
            h.GetYaxis().SetTitleSize(0.08)
            h.GetYaxis().SetTitleOffset( 0.95 )

            h.GetYaxis().SetLabelSize(0.06)
            h.GetYaxis().CenterTitle(1)

            h.GetXaxis().SetLabelSize(0.07)
            h.GetXaxis().SetTitleSize( 0.08 )
            h.GetXaxis().SetTitleOffset( 0.95 )
        else:
            h .Draw("e1hsame")
        h .Draw("e1hsame")
        first=False

for variation in args.variations:
    for i_point, point in enumerate(model.base_points):
        if h_pred[variation][tuple(point)]["is_nominal"]:
            h_pred_shape[variation][tuple(point)]["histo"].Draw("e1hsame")

bottomPad.RedrawAxis()

plot_directory_ = os.path.join( plot_directory, "propaganda_plots")
if not os.path.isdir(plot_directory_):
    try:
        os.makedirs( plot_directory_ )
    except IOError:
        pass

helpers.copyIndexPHP( plot_directory_ )

prefix = ""
if not (args.variations[0] is None):
    prefix = "-".join( args.variations )+'_' 

c1.Print( os.path.join( plot_directory_, prefix+"%s.png"%args.feature ) )
c1.Print( os.path.join( plot_directory_, prefix+"%s.pdf"%args.feature ) )
c1.Print( os.path.join( plot_directory_, prefix+"%s.svg"%args.feature ) )

syncer.sync()
