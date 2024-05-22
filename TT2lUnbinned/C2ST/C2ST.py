#!/usr/bin/env python
import pickle
import ROOT, os
import numpy as np
c1 = ROOT.TCanvas() # do this to avoid version conflict in png.h with keras import ...
c1.Draw()
c1.Print('/tmp/delete.png')

dir_path = os.path.dirname(os.path.realpath(__file__))
ROOT.gROOT.LoadMacro(os.path.join( dir_path, "../../tools/scripts/tdrstyle.C"))
ROOT.setTDRStyle()

import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
#argParser.add_argument('--config',             action='store', type=str,   default='tttt_3l', help="Name of the config file")
#argParser.add_argument('--small',              action='store_true', help="small?")
argParser.add_argument('--removePred',         action='store_true', help="remove prediction from 2nd sample")
argParser.add_argument('--max_n_tree',         action='store',  type=int, default=-1)
argParser.add_argument('--nJobs',              action='store', type=int, default=0, help="Shuffle weights")
argParser.add_argument('--job',                action='store', type=int, default=None, help="Shuffle weights (random seed)")
argParser.add_argument('--activation',         action='store', default='sigmoid', help="activation function?")
argParser.add_argument("--version",            action="store",      default="v4_for_paper",                 help="Which version?")
argParser.add_argument("--model",              action="store",      default="analytic",                 help="Which model?")
argParser.add_argument("--modelDir",          action="store",      default="models",                 help="Which model directory?")
argParser.add_argument("--variation",          action="store",      default=None, type=str,  help="variation")
argParser.add_argument("--era",                action="store",      default="Autumn18", choices = ["RunII", "Summer16_preVFP", "Summer16", "Fall17", "Autumn18"], type=str,  help="variation")
argParser.add_argument('--overwrite',          action='store',      default=None, choices = [None, "training", "data", "all"],  help="Overwrite output?")
#argParser.add_argument('--feature_plots',      action='store_true', help="feature_plots?")
argParser.add_argument("--nTraining",          action="store",      default=50000,       type=int,  help="number of training events")
#argParser.add_argument("--plot_directory",     action="store",      default="C2ST",                 help="plot sub-directory")

args = argParser.parse_args()

#if args.small:      args.name+="_small"

import sys

sys.path.insert(0, '..')
sys.path.insert(0, '../..')

from   tools import helpers
#import tools.syncer as syncer
import tools.user as user
import tools.logger as logger
logger = logger.get_logger("INFO", logFile = None )

# import the model
exec('import BPT.%s.%s as model'%( args.modelDir, args.model))

print("Set model era to:", args.era)
model.set_era( args.era )

cfg = model.bpt_cfg

feature_names = model.feature_names

## directory for plots
#plot_directory = os.path.join( user.plot_directory, args.plot_directory,
#    args.version, args.era,
#    args.model
#        + ("_"+args.variation if args.variation is not None else "")
#    )
#
#if not os.path.isdir(plot_directory):
#    try:
#        os.makedirs( plot_directory )
#    except IOError:
#        pass

training_data_filename = os.path.join(user.data_directory, args.version, args.era, args.model+("_"+args.variation if args.variation is not None else ""), "training_%i"%args.nTraining)+'.pkl'
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

from BPT.BoostedParametricTree import BoostedParametricTree

postfix = ("_"+args.version if args.version != "" else "") + ("_"+args.era if args.era != "RunII" else "")
if args.variation == None:
    bpt_name = "BPT_%s_nTraining_%i_nTrees_%i"%(args.model+postfix, args.nTraining, cfg["n_trees"])
else:
    bpt_name = "BPT_%s_%s_nTraining_%i_nTrees_%i"%(args.model+postfix, args.variation, args.nTraining, cfg["n_trees"])

filename = os.path.join(user.model_directory, 'BPT', bpt_name)+'.pkl'
try:
    print ("Loading %s from %s"%(bpt_name, filename))
    bpt = BoostedParametricTree.load(filename)
except (IOError, EOFError, ValueError):
    bpt = None

alt_point = [1.0]
alt_point_index = model.base_points.index(list(alt_point))

if args.max_n_tree>0:
    predicted_reweights = np.exp( np.dot( bpt.vectorized_predict(training_data[model.nominal_base_point]['features'],  max_n_tree = args.max_n_tree), bpt.VkA.transpose() ) )
else:
    predicted_reweights = np.exp( np.dot( bpt.vectorized_predict(training_data[model.nominal_base_point]['features']), bpt.VkA.transpose() ) )

learned_weights = predicted_reweights[:,alt_point_index]

from c2st_algo_nn import c2st_test

sample = training_data[(0.0,)]['features']#[:,f_mask]

# Example usage with weighted samples
weights1 = training_data[tuple(alt_point)]['weights'] 
weights2 = training_data[model.nominal_base_point]['weights'] 
if args.removePred:
    weights1 /= learned_weights

if args.nJobs>0:
    print("Shuffling with seed %i"%args.job)
    np.random.seed(args.job)
    weights_combined = np.concatenate([weights1, weights2])
    perm_indices = np.random.permutation(len(weights_combined))
    weights_combined = weights_combined[perm_indices]
    weights1     = weights_combined[:len(weights1)]
    weights2     = weights_combined[len(weights1):]

model, accuracy = c2st_test(sample, weights1, weights2, test_size=0.3, random_state=args.job)
print(f"Classifier Accuracy: {accuracy}")

# A significance test can be performed to check if the accuracy is significantly higher than 0.5
from scipy.stats import binom_test
n_test_samples = int(0.3 * len(sample) * 2)  # Number of test samples
p_value = binom_test(int(accuracy * n_test_samples), n_test_samples, p=0.5, alternative='greater')
print(f"p-value: {p_value}")

result = {
    #'model':model,
    'p_value':p_value,
    'n_test_samples':n_test_samples,
    'accuracy':accuracy,
}

filename = os.path.join( user.results_directory, 'C2ST', args.model+'_'+args.version, ("removePred_" if args.removePred else "")+("max_n_tree_%03i"%args.max_n_tree if args.max_n_tree>-1 else "")+("%04i"%args.job if args.nJobs>0 else "notShuffled")+".pkl")

os.makedirs( os.path.dirname( filename), exist_ok=True)

with open(filename, 'wb') as f:
    pickle.dump(result, f)

print (f"Written result ( accuracy: {accuracy}, p_value:{p_value}) to  {filename}")

#from weighted_energy_distcance import permutation_test
#
#N_max = 500
#
#features = ['nJetGood', 'tr_abs_delta_phi_ll_lab']
#f_mask   = [ model.feature_names.index(f) for f in features ]
# 
#observed_stat, p_value = permutation_test(
#    training_data[(0.0,)]['features'][:,f_mask][:N_max], 
#    training_data[(0.0,)]['features'][:,f_mask][:N_max], 
#    training_data[(1.0,)]['weights'][:N_max], 
#    training_data[(0.0,)]['weights'][:N_max],
#    )
#print(f"Observed Weighted Energy Distance: {observed_stat}, p-value: {p_value}")



##N_max = 300000
## FIXME ... only works in special case
#N_max = -1
#power = 5
#
#training_data[(1.0,)]['weights'] = (training_data[(1.0,)]['weights']/training_data[(0.0,)]['weights'])**power*training_data[(0.0,)]['weights']
#
################
### Plot Model #
################
#nominal_base_point_index = np.where(np.all(np.array(model.base_points)==np.array(model.nominal_base_point),axis=1))[0][0]
#
#colors = [ ROOT.kRed + 2, ROOT.kRed -4,ROOT.kCyan +2, ROOT.kCyan -4, ROOT.kMagenta+2,  ROOT.kMagenta-4,  ROOT.kBlue+2,     ROOT.kBlue-4,     ROOT.kGreen+2,    ROOT.kGreen-4, ROOT.kOrange+6, ROOT.kOrange+3]
#stuff = []
#if args.feature_plots:
#    h    = {}
#    for i_point, point in enumerate(model.base_points):
#
#        name     = '_'.join( [ (model.parameters[i_param]+'_%3.2f'%param).replace('.','p').replace('-','m') for i_param, param in enumerate(point)])
#        tex_name = '_'.join( [ (model.parameters[i_param]+' = %3.2f'%param) for i_param, param in enumerate(point)])
#        is_nominal = tuple(point) == tuple(model.nominal_base_point)
#        if is_nominal:
#            nominal_index = i_point
#            nominal_name  = name
#
#        if is_nominal:
#            tex_name += " (nominal)"
#
#        h[tuple(point)] = {'name':name}
#
#        for i_feature, feature in enumerate(feature_names):
#            h[tuple(point)][feature]        = ROOT.TH1F(name+'_'+feature+'_nom',    name+'_'+feature, *model.plot_options[feature]['binning'] )
#
#            binning = model.plot_options[feature]['binning']
#
#            if "weights" in training_data[tuple(point)]:
#                weights = training_data[tuple(point)]['weights']
#            #elif "weights" in training_data[model.nominal_base_point]:
#            #    weights = training_data[model.nominal_base_point]['weights']            
#            else:
#                weights = None
#
#            h[tuple(point)][feature] = helpers.make_TH1F( np.histogram(
#                training_data[tuple(point)]['features'][:,i_feature] if 'features' in training_data[tuple(point)] else training_data[model.nominal_base_point]['features'][:,i_feature],
#                bins=np.linspace(binning[1], binning[2], binning[0]+1), weights=weights
#                ))
#
#            color = colors[i_point] if not is_nominal else ROOT.kBlack
#            h[tuple(point)][feature].SetLineWidth(2)
#            h[tuple(point)][feature].SetLineColor( color )
#            h[tuple(point)][feature].SetMarkerStyle(0)
#            h[tuple(point)][feature].SetMarkerColor(color)
#            h[tuple(point)][feature].legendText = tex_name
#
#    for i_feature, feature in enumerate(feature_names):
#
#        norm = h[tuple(model.nominal_base_point)][feature].Integral()
#        if norm>0:
#            for point in model.base_points:
#                h[tuple(point)][feature].Scale(1./norm)
#
#        histos = [h[tuple(point)][feature] for point in model.base_points]
#        max_   = max( map( lambda h__:h__.GetMaximum(), histos ))
#
#        for logY in [True, False]:
#
#            c1 = ROOT.TCanvas("c1");
#            l = ROOT.TLegend(0.2,0.68,0.9,0.91)
#            l.SetNColumns(2)
#            l.SetFillStyle(0)
#            l.SetShadowColor(ROOT.kWhite)
#            l.SetBorderSize(0)
#            for i_histo, histo in enumerate(reversed(histos)):
#                histo.GetXaxis().SetTitle(model.plot_options[feature]['tex'])
#                histo.GetYaxis().SetTitle("1/#sigma_{SM}d#sigma/d%s"%model.plot_options[feature]['tex'])
#                if i_histo == 0:
#                    histo.Draw('hist')
#                    histo.GetYaxis().SetRangeUser( (0.01 if logY else 0), (100*max_ if logY else 2*max_))
#                    histo.Draw('hist')
#                else:
#                    histo.Draw('histsame')
#                l.AddEntry(histo, histo.legendText)
#                c1.SetLogy(logY)
#            l.Draw()
#
#            plot_directory_ = os.path.join( plot_directory, "feature_plots", "nTraining_%i"%args.nTraining, "log" if logY else "lin" )
#            helpers.copyIndexPHP( plot_directory_ )
#            c1.Print( os.path.join( plot_directory_, feature+'.png' ))
#
#        # Norm all shapes to 1
#        for i_histo, histo in enumerate(histos):
#            norm = histo.Integral()
#            if norm>0:
#                histo.Scale(1./histo.Integral())
#
#        # Divide all shapes by the SM
#        ref = histos[nominal_index].Clone()
#        for i_histo, histo in enumerate(histos):
#            histo.Divide(ref)
#
#        # Now plot shape differences
#        for logY in [True, False]:
#            c1 = ROOT.TCanvas("c1");
#            l = ROOT.TLegend(0.2,0.68,0.9,0.91)
#            l.SetNColumns(2)
#            l.SetFillStyle(0)
#            l.SetShadowColor(ROOT.kWhite)
#            l.SetBorderSize(0)
#
#            lower, higher = 0.8, 1.2
#            try:
#                lower, higher = model.shape_user_range["log" if logY else "lin"]
#            except:
#                pass
#
#            c1.SetLogy(logY)
#            for i_histo, histo in enumerate(reversed(histos)):
#                histo.GetXaxis().SetTitle(model.plot_options[feature]['tex'])
#                histo.GetYaxis().SetTitle("shape wrt. SM")
#                if i_histo == 0:
#                    histo.Draw('hist')
#                    histo.GetYaxis().SetRangeUser(lower, higher)
#                    histo.Draw('hist')
#                else:
#                    histo.Draw('histsame')
#                l.AddEntry(histo, histo.legendText)
#                c1.SetLogy(logY)
#            l.Draw()
#
#            plot_directory_ = os.path.join( plot_directory, "shape_plots", "nTraining_%i"%args.nTraining, "log" if logY else "lin" )
#            helpers.copyIndexPHP( plot_directory_ )
#            c1.Print( os.path.join( plot_directory_, feature+'.png' ))
#
#print ("Done with plots")
