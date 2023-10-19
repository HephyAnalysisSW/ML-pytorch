#!/usr/bin/env python

# Standard imports
import ROOT
import numpy as np
import math
import array
import sys, os, copy
import functools, operator

sys.path.insert(0, '../../..')
import tools.syncer as syncer
import tools.user as user
import tools.helpers as helpers


#ROOT.gStyle.SetPalette(ROOT.kBird)
dir_path = os.path.dirname(os.path.realpath(__file__))
ROOT.gROOT.LoadMacro(os.path.join( dir_path, "../../../tools/scripts/tdrstyle.C"))
ROOT.setTDRStyle()

# Parser
import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument("--plot_directory",     action="store",      default="TTCA",     help="plot sub-directory")
argParser.add_argument("--model",              action="store",      help="Which model?")
argParser.add_argument("--prefix",             action="store",      default="v2", type=str,  help="prefix")
argParser.add_argument("--small",              action="store_true"  )

args = argParser.parse_args()

# directory for plots
plot_directory = os.path.join( user.plot_directory, args.plot_directory, args.prefix, args.model )
os.makedirs( plot_directory, exist_ok=True)

exec('import models.%s as model'%(args.model))
#features, _, coeffs = model.getEvents(model.data_generator[-1])

def getEvents( data ):
    coeffs       = model.data_generator.vector_branch(     data, 'p_C', padding_target=len(model.weightInfo.combinations))
    features     = model.data_generator.scalar_branches(   data, model.feature_names )
    vectors      = None #{key:model.data_generator.vector_branch(data, key ) for key in vector_branches}

    return features, vectors, coeffs

if args.small:
   model.data_generator.input_files = model.data_generator.input_files[:10]
   args.plot_directory += '_small'

features, _, coeffs = getEvents(model.data_generator[-1])

def getWeights( eft, coeffs, lin=False):

    if lin:
        combs = list(filter( lambda c:len(c)<2, model.weightInfo.combinations))
    else:
        combs = model.weightInfo.combinations
    fac = np.array( [ functools.reduce( operator.mul, [ float(eft[v]) for v in comb ], 1 ) for comb in combs], dtype='float')
    #print (fac)
    return np.matmul(coeffs[:,:len(combs)], fac)

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

weights     = getWeights( model.make_eft(), coeffs)

feature_x = "parton_top12_dAbsEta"
i_feature_x = model.feature_names.index( feature_x )
binning_x = model.plot_options[feature_x]['binning']

for i_feature, feature in enumerate(model.feature_names):
    if feature == feature_x: continue
    binning = model.plot_options[feature]['binning']

    histo = helpers.make_TH2F( np.histogram2d(
            features[:,i_feature_x], 
            features[:,i_feature], 
            [np.linspace(binning_x[1], binning_x[2], binning_x[0]+1), np.linspace(binning[1], binning[2], binning[0]+1)], 
            weights=weights) )

    histo.GetXaxis().SetTitle(model.plot_options[feature_x]['tex'])
    histo.GetYaxis().SetTitle(model.plot_options[feature]['tex'])

    for logZ in [True, False]:

        c1 = ROOT.TCanvas("c1");
        histo.Draw('colz')
        c1.SetLogz(logZ)
        c1.SetLogy(False)
        c1.SetLogx(False)
        histo.Draw('colz')

        plot_directory_ = os.path.join( plot_directory, "feature_plots_2D", "log" if logZ else "lin" )
        helpers.copyIndexPHP( plot_directory_ )
        c1.Print( os.path.join( plot_directory_, feature_x+"_vs_"+feature+'.png' ))

print ("Done with 2D plots")
syncer.sync()

