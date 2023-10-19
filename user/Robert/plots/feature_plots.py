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

stuff = []
h    = {}
h_lin= {}

for i_eft, eft_plot_point in enumerate(model.eft_plot_points):
    eft = eft_plot_point['eft']

    if i_eft == 0:
        eft_sm     = eft

    name = ''
    name= '_'.join( [ (wc+'_%3.2f'%eft[wc]).replace('.','p').replace('-','m') for wc in model.wilson_coefficients if wc in eft ])
    tex_name = eft_plot_point['tex'] 

    if i_eft==0: name='SM'

    h[name]     = {}
    h_lin[name] = {}

    eft['name'] = name
    
    for i_feature, feature in enumerate(model.feature_names):
        h[name][feature]        = ROOT.TH1F(name+'_'+feature+'_nom',    name+'_'+feature, *model.plot_options[feature]['binning'] )
        h_lin[name][feature]    = ROOT.TH1F(name+'_'+feature+'_nom_lin',name+'_'+feature+'_lin', *model.plot_options[feature]['binning'] )

    weights     = getWeights( eft, coeffs)
    weights_lin = getWeights( eft, coeffs, lin=True)

    ## FIXME
    #if i_eft == 0:
    #    weights_sm = weights
    #mask = (np.log(np.abs(weights_lin/weights_sm))<0) & (weights_lin/weights_sm>0)
    #weights_lin[~mask] = 0

    for i_feature, feature in enumerate(model.feature_names):
        binning = model.plot_options[feature]['binning']

        h[name][feature] = helpers.make_TH1F( np.histogram(features[:,i_feature], np.linspace(binning[1], binning[2], binning[0]+1), weights=weights) )
        h_lin[name][feature] = helpers.make_TH1F( np.histogram(features[:,i_feature], np.linspace(binning[1], binning[2], binning[0]+1), weights=weights_lin) )

        #if feature=="parton_hadV_angle_phi": #FIXME
        #    h[name][feature] = helpers.make_TH1F( np.histogram(np.cos(features[:,i_feature]), np.linspace(-1,1, 50+1), weights=weights) )
        #    h_lin[name][feature] = helpers.make_TH1F( np.histogram(np.cos(features[:,i_feature]), np.linspace(-1,1, 50+1), weights=weights_lin) ) 
        #    if not model.plot_options[feature]['tex'].startswith('cos'):
        #        model.plot_options[feature]['tex'] = "cos(%s)"%(model.plot_options[feature]['tex'])

        h[name][feature].SetLineWidth(2)
        h[name][feature].SetLineColor( eft_plot_point['color'] )
        h[name][feature].SetMarkerStyle(0)
        h[name][feature].SetMarkerColor(eft_plot_point['color'])
        h[name][feature].legendText = tex_name
        h_lin[name][feature].SetLineWidth(2)
        h_lin[name][feature].SetLineColor( eft_plot_point['color'] )
        h_lin[name][feature].SetMarkerStyle(0)
        h_lin[name][feature].SetMarkerColor(eft_plot_point['color'])
        h_lin[name][feature].legendText = tex_name+(" (lin)" if name!="SM" else "")

for i_feature, feature in enumerate(model.feature_names):

    for _h in [h, h_lin]:
        norm = _h[model.eft_plot_points[0]['eft']['name']][feature].Integral()
        if norm>0:
            for eft_plot_point in model.eft_plot_points:
                _h[eft_plot_point['eft']['name']][feature].Scale(1./norm) 

    for postfix, _h in [ ("", h), ("_linEFT", h_lin)]:
        histos = [_h[eft_plot_point['eft']['name']][feature] for eft_plot_point in model.eft_plot_points]
        max_   = max( map( lambda h__:h__.GetMaximum(), histos ))
        # FIXME
        min_   = min( map( lambda h__:h__.GetMinimum(), histos ))
        if min_<0 and abs(min_)>max_:
            max_=abs(min_)

        for logY in [True, False]:

            c1 = ROOT.TCanvas("c1");
            l = ROOT.TLegend(0.2,0.75,0.9,0.91)
            l.SetNColumns(3)
            l.SetFillStyle(0)
            l.SetShadowColor(ROOT.kWhite)
            l.SetBorderSize(0)
            for i_histo, histo in enumerate(reversed(histos)):
                histo.GetXaxis().SetTitle(model.plot_options[feature]['tex'])
                histo.GetYaxis().SetTitle("1/#sigma_{SM}d#sigma/d%s"%model.plot_options[feature]['tex'])
                if i_histo == 0:
                    histo.Draw('hist')
                    #histo.GetYaxis().SetRangeUser( (0.001 if logY else -1.3*max_), (10*max_ if logY else 2*1.3*max_))
                    histo.GetYaxis().SetRangeUser( (0.001 if logY else 0), (10*max_ if logY else 1.5*max_))

                    histo.Draw('hist')
                else:
                    histo.Draw('histsame')
                l.AddEntry(histo, histo.legendText)
                c1.SetLogy(logY)
            l.Draw()

            plot_directory_ = os.path.join( plot_directory, "feature_plots", "log" if logY else "lin" )
            helpers.copyIndexPHP( plot_directory_ )
            c1.Print( os.path.join( plot_directory_, feature+postfix+'.png' ))

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
            l.SetNColumns(3)
            l.SetFillStyle(0)
            l.SetShadowColor(ROOT.kWhite)
            l.SetBorderSize(0)

            c1.SetLogy(logY)
            for i_histo, histo in enumerate(reversed(histos)):
                histo.GetXaxis().SetTitle(model.plot_options[feature]['tex'])
                histo.GetYaxis().SetTitle("shape wrt. SM")
                if i_histo == 0:
                    histo.Draw('hist')
                    histo.GetYaxis().SetRangeUser( (0.01 if logY else 0), (10 if logY else 2))
                    histo.Draw('hist')
                else:
                    histo.Draw('histsame')
                l.AddEntry(histo, histo.legendText)
                c1.SetLogy(logY)
            l.Draw()

            plot_directory_ = os.path.join( plot_directory, "shape_plots", "log" if logY else "lin" )
            helpers.copyIndexPHP( plot_directory_ )
            c1.Print( os.path.join( plot_directory_, feature+postfix+'.png' ))

print ("Done with plots")
syncer.sync()

