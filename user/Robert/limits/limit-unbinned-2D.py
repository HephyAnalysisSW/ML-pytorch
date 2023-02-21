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
import array

sys.path.insert(0, '../..')
sys.path.insert(0, '../../..')

#from   tools import helpers
import tools.syncer as syncer

# RootTools
dir_path = os.path.dirname(os.path.realpath(__file__))
ROOT.gROOT.LoadMacro(os.path.join( dir_path, "../../../tools/scripts/tdrstyle.C"))
ROOT.setTDRStyle()

# BIT
from BIT.MultiBoostedInformationTree import MultiBoostedInformationTree

# User
import user

# Parser
import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument("--plot_directory",     action="store",      default="BIT_semiLepTTbar_delphesJet",   help="plot sub-directory")
argParser.add_argument("--prefix",             action="store",      default="bit_semiLepTTbar_delphesJet",                 help="plot sub-directory")
argParser.add_argument("--model",              action="store",      default="semiLepTTbar_delphesJet",                 help="plot sub-directory")
argParser.add_argument("--WCs",                action="store",      nargs='*', default=["ctGRe", -1, 1, "ctGIm", -1, 1],                 help="Wilson coefficients")
argParser.add_argument("--nBins",              action="store",      type=int, default=30,                 help="Number of bins in each dimension")
argParser.add_argument("--nEvents",            action="store",      type=int, default=-1,             help="Number of events")
argParser.add_argument('--truth',              action='store_true', help="Truth?" )
argParser.add_argument('--lumi_factor',        action='store',      type=float, default=1.0, help="Lumi factor" )
argParser.add_argument('--bit',                action='store',      default='multiBit_semiLepTTbar_v3_ctGRe_ctGIm_ctWRe_ctWIm_nTraining_-1_nTrees_400', help="Which BIT?" )
argParser.add_argument('--ignoreR',            action='store_true', help="Use log(inclusive xsec ratio) as test statistic?" )
argParser.add_argument('--ignoreEFTScaling',   action='store_true', help="Ignore the xec effect of EFT, only use shapes?" )

args = argParser.parse_args()

WC1, theta1_min, theta1_max, WC2, theta2_min, theta2_max = [float(x[1]) if x[0] in [1,2,4,5] else x[1] for x in enumerate(args.WCs)]

# import the VH model
exec( "import models.%s as model"%args.model )

feature_names = model.feature_names

if args.ignoreR:
    sub_directory = "1bin" 
elif args.ignoreEFTScaling:
    sub_directory = "unbinned_shape" 
else:
    sub_directory = "unbinned" 
    
# directory for plots
plot_directory = os.path.join( user.plot_directory, args.plot_directory, args.model )

if not os.path.isdir(plot_directory):
    try:
        os.makedirs( plot_directory )
    except IOError:
        pass

features, weights = model.getEvents(args.nEvents)
weights = {tuple(sorted(key)):val for key, val in weights.items() }
nEvents  = len(features)
print(("Loaded data set of size %i" % nEvents ))

# normalize to SM event yield
lambda_expected_sm      = 100 # number of expected events 
lambda_current          = np.sum(weights[tuple()])
for key in list(weights.keys()):
    weights[key] = lambda_expected_sm/lambda_current*weights[key]
    
# predict total yields
xsec_coefficients  = {key:np.sum(weights[key]) for key in weights.keys()}
def lambda_tot( lin=False, **kwargs):
    result =  xsec_coefficients[tuple()]
    result += sum( [ (kwargs[coeff] - model.default_eft_parameters[coeff])*xsec_coefficients[(coeff,)] for coeff in list(kwargs.keys()) ])
    if not lin:
        result += sum( [ (.5 if coeff1!=coeff2 else 1)*(kwargs[coeff1] - model.default_eft_parameters[coeff1])*(kwargs[coeff2] - model.default_eft_parameters[coeff2])*xsec_coefficients[tuple(sorted((coeff1,coeff2)))] for coeff1 in list(kwargs.keys())  for coeff2 in list(kwargs.keys())])
    return result 

# xsec ratio
def lambda_ratio( lin=False, **kwargs):
    return lambda_tot( lin=lin, **kwargs ) / lambda_expected_sm

# compute weights for arbitrary WC
def make_weights( lin=False, **kwargs):
    result =  copy.deepcopy(weights[tuple()])
    result += sum( [ (kwargs[coeff] - model.default_eft_parameters[coeff])*weights[(coeff,)] for coeff in list(kwargs.keys()) ])
    if not lin:
        result += sum( [ (.5 if coeff1!=coeff2 else 1)*(kwargs[coeff1] - model.default_eft_parameters[coeff1])*(kwargs[coeff2] - model.default_eft_parameters[coeff2])*weights[tuple(sorted((coeff1,coeff2)))] for coeff1 in list(kwargs.keys())  for coeff2 in list(kwargs.keys())])
    return result 

filename = os.path.join(user.model_directory, args.bit)+'.pkl'
try:
    print ("Loading MultiBIT %s"%(filename))
    bit = MultiBoostedInformationTree.load(filename)
    predictions = bit.vectorized_predict(features)
    predictions = { der:predictions[:,i_der] for i_der, der in enumerate(bit.derivatives) }
except (IOError, EOFError, ValueError):
    bit         = None
    predictions = None

def make_logR_to_SM( order, predictions=predictions, truth=False,  **kwargs ):

    if args.ignoreR:
        return log(lambda_ratio(**kwargs))*np.ones(len(weights[()])) 

    eft      = model.make_eft(**kwargs)
    if order not in ["lin", "quad", "total"]:
        raise RuntimeError("Order %s not known" % order )
    result = np.zeros(nEvents)
    if order in ["lin", "total"]:
        for coeff in model.wilson_coefficients:
            if eft[coeff] == model.default_eft_parameters[coeff]: continue
            result += (eft[coeff] - model.default_eft_parameters[coeff])*( weights[(coeff,)]/weights[tuple()] if truth else predictions[(coeff,)])
    if order in ["quad", "total"]:
        for coeff1 in model.wilson_coefficients:
            if eft[coeff1] == model.default_eft_parameters[coeff1]: continue
            for coeff2 in model.wilson_coefficients:
                if eft[coeff2] == model.default_eft_parameters[coeff2]: continue
                result += (.5 if coeff1!=coeff2 else 1)*(eft[coeff1] - model.default_eft_parameters[coeff1])*(eft[coeff2] - model.default_eft_parameters[coeff2])*( weights[tuple(sorted((coeff1,coeff2)))]/weights[tuple()] if truth else predictions[tuple(sorted((coeff1,coeff2)))])

    result += 1
    neg_frac = len(result[result<0])/float(len(result))
    if neg_frac>10**-3:
        print("Fraction of negative test statistics for %s: %3.2f"% ( order, neg_frac ))
    return 0.5*np.log( result**2 )

event_indices = np.arange(nEvents)
def make_toys( yield_per_toy, n_toys, lin=False, **kwargs):
    weights_      = make_weights(lin=lin, **kwargs) 
    biased_sample = np.random.choice( event_indices, size=10*nEvents,  p = weights_/np.sum(weights_) )

    return [ np.array(np.random.choice( biased_sample, size=n_observed )) for n_observed in np.random.poisson(yield_per_toy, n_toys) ]

tex = ROOT.TLatex()
tex.SetNDC()
tex.SetTextSize(0.04)

colors   = [ ROOT.kRed, ROOT.kBlue, ROOT.kBlack, ROOT.kBlue, ROOT.kRed]
#extended = True
n_toys   = 500000

# do not make the following inconsistent
levels          = [ 0.68, 0.95 ]
#quantile_levels = [ 0.32, 0.05 ]
#quantile_levels = [0.05, 0.32]

# P(n|lambda) = exp(-lambda)lambda^n/n!
#    lambda_theta = lambda_tot(cHWtil=.3)
#    lambda_0     = lambda_tot()

def PoissonExclusionPower( lambda_theta, lambda_0 ):
    lambda_theta = args.lumi_factor*lambda_tot(**eft)
    lambda_0     = args.lumi_factor*lambda_tot()

    n_lambda_theta = np.random.poisson( lambda_theta, n_toys )
    n_lambda_0     = np.random.poisson( lambda_0, n_toys )

    q_null = lambda_theta - lambda_0 - n_lambda_theta*log(lambda_theta/float(lambda_0) ) 
    q_alt  = lambda_theta - lambda_0 - n_lambda_0*log(lambda_theta/float(lambda_0) ) 

    q_null, q_alt = (q_null - np.mean(q_null) )/sqrt(np.var(q_null)), (q_alt  - np.mean(q_null) )/sqrt(np.var(q_null))

    #print "Null mean, sigma",np.mean(q_null), sqrt(np.var(q_null)), "Alt mean, sigma", np.mean(q_alt), sqrt(np.var(q_alt))

    quantiles_null  = np.quantile( q_null, levels )
    sizes = {level:len(q_null[q_null<=quantiles_null[i_level]])/float(len(q_null)) for i_level, level in enumerate(levels) }
    powers = {level:np.count_nonzero(q_alt>quantiles_null[i_level])/float(n_toys) for i_level, level in enumerate( levels ) }

    return {'size':sizes, 'power':powers}

def getContours( h, level):
    _h     = h.Clone()
    _h.Smooth(1, "k5b")
    ctmp = ROOT.TCanvas()
    _h.SetContour(1,array.array('d', [level]))
    _h.Draw("contzlist")
    _h.GetZaxis().SetRangeUser(0.0001,1)
    ctmp.Update()
    contours = ROOT.gROOT.GetListOfSpecials().FindObject("contours")
    return contours.At(0).Clone()

###################################
##          Training plot         #
###################################
#
#predictions_iterations = { der:np.cumsum(bits[der].vectorized_predict(features, summed = False), axis=0) for der in list(bits.keys()) } 
#n_iterations   = len(list(predictions_iterations.values())[0])
#h_power = {level:ROOT.TH1F("power", "power", n_iterations, 0 ,n_iterations ) for level in levels}
#for h in list(h_power.values()):
#    h.style = styles.lineStyle( ROOT.kBlack, width = 2)
##eft = {'cHWtil':0.35, 'cHW':-0.4}
#for eft in [ 
#        #{'cHWtil':0.1,  'cHW':0,     'cHQ3':0},
#        #{'cHWtil':0.25,  'cHW':0,     'cHQ3':0},
#        {'cHWtil':0,  'cHW':0.15,     'cHQ3':-0.015},
#        {'cHWtil':0,  'cHW':0.15,     'cHQ3':0},
#        {'cHWtil':0,  'cHW':0,        'cHQ3':-0.015},
#        {'cHWtil':0,  'cHW':-.1,      'cHQ3':0},
#        {'cHWtil':0,  'cHW':-.15,      'cHQ3':0.01},
#        {'cHWtil':0.2,  'cHW':0,      'cHQ3':0.},
#        {'cHWtil':-0.2,  'cHW':0,      'cHQ3':0.},
#        #{'cHWtil':0.15,  'cHW':0,     'cHQ3':0},
#        #{'cHW':0.3,  'cHWtil':0,     'cHQ3':0},
#        #{'cHW':0.25,  'cHWtil':0,     'cHQ3':0},
#        #{'cHW':0.20,  'cHWtil':0,     'cHQ3':0},
#        #{'cHW':0.15,  'cHWtil':0,     'cHQ3':0},
#
#        #{'cHQ3':0.03,  'cHWtil':0,      'cHW':0},
#        #{'cHQ3':0.025,  'cHWtil':0,     'cHW':0},
#        #{'cHQ3':0.020,  'cHWtil':0,     'cHW':0},
#        #{'cHQ3':0.015,  'cHWtil':0,     'cHW':0},
#        #{'cHWtil':0.15, 'cHW':-0.15, 'cHQ3':0},
#        #{'cHWtil':0.35, 'cHW':-0.4,  'cHQ3':0},
#        ]:
#    power_truth = {}
#    theta_toys= make_toys( args.lumi_factor*lambda_tot(**eft), n_toys, **eft)
#    sm_toys   = make_toys( args.lumi_factor*lambda_tot(), n_toys ) 
#    const     = args.lumi_factor*(lambda_tot(**eft) - lambda_tot())
#    for iteration in range( -1, n_iterations ):
#        if iteration<0:
#            event_logR_to_SM = make_logR_to_SM( "total", truth=True, **eft)
#        else:
#            predictions_i = {key:(value[iteration] if iteration<bits[key].n_trees else value[bits[key].n_trees-1]) for key, value in predictions_iterations.items()}
#            event_logR_to_SM = make_logR_to_SM( "total", predictions=predictions_i, truth=False, **eft)
#    
#        q_null = const-np.array([np.sum( event_logR_to_SM[toy_] ) for toy_ in theta_toys ]) #NULL
#        q_alt  = const-np.array([np.sum( event_logR_to_SM[toy_] ) for toy_ in sm_toys ])    #ALT
#        # calibration according to the null
#        if True:
#            n = float(len(q_null))
#            mean_q_null     = np.sum(q_null)/n
#            sigma_q_null    = sqrt( np.sum((q_null - mean_q_null)**2)/(n-1) ) 
#            q_alt  = (q_alt  - mean_q_null)/sigma_q_null
#            q_null = (q_null - mean_q_null)/sigma_q_null
#
#        quantiles_null  = np.quantile( q_null, levels )
#        #quantiles_alt   = np.quantile( q_alt, levels )
#        sizes = {level:len(q_null[q_null<quantiles_null[i_level]])/float(len(q_null)) for i_level, level in enumerate(levels) }
#        #power_histo     = np.histogram( q_given_theta, quantiles_SM)
#        
#        powers = {level:np.count_nonzero(q_alt>=quantiles_null[i_level])/float(n_toys) for i_level, level in enumerate(levels) }
#
#        for  i_level, level in enumerate(levels):
#            if iteration>=0:
#                h_power[level].SetBinContent( h_power[level].FindBin( iteration ), powers[level] )
#            else:
#                power_truth[level] = powers[level]
#            print("iteration",iteration, "size", sizes[levels[i_level]], "power", round(powers[level],3))
#
#    for i_level, level in enumerate(levels):
#        plot = Plot.fromHisto(name = "power_evolution_cHW_%3.2f_cHWtil_%3.2f_cHQ3_%3.2f_level_%3.2f"%( eft['cHW'], eft['cHWtil'], eft['cHQ3'], level), histos = [[h_power[level]]], texX = "Boosting iteration", texY = "power" )
#        line = ROOT.TLine(0,power_truth[level],n_iterations,power_truth[level])
#        line.SetLineWidth(2)
#        plotting.draw(plot, plot_directory = os.path.join( plot_directory, sub_directory), logY = False, logX = False, copyIndexPHP=True, drawObjects = [line], yRange = (0,1))
#
#syncer.sync()

##################################
#            2D plot             #
##################################

step1 = (theta1_max-theta1_min)/args.nBins
step2 = (theta2_max-theta2_min)/args.nBins
theta1_vals = np.arange(theta1_min, theta1_max+step1, (theta1_max-theta1_min)/args.nBins) 
theta2_vals = np.arange(theta2_min, theta2_max+step2, (theta2_max-theta2_min)/args.nBins) 

test_statistics = ["total"]

power = {}
for test_statistic in test_statistics: 
#for test_statistic in ["total"]: 

    truth_txt = "truth" if args.truth else "predicted"
    print("Test statistic", test_statistic, "truth?", args.truth)

    power[test_statistic] = {level:ROOT.TH2D("power_"+test_statistic, "power_"+test_statistic, len(theta1_vals)-1, array.array('d', theta1_vals), len(theta2_vals)-1, array.array('d', theta2_vals)) for level in levels}

    min_, max_ = float('inf'), -float('inf')

    sm_toys = make_toys( args.lumi_factor*lambda_tot(), n_toys )
 
    for i_theta1, theta1 in enumerate( theta1_vals ):
        #for i_theta1, theta1 in enumerate( [0] ):
        for i_theta2, theta2 in enumerate( theta2_vals ):
            #for i_theta2, theta2 in enumerate( [.01, .05, .1, .15, .2] ):

            if theta1==theta2==0: continue

            eft     = {WC1:theta1, WC2:theta2}
            event_logR_to_SM = make_logR_to_SM( test_statistic, truth=args.truth, **eft )
            const   = args.lumi_factor*(lambda_tot(**eft) - lambda_tot())

            q_null = const-np.array([np.sum( event_logR_to_SM[toy_] ) for toy_ in make_toys( args.lumi_factor*(lambda_tot(**eft) if not args.ignoreEFTScaling else lambda_tot()), n_toys, **eft) ]) #NULL
            q_alt  = const-np.array([np.sum( event_logR_to_SM[toy_] ) for toy_ in sm_toys ])    #ALT

            # calibration according to the null
            if True:
                mean_q_null     = np.mean(q_null)
                sigma_q_null    = sqrt( np.var(q_null ))
                q_alt  = (q_alt  - mean_q_null)/sigma_q_null
                q_null = (q_null - mean_q_null)/sigma_q_null

            quantiles_null  = np.quantile( q_null, levels )
            #quantiles_alt   = np.quantile( q_alt, levels )
            sizes = {level:len(q_null[q_null<=quantiles_null[i_level]])/float(len(q_null)) for i_level, level in enumerate(levels) }
            #power_histo     = np.histogram( q_given_theta, quantiles_SM)

            for i_level, level in enumerate(levels):
                power_ = np.count_nonzero(q_alt>quantiles_null[i_level])/float(n_toys)
                power[test_statistic][level].SetBinContent( power[test_statistic][level].FindBin( theta1, theta2 ), power_ )
                print("theta", round(theta1,3), round(theta2,4), "size", sizes[level], "power", round(power_,4), test_statistic, WC1, WC2,  "truth", args.truth)

            pois = PoissonExclusionPower(args.lumi_factor*lambda_tot(**eft), args.lumi_factor*lambda_tot())
            for level in levels:
                print("PoissonExclusionpower (alphs=%4.3f): %4.3f" % ( pois['size'][level], pois['power'][level] ))
            print()

colors   = { 'quad':ROOT.kRed, 'lin':ROOT.kBlue, 'total':ROOT.kBlack}

contours = { key:{level:getContours( power[key][level], 0.5 ) for level in levels } for key in list(power.keys())} 
contour_objects = []
for test_statistic in list(contours.keys()):
    for level in list(contours[test_statistic].keys()):
        for i_tgraph in range(contours[test_statistic][level].GetSize()):
            tgr = contours[test_statistic][level].At(i_tgraph)
            print(i_tgraph, tgr, test_statistic, level)

            tgr.SetLineColor(colors[test_statistic])
            tgr.SetLineWidth(2)
            tgr.SetLineStyle(ROOT.kDashed if level!=0.95 else 1)
            tgr.SetMarkerStyle(0)
            contour_objects.append( tgr )

for test_statistic in test_statistics:
    for level in levels: 

        filename = "power_%s_%s_vs_%s_%s_lumi_factor_%3.2f_level_%3.2f"%(test_statistic, WC1, WC2, ("truth" if args.truth else "predicted"), args.lumi_factor, level)
        dirname  = os.path.join( plot_directory, sub_directory)
        ROOT.gStyle.SetPalette(58)

        c1 = ROOT.TCanvas()
        h = power[test_statistic][level]
        h.GetXaxis().SetTitle(WC1)
        h.GetYaxis().SetTitle(WC2)

        c1.SetBottomMargin(0.13)
        c1.SetLeftMargin(0.15)
        c1.SetTopMargin(0.07)
        c1.SetRightMargin(0.16)
        c1.SetLogz()
        h.GetXaxis().SetTitleFont(43)
        h.GetYaxis().SetTitleFont(43)
        h.GetXaxis().SetLabelFont(43)
        h.GetYaxis().SetLabelFont(43)
        h.GetXaxis().SetTitleSize(24)
        h.GetYaxis().SetTitleSize(24)
        h.GetXaxis().SetLabelSize(20)
        h.GetYaxis().SetLabelSize(20)
        h.Draw("COLZ")

        c1.Update()
        palette = h.GetListOfFunctions().FindObject("palette")
        palette.SetX1NDC(0.85)
        palette.SetX2NDC(0.90)
        palette.SetY1NDC(0.13)
        palette.SetY2NDC(0.93)
        c1.Modified()
        c1.Update()

        for o in contour_objects:
            o.Draw("same")

        if not os.path.exists(dirname):
            try:
                os.path.makedirs(dirname)
            except:
                pass

        for extension in ["root", "png", "pdf"]:
            c1.Print( os.path.join( dirname, filename+"."+extension) )

        #plotting.draw2D(plot2D, plot_directory = os.path.join( plot_directory, sub_directory), histModifications = [lambda h:ROOT.gStyle.SetPalette(58)], logY = False, logX = False, logZ = True, copyIndexPHP=True, drawObjects = contour_objects, zRange = (0.01,1))

syncer.sync()
