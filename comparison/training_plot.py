# Standard imports
import ROOT
import numpy as np
import random
import cProfile
import time
import os, sys
import sys
sys.path.insert( 0, '..')
from math import log, exp, sin, cos, sqrt, pi
import copy
import array

dir_path = os.path.dirname(os.path.realpath(__file__))
ROOT.gROOT.LoadMacro(os.path.join( dir_path, "../tools/scripts/tdrstyle.C"))
ROOT.setTDRStyle()

# BIT
from BIT.MultiBoostedInformationTree import MultiBoostedInformationTree
from NN.CholeskyNN import CholeskyNN 
import torch
torch.set_grad_enabled(False)

# User
import tools.user as user
import tools.syncer as syncer

# Parser
import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument("--plot_directory",     action="store",      default="choleskyNN_vs_BIT",       help="plot sub-directory")
argParser.add_argument("--model",              action="store",      default="ZH_Nakamura",             help="Name of the model")
argParser.add_argument("--nEvents",            action="store",      type=int, default=50000,             help="Number of events")
argParser.add_argument("--nToys",              action="store",      type=int, default=100,             help="Number of toys")
argParser.add_argument('--lumi_factor',        action='store',      type=float, default=1.0, help="Lumi factor" )

args = argParser.parse_args()
exec('import models.%s as model'%args.model)

feature_names = model.feature_names

# directory for plots
plot_directory = os.path.join( user.plot_directory, args.plot_directory, args.model )

if not os.path.isdir(plot_directory):
    try:
        os.makedirs( plot_directory )
    except IOError:
        pass

features = model.getEvents(args.nEvents)

#if args.model.startswith("ZH"):
#    adhoc = (features[:,model.feature_names.index('fLL')]>0.2) & (features[:,model.feature_names.index('f2TT')]<3) & (features[:,model.feature_names.index('cos_theta')]>-0.9) & (features[:,model.feature_names.index('cos_theta')]<0.9) & (features[:,model.feature_names.index('f1TT')]>-0.9) & (features[:,model.feature_names.index('f1TT')]<0.9)  & (features[:,model.feature_names.index('f2TT')]<3.5) 
#    features = features[adhoc]

nEvents  = len(features)
weights  = model.getWeights(features, eft=model.default_eft_parameters)
print ("Created data set of size %i" % nEvents )

# normalize to SM event yield
lambda_expected_sm      = 90.13 if args.model.startswith("ZH") else 599.87 #Delphes, ptZ>200
lambda_current          = np.sum(weights[tuple()])
for key in weights.keys():
    weights[key] = lambda_expected_sm/lambda_current*weights[key]
    
# predict total yields
sigma_coefficients  = {key:np.sum(weights[key]) for key in model.derivatives}
def lambda_tot( lin=False, **kwargs):
    result =  sigma_coefficients[tuple()]
    result += sum( [ (kwargs[coeff] - model.default_eft_parameters[coeff])*sigma_coefficients[(coeff,)] for coeff in kwargs.keys() ])
    if not lin:
        result += sum( [ .5*(kwargs[coeff1] - model.default_eft_parameters[coeff1])*(kwargs[coeff2] - model.default_eft_parameters[coeff2])*sigma_coefficients[tuple(sorted((coeff1,coeff2)))] for coeff1 in kwargs.keys()  for coeff2 in kwargs.keys()])
    return result 

# xsec ratio
def lambda_ratio( lin=False, **kwargs):
    return lambda_tot( lin=lin, **kwargs ) / lambda_expected_sm

# compute weights for arbitrary WC
def make_weights( lin=False, **kwargs):
    result =  copy.deepcopy(weights[tuple()])
    result += sum( [ (kwargs[coeff] - model.default_eft_parameters[coeff])*weights[(coeff,)] for coeff in kwargs.keys() ])
    if not lin:
        result += sum( [ .5*(kwargs[coeff1] - model.default_eft_parameters[coeff1])*(kwargs[coeff2] - model.default_eft_parameters[coeff2])*weights[tuple(sorted((coeff1,coeff2)))] for coeff1 in kwargs.keys()  for coeff2 in kwargs.keys()])
    return result 

def make_logR_to_SM( order, truth=False, predictions=None, **kwargs ):

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
                result += .5*(eft[coeff1] - model.default_eft_parameters[coeff1])*(eft[coeff2] - model.default_eft_parameters[coeff2])*( weights[tuple(sorted((coeff1,coeff2)))]/weights[tuple()] if truth else predictions[tuple(sorted((coeff1,coeff2)))])

    result += 1
    neg_frac = len(result[result<0])/float(len(result))
    if neg_frac>10**-3:
        print ("Fraction of negative test statistics for %s: %3.2f"% ( order, neg_frac ))
    return 0.5*np.log( result**2 )

event_indices = np.arange(nEvents)
def make_toys( yield_per_toy, n_toys, lin=False, **kwargs):
    weights_      = make_weights(lin=lin, **kwargs) 
    biased_sample = np.random.choice( event_indices, size=50*nEvents,  p = weights_/np.sum(weights_) )

    return np.array( [ np.random.choice( biased_sample, size=n_observed ) for n_observed in np.random.poisson(yield_per_toy, n_toys) ])

tex = ROOT.TLatex()
tex.SetNDC()
tex.SetTextSize(0.04)

colors   = [ ROOT.kRed, ROOT.kBlue, ROOT.kBlack, ROOT.kBlue, ROOT.kRed]
#extended = True
n_toys   = 50000

# do not make the following inconsistent
level          = 0.95

# precompute BITS
h_power     = {}

cfgs = [
        { "name":"cnn", "cfg":"choleskyNN_ZH_Nakamura_bpg_0p01_0p5_1_cHW_nTraining_100000" ,   "tex":"cNN", "color":ROOT.kRed},
        { "name":"bit", "cfg":"multiBit_ZH_Nakamura_v4_cHW_nTraining_500000_nTrees_200",       "tex":"BIT", "color":ROOT.kBlue},
    ]

model_directory = '/groups/hephy/cms/robert.schoefbeck/NN/models/'
for cfg in cfgs:
    if 'bit' in cfg['name']:
        cfg['model']          = MultiBoostedInformationTree.load( os.path.join( model_directory, cfg['cfg']+'.pkl' ))
        cfg['epochs']         = list(range(1,len(cfg['model'].trees)+1)) 
        cfg['predictions']    = np.cumsum( cfg['model'].vectorized_predict( features, summed = False), axis=0)
        cfg['predictions']    = {epoch:{comb:cfg['predictions'][i_epoch,:,i_comb] for i_comb, comb in enumerate(cfg['model'].derivatives) } for i_epoch, epoch in enumerate(cfg['epochs'])} 
        cfg['training_time']  = {cfg['epochs'][i]:t for i,t in enumerate(np.cumsum([cfg['model'].trees[i_tree].training_time for i_tree in range(len(cfg['model'].trees)) ])) }

    elif 'cnn' in cfg['name']:
        cfg['model'] = CholeskyNN.load(os.path.join( model_directory, cfg['cfg']+'.pkl' ))
        cfg['epochs'] = list( cfg['model'].snapshots.keys() )
        cfg['epochs'].sort()
        cfg['predictions'] = np.array([cfg['model'].dict_to_derivatives(cfg['model'].load_snapshot(cfg['model'].snapshots[i_snapshot]).predict( features )) for i_snapshot in cfg['epochs'] ] )
        #cfg['predictions'] = {comb:cfg['predictions'][:,i_comb] for i_comb, comb in enumerate(cfg['model'].combinations) }
        cfg['predictions'] = {epoch:{comb:cfg['predictions'][i_epoch,:,i_comb] for i_comb, comb in enumerate(cfg['model'].combinations) } for i_epoch, epoch in enumerate(cfg['epochs'])} 
        cfg['training_time'] = {cfg['model'].monitoring[i]['epoch']:cfg['model'].monitoring[i]['training_time'] for i in range(len(cfg['model'].monitoring)) }

#eft = {'cHWtil':0.35, 'cHW':-0.4}
#        {'cHWtil':0.1,  'cHW':0,     'cHQ3':0},
#        {'cHWtil':0.25,  'cHW':0,     'cHQ3':0},
#        {'cHWtil':0.15,  'cHW':0,     'cHQ3':0},
#        {'cHW':0.3,  'cHWtil':0,     'cHQ3':0},
#        {'cHW':0.25,  'cHWtil':0,     'cHQ3':0},
#        {'cHW':0.20,  'cHWtil':0,     'cHQ3':0},
#        {'cHW':0.15,  'cHWtil':0,     'cHQ3':0},

        #{'cHQ3':0.03,  'cHWtil':0,      'cHW':0},
        #{'cHQ3':0.025,  'cHWtil':0,     'cHW':0},
        #{'cHQ3':0.020,  'cHWtil':0,     'cHW':0},
        #{'cHQ3':0.015,  'cHWtil':0,     'cHW':0},
        #{'cHWtil':0.15, 'cHW':-0.15, 'cHQ3':0},
        #{'cHWtil':0.35, 'cHW':-0.4,  'cHQ3':0},

eft         = {'cHWtil':0.0,  'cHW':0.3, 'cHQ3':0}
const       = args.lumi_factor*(lambda_tot(**eft) - lambda_tot())

if True:
    i_plot=0

    sm_toys = make_toys( args.lumi_factor*lambda_tot(), n_toys ) 
    eft_toys= make_toys( args.lumi_factor*lambda_tot(**eft), n_toys, **eft)

    for cfg in cfgs:

        #n_iterations   = len(predictions_iterations.values()[0])
        #cfg['tGraph'] = ROOT.TGraph("power", "power", len(cfg['epochs']) )
        cfg['epochs'].insert(0,-1)
        cfg['power'] = {}
        for epoch in cfg['epochs']:
            if epoch<0:
                event_logR_to_SM = make_logR_to_SM( "total", truth=True, **eft)
            else:
                #predictions_i = {key:(value[iteration] if iteration<bits[prefix][key].n_trees else value[bits[prefix][key].n_trees-1]) for key, value in cfg['predictions'].items()}
                event_logR_to_SM = make_logR_to_SM( "total", truth=False, predictions=cfg['predictions'][epoch], **eft)

            q_null = const-np.array([np.sum( event_logR_to_SM[toy_] ) for toy_ in eft_toys ]) #NULL
            q_alt  = const-np.array([np.sum( event_logR_to_SM[toy_] ) for toy_ in sm_toys ])    #ALT
            # calibration according to the null
            if True:
                n = float(len(q_null))
                mean_q_null     = np.sum(q_null)/n
                sigma_q_null    = sqrt( np.sum((q_null - mean_q_null)**2)/(n-1) )
                q_alt  = (q_alt  - mean_q_null)/sigma_q_null
                q_null = (q_null - mean_q_null)/sigma_q_null

            # Exclusion: The null hypothesis is the BSM point, the alternate is the SM.
            quantile_alt  = np.quantile( q_alt, level )
            quantile_null = np.quantile( q_null, level )
            size_         = np.count_nonzero(q_null>=quantile_null)/float(n_toys)
            #power_histo     = np.histogram( q_theta_given_theta, quantile_SM)

            power_ = np.count_nonzero(q_alt>=quantile_null)/float(n_toys)
            if epoch>=0:
                #h_power[prefix].SetBinContent( h_power[prefix].FindBin( iteration ), truth-power_ )
                cfg['power'][epoch] = truth-power_
            else:
                truth = power_
            if epoch>=0:
                print ("epoch",epoch, "size", size_, "power", round(power_,3), "t_train", round(cfg['training_time'][epoch]) )
            else:
                print ("epoch",epoch, "size", size_, "power", round(power_,3), "(truth)")

c1 = ROOT.TCanvas("c1");

l = ROOT.TLegend(0.2,0.8,0.9,0.85)
l.SetNColumns(2)
l.SetFillStyle(0)
l.SetShadowColor(ROOT.kWhite)
l.SetBorderSize(0)

for i_cfg, cfg in enumerate(cfgs):

    cfg['tGraph'] = ROOT.TGraph( len(cfg['epochs'])-1, array.array('d',[cfg['training_time'][epoch] for epoch in cfg['epochs'] if epoch>=0]), array.array( 'd', [ cfg['power'][epoch] for epoch in cfg['epochs'] if epoch>=0 ] ))

    cfg['tGraph'].SetLineColor( cfg['color'] )
    cfg['tGraph'].SetLineWidth(2)
    cfg['tGraph'].SetMarkerColor( cfg['color'] )
    cfg['tGraph'].SetMarkerStyle()
    
    cfg['tGraph'].GetXaxis().SetTitle("T_{train}")
    cfg['tGraph'].GetYaxis().SetTitle("Power")

    l.AddEntry(cfg['tGraph'], cfg['tex'])

    if i_cfg == 0:
        cfg['tGraph'].Draw("AL")
    else:
        cfg['tGraph'].Draw("Lsame")

l.Draw()

plot_directory_ = os.path.join( plot_directory, "comparison" )
c1.Print(os.path.join(plot_directory_, "comparison.png"))

