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
argParser.add_argument("--prefix",             action="store",      default="v4", type=str,  help="prefix")
argParser.add_argument("--small",              action="store_true"  )

args = argParser.parse_args()

exec('import models.%s as model'%(args.model))
#features, _, coeffs = model.getEvents(model.data_generator[-1])

if args.small:
   model.data_generator.input_files = model.data_generator.input_files[:1]
   args.plot_directory += '_small'

# directory for plots
plot_directory = os.path.join( user.plot_directory, args.plot_directory, args.prefix, args.model )
os.makedirs( plot_directory, exist_ok=True)

def getEvents( data ):
    coeffs       = model.data_generator.vector_branch(     data, 'p_C', padding_target=len(model.weightInfo.combinations))
    features     = model.data_generator.scalar_branches(   data, model.feature_names )
    vectors      = None #{key:model.data_generator.vector_branch(data, key ) for key in vector_branches}

    return features, vectors, coeffs

features, _, coeffs = getEvents(model.data_generator[-1])

def getWeights( eft, coeffs, lin=False):

    if lin:
        combs = list(filter( lambda c:len(c)<2, model.weightInfo.combinations))
    else:
        combs = model.weightInfo.combinations
    #fac = np.array( [ functools.reduce( operator.mul, [ (float(eft[v]) - model.weightInfo.ref_point_coordinates[v]) for v in comb ], 1 ) for comb in combs], dtype='float')
    fac = np.array( [ functools.reduce( operator.mul, [ float(eft[v]) for v in comb ], 1 ) for comb in combs], dtype='float')
    return np.matmul(coeffs[:,:len(combs)], fac)

def getDerivatives( eft, coeffs):
    coeff_mat = np.zeros( (len(model.weightInfo.variables), len(model.weightInfo.combinations)) )
    for i_var, variable in enumerate(model.weightInfo.variables): 
        for i_comb, comb in enumerate(model.weightInfo.combinations):
            if len(comb)==1:
                 if variable==comb[0]:
                     coeff_mat[i_var, i_comb]=1
            elif len(comb)==2:
                if variable not in comb:
                    continue
                coeff_mat[model.weightInfo.variables.index(comb[0]), i_comb] += float(eft[comb[1]]) 
                coeff_mat[model.weightInfo.variables.index(comb[1]), i_comb] += float(eft[comb[0]]) 
    return np.dot( coeffs, coeff_mat.transpose()) 

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

#####################
## Correlation plot #
#####################

excluded_features = np.array(list(set(np.argwhere(np.isnan(features))[:,1])))
if len(excluded_features)>0:
    features = np.delete( features, excluded_features, axis=1)
    feature_names = np.delete( model.feature_names, excluded_features)
    print ("Excluded features because of NaNs: ", ", ".join( [model.feature_names[i] for i in excluded_features]) )
else:
    feature_names = model.feature_names
    
mask            = np.ones(len(features), np.bool)
mask[np.argwhere(np.isnan(features))[:,0]] = 0
features        = features[mask]
coeffs          = coeffs[mask]

sm_weights  = getWeights( model.make_eft(), coeffs )
norm        = np.sum(sm_weights) 
means       = np.dot(sm_weights, features)/norm
covariance  = np.einsum('i,ij,ik', sm_weights, features - means, features - means)
correlation = np.array( [ [ covariance[i][j]/np.sqrt(covariance[i][i]*covariance[j][j]) for i in range(covariance.shape[0])] for j in range(covariance.shape[1])] )

n_features = covariance.shape[0]
h_correlation = ROOT.TH2F("corr", "corr", n_features, 0, n_features, n_features, 0, n_features)
for i_x in range(n_features):
    h_correlation.GetXaxis().SetBinLabel(i_x+1, model.plot_options[feature_names[i_x]]['tex'])
    h_correlation.GetYaxis().SetBinLabel(i_x+1, model.plot_options[feature_names[i_x]]['tex'])
    for i_y in range(n_features):
        h_correlation.SetBinContent( i_x+1, i_y+1, correlation[i_x,i_y] )

#h_correlation.GetZaxis().SetRangeUser(-1,1)
h_correlation.GetXaxis().LabelsOption("v")
h_correlation.GetYaxis().LabelsOption("v")
ROOT.gStyle.SetPalette(ROOT.kRainBow)
c1 = ROOT.TCanvas("c", "c", n_features*60, n_features*60)
h_correlation.Draw("COLZ")
h_correlation.GetXaxis().SetLabelSize(0.017)
h_correlation.GetYaxis().SetLabelSize(0.017)
h_correlation.GetZaxis().SetLabelSize(0.025)
c1.SetRightMargin(0.15)

ROOT.gPad.Update()
palette = h_correlation.GetListOfFunctions().FindObject("palette")

palette.SetX1NDC(0.88)
palette.SetX2NDC(0.92)
palette.SetY1NDC(0.13)
palette.SetY2NDC(0.95)
ROOT.gPad.Modified()
ROOT.gPad.Update()

c1.Print( os.path.join( plot_directory, "correlation_plots", "feature_correlation.png" ))
c1.Print( os.path.join( plot_directory, "correlation_plots", "feature_correlation.pdf" ))
syncer.sync()


derivatives = getDerivatives( model.make_eft(), coeffs)

total_der   =  derivatives.sum(axis=0)
total_yield = sm_weights.sum()

def ev_tex( ev ):
    max_ = np.max(np.abs(ev))
    sstr = ""
    for i_val, val in enumerate(ev):
        if abs(val)>10**-3*max_:
            sstr+= ("%+4.3f %s"%(val, model.tex[model.weightInfo.variables[i_val]]))
    return sstr.lstrip("+")

FI_total = 1./total_yield*np.outer(total_der,total_der)
#var_mask = [v in ['ctGRe', 'cQj18', 'ctj8'] for v in model.weightInfo.variables]
var_mask = [True for v in model.weightInfo.variables]
FI_total = FI_total[:,var_mask][var_mask,:]

#helpers.weighted_quantile( values=features[:,0], quantiles=np.linspace(0,1,11), sample_weights=sm_weights)

from scipy.linalg import eigh

w, vr = eigh(FI_total)


print ("Total")
for i_ew, ew in enumerate(reversed(w)):
    ev = vr[:,len(w)-i_ew-1]
    if not ew>10**-4*w[-1]: 
        continue
    print (i_ew, np.sqrt(ew), ev_tex(ev))

print()

Nbins       = 10
max_counter = 6
counter     = 0
FI_best     = FI_total
while True:
    print ("Iteration", counter)
    best_ew = 0
    for i_feature, feature_name in enumerate(feature_names):

        #print ("Feature", feature_name)

        thresholds = helpers.weighted_quantile(  
                values=features[:,i_feature], 
                quantiles=np.linspace(0,1,Nbins+1), sample_weight=sm_weights)

        binned_yields, _      = np.histogram( features[:,i_feature], thresholds, weights=sm_weights )

        binned_derivatives = np.zeros((len(model.weightInfo.variables), 10))
        for i_variable, variable in enumerate(model.weightInfo.variables):
            binned_derivatives[i_variable], _ = np.histogram( features[:,i_feature], thresholds, weights=derivatives[:,i_variable] )

        mask = binned_yields!=0

        FI = np.einsum('i,ji,ki',1./binned_yields[mask], binned_derivatives[:, mask], binned_derivatives[:, mask])[:,var_mask][var_mask,:]

        if np.isnan(FI.sum()): continue  

        w, vr = eigh(FI-FI_best)

        for i_ew, ew in enumerate(reversed(w)):
            ev = vr[:,len(w)-i_ew-1]
            if not ew>10**-4*w[-1]: 
                continue
            #print (i_ew, np.sqrt(ew), ev_tex(ev))

            if ew>best_ew:
                #print ("Found new best: feature", feature_name)
                #print (i_ew, np.sqrt(ew), ev_tex(ev))
                FI_best_  = FI
                best_ew   = ew
                best_evec = ev
                best_feature = feature_name

    counter+=1
    print("Done with iteration: ", counter, "best_ew",best_ew)
    if best_ew>0:
        print ("Best feature:,", best_feature, "constrainig best evec:\n", ev_tex(best_evec))
        
        FI_best += FI_best_
     
    print()

    if counter>max_counter: break
    
