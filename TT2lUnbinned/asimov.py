#!/usr/bin/env python

# Standard imports
import ROOT
import numpy as np
import math
import array
import sys, os, copy
import functools
import operator
import itertools
import re
import scipy
import pickle

sys.path.insert(0, '..')
import tools.syncer as syncer
import tools.user as user
import tools.helpers as helpers


#ROOT.gStyle.SetPalette(ROOT.kBird)
dir_path = os.path.dirname(os.path.realpath(__file__))
ROOT.gROOT.LoadMacro(os.path.join( dir_path, "../tools/scripts/tdrstyle.C"))
ROOT.setTDRStyle()

# Parser
import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--logLevel',           action='store',      default='INFO',          nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], help="Log level for logging")
argParser.add_argument("--plot_directory",     action="store",      default="TTCA",     help="plot sub-directory")
argParser.add_argument("--data_model",         action="store",      default = "TT2lUnbinned", help="Which data model?")
argParser.add_argument("--prefix",             action="store",      default="v2", type=str,  help="prefix")
argParser.add_argument("--bit_name",           action="store",      default="multiBit_TT2lUnbinned_TK_False_LK_False_CA_False_SC_False_v1_coeffs_ctGRe_ctGIm_cQj18_cQj38_ctj8_nTraining_-1_nTrees_300", type=str,  help="prefix")
argParser.add_argument("--small",              action="store_true"  )
argParser.add_argument("--overwrite",          action="store_true", help = "Overwrite?"  )
argParser.add_argument("--cmds",               action="store_true"  )

argParser.add_argument("--wc1",                action="store",      default = "ctGRe", help="Which wilson coefficient?")
argParser.add_argument("--low1",               action="store",      default = -0.7, type=float, help="Which wilson coefficient?")
argParser.add_argument("--high1",              action="store",      default = 0.7, type=float, help="Which wilson coefficient?")
argParser.add_argument("--nBins",              action="store",      default = 35, type=int, help="Which wilson coefficient?")
argParser.add_argument("--wc2",                action="store",      default = None, help="Which wilson coefficient?")
argParser.add_argument("--low2",               action="store",      default = -0.7, type=float, help="Which wilson coefficient?")
argParser.add_argument("--high2",              action="store",      default = 0.7, type=float, help="Which wilson coefficient?")

args = argParser.parse_args()

if args.cmds:
    for TK in ["False", "True"]:
        for LK in ["False", "True"]:
            for CA in ["False", "True"]:
                for SC in ["False", "True"]:
                    if TK==LK==CA==SC=="True": continue
                    for wc1 in ["ctGRe", "ctGIm", "cQj18", "cQj38", "ctj8"]:
                        print("python asimov.py --bit_name multiBit_TT2lUnbinned_TK_{TK}_LK_{LK}_CA_{CA}_SC_{SC}_v1_coeffs_ctGRe_ctGIm_cQj18_cQj38_ctj8_nTraining_-1_nTrees_300 --wc1 {wc1}".format(TK=TK, LK=LK, CA=CA, SC=SC, wc1=wc1))
                        for wc2 in ["ctGRe", "ctGIm", "cQj18", "cQj38", "ctj8"]:
                            if wc1>=wc2:continue
                            print("python asimov.py --bit_name multiBit_TT2lUnbinned_TK_{TK}_LK_{LK}_CA_{CA}_SC_{SC}_v1_coeffs_ctGRe_ctGIm_cQj18_cQj38_ctj8_nTraining_-1_nTrees_300 --wc1 {wc1} --wc2 {wc2}".format(TK=TK, LK=LK, CA=CA, SC=SC, wc1=wc1,wc2=wc2))
    sys.exit(0)
 
#Logger
import tools.logger as logger_
logger = logger_.get_logger(args.logLevel, logFile = None )

# directory for plots
plot_directory = os.path.join( user.plot_directory, args.plot_directory, args.data_model)#, args.physics_model )
os.makedirs( plot_directory, exist_ok=True)

results_directory = os.path.join( user.results_directory, args.data_model, args.bit_name )
os.makedirs( results_directory, exist_ok=True)
results_filename = os.path.join( results_directory, args.wc1 + ("_vs_"+args.wc2 if args.wc2 is not None else"") + ".pkl")
if os.path.exists( results_filename ) and not args.overwrite:
    logger.info( "Found %s. Quit.", results_filename )
    sys.exit(0)

# BIT
#bit_name = "multiBit_TT2lUnbinned_TK_False_LK_False_CA_False_SC_False_v1_coeffs_ctGRe_ctGIm_cQj18_cQj38_ctj8_nTraining_-1_nTrees_300"
#bit_name = "multiBit_TT2lUnbinned_TK_False_LK_False_CA_True_SC_False_v1_coeffs_ctGRe_ctGIm_cQj18_cQj38_ctj8_nTraining_-1_nTrees_300"

## FIXME ... this is bad
#sstr  = re.sub(r'^.*?_TT2lUnbinned_', '', args.bit_name) 
#flags = re.sub(r'_v.*_coeffs_.*', '', sstr).split('_')
#data_model_flags = {key:val for key, val in zip( [ "top_kinematics", "lepton_kinematics", "asymmetry", "spin_correlation" ], list(map( (lambda f:f=="True"), flags[1::2])))}

from BIT.MultiBoostedInformationTree import MultiBoostedInformationTree
filename = os.path.join(user.model_directory, args.bit_name)+'.pkl'
logger.info ("Loading %s for %s"%(args.bit_name, filename))
bit = MultiBoostedInformationTree.load(filename)

exec('import data_models.%s as data_model'%(args.data_model))
data_model_ = data_model.DataModel()

if args.small:
   data_model.data_generator.input_files = data_model.data_generator.input_files[:10]
   args.plot_directory += '_small'

sstr                = re.sub(r'^.*?_coeffs_', '', args.bit_name)
wilson_coefficients = re.sub(r'_nTraining_.*', '', sstr).split('_')

def make_eft( **kwargs ):
    result = {wc:0 for wc in wilson_coefficients}
    if any( k not in wilson_coefficients for k in kwargs.keys()):
        raise RuntimeError( "Unknown Wilson coefficients: %r" % ( [k for k in kwargs.keys() if k not in wilson_coefficients]))
    result.update( kwargs )
    return result

logger.info ("Loading events with WC %s"%(",".join(wilson_coefficients)))

features, weights = data_model_.getEvents(-1, wilson_coefficients=wilson_coefficients, feature_names = bit.feature_names)

logger.info ("Computing BIT predictions for %i events from these %i features: %s"%( features.shape[0], features.shape[1], ", ".join(bit.feature_names)) )
bit_predictions = bit.vectorized_predict( features )
logger.info ("Done.")

def make_combinations( coefficients ):
    return list(itertools.combinations_with_replacement(coefficients,1))+list(itertools.combinations_with_replacement(coefficients,2))

bit_predicted_weights = {comb:weights[()]*bit_predictions[:, bit.derivatives.index(bit.sort_comb(comb))] for comb in make_combinations(wilson_coefficients)}
bit_predicted_weights[()] = weights[()] 

class Lumi:
    def __init__( self, value, uncertainty):
        self.value      = value
        self.uncertainy = uncertainty

        self.nuisances             = ["lumi_unc"]

    def __call__( self, **kwargs ):
        #if any( [k not in self.nuisances for k in kwargs.keys()]):
        #    raise RuntimeError( "Unknown nuisance:", [k for k in kwargs.keys() if k not in self.nuisances] )
        unc_fac = self.uncertainy**kwargs["lumi_unc"] if "lumi_unc" in kwargs else 1 
        return self.value*unc_fac

class CrossSection:
    def __init__( self, value, uncertainty, weights, wilson_coefficients):
        self.value        = value
        self.wilson_coefficients = wilson_coefficients
        #self.combinations = [tuple()] + make_combinations(self.wilson_coefficients)  
        self.weights  = weights 

        self.norm       = np.sum(weights[()])
        self.uncertainy = uncertainty

        self.uncertainty_reweights = {}
        self.nuisances             = ["xsec"]

    @property
    def parameters( self ):
        return self.nuisances + self.wilson_coefficients

    def addLnN( self, name, weights):
        self.uncertainty_reweights[name] = ( weights )
        self.nuisances.append( name )

    def __call__( self, lin=False, **kwargs ):
      
        eft_keys = [k for k in self.wilson_coefficients if k in kwargs.keys()]
        sys_keys = [k for k in kwargs.keys() if k not in self.wilson_coefficients and k in self.nuisances]

        combs = [tuple()] + make_combinations(eft_keys)
        if lin:
            combs = [ comb for comb in combs if len(comb)<2 ]
        #print (combs)
        fac = np.array( [ functools.reduce( operator.mul, [ float(kwargs[c]) for c in comb ], 1 ) for comb in combs], dtype='float')
        #print (fac.shape, fac)
        #print (self.weights)
        weights = np.array( [self.weights[comb] for comb in combs]).transpose()
        #print (weights.shape, weights)
        eft_weights = np.matmul( weights, fac)
        #self.weights = np.matmul(np.stack( [self.weights[comb] for comb in combs], axis=1), fac)
                 
        if any( [k not in self.nuisances for k in sys_keys]):
            raise RuntimeError( "Unknown nuisance:", [k for k in sys_keys if k not in self.nuisances] )
        unc_fac = self.uncertainy**kwargs["xsec"] if "xsec" in kwargs else 1 
        return self.value*eft_weights/self.norm*unc_fac*np.prod( [self.uncertainty_reweights[key]**kwargs[key] for key in sys_keys if key!="xsec"], axis=0)

# Using truth
#crossSection = CrossSection( value=1, weights=weights, wilson_coefficients=wilson_coefficients, uncertainty=1.3)
#crossSection()

# using predictions
crossSection = CrossSection( value=20000, weights=bit_predicted_weights, wilson_coefficients=wilson_coefficients, uncertainty=1.5)
crossSection()
 
# lepton efficiency uncertainty # FIXME need a better data model
if 'recoLep0_pt' in data_model_.feature_names: 
    lepton_weights = 1.01 + 0.04*( features[:, data_model_.feature_names.index('recoLep0_pt')] -10 )/300
    lepton_weights[lepton_weights>1.1]=1.1
    lepton_weights[lepton_weights<0.9]=0.9

    crossSection.addLnN( "lepId", lepton_weights )

lumi         = Lumi( value=1, uncertainty=1.04)

class MakeHypothesis:
    def __init__(self, lumi, crossSection):
    
        self.lumi = lumi
        self.crossSection = crossSection    
        self.default_eft = make_eft()

        # Let's just make sure we have no naming clash
        assert len(set(lumi.nuisances).intersection(crossSection.nuisances))==0, "Ambigous parameters!"
        assert len(set(self.default_eft.keys()).intersection(crossSection.nuisances))==0, "Ambigous parameters!"
        assert len(set(self.default_eft.keys()).intersection(lumi.nuisances))==0, "Ambigous parameters!"

    def __call__(self, **kwargs):
        hypo = copy.deepcopy( self.default_eft )
        for factor in [lumi, crossSection]:
            hypo.update( {val:0 for val in factor.nuisances } )

        for key, value in kwargs.items():
            if key not in hypo.keys():
                raise RuntimeError("Unknown argument: %s" % key)
            hypo[key] = value

        return hypo 

makeHypo = MakeHypothesis(lumi, crossSection)

#null = makeHypo()
#alt  = makeHypo(ctGRe = 1)

from iminuit import Minuit
from iminuit.util import describe
from typing import Annotated

class AsimovNonCentrality:

    def __init__( self, lumi, crossSection, null, alt):
        self.lumi         = lumi
        self.crossSection = crossSection 
        self.null = null
        self.alt  = alt

        self._ignore   = []
        self._frozen   = []
        self._floating = []

        self.nuisances           = self.crossSection.nuisances + self.lumi.nuisances
        self.parameters          = self.nuisances + self.crossSection.wilson_coefficients
        self.wilson_coefficients = self.crossSection.wilson_coefficients

    def ignore( self, var ):
        if type(var)==type([]) or type(var)==type(()):
            for v in var:
                self.ignore(v)
        elif type(var)==type(""):
            if var not in self.null:
                raise RuntimeError
            if var not in self.alt:
                raise RuntimeError
            if var not in self.parameters:
                raise RuntimeError
            if var in self._frozen or var in self._floating:
                raise RuntimeError
            if var not in self._ignore:
                self._ignore.append( var )
        else:
            raise RuntimeError( "Don't know what to do with %r"%var )

    def freeze( self, **kwargs):
        for var, val in kwargs.items():
            if var not in self.null:
                raise RuntimeError
            if var not in self.alt:
                raise RuntimeError
            if var not in self.parameters:
                raise RuntimeError
            if var in self._ignore or var in self._floating:
                raise RuntimeError
            self.null[var] = val
            self._frozen.append(var)

    def float( self, var ):
        if type(var)==type([]) or type(var)==type(()):
            for v in var:
                self.float(v)
        elif type(var)==type(""):
            if var not in self.null:
                raise RuntimeError
            if var not in self.alt:
                raise RuntimeError
            if var not in self.parameters:
                raise RuntimeError
            if var in self._frozen or var in self._ignore:
                raise RuntimeError
            if var not in self._floating:
                self._floating.append( var )
        else:
            raise RuntimeError( "Don't know what to do with %r"%var )

    @property
    def variables(self):
        return [v for v in self.parameters if v not in self._ignore+self._frozen]

    @property
    def _penalized(self):
        return [v for v in self.parameters if v not in self._ignore+self._frozen+self.wilson_coefficients+self._floating]

    def __call__( self, null=None):
        _null = null if null is not None else self.null
        w_null = self.crossSection(**_null) 
        w_alt  = self.crossSection(**self.alt)
        l_null = self.lumi(**_null)
        l_alt  = self.lumi(**self.alt)

        disc_frac = (w_alt<0).sum()/len(w_alt)
        if disc_frac>0.01: 
            logger.warning ( "Discarded fraction of events because prediction is negative: %4.3f" % disc_frac ) 
        penalties = np.sum( [_null[var]**2 for var in self._penalized ] ) 
        return -2*( 
                      -l_null*w_null.sum() +l_alt*w_alt.sum() +
                    + (l_alt*w_alt*np.log(l_null/l_alt*np.where( (w_alt>0) & (w_null>0), w_null/w_alt, 1))).sum() 
                ) + penalties

class MinuitInterface:
    # 1 for LSQ, 0.5 for NLL: 
    # https://iminuit.readthedocs.io/en/stable/reference.html#iminuit.Minuit.errordef
    errordef = Minuit.LEAST_SQUARES  # for Minuit to compute errors correctly

    def __init__( self, asimov_object):

        self._parameters   = {v:None for v in asimov_object.variables}
        self.asimov_object = asimov_object

    def __call__(self, *par):
        hypo = copy.deepcopy( self.asimov_object.null)
        hypo.update( {k:v for k, v in zip(self.asimov_object.variables, par)} )

        logger.debug ("Calling Asimov Object")
        for name, val in hypo.items():
            post_string = " "
            if name in self.asimov_object._ignore:
                post_string+="[ignored]" 
            if name in self.asimov_object._frozen:
                post_string+="[frozen]" 
            if name in self.asimov_object._floating:
                post_string+="[floating]" 
            if name in self.asimov_object._penalized:
                post_string+="[penalized]" 

            logger.debug ("%15s"%name + " %+4.3f (null) %+4.3f (alt)"%( val, self.asimov_object.alt[name]) + post_string)

        res = self.asimov_object(hypo)
        return res

    @property
    def defaults( self ):
        return (self.asimov_object.null[v] for v in self.asimov_object.variables) 

    def fit( self ):
        m = Minuit(self, *self.defaults)

        m.migrad()

        logger.info (m)

        return m.values.to_dict()

results = []

#assert False, ""

for wc1_val in np.linspace(args.low1,args.high1,args.nBins):
    for wc2_val in ( np.linspace(args.low2,args.high2,args.nBins) if args.wc2 is not None else [None]):

        logger.info ("WC: %s=%3.2f"%(args.wc1, wc1_val) +" "+("%s=%3.2f"%(args.wc2, wc2_val) if  args.wc2 is not None else "" ))
        asimovNonCentrality = AsimovNonCentrality(
            lumi=lumi, 
            crossSection=crossSection,
            null = makeHypo(),
            alt  = makeHypo(),
            )

        frozen_param = {args.wc1:wc1_val}
        if args.wc2 is not None:
            frozen_param[args.wc2] = wc2_val
        #asimovNonCentrality.ignore( asimovNonCentrality.nuisances )
        asimovNonCentrality.freeze( **frozen_param )
        asimovNonCentrality.float("xsec")

        prefit = asimovNonCentrality()

        asimovNonCentrality_interface = MinuitInterface( asimovNonCentrality )

        fit_result = asimovNonCentrality_interface.fit()

        logger.info ("Prefit %5.4f " %prefit)
        non_centrality = asimovNonCentrality( makeHypo(**fit_result, **frozen_param) )
        logger.info ("Postfit %5.4f" % non_centrality)

        median_qTheta_Alt = scipy.stats.ncx2.median(df=1,nc=non_centrality)
        logger.info("median_qTheta_Alt %5.4f" % median_qTheta_Alt)

        results.append( {'wc1':args.wc1, 'val1':wc1_val, 'wc2':args.wc2, 'val2':wc2_val, 'prefit':prefit, 'postfit':non_centrality, 'median_qTheta_Alt':median_qTheta_Alt} )

pickle.dump( results, open(results_filename, 'wb') ) 
