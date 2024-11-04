import sys
import numpy as np
import copy

sys.path.insert(0, '..')
sys.path.insert(0, '../..')

import logging
logger = logging.getLogger('ML')

import scipy
def median_expected_pValue(df, nc):
    return 1-scipy.stats.chi2.cdf( scipy.stats.ncx2.median(df=df,nc=nc), df=df )

def make_combinations( coefficients ):
    import itertools
    return list(itertools.combinations_with_replacement(coefficients,1))+list(itertools.combinations_with_replacement(coefficients,2))

class ModelParameter:
    def __init__( self, name, val=0., isWC=False, isPOI=False, isFrozen=False, isPenalized=False, isIgnored=False):
        self.name       = name
        self.isFrozen   = isFrozen
        self.isPenalized= isPenalized
        self.isPOI      = isPOI
        self.isWC       = isWC
        self.isIgnored  = isIgnored
        self.val        = val

    def __repr__( self ):
        status = []
        if self.isWC:
            status.append('WC')
        if self.isPOI:
            status.append('POI')
        else:
            status.append('Nuis.')
        if self.isFrozen:
            status.append('frozen')
        if self.isPenalized:
            status.append('pen.')
        if self.isIgnored:
            status.append('ign.')
        return '<'+self.name+"("+",".join(status)+")={:e}".format(self.val)+'>'

    def __str__( self ):
        return self.__repr__().rstrip('<').lstrip('>')

    def __call__(self):
        return self.val

    @classmethod
    def makePenalizedNuisance( cls, name):
        return cls(name=name, isPenalized=True)

class MultiplicativeUncertainty:

    def __init__( self, name, alpha ):
        self.name       = name
        self.parameters = [name]
        self.alpha      = alpha

    def makePenalizedNuisances( self ):
        return [ModelParameter.makePenalizedNuisance(self.name)]

    def __call__( self, hypothesis ):
        return self.alpha**(hypothesis[self.name].val)

from BPT.BoostedParametricTree import BoostedParametricTree
class BPTUncertainty:

    def __init__( self, name, bpt_file, renameParams=None):
    
        self.name = name
        self.bpt  = BoostedParametricTree.load( bpt_file )
        self.initialized = False

        # translate parameter names
        self.translate_param_names = {}
        self.parameters  = []
        for i_p, p in enumerate(self.bpt.parameters):
            if renameParams is not None:
                if len(self.bpt.parameters)==1:
                    new_name = renameParams
                else:
                    new_name = renameParams+"_"+str(i_p)
            else:
                new_name = p
            self.translate_param_names[p] = new_name 
            self.translate_param_names[new_name] = p
            self.parameters.append( new_name )
 
    def makePenalizedNuisances( self ):
        return list(map( ModelParameter.makePenalizedNuisance, self.parameters ))

    @property
    def feature_names( self ):
        return self.bpt.feature_names

    def initialize( self, dataset, translation = None):
        logger.info( "Initializing BPT predictions for %s and %i events", self.name, len( dataset ) )
        self.bpt_predictions = dataset.BPTPrediction( self.bpt, translation=translation)
        self.initialized     = True

    def __call__( self, hypothesis):

        if any( [ not param in hypothesis for param in self.parameters ]):
            raise RuntimeError( "A parameter is missing in the hypothesis. Need %s" % (",".join( self.parameters) ) )
 
        if not self.initialized:
            raise RuntimeError( "Must initialize BPTUncertainty %s with a dataset" % self.name )

        return np.exp(np.sum( [ np.prod([hypothesis[self.translate_param_names[var]].val for var in comb])*self.bpt_predictions[:,i_comb] for i_comb, comb in enumerate(self.bpt.combinations)], axis=0))

class Hypothesis:
    def __init__(self, parameters, name=None):

        self.parameters = parameters
        self.name = name 

        self.check()

    def append(self, parameters):
        if type(parameters)==type([]): 
            self.parameters.extend(parameters)
        else:
            self.parameters.append(parameters)

    def check( self ):
        # Sanity
        for p in self.parameters:
            if p.isPenalized and p.isPOI:
                p.isPenalized=False
                logger.warning("Warning: A POI %s was penalized. Remove the penalty, because it is a POI.", p.name)

        # Check that all parameters have unique name
        p_names = [ p.name for p in self.parameters]
        if len(p_names)<len(set(p_names)):
            logger.info( "All parameters:", ",".join( p_names ) )
            raise RuntimeError("Ambigous parameter names found (see above)!")

    def __contains__( self, key ):
        return (key in [p.name for p in self.parameters])

    def __getitem__( self, key ):
        for p in self.parameters:
            if key==p.name:
                return p

        raise AttributeError("'Hypothesis' has no key %s"%key) 

    @property
    def POIs( self ):
        return [p for p in self.parameters if p.isPOI]

    @property
    def WCs( self ):
        return [p for p in self.parameters if p.isWC]

    @property
    def penalized( self ):
        return [p for p in self.parameters if p.isPenalized]

    @property
    def nuisances( self ):
        return [p for p in self.parameters if not p.isPOI]

    def print( self ):
        print("Hypothesis (%s)"%( "unnamed" if self.name is None else self.name))
        print()
        for i_p, p in enumerate(self.POIs):
            print ("%00i"%i_p, p)
        print()
        for i_p, p in enumerate(self.nuisances):
            print ("%00i"%(i_p+len(self.POIs)), p)

    def modify( self, **kwargs ):
        for key, val in kwargs.items():
            param = self[key]
            if param.isFrozen:
                raise RuntimeError( "Can not modify %s. It is frozen."%key )
            self[key].val = val 
        return self

    def cloneModify( self, **kwargs ):
        res = copy.deepcopy( self )
        return res.modify(**kwargs)

    def clone( self ):
        return copy.deepcopy( self )

    def cloneSM( self ):
        res = copy.deepcopy( self )
        for param in res.parameters:
            if param.val!=0.:
                if param.isFrozen:
                    logger.warning("Set a frozen parameter %s=%3.2f to zero!", param.name, param.val)
                param.val = 0.
        return res

    def cloneFreeze( self, **kwargs):
        res = copy.deepcopy(self)
        for key, val in kwargs.items():
            res[key].val = val
            res[key].isFrozen = True
        return res

class NormalizedSMEFTData:

    def __init__( self, features, weights, inclusiveExpectation=None):

        if type(features)==type({}):
            self.feature_names = list(features.keys())
            self.features      = np.column_stack( [features[k] for k in self.feature_names] )
        else:
            self.features = features

        self.weights  = weights

        if inclusiveExpectation is not None:
            scaling = inclusiveExpectation/weights[()].sum()
            self.weights = {k:v*scaling for k,v in self.weights.items()}
            logger.info( "Scaled %i input events to an expectation of %3.2f", len(self.weights[()]), inclusiveExpectation)

    def __len__( self ):
        return self.features.shape[0] 

    def get_features_by_name( self, feature_names ):
        if not hasattr( self, "feature_names"):
            logger.error( "Feature names were not provided to this object" )
            raise RuntimeError

        unknown = [f for f in feature_names if f not in self.feature_names]
        if len(unknown)>0:
            logger.error( "Do not know about these features: %s", ",".join(unknown) )
            raise RuntimeError
        return self.features[:, [self.feature_names.index( feature ) for feature in feature_names]]
             
    def BITPrediction( self, bit, combinations):

        #assert features.shape[1]==len(bit.feature_names), "Length of features not consistent!"
        #assert len(SM_weights.shape)==1, "SM_weights are not a 1D array"
        #assert len(features)==SM_weights.shape[0], "Length of features not consistent!"

        logger.info("Computing BIT predictions for %i events from these %i features: %s"%( len(self), len(bit.feature_names), ", ".join(bit.feature_names)) )
        bit_predictions = bit.vectorized_predict( self.get_features_by_name( bit.feature_names ) if hasattr( self, "feature_names") else self.features )
        logger.info ("Done.")

        bit_predicted_weights = {comb:self.weights[()]*bit_predictions[:, bit.derivatives.index(bit.sort_comb(comb))] for comb in combinations}
        bit_predicted_weights[()] = self.weights[()] 
    
        return bit_predicted_weights  

    def BPTPrediction( self, bpt, translation=None):
        if translation == None:
            feature_names = bpt.feature_names
        else:
            feature_names = [translation[feature] if feature in translation else feature for feature in bpt.feature_names]
            logger.info("Using translation: %s", " ".join( ["%s -> %s"%(key, val) for key, val in translation.items()]) )
        logger.info("Computing BPT predictions for %i events from these %i features: %s"%( len(self), len(bpt.feature_names), ", ".join(bpt.feature_names)) )
        return bpt.vectorized_predict( self.get_features_by_name( feature_names ) )

def SMEFTweight( weights, hypothesis, relative=False):
    import numpy as np
    if relative:
        return np.sum( [np.prod([hypothesis[wc].val for wc in key])*value/weights[()] for key, value in weights.items()], axis=0 )
    else:
        return np.sum( [np.prod([hypothesis[wc].val for wc in key])*value for key, value in weights.items()], axis=0 )

class AsimovNonCentrality:

    def __init__( self, model_weight_func, null, alt=None, debug=False):

        self.model_weight_func = model_weight_func
        self.null = null
        self.null.name  = "Null (BSM)"
        self.alt        = null.cloneSM() if alt is None else alt
        self.alt.name   = "Alternate (SM)"

        self.debug = debug
        if self.debug:
            print ("End of constructor Null:")
            self.null.print()
            print ("End of constructor Alt:")
            self.alt.print()

    def __call__( self, null=None):
        _null = null if null is not None else self.null

        w_null = self.model_weight_func(_null) 
        w_alt  = self.model_weight_func(self.alt) 

        neg_frac = (w_null<0).sum()/len(w_null)
        #if return_neg_frac: return neg_frac

        if neg_frac>0.01: 
            logger.warning ( "Discarded fraction of events because prediction is negative: %4.3f" % neg_frac ) 
        penalties = np.sum( [par.val**2 for par in _null.penalized  ] ) 
        return -2*( 
                      -w_null.sum() + w_alt.sum() +
                    + (w_alt*np.log(np.where( (w_alt>0) & (w_null>0), w_null/w_alt, 1))).sum() 
                ) + penalties

    @property
    def variables(self):
        return [v for v in self.null.parameters if not (v.isFrozen or v.isIgnored)]

class Toy:

    def __init__( self, expectation, choice, model_weight_func, null, alt=None, debug=False):

        self.model_weight_func = model_weight_func
        self.null = null
        self.null.name  = "Null (BSM)"
        self.alt        = null.cloneSM() if alt is None else alt
        self.alt.name   = "Alternate (SM)"

        self.expectation= expectation
        self.choice     = choice 

        self.debug = debug
        if self.debug:
            print ("End of constructor Null:")
            self.null.print()
            print ("End of constructor Alt:")
            self.alt.print()

    def __call__( self, null=None):
        _null = null if null is not None else self.null

        w_null = self.model_weight_func(_null) 
        w_alt  = self.model_weight_func(self.alt) 

        sim_scale  = self.expectation/w_alt.sum()

        neg_frac = (w_null<0).sum()/len(w_null)
        #if return_neg_frac: return neg_frac

        if neg_frac>0.01: 
            logger.warning ( "Discarded fraction of events because prediction is negative: %4.3f" % neg_frac ) 
        penalties = np.sum( [par.val**2 for par in _null.penalized  ] ) 
        return -2*( 
                      sim_scale*(-w_null.sum() + w_alt.sum()) +
                    + (np.log(np.where( (w_alt[self.choice]>0) & (w_null[self.choice]>0), w_null[self.choice]/w_alt[self.choice], 1))).sum() 
                ) + penalties

    @property
    def variables(self):
        return [v for v in self.null.parameters if not (v.isFrozen or v.isIgnored)]


from iminuit import Minuit
from iminuit.util import describe
from typing import Annotated

class MinuitInterface:
    # 1 for LSQ, 0.5 for NLL: 
    # https://iminuit.readthedocs.io/en/stable/reference.html#iminuit.Minuit.errordef
    errordef = Minuit.LEAST_SQUARES  # for Minuit to compute errors correctly

    def __init__( self, asimov_object, debug=False):
        self.asimov_object = asimov_object
        self.debug         = debug

    def __call__(self, *par):
        hypo = self.asimov_object.null.clone()
        for i_var, var in enumerate(self.asimov_object.variables):
            hypo[var.name].val = par[i_var]
        #hypo.update( {k:v for k, v in zip(self.asimov_object.variables, par)} )

        logger.debug ("Calling Asimov Object")
        if self.debug:
            hypo.print()

        res = self.asimov_object(null=hypo)
        return res

    def fit( self ):

        res = {'preFit_hypothesis':self.asimov_object.null.clone(), 'preFit_nonCentrality':self.asimov_object(self.asimov_object.null),}
        if len( self.asimov_object.variables) == 0:
            logger.warning( "No parameters to fit!")
            res['minuit'] = None
            return res

        m = Minuit(self, *[par.val for par in self.asimov_object.variables], name=[par.name for par in self.asimov_object.variables])

        m.migrad()

        logger.info (m)

        res.update( {
            'minuit':m, 
            'hypothesis':self.asimov_object.null.cloneModify( **{ par.name:val for par, val in zip(self.asimov_object.variables, list(m.values))} ),
            'nonCentrality':m.fval,
        })
        return res
