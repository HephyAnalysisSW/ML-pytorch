import sys
import itertools

sys.path.insert(0, '..')
sys.path.insert(0, '../..')

# Logger
import tools.logger as logger_
logger = logger_.get_logger("INFO", logFile = None )

import Modeling

logger.info ("LOADING DATA MODEL: TT2l_EFT_delphes") 
import data_models.TT2l_EFT_delphes as data_model
logger.info("All Wilson coefficients: "+",".join( data_model.wilson_coefficients ) )

logger.info ("LOADING BIT") 
from BIT.MultiBoostedInformationTree import MultiBoostedInformationTree

bit_name = "/groups/hephy/cms/robert.schoefbeck/NN/models/multiBit_TT2l_EFT_delphes_TK_False_LK_False_CA_False_SC_False_v1.1_coeffs_ctGRe_ctGIm_cQj18_cQj38_ctj8_nTraining_-1_nTrees_300.pkl"
logger.info ("Using BIT training: %s", bit_name)
bit = MultiBoostedInformationTree.load(bit_name)
logger.info ("We have predictions for these BIT derivatives: %s", bit.derivatives)

wilson_coefficients = list(set(sum(map(list,bit.derivatives), [])))
# Sanity: Check that the Wilson coefficients from the data model were learnt
for coeff in wilson_coefficients:
    if coeff not in data_model.wilson_coefficients:
        logger.info ("BIT model contains derivative we don't have in the data model.")
        raise RuntimeError 

all_features = bit.feature_names 

# systematics
scale_2D     = Modeling.BPTUncertainty( "scale_2D", "/groups/hephy/cms/robert.schoefbeck/NN/models/BPT/BPT_TTLep_scale_2D_paper_v4_Autumn18_nTraining_-1_nTrees_300.pkl" )
all_features+= scale_2D.feature_names

lumi         = Modeling.MultiplicativeUncertainty( "lumi", 1.05 ) 
xsec         = Modeling.MultiplicativeUncertainty( "xsec", 1.20 ) 

# Steerables
WC_POIs              = ['ctGRe']
WC_Nuis              = ['ctGIm']
inclusiveExpectation = 1000

combinations = Modeling.make_combinations(WC_POIs+WC_Nuis)

logger.info ("LOADING DATA") 
data_model_ = data_model.DataModel()

trueSMEFTData = Modeling.NormalizedSMEFTData( 
                    *data_model.DataModel().getEvents(-1, wilson_coefficients=WC_POIs+WC_Nuis, 
                            feature_names = all_features, feature_dicts=True), 
                     inclusiveExpectation=inclusiveExpectation )

bitPrediction = trueSMEFTData.BITPrediction ( 
                        bit          = bit,
                        combinations = combinations)
# BIT computation:
#Modeling.SMEFTweight(bitPrediction, hypothesis.modify(ctGRe=.4)).sum()

# scale systematic
scale_2D.initialize( trueSMEFTData )


hypothesis = Modeling.Hypothesis(  
                            list(map( lambda p: Modeling.ModelParameter(p, isWC=True, isPOI=True), WC_POIs )) +
                            list(map( lambda p: Modeling.ModelParameter(p, isWC=True, ), WC_Nuis )) +
                            scale_2D.makePenalizedNuisances() +
                            lumi.makePenalizedNuisances()+ 
                            xsec.makePenalizedNuisances() 
                            )

#trueSMEFTData.SMEFTweight(hypothesis.modify(ctGRe=0.,ctGIm=0.)).sum()
hypothesis.print()

def model_weights( hypo ):
    return xsec(hypo)*lumi(hypo)*Modeling.SMEFTweight(bitPrediction, hypo)*scale_2D(hypo)

asimov = Modeling.AsimovNonCentrality( model_weights, null=hypothesis, alt=hypothesis.cloneModify(ctGRe=1, ctGIm=1))

minuit = Modeling.MinuitInterface( asimov )

res = minuit.fit()

res.print()

