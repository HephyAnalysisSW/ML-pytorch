import sys, os
import pickle
import itertools
import numpy as np

sys.path.insert(0, '..')
sys.path.insert(0, '../..')

from tools.user import results_directory

import Modeling

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--overwrite', action='store_true', default=False, help="overwrite results")
parser.add_argument('--logLevel',  action='store', nargs='?',  choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'],   default='INFO', help="Log level for logging" )

parser.add_argument('--version',  action='store',   default = 'v1_1D', help="Version")

parser.add_argument('--marginalized',   action='store_true',   help="Marginalized?")
parser.add_argument('--th',             action='store_true',  help="Theoretical systematics?")
parser.add_argument('--mod',            action='store_true',  help="Modeling systematics?")
parser.add_argument('--exp',            action='store_true',  help="Experimental systematics?")
parser.add_argument('--top_kinematics',       action='store_true')
parser.add_argument('--lepton_kinematics',    action='store_true')
parser.add_argument('--asymmetry',            action='store_true')
parser.add_argument('--spin_correlation',     action='store_true')

args = parser.parse_args()

# Logger
import tools.logger as logger_
logger = logger_.get_logger(args.logLevel, logFile = None )

sub_directory = []

logger.info ("LOADING DATA MODEL: TT2l_EFT_delphes")
import data_models.TT2l_EFT_delphes as data_model
logger.info("All Wilson coefficients: "+",".join( data_model.wilson_coefficients ) )

logger.info ("LOADING BIT")
from BIT.MultiBoostedInformationTree import MultiBoostedInformationTree

bit_id = "TK_%r_LK_%r_CA_%r_SC_%r"%( args.top_kinematics, args.lepton_kinematics, args.asymmetry, args.spin_correlation)


logger.info ("LOADING DATA MODEL: TT2l_EFT_delphes") 
import data_models.TT2l_EFT_delphes as data_model
logger.info("All Wilson coefficients: "+",".join( data_model.wilson_coefficients ) )

output_directory = os.path.join( results_directory, "TT2lUnbinned/ref_toy", args.version )

bit_name = "/groups/hephy/cms/robert.schoefbeck/NN/models/multiBit_TT2l_EFT_delphes_%s_v1.1_coeffs_ctGRe_ctGIm_cQj18_cQj38_ctj8_nTraining_-1_nTrees_300.pkl"%bit_id
logger.info ("Using BIT training: %s", bit_name)
bit = MultiBoostedInformationTree.load(bit_name)
logger.info ("We have predictions for these BIT derivatives: %s", bit.derivatives)

#wilson_coefficients = list(set(sum(map(list,bit.derivatives), [])))
## Sanity: Check that the Wilson coefficients from the data model were learnt
#for coeff in wilson_coefficients:
#    if coeff not in data_model.wilson_coefficients:
#        logger.info ("BIT model contains derivative we don't have in the data model.")
#        raise RuntimeError

all_features = bit.feature_names

# STEERABLES
WCs = ['ctGRe', 'ctGIm', 'cQj18', 'cQj38', 'ctj8']
inclusiveExpectation = 0.309*137*1000

combinations = Modeling.make_combinations(WCs)

hypothesis = Modeling.Hypothesis( [Modeling.ModelParameter( "hf", val=0, isPOI=True )] ) 

all_features.extend(['ht', 'jet0_pt', 'jet1_pt', 'nrecoJet'])
bTagSys_hf       = Modeling.BPTUncertainty("bTagSys_hf", "/groups/hephy/cms/robert.schoefbeck/NN/models/BPT/BPT_TTLep_bTagSys_v4_Autumn18_hf_nTraining_-1_nTrees_300.pkl", renameParams="hf")

#hypothesis.append(bTagSys_hf         .makePenalizedNuisances())

hypothesis.print()

logger.info ("LOADING DATA") 
data_model_ = data_model.DataModel()

trueSMEFTData = Modeling.NormalizedSMEFTData( 
                    *data_model.DataModel().getEvents(-1, wilson_coefficients=[], 
                            feature_names = list(set(all_features)), feature_dicts=True), 
                     inclusiveExpectation=inclusiveExpectation )

bTagSys_hf         .initialize( trueSMEFTData, translation = {"nJetGood":"nrecoJet", "l1_pt":"recoLep0_pt", "l2_pt":"recoLep1_pt"}) 

def model_weights( hypo ):
    #TTBAR_SMEFT_weight = Modeling.SMEFTweight(bitPrediction, hypo)

    res = np.ones( trueSMEFTData.features.shape[0] ) 

    btag = bTagSys_hf(hypo)
    res *= btag

    return res

    #return lumiUnc(hypo)*leptonSF(hypo)*jec*btag*( xsecUnc(hypo)*TTBAR_SMEFT_weight*scaleUnc_2D(hypo)*PS_FSR(hypo)*PS_ISR(hypo)*MGvsPow(hypo) + scale_DYNorm_fraction*(alpha_DYUnc_norm**hypo['gDY'].val)*DYUnc_norm_weight*TTBAR_SM_weight )

#hypothesis = hypothesis.cloneFreeze(ctGIm=0, xsec=0, ren=0, fac=0,gDY=0,fsr=0,isr=0,gPowheg=0,lSF=0,jesAbsBias=0,jesFlavQCD=0,jesPU=0,jesRelBal=0,jesECAL=0)
#asimov = Modeling.AsimovNonCentrality( model_weights, 
#    null=hypothesis.cloneFreeze(ctGRe=.1), 
#    alt =hypothesis)#, alt=hypothesis.cloneModify(ctGRe=1, ctGIm=1))

results = {}
#logger.info("Fitting %s", " ".join( ["%s=%3.2f"%(w,v) for w, v in model_point_dict.items()]) )
asimov = Modeling.AsimovNonCentrality( model_weights, 
    null=hypothesis.cloneFreeze(), 
    alt =hypothesis)#, alt=hypothesis.cloneModify(ctGRe=1, ctGIm=1))

norms = []
vals = []
sigmas = []
for n_toy in range(1000):
    size=10000
    choice = np.zeros(trueSMEFTData.weights[()].shape[0], dtype='bool')
    choice[:size] = True
    np.random.shuffle(choice)

    toy = Modeling.Toy( 
        expectation = size,
        choice = choice,
        model_weight_func = model_weights, 
        null=hypothesis.cloneFreeze(), 
        alt =hypothesis)#, alt=hypothesis.cloneModify(ctGRe=1, ctGIm=1))

    minuit = Modeling.MinuitInterface( toy )

    res = minuit.fit()
    val = res['minuit'].values[0]
    sigma =  np.sqrt(res['minuit'].covariance[0,0])

    norms.append(val/sigma)
    vals.append(val)
    sigmas.append(sigma)

    print (val/sigma, val, sigma)

import ROOT
import tools.helpers
h = tools.helpers.make_TH1F(np.histogram(norms, np.linspace(-3,3,11)))
for i_b in range(1, 1+h.GetNbinsX()):
    h.SetBinError( i_b, np.sqrt(h.GetBinContent(i_b)))
g = ROOT.TF1("f2", "300*[0]*exp(-((x-[1]))**2/2.)",-3,3)
h.Fit(g,"S")

h.Draw()
c1.Print("h3.png")
