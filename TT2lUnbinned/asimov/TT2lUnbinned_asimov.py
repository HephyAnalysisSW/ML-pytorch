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

parser.add_argument("--wc1",   action="store",      default = "ctGRe", help="Which wilson coefficient?")
parser.add_argument("--low1",  action="store",      default = -0.5, type=float, help="Lower range")
parser.add_argument("--high1", action="store",      default = 0.5, type=float, help="Upper range")
parser.add_argument("--nBins1", action="store",      default = 21, type=int, help="Upper range")

parser.add_argument("--wc2",   action="store",      default = None, help="Which second wilson coefficient?")
parser.add_argument("--low2",  action="store",      default = None, type=float, help="Lower range")
parser.add_argument("--high2", action="store",      default = None, type=float, help="Upper range")
parser.add_argument("--nBins2", action="store",     default = None, type=int, help="Upper range")

parser.add_argument("--wc2_val",  action="store",      default = None, type=float, help="Value of the 2nd Wilson coefficient")

args = parser.parse_args()

# Logger
import tools.logger as logger_
logger = logger_.get_logger(args.logLevel, logFile = None )

# Idea: If an array for the 2nd WC is provided, append to a jobs file instead!
if args.high2 is not None:

    with open('jobs.sh', 'a+') as job_file:
        for wc2_val in np.linspace(args.low2, args.high2, args.nBins2):

            arguments = ["--version %s"%args.version]
            if args.overwrite:
                arguments.append("--overwrite")
            if args.marginalized:
                arguments.append("--marginalized")
            if args.th:
                arguments.append("--th")
            if args.mod:
                arguments.append("--mod")
            if args.exp:
                arguments.append("--exp")
            if args.top_kinematics:
                arguments.append("--top_kinematics")
            if args.lepton_kinematics:
                arguments.append("--lepton_kinematics")
            if args.asymmetry:
                arguments.append("--asymmetry")
            if args.spin_correlation:
                arguments.append("--spin_correlation")
           
            arguments.append("--wc1 %s"%args.wc1) 
            arguments.append("--low1 %.5f"%args.low1 ) 
            arguments.append("--high1 %.5f"%args.high1 ) 
            arguments.append("--nBins1 %i"%args.nBins1 ) 

            arguments.append("--wc2 %s"%args.wc2) 
            arguments.append("--wc2_val %.5f"%wc2_val ) 

            job_file.write('python TT2lUnbinned_asimov.py '+' '.join(arguments)+'\n')
    logger.info( "Appended to jobs.sh" )
    sys.exit(0)

sub_directory = []

# Uncertainty grouping
lSF    = False
JEC    = False
BTag   = False
lumi   = False
scale  = False
DYnorm = False
xsec   = False
PS     = False
MG     = False

if args.marginalized:
    sub_directory.append("marginalized")

if args.th:
    sub_directory.append("th")
    scale  = True
    DYnorm = False
    xsec   = True

if args.mod:
    sub_directory.append("mod")
    PS     = True
    MG     = True

if args.exp:
    sub_directory.append("exp")
    lSF    = True
    JEC    = True
    BTag   = True
    lumi   = True


logger.info ("LOADING DATA MODEL: TT2l_EFT_delphes") 
import data_models.TT2l_EFT_delphes as data_model
logger.info("All Wilson coefficients: "+",".join( data_model.wilson_coefficients ) )

logger.info ("LOADING BIT") 
from BIT.MultiBoostedInformationTree import MultiBoostedInformationTree

bit_id = "TK_%r_LK_%r_CA_%r_SC_%r"%( args.top_kinematics, args.lepton_kinematics, args.asymmetry, args.spin_correlation)

if "True" in bit_id:
    args.version+= "_"+bit_id

output_directory = os.path.join( results_directory, "TT2lUnbinned/limits", args.version )

#bit_name = "/groups/hephy/cms/robert.schoefbeck/NN/models/multiBit_TT2l_EFT_delphes_TK_False_LK_False_CA_False_SC_False_v1.1_coeffs_ctGRe_ctGIm_cQj18_cQj38_ctj8_nTraining_-1_nTrees_300.pkl"
bit_name = "/groups/hephy/cms/robert.schoefbeck/NN/models/multiBit_TT2l_EFT_delphes_%s_v1.1_coeffs_ctGRe_ctGIm_cQj18_cQj38_ctj8_nTraining_-1_nTrees_300.pkl"%bit_id
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

# STEERABLES
WCs = ['ctGRe', 'ctGIm', 'cQj18', 'cQj38', 'ctj8']
inclusiveExpectation = 0.309*137*1000

combinations = Modeling.make_combinations(WCs)

hypothesis = Modeling.Hypothesis( list(map( lambda p: Modeling.ModelParameter(p, isWC=True), WCs )) ) 

hypothesis[args.wc1].isPOI = True
if args.wc2 is not None:
    hypothesis[args.wc2].isPOI = True

for wc in WCs:
    if wc==args.wc1:
        continue
    if args.wc2 is not None and wc==args.wc2:
        continue
    if not args.marginalized:
        hypothesis[wc].isFrozen = True
 
# SYSTEMATICS
if lumi:
    lumiUnc         = Modeling.MultiplicativeUncertainty( "lumi", 1.05 )
    hypothesis.append( lumiUnc.makePenalizedNuisances() )

if xsec:
    xsecUnc         = Modeling.MultiplicativeUncertainty( "xsec", 1.20 ) 
    hypothesis.append( xsecUnc.makePenalizedNuisances() )

if scale:
    # Scale
    scaleUnc_2D     = Modeling.BPTUncertainty( "scaleUnc_2D", "/groups/hephy/cms/robert.schoefbeck/NN/models/BPT/BPT_TTLep_scale_2D_paper_v4_Autumn18_nTraining_-1_nTrees_300.pkl" )
    all_features+= scaleUnc_2D.feature_names

    hypothesis.append(scaleUnc_2D.makePenalizedNuisances())

if DYnorm:
    # DY norm
    DYUnc_norm     = Modeling.BPTUncertainty( "DYUnc_norm", "/groups/hephy/cms/robert.schoefbeck/NN/models/BPT/BPT_delphes_TTLep_DY_v4.7_nTraining_-1_nTrees_300.pkl" )
    all_features  += DYUnc_norm.feature_names
    alpha_DYUnc_norm = 1.2

    hypothesis.append( DYUnc_norm.makePenalizedNuisances() )

if PS:
    # PS weights
    PS_FSR     = Modeling.BPTUncertainty( "PS_FSR", "/groups/hephy/cms/robert.schoefbeck/NN/models/BPT/BPT_delphes_TTLep_PS_FSR_scale_v4.1_nTraining_-1_nTrees_300.pkl" )
    PS_ISR     = Modeling.BPTUncertainty( "PS_ISR", "/groups/hephy/cms/robert.schoefbeck/NN/models/BPT/BPT_delphes_TTLep_PS_ISR_scale_v4.1_nTraining_-1_nTrees_300.pkl" )
    all_features += PS_FSR.feature_names

    hypothesis.append(PS_FSR.makePenalizedNuisances())
    hypothesis.append(PS_ISR.makePenalizedNuisances())

if MG:
    # Madgraph vs. Powheg modeling
    MGvsPow    = Modeling.BPTUncertainty( "MGvsPow", "/groups/hephy/cms/robert.schoefbeck/NN/models/BPT/BPT_delphes_TTLep_MG_vs_Pow_v4.1_nTraining_-1_nTrees_300.pkl")
    hypothesis.append( MGvsPow.makePenalizedNuisances() )

if lSF:
    leptonSF   = Modeling.BPTUncertainty( "leptonSF", "/groups/hephy/cms/robert.schoefbeck/NN/models/BPT/BPT_TTLep_leptonSF_v4_for_paper_Autumn18_nTraining_-1_nTrees_300.pkl", renameParams = "lSF")
    hypothesis.append( leptonSF.makePenalizedNuisances() )

if JEC:

    all_features.append( "nrecoJet" )

    jesAbsoluteMPFBias  = Modeling.BPTUncertainty("jesAbsoluteMPFBias", "/groups/hephy/cms/robert.schoefbeck/NN/models/BPT/BPT_TTLep_JERC_linear_paper_v4.1_jesAbsoluteMPFBias_nTraining_-1_nTrees_300.pkl", renameParams = "jesAbsBias")
    jesFlavorQCD        = Modeling.BPTUncertainty("jesFlavorQCD", "/groups/hephy/cms/robert.schoefbeck/NN/models/BPT/BPT_TTLep_JERC_linear_paper_v4.1_jesFlavorQCD_nTraining_-1_nTrees_300.pkl", renameParams = "jesFlavQCD")
    jesPileUpDataMC     = Modeling.BPTUncertainty("jesPileUpDataMC", "/groups/hephy/cms/robert.schoefbeck/NN/models/BPT/BPT_TTLep_JERC_linear_paper_v4.1_jesPileUpDataMC_nTraining_-1_nTrees_300.pkl", renameParams = "jesPU")
    jesRelativeBal      = Modeling.BPTUncertainty("jesRelativeBal", "/groups/hephy/cms/robert.schoefbeck/NN/models/BPT/BPT_TTLep_JERC_linear_paper_v4.1_jesRelativeBal_nTraining_-1_nTrees_300.pkl", renameParams = "jesRelBal")
    jesSinglePionECAL   = Modeling.BPTUncertainty("jesSinglePionECAL", "/groups/hephy/cms/robert.schoefbeck/NN/models/BPT/BPT_TTLep_JERC_linear_paper_v4.1_jesSinglePionECAL_nTraining_-1_nTrees_300.pkl", renameParams = "jesECAL")

    hypothesis.append( jesAbsoluteMPFBias .makePenalizedNuisances())
    hypothesis.append( jesFlavorQCD       .makePenalizedNuisances())
    hypothesis.append( jesPileUpDataMC    .makePenalizedNuisances())
    hypothesis.append( jesRelativeBal     .makePenalizedNuisances())
    hypothesis.append( jesSinglePionECAL  .makePenalizedNuisances())

if BTag:
    all_features.extend(['ht', 'jet0_pt', 'jet1_pt'])
    bTagSys_hf       = Modeling.BPTUncertainty("bTagSys_hf", "/groups/hephy/cms/robert.schoefbeck/NN/models/BPT/BPT_TTLep_bTagSys_v4_Autumn18_hf_nTraining_-1_nTrees_300.pkl", renameParams="hf")
    bTagSys_cferr1   = Modeling.BPTUncertainty("bTagSys_cferr1", "/groups/hephy/cms/robert.schoefbeck/NN/models/BPT/BPT_TTLep_bTagSys_v4_Autumn18_cferr1_nTraining_-1_nTrees_300.pkl", renameParams="cferr1")
    bTagSys_cferr2   = Modeling.BPTUncertainty("bTagSys_cferr2", "/groups/hephy/cms/robert.schoefbeck/NN/models/BPT/BPT_TTLep_bTagSys_v4_Autumn18_cferr2_nTraining_-1_nTrees_300.pkl", renameParams="cferr2")
    bTagSys_hfstats1 = Modeling.BPTUncertainty("bTagSys_hfstats1", "/groups/hephy/cms/robert.schoefbeck/NN/models/BPT/BPT_TTLep_bTagSys_v4_Autumn18_hfstats1_nTraining_-1_nTrees_300.pkl", renameParams="hfstats1")
    bTagSys_hfstats2 = Modeling.BPTUncertainty("bTagSys_hfstats2", "/groups/hephy/cms/robert.schoefbeck/NN/models/BPT/BPT_TTLep_bTagSys_v4_Autumn18_hfstats2_nTraining_-1_nTrees_300.pkl", renameParams="hfstats2")
    bTagSys_lf       = Modeling.BPTUncertainty("bTagSys_lf", "/groups/hephy/cms/robert.schoefbeck/NN/models/BPT/BPT_TTLep_bTagSys_v4_Autumn18_lf_nTraining_-1_nTrees_300.pkl", renameParams="lf")
    bTagSys_lfstats1 = Modeling.BPTUncertainty("bTagSys_lfstats1", "/groups/hephy/cms/robert.schoefbeck/NN/models/BPT/BPT_TTLep_bTagSys_v4_Autumn18_lfstats1_nTraining_-1_nTrees_300.pkl", renameParams="lfstats1")
    bTagSys_lfstats2 = Modeling.BPTUncertainty("bTagSys_lfstats2", "/groups/hephy/cms/robert.schoefbeck/NN/models/BPT/BPT_TTLep_bTagSys_v4_Autumn18_lfstats2_nTraining_-1_nTrees_300.pkl", renameParams="lfstats2")

    hypothesis.append(bTagSys_hf         .makePenalizedNuisances())
    hypothesis.append(bTagSys_cferr1     .makePenalizedNuisances())
    hypothesis.append(bTagSys_cferr2     .makePenalizedNuisances())
    hypothesis.append(bTagSys_hfstats1   .makePenalizedNuisances())
    hypothesis.append(bTagSys_hfstats2   .makePenalizedNuisances())
    hypothesis.append(bTagSys_lf         .makePenalizedNuisances())
    hypothesis.append(bTagSys_lfstats1   .makePenalizedNuisances())
    hypothesis.append(bTagSys_lfstats2   .makePenalizedNuisances())

hypothesis.print()

logger.info ("LOADING DATA") 
data_model_ = data_model.DataModel()

trueSMEFTData = Modeling.NormalizedSMEFTData( 
                    *data_model.DataModel().getEvents(-1, wilson_coefficients=WCs, 
                            feature_names = list(set(all_features)), feature_dicts=True), 
                     inclusiveExpectation=inclusiveExpectation )

bitPrediction = trueSMEFTData.BITPrediction ( 
                        bit          = bit,
                        combinations = combinations)
if scale:
    scaleUnc_2D.initialize( trueSMEFTData )

if DYnorm:
    DYUnc_norm.initialize( trueSMEFTData )

    # Scaling the DY fraction to 10%
    target_DYnorm_fraction = 0.1
    hypothesis_SM = hypothesis.cloneSM()
    hypothesis_SM_DY1 = hypothesis_SM.cloneModify(gDY=1)
    DYUnc_norm_weight = DYUnc_norm(hypothesis_SM_DY1)
    TTBAR_SM_weight   = Modeling.SMEFTweight(bitPrediction, hypothesis_SM)

    relative_DYnorm_fraction     = (DYUnc_norm_weight*TTBAR_SM_weight).sum()/TTBAR_SM_weight.sum()
    scale_DYNorm_fraction = target_DYnorm_fraction/relative_DYnorm_fraction

if PS:
    PS_FSR.initialize( trueSMEFTData )
    PS_ISR.initialize( trueSMEFTData )

if MG:
    MGvsPow.initialize( trueSMEFTData )

if lSF:
    leptonSF.initialize( trueSMEFTData, translation = {"nJetGood":"nrecoJet"})

if JEC:
    jesAbsoluteMPFBias .initialize( trueSMEFTData, translation = {"nJetGood":"nrecoJet"}) 
    jesFlavorQCD       .initialize( trueSMEFTData, translation = {"nJetGood":"nrecoJet"}) 
    jesPileUpDataMC    .initialize( trueSMEFTData, translation = {"nJetGood":"nrecoJet"}) 
    jesRelativeBal     .initialize( trueSMEFTData, translation = {"nJetGood":"nrecoJet"}) 
    jesSinglePionECAL  .initialize( trueSMEFTData, translation = {"nJetGood":"nrecoJet"})

if BTag: 
    bTagSys_hf         .initialize( trueSMEFTData, translation = {"nJetGood":"nrecoJet", "l1_pt":"recoLep0_pt", "l2_pt":"recoLep1_pt"}) 
    bTagSys_cferr1     .initialize( trueSMEFTData, translation = {"nJetGood":"nrecoJet", "l1_pt":"recoLep0_pt", "l2_pt":"recoLep1_pt"}) 
    bTagSys_cferr2     .initialize( trueSMEFTData, translation = {"nJetGood":"nrecoJet", "l1_pt":"recoLep0_pt", "l2_pt":"recoLep1_pt"}) 
    bTagSys_hfstats1   .initialize( trueSMEFTData, translation = {"nJetGood":"nrecoJet", "l1_pt":"recoLep0_pt", "l2_pt":"recoLep1_pt"}) 
    bTagSys_hfstats2   .initialize( trueSMEFTData, translation = {"nJetGood":"nrecoJet", "l1_pt":"recoLep0_pt", "l2_pt":"recoLep1_pt"}) 
    bTagSys_lf         .initialize( trueSMEFTData, translation = {"nJetGood":"nrecoJet", "l1_pt":"recoLep0_pt", "l2_pt":"recoLep1_pt"}) 
    bTagSys_lfstats1   .initialize( trueSMEFTData, translation = {"nJetGood":"nrecoJet", "l1_pt":"recoLep0_pt", "l2_pt":"recoLep1_pt"}) 
    bTagSys_lfstats2   .initialize( trueSMEFTData, translation = {"nJetGood":"nrecoJet", "l1_pt":"recoLep0_pt", "l2_pt":"recoLep1_pt"}) 

def model_weights( hypo ):
    TTBAR_SMEFT_weight = Modeling.SMEFTweight(bitPrediction, hypo)

    res = TTBAR_SMEFT_weight

    if xsec:
        res *= xsecUnc(hypo)
    if scale:
        res *= scaleUnc_2D(hypo)
    if MG:
        res *= MGvsPow(hypo)
    if PS:
        res *= PS_FSR(hypo)*PS_ISR(hypo)
    if DYnorm:
        res += scale_DYNorm_fraction*(alpha_DYUnc_norm**hypo['gDY'].val)*DYUnc_norm_weight*TTBAR_SM_weight
    if BTag:
        btag = bTagSys_hf(hypo)*bTagSys_cferr1(hypo)*bTagSys_cferr2(hypo)*bTagSys_hfstats1(hypo)*bTagSys_hfstats2(hypo)*bTagSys_lf(hypo)*bTagSys_lfstats1(hypo)*bTagSys_lfstats2(hypo)
        res *= btag
    if JEC: 
        jec =  jesAbsoluteMPFBias(hypo)*jesFlavorQCD(hypo)*jesPileUpDataMC(hypo)*jesRelativeBal(hypo)*jesSinglePionECAL(hypo)
        res *= jec
    if lSF:
        res *= leptonSF(hypo)
    if lumi:
        res *= lumiUnc(hypo)

    return res

    #return lumiUnc(hypo)*leptonSF(hypo)*jec*btag*( xsecUnc(hypo)*TTBAR_SMEFT_weight*scaleUnc_2D(hypo)*PS_FSR(hypo)*PS_ISR(hypo)*MGvsPow(hypo) + scale_DYNorm_fraction*(alpha_DYUnc_norm**hypo['gDY'].val)*DYUnc_norm_weight*TTBAR_SM_weight )

#hypothesis = hypothesis.cloneFreeze(ctGIm=0, xsec=0, ren=0, fac=0,gDY=0,fsr=0,isr=0,gPowheg=0,lSF=0,jesAbsBias=0,jesFlavQCD=0,jesPU=0,jesRelBal=0,jesECAL=0)
#asimov = Modeling.AsimovNonCentrality( model_weights, 
#    null=hypothesis.cloneFreeze(ctGRe=.1), 
#    alt =hypothesis)#, alt=hypothesis.cloneModify(ctGRe=1, ctGIm=1))

sub_dir = "def" if len(sub_directory)==0 else "_".join(sub_directory)
out_dir = os.path.join( output_directory, sub_dir)
if not os.path.exists( out_dir ):
    os.makedirs( out_dir, exist_ok=True )

if args.wc2 is not None:
    fname_postfix = "_"+args.wc2 +"_" + ("%.5f"%args.wc2_val).replace('.','p').replace('-','m')
else:
    fname_postfix = ""

results = {}
for wc1_val in np.linspace(args.low1, args.high1, args.nBins1):

    filename = os.path.join( out_dir, args.wc1 +"_" + ("%.5f"%wc1_val).replace('.','p').replace('-','m') + fname_postfix + ".pkl" )
    if os.path.exists(filename) and not args.overwrite:
        logger.info("File %s exists. Do nothing.", filename)
        continue

    model_point_dict = {args.wc1:wc1_val}
    if args.wc2 is not None:
        model_point_dict[args.wc2] = args.wc2_val

    logger.info("Fitting %s", " ".join( ["%s=%3.2f"%(w,v) for w, v in model_point_dict.items()]) )
    asimov = Modeling.AsimovNonCentrality( model_weights, 
        null=hypothesis.cloneFreeze(**model_point_dict), 
        alt =hypothesis)#, alt=hypothesis.cloneModify(ctGRe=1, ctGIm=1))

    minuit = Modeling.MinuitInterface( asimov )

    res = minuit.fit()
    if res['minuit'] is not None:
        res['hypothesis'].print()
        print( "nonCentrality", res['nonCentrality'], "median expected CL:", Modeling.median_expected_pValue(1, res['nonCentrality']))
    else:
        print( "Fit did not run (no parameters)" )
        print( "preFit nonCentrality", res['preFit_nonCentrality'],  "median expected CL:", Modeling.median_expected_pValue(1, res['preFit_nonCentrality']))
    
    with open(filename,'wb') as file_:
        pickle.dump( res, file_ )

    logger.info( "Written result to %s", filename )

    results[(args.wc1, wc1_val)] = res

#res = Modeling.MinuitInterface( Modeling.AsimovNonCentrality( model_weights, null=hypothesis)).fit()
