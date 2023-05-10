import pickle
import random
import ROOT

if __name__=="__main__":
    import sys
    sys.path.append('..')
    sys.path.append('../..')

from tools.DataGenerator import DataGenerator
from tools.WeightInfo    import WeightInfo

selection = lambda ar: (ar.genJet_pt>500) & (ar.genJet_SDmass>0) & (abs(ar.dR_genJet_maxq1q2b)<0.6) & (ar.genJet_SDsubjet1_mass>=0)
# -> https://schoef.web.cern.ch/schoef/pytorch/choleskyNN/genTops/training_plots/choleskyNN_genTops_ctWRe_nTraining_519075/lin/epoch.gif

data_generator =  DataGenerator(
    input_files = ["/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v6/tschRefPointNoWidthRW/tschRefPointNoWidthRW_*.root"],
        n_split = 1,
        splitting_strategy = "files",
        selection   = selection,
        branches = [
                    #"genQ1_pt", "genQ1_eta", "genQ1_phi", "genQ1_mass", "genQ1_pdgId", "genQ2_pt", "genQ2_eta", "genQ2_phi", "genQ2_mass", "genQ2_pdgId", 
                    #"genb_pt", "genb_eta", "genb_phi", "genb_mass", "genb_pdgId", 
                    #"genW_pt", "genW_eta", "genW_phi", "genW_mass", "genJet_pt", "genJet_eta", "genJet_phi", "genJet_mass", "genJet_nConstituents", 
                    #"genJet_SDmass", "genJet_SDsubjet0_eta", "genJet_SDsubjet0_deltaEta", "genJet_SDsubjet0_phi", "genJet_SDsubjet0_deltaPhi", "genJet_SDsubjet0_deltaR", 
                    #"genJet_SDsubjet0_mass", "genJet_SDsubjet1_eta", "genJet_SDsubjet1_deltaEta", "genJet_SDsubjet1_phi", "genJet_SDsubjet1_deltaPhi", 
                    #"genJet_SDsubjet1_deltaR", "genJet_SDsubjet1_mass", 
                    #"genJet_tau1", "genJet_tau2", "genJet_tau3", "genJet_tau4", "genJet_tau21", "genJet_tau32", 
                    #"genJet_ecf1", "genJet_ecf2", "genJet_ecf3", "genJet_ecfC1", "genJet_ecfC2", "genJet_ecfC3", "genJet_ecfD", "genJet_ecfDbeta2", "genJet_ecfM1", "genJet_ecfM2", "genJet_ecfM3", "genJet_ecfM1beta2", "genJet_ecfM2beta2", "genJet_ecfM3beta2", "genJet_ecfN1", "genJet_ecfN2", "genJet_ecfN3", "genJet_ecfN1beta2", "genJet_ecfN2beta2", "genJet_ecfN3beta2", "genJet_ecfU1", "genJet_ecfU2", "genJet_ecfU3", "genJet_ecfU1beta2", "genJet_ecfU2beta2", "genJet_ecfU3beta2", 
                    #"dR_genJet_Q1", "dR_genJet_Q2", "dR_genJet_W", "dR_genJet_b", "dR_genJet_top", "dR_genJet_maxQ1Q2b",
                    "genJet_pt", "genJet_SDmass", "dR_genJet_maxq1q2b", "genJet_SDsubjet1_mass", 

                    "ngen", "gen_pt", "gen_etarel", "gen_phirel", "gen_eta", "gen_phi", "gen_pdgId", "gen_charge", "gen_type", 
                    "p_C"]
    )

reweight_pkl = '/eos/vbc/group/cms/robert.schoefbeck/gridpacks/ParticleNet/t-sch-RefPoint-noWidthRW_reweight_card.pkl'
weightInfo = WeightInfo(reweight_pkl)
weightInfo.set_order(2)
default_eft_parameters = {p:0 for p in weightInfo.variables}
def make_eft(**kwargs):
    result = { key:val for key, val in default_eft_parameters.items() }
    for key, val in kwargs.items():
        if not key in weightInfo.variables+["Lambda"]:
            raise RuntimeError ("Wilson coefficient not known.")
        else:
            result[key] = float(val)
    return result

random_eft = make_eft(**{v:random.random() for v in weightInfo.variables} )
sm         = make_eft()

if __name__=="__main__":
    wilson_coefficients = weightInfo.variables 
else:
    wilson_coefficients = ['ctWRe']

vector_branches = ["gen_pt", "gen_etarel", "gen_phirel"]

feature_names = [   #"genJet_pt", "genJet_mass", "genJet_nConstituents",
                    #"genJet_SDmass", "genJet_SDsubjet0_deltaEta", "genJet_SDsubjet0_deltaPhi", "genJet_SDsubjet0_deltaR",
                    #"genJet_SDsubjet0_mass", "genJet_SDsubjet1_deltaEta", "genJet_SDsubjet1_deltaPhi",
                    #"genJet_SDsubjet1_deltaR", "genJet_SDsubjet1_mass",
                    #"genJet_tau1", "genJet_tau2", "genJet_tau3", "genJet_tau4", "genJet_tau21", "genJet_tau32", 
                    #"genJet_ecf1", "genJet_ecf2", "genJet_ecf3", "genJet_ecfC1", "genJet_ecfC2", "genJet_ecfC3", "genJet_ecfD", "genJet_ecfDbeta2", "genJet_ecfM1", "genJet_ecfM2", "genJet_ecfM3", "genJet_ecfM1beta2", "genJet_ecfM2beta2", "genJet_ecfM3beta2", "genJet_ecfN1", "genJet_ecfN2", "genJet_ecfN3", "genJet_ecfN1beta2", "genJet_ecfN2beta2", "genJet_ecfN3beta2", "genJet_ecfU1", "genJet_ecfU2", "genJet_ecfU3", "genJet_ecfU1beta2", "genJet_ecfU2beta2", "genJet_ecfU3beta2",

                    "genJet_pt", "genJet_SDmass", "dR_genJet_maxq1q2b", "genJet_SDsubjet1_mass", 
                    ]

def make_combinations( coefficients ):
    combinations = []
    for comb in weightInfo.combinations:
        good = True
        for k in comb:
            if k not in coefficients:
                good = False
                break
        if good:
            combinations.append(comb)
    return combinations

def getEvents( nTraining ):
    data_generator.load(-1, small=nTraining )
    combinations = make_combinations( wilson_coefficients )
    coeffs = data_generator.vector_branch('p_C')
    return {key:data_generator.vector_branch( key ) for key in vector_branches}, {comb:coeffs[:,weightInfo.combinations.index(comb)] for comb in combinations}

tex = {"ctWRe":"C_{tW}^{Re}", "ctWIm":"C_{tW}^{Im}", "ctBIm":"C_{tB}^{Im}", "ctBRe":"C_{tB}^{Re}", "cHt":"C_{Ht}", 'cHtbRe':'C_{Htb}^{Re}', 'cHtbIm':'C_{Htb}^{Im}', 'cHQ3':'C_{HQ}^{(3)}'}

#['ctWRe', 'ctBRe', 'cHQ3', 'cHt', 'cHtbRe', 'ctWIm', 'ctBIm', 'cHtbIm']

eft_plot_points = [
    {'color':ROOT.kBlack,       'eft':sm, 'tex':"SM"},
    {'color':ROOT.kMagenta-4,   'eft':make_eft(ctWRe=3),   'tex':"c_{tW}^{Re}=3",   },
    {'color':ROOT.kMagenta+2,   'eft':make_eft(ctWIm=5),   'tex':"c_{tW}^{Im}=5",   },
    {'color':ROOT.kGreen-4,     'eft':make_eft(cHtbRe=5),  'tex':"c_{Htb}^{Re}=5",  },
    {'color':ROOT.kGreen+2,     'eft':make_eft(cHtbIm=5),  'tex':"c_{Htb}^{Im}=5",  },
    {'color':ROOT.kBlue+2,      'eft':make_eft(cHQ3=5),    'tex':"c_{HQ}^{(3)}=5",  },
    ]
plot_options =  {
    "genJet_pt"                 :{'binning':[50,500,2000], 'tex':'p_{T}(jet)'},
    "genJet_mass"               :{'binning':[50,150,200], 'tex':'M(jet) unpruned'},
    "genJet_nConstituents"      :{'binning':[50,30,230], 'tex':'n-constituents'},
    "genJet_SDmass"             :{'binning':[50,150,200], 'tex':'M_{SD}(jet)'},
    "genJet_SDsubjet0_deltaEta" :{'binning':[50,-0.6,0.6], 'tex':'#Delta#eta(jet,jet_{SD,0})'},
    "genJet_SDsubjet0_deltaPhi" :{'binning':[50,-0.6,0.6], 'tex':'#Delta#phi(jet,jet_{SD,0})'},
    "genJet_SDsubjet0_deltaR"   :{'binning':[50,0,0.7], 'tex':'#Delta R(jet,jet_{SD,0})'},
    "genJet_SDsubjet0_mass"     :{'binning':[50,0,200], 'tex':'M_{SD}(jet_{0})'},
    "genJet_SDsubjet1_deltaEta" :{'binning':[50,-0.6,0.6], 'tex':'#Delta#eta(jet,jet_{SD,1})'},
    "genJet_SDsubjet1_deltaPhi" :{'binning':[50,-0.6,0.6], 'tex':'#Delta#phi(jet,jet_{SD,1})'},
    "genJet_SDsubjet1_deltaR"   :{'binning':[50,0,0.7], 'tex':'#Delta R(jet,jet_{SD,1})'},
    "genJet_SDsubjet1_mass"     :{'binning':[50,0,200], 'tex':'M_{SD}(jet_{1})'},
    "genJet_tau1"               :{'binning':[50,0,1], 'tex':'#tau_{1}'},
    "genJet_tau2"               :{'binning':[50,0,.5],'tex':'#tau_{2}'},
    "genJet_tau3"               :{'binning':[50,0,.3],'tex':'#tau_{3}'},
    "genJet_tau4"               :{'binning':[50,0,.3],'tex':'#tau_{4}'},
    "genJet_tau21"              :{'binning':[50,0,1], 'tex':'#tau_{21}'},
    "genJet_tau32"              :{'binning':[50,0,1], 'tex':'#tau_{32}'},
#https://github.com/cms-sw/cmssw/blob/6d2f66057131baacc2fcbdd203588c41c885b42c/RecoJets/JetProducers/python/ECF_cff.py
    "genJet_ecf1"               :{'binning':[50,0,2000], 'tex':"ecf1"},
    "genJet_ecf2"               :{'binning':[50,0,400000], 'tex':"ecf2"},
    "genJet_ecf3"               :{'binning':[50,0,4000000], 'tex':"ecf3"},
    "genJet_ecfC1"              :{'binning':[50,0,.5], 'tex':"ecfC1"},
    "genJet_ecfC2"              :{'binning':[50,0,.5], 'tex':"ecfC2"},
    "genJet_ecfC3"              :{'binning':[50,0,.5], 'tex':"ecfC3"},
    "genJet_ecfD"               :{'binning':[50,0,8], 'tex':"ecfD"},
    "genJet_ecfDbeta2"          :{'binning':[50,0,20], 'tex':"ecfDbeta2"},
    "genJet_ecfM1"              :{'binning':[50,0,0.35], 'tex':"ecfM1"},
    "genJet_ecfM2"              :{'binning':[50,0,0.2], 'tex':"ecfM2"},
    "genJet_ecfM3"              :{'binning':[50,0,0.2], 'tex':"ecfM3"},
    "genJet_ecfM1beta2"         :{'binning':[50,0,0.35], 'tex':"ecfM1beta2"},
    "genJet_ecfM2beta2"         :{'binning':[50,0,0.2], 'tex':"ecfM2beta2"},
    "genJet_ecfM3beta2"         :{'binning':[50,0,0.2], 'tex':"ecfM3beta2"},
    "genJet_ecfN1"              :{'binning':[50,0,0.5], 'tex':"ecfN1"},
    "genJet_ecfN2"              :{'binning':[50,0,0.5], 'tex':"ecfN2"},
    "genJet_ecfN3"              :{'binning':[50,0,5], 'tex':"ecfN3"},
    "genJet_ecfN1beta2"         :{'binning':[50,0,0.5], 'tex':"ecfN1beta2"},
    "genJet_ecfN2beta2"         :{'binning':[50,0,0.5], 'tex':"ecfN2beta2"},
    "genJet_ecfN3beta2"         :{'binning':[50,0,5], 'tex':"ecfN3beta2"},
    "genJet_ecfU1"              :{'binning':[50,0,0.5], 'tex':"ecfU1"},
    "genJet_ecfU2"              :{'binning':[50,0,0.04], 'tex':"ecfU2"},
    "genJet_ecfU3"              :{'binning':[50,0,0.004], 'tex':"ecfU3"},
    "genJet_ecfU1beta2"         :{'binning':[50,0,0.5], 'tex':"ecfU1beta2"},
    "genJet_ecfU2beta2"         :{'binning':[50,0,0.04], 'tex':"ecfU2beta2"},
    "genJet_ecfU3beta2"         :{'binning':[50,0,0.004], 'tex':"ecfU3beta2"},
}

multi_bit_cfg = {'n_trees': 100,
                 'max_depth': 4,
                 'learning_rate': 0.20,
                 'min_size': 15 }

if __name__=="__main__":
   
    # load some events and their weights 
    x, w = getEvents(1000)

    # x are a list of feature-vectors such that x[0] are the features of the first event. Their branch-names are stored in feature_names.
    # w are a dictionary with the weight-coefficients. The key tuple(), i.e., the empty n-tuple, is the constant term. The key ('ctWRe', ), i.e., the coefficient 
    # that is an tuple of length one is the linear derivative (in this case wrt to ctWRe). The quadratic derivatives are stored in the keys ('ctWRe', 'ctWRe') etc.
    # The list of Wilson coefficients is in: weightInfo.variables
    # The list of all derivatives (i.e., the list of all combiations of all variables up to length 2) is weightInfo.combinations. It includes the zeroth derivative, i.e., the constant term.

    # Let us scale all the weights to reasonable numbers. They come out of MG very small because the cross section of the process I used to genereate the top quarks is so small: s-channel single-top
    # Let us add up the constant terms of all events and normalize the sample to the number of events. (Arbitrary choice)
    const = (len(w[()])/w[()].sum())
    for k in w.keys():
        w[k] *= const 

    ##let's remove the most extreme weight derivatives ... cosmetics for the propaganda plots
    #from   tools import helpers 
    #len_before = len(x)
    #auto_clip = 0.001
    #x, w = helpers.clip_quantile(x, auto_clip, weights = w )
    #print ("Auto clip efficiency (training) %4.3f is %4.3f"%( auto_clip, len(x)/len_before ) )

    print ("Wilson coefficients:", weightInfo.variables )
    print ("Features of the first event:\n" + "\n".join( ["%25s = %4.3f"%(name, value) for name, value in zip(feature_names, x[0])] ) )
    prstr = {0:'constant', 1:'linear', 2:'quadratic'}
    print ("Weight coefficients(!) of the first event:\n"+"\n".join( ["%30s = %4.3E"%( prstr[len(comb)] + " " +",".join(comb), w[comb][0]) for comb in weightInfo.combinations] ) )

    # Let us compute the quadratic weight for ctWRe=1:
    import copy
    eft_sm  = make_eft()
    eft_bsm = make_eft(ctWRe=1)
    # constant term
    reweight = copy.deepcopy(w[()])
    # linear term
    for param1 in wilson_coefficients:
        reweight += (eft_bsm[param1]-eft_sm[param1])*w[(param1,)] 
    # quadratic term
    for param1 in wilson_coefficients:
        if eft_bsm[param1]-eft_sm[param1] ==0: continue
        for param2 in wilson_coefficients:
            if eft_bsm[param2]-eft_sm[param2] ==0: continue
            reweight += .5*(eft_bsm[param1]-eft_sm[param1])*(eft_bsm[param2]-eft_sm[param2])*w[tuple(sorted((param1,param2)))]

    print ("w(ctWRe=1) for the first event is: ", reweight[0])

    # Let us compute the weight ratio we can use for the training:
    target_ctWRe        = w[('ctWRe',)]/w[()]
    target_ctWRe_ctWRe  = w[('ctWRe','ctWRe')]/w[()]

    # NOTE!! These "target" branches are already written to the training data! No need to compute
