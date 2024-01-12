import pickle
import random
import ROOT
from math import pi
if __name__=="__main__":
    import sys
    sys.path.append('..')

from tools.DataGenerator import DataGenerator
from tools.WeightInfo    import WeightInfo

selection = lambda ar: (ar.nrecoLep==2) & (ar.recoLepNeg_pt>0) & (ar.recoLepPos_pt>0) #& (ar.recoBj01_pt > 0) & (ar.recoBj01_mass>0) 

feature_names = [   
            "parton_top1_pt", 
            "parton_top1_f1_pt", 
            "parton_top1_f2_pt", 
            "parton_top1_b_pt", 
            "parton_top1_W_pt", 

            "parton_top12_pt",
            "parton_top12_mass",

            "recoLep0_pt", 
            "recoLep1_pt", 
            "recoLepPos_pt", 
            "recoLepNeg_pt", 

            "recoLep01_pt", 
            "recoLep01_mass", 

            "parton_top2_pt", 
            "parton_top2_f1_pt", 
            "parton_top2_f2_pt", 
            "parton_top2_b_pt", 
            "parton_top2_W_pt", 

            "parton_top1_eta",  
            "parton_top1_b_eta", 
            "parton_top1_W_eta",  
            "parton_top1_f1_eta", 
            "parton_top1_f2_eta", 

            "recoLep0_eta", 
            "recoLep1_eta", 
            "recoLepPos_eta", 
            "recoLepNeg_eta", 

            "parton_top2_eta",  
            "parton_top2_f1_eta", 
            "parton_top2_f2_eta", 
            "parton_top2_b_eta", 
            "parton_top2_W_eta", 

            "parton_top12_eta",
            "parton_top12_dEta",
            "parton_top12_dAbsEta",

            "recoLep_dEta", 
            "recoLep_dAbsEta",

            "recoLep_dPhi", 

            "parton_top1_decayAngle_theta", 
            "parton_top1_decayAngle_phi", 
            "parton_top2_decayAngle_theta", 
            "parton_top2_decayAngle_phi", 

            "parton_cosThetaPlus_n", "parton_cosThetaMinus_n", "parton_cosThetaPlus_r", "parton_cosThetaMinus_r", "parton_cosThetaPlus_k", "parton_cosThetaMinus_k", "parton_cosThetaPlus_r_star", "parton_cosThetaMinus_r_star", "parton_cosThetaPlus_k_star", "parton_cosThetaMinus_k_star", 
            
            "parton_xi_nn", "parton_xi_rr", "parton_xi_kk", "parton_xi_nr_plus", "parton_xi_nr_minus", "parton_xi_rk_plus", "parton_xi_rk_minus", "parton_xi_nk_plus", "parton_xi_nk_minus", 
            "parton_xi_r_star_k", "parton_xi_k_r_star", "parton_xi_kk_star",
            "parton_cos_phi", "parton_cos_phi_lab", "parton_abs_delta_phi_ll_lab",
            "nBTag",
    ]

observers = []

data_generator =  DataGenerator(
    input_files = ["/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v4/TT01j2lRef_HT500/TT01j2lRef_HT500_*.root"],
        n_split = 100,
        splitting_strategy = "events",
        selection   = selection,
        branches = ["p_C", "nrecoLep"] +feature_names   ) 

reweight_pkl = '/eos/vbc/group/cms/robert.schoefbeck/gridpacks/CA/TT01j2lCARef_HT500_reweight_card.pkl'
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

wilson_coefficients = weightInfo.variables 

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

def getEvents( nTraining, return_observers = False ):

    index = -1

    coeffs       = data_generator.vector_branch( data_generator[index], 'p_C', padding_target=len(weightInfo.combinations))[:nTraining]
    features     = data_generator.scalar_branches( data_generator[index], feature_names )[:nTraining]
    vectors      = None #{key:model.data_generator.vector_branch(data, key ) for key in vector_branches}
    combinations = make_combinations( wilson_coefficients )

    if return_observers:
        observers_ = data_generator.scalar_branches( data_generator[index], observers )[:nTraining]
        return features, {comb:coeffs[:,weightInfo.combinations.index(comb)] for comb in combinations}, observers_
    else: 
        return features, {comb:coeffs[:,weightInfo.combinations.index(comb)] for comb in combinations}

tex = {"ctGIm":"C_{tG}^{Im}", "ctGRe":"C_{tG}^{Re}", "cQj18":"c_{Qj}^{18}" ,"cQj38":"c_{Qj}^{38}" ,"cQj11":"c_{Qj}^{11}" ,"cjj31":"c_{jj}^{31}" ,"ctu8":"c_{tu}^{8}" ,"ctd8":"c_{td}^{8}" ,"ctj8":"c_{tj}^{8}" ,"cQu8":"c_{Qu}^{8}" ,"cQd8":"c_{Qd}^{8}" ,"ctu1":"c_{tu}^{1}" ,"ctd1":"c_{td}^{1}" ,"ctj1":"c_{tj}^{1}" ,"cQu1":"c_{Qu}^{1}" ,"cQd1":"c_{Qd}^{1}"}

plot_points = [
    {'color':ROOT.kBlack,       'point':sm, 'tex':"SM"},

    {'color':ROOT.kMagenta-4,   'point':make_eft(ctGRe=1), 'tex':"C_{tG}^{Re}=1"},
    {'color':ROOT.kMagenta-6,   'point':make_eft(ctGIm=1), 'tex':"C_{tG}^{Im}=1"},
    {'color':ROOT.kMagenta+2,   'point':make_eft(cQj18=10), 'tex':"c_{Qj}^{18}=10"},
    {'color':ROOT.kGreen-4,     'point':make_eft(cQj38=10), 'tex':"c_{Qj}^{38}=10"},
    {'color':ROOT.kGreen+2,     'point':make_eft(cQj11=10), 'tex':"c_{Qj}^{11}=10"},
    {'color':ROOT.kBlue-4,      'point':make_eft(cjj31=10), 'tex':"c_{jj}^{31}=10"},
    {'color':ROOT.kBlue+2,      'point':make_eft(ctu8=10), 'tex':"c_{tu}^{8}=10"  },
    {'color':ROOT.kRed-4,       'point':make_eft(ctd8=10), 'tex':"c_{td}^{8}=10"  },
    {'color':ROOT.kRed+2,       'point':make_eft(ctj8=10), 'tex':"c_{tj}^{8}=10"  },
    {'color':ROOT.kOrange-4,    'point':make_eft(cQu8=10), 'tex':"c_{Qu}^{8}=10"  },
    {'color':ROOT.kOrange+2,    'point':make_eft(cQd8=10), 'tex':"c_{Qd}^{8}=10"  },
    {'color':ROOT.kCyan-4,      'point':make_eft(ctu1=10), 'tex':"c_{tu}^{1}=10"  },
    {'color':ROOT.kCyan+2,      'point':make_eft(ctd1=10), 'tex':"c_{td}^{1}=10"  },
#    {'color':ROOT.kYellow-4,    'point':make_eft(ctj1=10), 'tex':"c_{tj}^{1}=10"  },
#    {'color':ROOT.kYellow+2,    'point':make_eft(cQu1=10), 'tex':"c_{Qu}^{1}=10"  },
    {'color':ROOT.kPink,        'point':make_eft(cQd1=10), 'tex':"c_{Qd}^{1}=10"  },
]

plot_options =  {
    "parton_top1_decayAngle_theta" :{'binning':[30,0,pi], 'tex':'#theta(t_{1})'},
    "parton_top1_decayAngle_phi"   :{'binning':[30,-pi,pi], 'tex':'#phi(t_{1})'},
    "parton_top1_pt" :{'binning':[50,0,1500], 'tex':'p_{T}(t_{1})'},
    "parton_top1_eta" :{'binning':[30,-3,3], 'tex':'#eta(t_{1})'},
    "parton_top1_f1_pt" :{'binning':[30,0,800], 'tex':'p_{T}(f_{1}(t_{1}))'},
    "parton_top1_f1_eta" :{'binning':[30,-3,3], 'tex':'#eta(f_{2}(t_{1}))'},
    "parton_top1_f2_pt" :{'binning':[30,0,800], 'tex':'p_{T}(f_{2} (t_{1}))'},
    "parton_top1_f2_eta" :{'binning':[30,-3,3], 'tex':'#eta(f_{2}(t_{1}))'},
    "parton_top1_b_pt" :{'binning':[50,0,800], 'tex':'p_{T}(b (t_{1}))'},
    "parton_top1_b_eta" :{'binning':[30,-3,3], 'tex':'#eta(b(t_{1}))'},
    "parton_top1_W_pt" :{'binning':[30,0,1000], 'tex':'p_{T}(W (t_{1}))'},
    "parton_top1_W_eta" :{'binning':[30,-3,3], 'tex':'#eta(W(t_{1}))'},

    "parton_top2_decayAngle_theta" :{'binning':[30,0,pi], 'tex':'#theta(t_{2})'},
    "parton_top2_decayAngle_phi"   :{'binning':[30,-pi,pi], 'tex':'#phi(t_{2})'},
    "parton_top2_pt" :{'binning':[50,0,1500], 'tex':'p_{T}(t_{2})'},
    "parton_top2_eta" :{'binning':[30,-3,3], 'tex':'#eta(t_{2})'},
    "parton_top2_f1_pt" :{'binning':[30,0,800], 'tex':'p_{T}(f_{1}(t_{2}))'},
    "parton_top2_f1_eta" :{'binning':[30,-3,3], 'tex':'#eta(f_{1}(t_{2}))'},
    "parton_top2_f2_pt" :{'binning':[30,0,800], 'tex':'p_{T}(f_{2} (t_{2}))'},
    "parton_top2_f2_eta" :{'binning':[30,-3,3], 'tex':'#eta(f_{2}(t_{2}))'},
    "parton_top2_b_pt" :{'binning':[50,0,800], 'tex':'p_{T}(b (t_{2}))'},
    "parton_top2_b_eta" :{'binning':[30,-3,3], 'tex':'#eta(b(t_{2}))'},
    "parton_top2_W_pt" :{'binning':[30,0,1000], 'tex':'p_{T}(W (t_{2}))'},
    "parton_top2_W_eta" :{'binning':[30,-3,3], 'tex':'#eta(W(t_{2}))'},

    "parton_top12_pt":{'binning':[50,0,1000], 'tex':'p_{T}(t#bar{t})'},
    "parton_top12_mass":{'binning':[50,0,2000], 'tex':'M(t#bar{t})'},
    "parton_top12_eta":{'binning':[30,-3,3], 'tex':'#eta(t#bar{t})'},
    "parton_top12_dEta":{'binning':[30,-3,3], 'tex':'#Delta#eta(t#bar{t})'},
    "parton_top12_dAbsEta":{'binning':[30,-3,3], 'tex':'#Delta|#eta|(t#bar{t})'},

    "parton_cosThetaPlus_n"     :{'binning':[30,-1,1], 'tex':'cos#theta^{+}_{n}'},
    "parton_cosThetaMinus_n"    :{'binning':[30,-1,1], 'tex':'cos#theta^{-}_{n}'},
    "parton_cosThetaPlus_r"     :{'binning':[30,-1,1], 'tex':'cos#theta^{+}_{r}'},
    "parton_cosThetaMinus_r"    :{'binning':[30,-1,1], 'tex':'cos#theta^{-}_{r}'},
    "parton_cosThetaPlus_k"     :{'binning':[30,-1,1], 'tex':'cos#theta^{+}_{k}'},
    "parton_cosThetaMinus_k"    :{'binning':[30,-1,1], 'tex':'cos#theta^{-}_{k}'},
    "parton_cosThetaPlus_r_star"    :{'binning':[30,-1,1], 'tex':'cos#theta^{+*}_{n}'},
    "parton_cosThetaMinus_r_star"   :{'binning':[30,-1,1], 'tex':'cos#theta^{-*}_{n}'},
    "parton_cosThetaPlus_k_star"    :{'binning':[30,-1,1], 'tex':'cos#theta^{+*}_{k}'},
    "parton_cosThetaMinus_k_star"   :{'binning':[30,-1,1], 'tex':'cos#theta^{-*}_{k}'},
    "parton_xi_nn"              :{'binning':[30,-1,1], 'tex':'#xi_{nn}'},
    "parton_xi_rr"              :{'binning':[30,-1,1], 'tex':'#xi_{rr}'},
    "parton_xi_kk"              :{'binning':[30,-1,1], 'tex':'#xi_{kk}'},
    "parton_xi_nr_plus"         :{'binning':[30,-1,1], 'tex':'#xi_{nr}^{+}'},
    "parton_xi_nr_minus"        :{'binning':[30,-1,1], 'tex':'#xi_{nr}^{-}'},
    "parton_xi_rk_plus"         :{'binning':[30,-1,1], 'tex':'#xi_{rk}^{+}'},
    "parton_xi_rk_minus"        :{'binning':[30,-1,1], 'tex':'#xi_{rk}^{-}'},
    "parton_xi_nk_plus"         :{'binning':[30,-1,1], 'tex':'#xi_{nk}^{+}'},
    "parton_xi_nk_minus"        :{'binning':[30,-1,1], 'tex':'#xi_{nk}^{-}'},

    "parton_xi_r_star_k"        :{'binning':[30,-1,1], 'tex':'#xi_{r^{*}k}'},
    "parton_xi_k_r_star"        :{'binning':[30,-1,1], 'tex':'#xi_{kr^{*}}'},
    "parton_xi_kk_star"         :{'binning':[30,-1,1], 'tex':'#xi_{kk^{*}}'},

    "parton_cos_phi"            :{'binning':[30,-1,1], 'tex':'cos(#phi)'},
    "parton_cos_phi_lab"        :{'binning':[30,-1,1], 'tex':'cos(#phi lab)'},
    "parton_abs_delta_phi_ll_lab":{'binning':[30,0,pi], 'tex':'|#Delta(#phi(l,l))|'},

    "recoBj01_pt"     :{'binning':[30,0,1500], 'tex':'p_{T}(bb)'},
    "recoBj01_mass"   :{'binning':[30,0,1500], 'tex':'M(bb)'},
    "nBTag"           :{'binning':[4,0,4], 'tex':'N_{b}'},
    "recoLep0_pt"     :{'binning':[30,0,200], 'tex':'p_{T}(l_{1})'},
    "recoLep0_eta"   :{'binning':[30,-3,3], 'tex':'#eta(l_{1})'},
    "recoLep1_pt"   :{'binning':[30,0,200], 'tex':'p_{T}(l_{2})'},
    "recoLep1_eta"   :{'binning':[30,-3,3], 'tex':'#eta(l_{2})'},
    "recoLepPos_pt"   :{'binning':[30,0,200], 'tex':'p_{T}(l^{+})'},
    "recoLepPos_eta"   :{'binning':[30,-3,3], 'tex':'#eta(l^{+})'},
    "recoLepNeg_pt"   :{'binning':[30,0,200], 'tex':'p_{T}(l^{-})'},
    "recoLepNeg_eta"   :{'binning':[30,-3,3], 'tex':'#eta(l^{-})'},
    "recoLep01_pt"   :{'binning':[30,0,200], 'tex':'p_{T}(ll)'},
    "recoLep01_mass"   :{'binning':[30,0,200], 'tex':'M(ll)'},
    "recoLep_dPhi"     :{'binning':[30,-pi,pi], 'tex':'#Delta#phi(ll)'},
    "recoLep_dEta"     :{'binning':[30,-2.5,2.5], 'tex':'#Delta#eta(ll)'},
    "recoLep_dAbsEta"     :{'binning':[30,-2.5,2.5], 'tex':'#Delta|#eta|(ll)'},
}

multi_bit_cfg = {'n_trees': 300,
                 'max_depth': 4,
                 'learning_rate': 0.20,
                 'min_size': 25 }
