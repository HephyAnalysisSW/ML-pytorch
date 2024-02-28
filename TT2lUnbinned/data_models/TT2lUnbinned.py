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

top_kinematics_features = [
            "parton_top1_pt", 
            "parton_top2_pt", 
            "parton_top1_eta", 
            "parton_top2_eta", 

            "parton_top12_pt",
            "parton_top12_mass",
    ]

lepton_kinematics_features = [
            "recoLep0_pt", 
            "recoLep1_pt", 
            "recoLepPos_pt", 
            "recoLepNeg_pt", 

            "recoLep01_pt", 
            "recoLep01_mass", 
    ]

asymmetry_features = [
            "parton_top12_dEta",
            "parton_top12_dAbsEta",

            "recoLep_dEta", 
            "recoLep_dAbsEta",
]

spin_correlation_features = [
            "parton_cosThetaPlus_n", "parton_cosThetaMinus_n", "parton_cosThetaPlus_r", "parton_cosThetaMinus_r", "parton_cosThetaPlus_k", "parton_cosThetaMinus_k", "parton_cosThetaPlus_r_star", "parton_cosThetaMinus_r_star", "parton_cosThetaPlus_k_star", "parton_cosThetaMinus_k_star", 
            
            "parton_xi_nn", "parton_xi_rr", "parton_xi_kk", "parton_xi_nr_plus", "parton_xi_nr_minus", "parton_xi_rk_plus", "parton_xi_rk_minus", "parton_xi_nk_plus", "parton_xi_nk_minus", 
            "parton_xi_r_star_k", "parton_xi_k_r_star", "parton_xi_kk_star",
            "parton_cos_phi", "parton_cos_phi_lab", "parton_abs_delta_phi_ll_lab",
]

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
        branches = ["p_C", "nrecoLep"] + feature_names   ) 

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

class DataModel:
    
    def __init__( self, top_kinematics=False, lepton_kinematics = False, asymmetry = False, spin_correlation = False):

        self.top_kinematics     = top_kinematics
        self.lepton_kinematics  = lepton_kinematics
        self.asymmetry          = asymmetry
        self.spin_correlation   = spin_correlation

        if any([top_kinematics, lepton_kinematics, asymmetry, spin_correlation]): 
            self.feature_names = []
            if top_kinematics:
                self.feature_names += top_kinematics_features
            if lepton_kinematics:
                self.feature_names += lepton_kinematics_features
            if asymmetry:
                self.feature_names += asymmetry_features
            if spin_correlation:
                self.feature_names += spin_correlation_features
        else:
            self.feature_names = feature_names

    @property
    def name(self):
        return "TK_%r_LK_%r_CA_%r_SC_%r"%( self.top_kinematics, self.lepton_kinematics, self.asymmetry, self.spin_correlation) 

    def getEvents( self, nTraining, return_observers = False,  wilson_coefficients = None, feature_names=None):

        index = -1
        if wilson_coefficients is None:
            wilson_coefficients = weightInfo.variables

        coeffs       = data_generator.vector_branch( data_generator[index], 'p_C', padding_target=len(weightInfo.combinations))[:nTraining]
        features     = data_generator.scalar_branches( data_generator[index], self.feature_names if feature_names is None else feature_names)[:nTraining]
        vectors      = None #{key:model.data_generator.vector_branch(data, key ) for key in vector_branches}
        combinations = make_combinations( [ w for w in weightInfo.variables if w in wilson_coefficients] )

        if return_observers:
            observers_ = data_generator.scalar_branches( data_generator[index], observers )[:nTraining]
            return features, {comb:coeffs[:,weightInfo.combinations.index(comb)] for comb in combinations}, observers_
        else: 
            return features, {comb:coeffs[:,weightInfo.combinations.index(comb)] for comb in combinations}
    

eft_plot_points = [
    {'color':ROOT.kBlack,       'eft':sm, 'tex':"SM"},

    {'color':ROOT.kMagenta-4,   'eft':make_eft(ctGRe=1), 'tex':"C_{tG}^{Re}=1"},
    {'color':ROOT.kMagenta-6,   'eft':make_eft(ctGIm=1), 'tex':"C_{tG}^{Im}=1"},
    {'color':ROOT.kMagenta+2,   'eft':make_eft(cQj18=10), 'tex':"c_{Qj}^{18}=10"},
    {'color':ROOT.kGreen-4,     'eft':make_eft(cQj38=10), 'tex':"c_{Qj}^{38}=10"},
    {'color':ROOT.kGreen+2,     'eft':make_eft(cQj11=10), 'tex':"c_{Qj}^{11}=10"},
    {'color':ROOT.kBlue-4,      'eft':make_eft(cjj31=10), 'tex':"c_{jj}^{31}=10"},
    {'color':ROOT.kBlue+2,      'eft':make_eft(ctu8=10), 'tex':"c_{tu}^{8}=10"  },
    {'color':ROOT.kRed-4,       'eft':make_eft(ctd8=10), 'tex':"c_{td}^{8}=10"  },
    {'color':ROOT.kRed+2,       'eft':make_eft(ctj8=10), 'tex':"c_{tj}^{8}=10"  },
    {'color':ROOT.kOrange-4,    'eft':make_eft(cQu8=10), 'tex':"c_{Qu}^{8}=10"  },
    {'color':ROOT.kOrange+2,    'eft':make_eft(cQd8=10), 'tex':"c_{Qd}^{8}=10"  },
    {'color':ROOT.kCyan-4,      'eft':make_eft(ctu1=10), 'tex':"c_{tu}^{1}=10"  },
    {'color':ROOT.kCyan+2,      'eft':make_eft(ctd1=10), 'tex':"c_{td}^{1}=10"  },
#    {'color':ROOT.kYellow-4,    'eft':make_eft(ctj1=10), 'tex':"c_{tj}^{1}=10"  },
#    {'color':ROOT.kYellow+2,    'eft':make_eft(cQu1=10), 'tex':"c_{Qu}^{1}=10"  },
    {'color':ROOT.kPink,        'eft':make_eft(cQd1=10), 'tex':"c_{Qd}^{1}=10"  },
]

multi_bit_cfg = {'n_trees': 300,
                 'max_depth': 4,
                 'learning_rate': 0.20,
                 'min_size': 25 }
