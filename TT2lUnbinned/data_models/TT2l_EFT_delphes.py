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
            "tr_ttbar_pt", 
            "tr_ttbar_mass", 

            "tr_top_pt", 
            "tr_topBar_pt", 
            "tr_top_eta", 
            "tr_topBar_eta", 
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
            "tr_ttbar_dEta",
            "tr_ttbar_dAbsEta",
            "recoLep_dEta", 
            "recoLep_dAbsEta",
]

spin_correlation_features = [
            "tr_cosThetaPlus_n", "tr_cosThetaMinus_n", "tr_cosThetaPlus_r", "tr_cosThetaMinus_r", "tr_cosThetaPlus_k", "tr_cosThetaMinus_k", "tr_cosThetaPlus_r_star", "tr_cosThetaMinus_r_star", "tr_cosThetaPlus_k_star", "tr_cosThetaMinus_k_star", 
            
            "tr_xi_nn", "tr_xi_rr", "tr_xi_kk", "tr_xi_nr_plus", "tr_xi_nr_minus", "tr_xi_rk_plus", "tr_xi_rk_minus", "tr_xi_nk_plus", "tr_xi_nk_minus", 
            "tr_xi_r_star_k", "tr_xi_k_r_star", "tr_xi_kk_star",
            "tr_cos_phi", "tr_cos_phi_lab", "tr_abs_delta_phi_ll_lab",
]

feature_names = top_kinematics_features + lepton_kinematics_features + asymmetry_features + spin_correlation_features + [   
            "nBTag", "nrecoJet", "nrecoLep", "jet0_pt", "jet1_pt", "ht", "recoLep0_pt", "recoLep1_pt", 
        ]

observers = []

data_generator =  DataGenerator(
    #input_files = [ "/eos/vbc/group/cms/robert.schoefbeck/TT2lUnbinned/training-ntuples-delphes-v1/MVA-training/EFT_delphes_dilep-offZ-njet3p-btag2p-mtt750/TT01j2lCAOldRef_Mtt500_small/TT01j2lCAOldRef_Mtt500_small.root"],
    #input_files = [ "/users/robert.schoefbeck/ML-pytorch/TT2lUnbinned/asimov/TT01j2lCAOldRef_Mtt500_small.root"],
    #input_files = [ "/eos/vbc/group/cms/robert.schoefbeck/TT2lUnbinned/training-ntuples-delphes-v1/MVA-training/EFT_delphes_dilep-offZ-njet3p-btag2p-mtt750/TT01j2lCAOldRef_Mtt500_20percent/TT01j2lCAOldRef_Mtt500_20percent.root"],
    input_files = [ "/eos/vbc/group/cms/robert.schoefbeck/TT2lUnbinned/training-ntuples-delphes-v2/MVA-training/EFT_delphes_dilep-offZ-njet3p-btag2p-mtt750/TT01j2lCAOldRef_Mtt500_50percent/TT01j2lCAOldRef_Mtt500_50percent.root"],
    #input_files = [ "/eos/vbc/group/cms/robert.schoefbeck/TT2lUnbinned/training-ntuples-delphes-v1/MVA-training/EFT_delphes_dilep-offZ-njet3p-btag2p-mtt750/TT01j2lCAOldRef_Mtt500_ext/TT01j2lCAOldRef_Mtt500_ext.root"],
        n_split = 1,
        splitting_strategy = "events",
        selection   = selection,
        branches = ["p_C"] + feature_names   ) 

reweight_pkl = '/eos/vbc/group/cms/robert.schoefbeck/gridpacks/CA_v4/TT01j2lCAOldRef_reweight_card.pkl'
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

    def getEvents( self, nTraining, return_observers = False,  wilson_coefficients = None, feature_names=None, feature_dicts=False):

        index = -1
        if wilson_coefficients is None:
            wilson_coefficients = weightInfo.variables

        coeffs       = data_generator.vector_branch( data_generator[index], 'p_C', padding_target=len(weightInfo.combinations))[:nTraining]
        features     = data_generator.scalar_branches( data_generator[index], self.feature_names if feature_names is None else feature_names)[:nTraining]
        vectors      = None #{key:model.data_generator.vector_branch(data, key ) for key in vector_branches}
        combinations = make_combinations( [ w for w in weightInfo.variables if w in wilson_coefficients] )

        if feature_dicts:
            features = {f:features[:,i_f] for i_f,f in enumerate(feature_names)}

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
    {'color':ROOT.kBlue-4,      'eft':make_eft(cQj31=10), 'tex':"c_{jj}^{31}=10"},
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
                 'min_size': 50 }
