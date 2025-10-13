import pickle
import random
import ROOT
from math import pi
if __name__=="__main__":
    import sys
    sys.path.append('..')

from tools.DataGenerator import DataGenerator
from tools.PDFParametrization import PDFParametrization

selection = None #lambda ar: (ar.nrecoLep==2) & (ar.recoLepNeg_pt>0) & (ar.recoLepPos_pt>0) #& (ar.recoBj01_pt > 0) & (ar.recoBj01_mass>0) 

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
            "nBTag", "nrecoJet", "jet0_pt", "jet1_pt", "ht", "recoLep0_pt", "recoLep1_pt", 
        ]

observers = ["Generator_x1", "Generator_x2", "Generator_id1", "Generator_id2"]

pdf = PDFParametrization(n = 5)

default_pdf_parameters = {p:0 for p in pdf.variables}
def make_pdf(**kwargs):
    result = { key:val for key, val in default_pdf_parameters.items() }
    for key, val in kwargs.items():
        if not key in pdf.variables:
            raise RuntimeError ("Coefficient not known.")
        else:
            result[key] = float(val)
    return result

sm         = make_pdf()

data_generator =  DataGenerator(
    input_files = [ "/scratch-cbe/users/robert.schoefbeck/TT2lUnbinned/training-ntuples-v7/MVA-training/PDF_tr-minDLmass20-dilepM-offZ1-njet3p-btagM2p/TTLep_Summer16_preVFP/TTLep_Summer16_preVFP.root"],
        n_split = 1,
        splitting_strategy = "events",
        selection   = selection,
        branches = ["Generator_x1", "Generator_x2", "Generator_id1", "Generator_id2"] + feature_names   ) 

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

    def getEvents( self, nTraining, return_observers = False, feature_names=None, feature_dicts=False):

        index = -1

        generator    = data_generator.scalar_branches( data_generator[index], ['Generator_x1', 'Generator_x2', 'Generator_id1', 'Generator_id2'])[:nTraining]
        features     = data_generator.scalar_branches( data_generator[index], self.feature_names if feature_names is None else feature_names)[:nTraining]
        vectors      = None #{key:model.data_generator.vector_branch(data, key ) for key in vector_branches}

        derivatives = pdf.derivatives( x1=generator[:,0], x2=generator[:,1], id1=generator[:,2], id2=generator[:,3] )  

        if feature_dicts:
            features = {f:features[:,i_f] for i_f,f in enumerate(feature_names)}

        if return_observers:
            observers_ = data_generator.scalar_branches( data_generator[index], observers )[:nTraining]
            return features, {comb:derivatives[:,pdf.combinations.index(comb)] for comb in pdf.combinations}, observers_
        else: 
            return features, {comb:derivatives[:,pdf.combinations.index(comb)] for comb in pdf.combinations}


pdf_plot_points = [
    {'color':ROOT.kBlack,       'pdf':sm, 'tex':"SM"},

    {'color':ROOT.kGreen+2,     'pdf':make_pdf(c1=.1), 'tex':"c_{1}=.1"},
    {'color':ROOT.kGreen-4,     'pdf':make_pdf(c1=-.1), 'tex':"c_{1}=-.1"},
    {'color':ROOT.kBlue-7,      'pdf':make_pdf(c2=.1), 'tex':"c_{2}=.1"},
    {'color':ROOT.kBlue+1,      'pdf':make_pdf(c2=-.1), 'tex':"c_{2}=-.1"},
    {'color':ROOT.kRed+1,       'pdf':make_pdf(c3=.1), 'tex':"c_{3}=.1"},
    {'color':ROOT.kRed-7,       'pdf':make_pdf(c3=-.1), 'tex':"c_{3}=-.1"},
    {'color':ROOT.kCyan+2,      'pdf':make_pdf(c4=.1), 'tex':"c_{4}=.1"},
    {'color':ROOT.kCyan-7,      'pdf':make_pdf(c4=-.1), 'tex':"c_{4}=-.1"},
    {'color':ROOT.kOrange,      'pdf':make_pdf(c5=.1), 'tex':"c_{5}=.1"},
    {'color':ROOT.kOrange+1,    'pdf':make_pdf(c5=-.1), 'tex':"c_{5}=-.1"},
]

multi_bit_cfg = {'n_trees': 300,
                 'max_depth': 4,
                 'learning_rate': 0.20,
                 'min_size': 50,
                 'learn_global_score': True,
                    }
jax_bit_cfg = {'n_trees': 300,
                 'max_depth': 4,
                 'learning_rate': 0.20,
                 'min_size': 50,
                 'learn_global_score': True,
                 'max_n_split':64,
                 'loss':"CrossEntropy",
                    }
