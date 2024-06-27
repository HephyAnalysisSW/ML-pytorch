import os

input_selection = "tr-minDLmass20-dilepM-offZ1-njet3p-btagM2p-mtt750"
#input_selection = "tr-minDLmass20-dilepL-offZ1-njet3p-btag2p-ht500"

location = "/eos/vbc/group/cms/robert.schoefbeck/TT2lUnbinned/training-ntuples-v6/MVA-training/"

data_location =  os.path.join( location, "EFT_for_paper_%s/"%input_selection)

selection = None #lambda ar: (ar.overflow_counter_v1>=-1) #NO selection

feature_names = [

            "tr_ttbar_pt",
            "tr_ttbar_mass",

            "tr_top_pt",
            "tr_topBar_pt",
            "tr_top_eta",
            "tr_topBar_eta",

            "recoLep0_pt",
            "recoLep1_pt",
            "recoLepPos_pt",
            "recoLepNeg_pt",

            "recoLep01_pt",
            "recoLep01_mass",

            "tr_ttbar_dEta",
            "tr_ttbar_dAbsEta",
            "recoLep_dEta",
            "recoLep_dAbsEta",

            "tr_cosThetaPlus_n", "tr_cosThetaMinus_n", "tr_cosThetaPlus_r", "tr_cosThetaMinus_r", "tr_cosThetaPlus_k", "tr_cosThetaMinus_k", "tr_cosThetaPlus_r_star", "tr_cosThetaMinus_r_star", "tr_cosThetaPlus_k_star", "tr_cosThetaMinus_k_star",

            "tr_xi_nn", "tr_xi_rr", "tr_xi_kk", "tr_xi_nr_plus", "tr_xi_nr_minus", "tr_xi_rk_plus", "tr_xi_rk_minus", "tr_xi_nk_plus", "tr_xi_nk_minus",
            "tr_xi_r_star_k", "tr_xi_k_r_star", "tr_xi_kk_star",
            "tr_cos_phi", "tr_cos_phi_lab", "tr_abs_delta_phi_ll_lab",

            "nBTag",
            "nJetGood", #change name when ntuple is updated
#            "ht"
]
