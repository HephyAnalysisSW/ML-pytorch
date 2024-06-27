import os

#input_selection = "dilep-offZ-njet3p-btag2p-mtt0-ht500"
#input_selection = "tr-minDLmass20-dilepL-offZ1-njet3p-btag2p-ht500"
input_selection = "dilep-offZ-njet3p-btag2p-mtt750"

location = "/eos/vbc/group/cms/robert.schoefbeck/TT2lUnbinned/training-ntuples-delphes-v1/MVA-training/"

data_locations = {
    "RunII"   :         os.path.join( location, "EFT_delphes_%s/"%input_selection),
    #"Summer16_preVFP":  os.path.join( location, "EFT_Summer16_preVFP_%s/"%input_selection),
    #"Summer16":         os.path.join( location, "EFT_Summer16_%s/"%input_selection),
    #"Fall17"  :         os.path.join( location, "EFT_Fall17_%s/"%input_selection),
    #"Autumn18":         os.path.join( location, "EFT_Autumn18_%s/"%input_selection),
}

selection = None #lambda ar: (ar.overflow_counter_v1==0)
#selection = lambda ar: (ar.tr_ttbar_pt<800)

feature_names_tr = [
        "tr_cosThetaPlus_n",
        "tr_cosThetaMinus_n",
        "tr_cosThetaPlus_r",
        "tr_cosThetaMinus_r",
        "tr_cosThetaPlus_k",
        "tr_cosThetaMinus_k",
        "tr_cosThetaPlus_r_star",
        "tr_cosThetaMinus_r_star",
        "tr_cosThetaPlus_k_star",
        "tr_cosThetaMinus_k_star",
        "tr_xi_nn",
        "tr_xi_rr",
        "tr_xi_kk",
        "tr_xi_nr_plus",
        "tr_xi_nr_minus",
        "tr_xi_rk_plus",
        "tr_xi_rk_minus",
        "tr_xi_nk_plus",
        "tr_xi_nk_minus",

        "tr_xi_r_star_k",
        "tr_xi_k_r_star",
        "tr_xi_kk_star",

        "tr_cos_phi",
        "tr_cos_phi_lab",
        "tr_abs_delta_phi_ll_lab",
        "tr_ttbar_dAbsEta",

        "tr_ttbar_mass",
]
feature_names = feature_names_tr + [
        "recoLep0_pt",
        "recoLep1_pt",
        "jet0_pt",
        "jet1_pt",
        #"jet2_pt",
        "nrecoJet",
        "nBTag",
]

feature_names_dy_reduced = [

]
