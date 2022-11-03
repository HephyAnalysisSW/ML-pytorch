

lstm             = False
lstm_jets_maxN   = 10
lstm_jetVars     = ['pt/F', 'eta/F', 'phi/F', 'btagDeepFlavB/F', 'btagDeepFlavC/F', 'chEmEF/F', 'chHEF/F', 'neEmEF/F', 'neHEF/F', 'muEF/F', 'puId/F', 'qgl/F']
lstm_jetVarNames = [x.split('/')[0] for x in lstm_jetVars]


samples   = ["TTTT", "TTLep_bb", "TTLep_cc", "TTLep_other"] 

directory = "/eos/vbc/group/cms/robert.schoefbeck/TMB/training-ntuples-tttt-v2/MVA-training/tttt_2l/"

content_list = ["mva_nJetGood",
                "mva_nBTag",
                "mva_nlep",
                "mva_mT_l1",
                "mva_mT_l2",
                "mva_ml_12",
                "mva_met_pt",
                "mva_l1_pt",
                "mva_l1_eta",
                "mva_l2_pt",
                "mva_l2_eta",
                "mva_mj_12",
                "mva_mlj_11",              
                "mva_mlj_12",                
                "mva_dPhil_12",              
                "mva_dPhij_12",              
                "mva_dEtal_12",              
                "mva_dEtaj_12",              
                "mva_ht",                    
                "mva_htb",                   
                "mva_ht_ratio",              
                "mva_jet0_pt",               
                "mva_jet0_eta",              
                "mva_jet0_btagDeepFlavB",     
                "mva_jet1_pt",               
                "mva_jet1_eta",              
                "mva_jet1_btagDeepFlavB",      
                "mva_jet2_pt",               
                "mva_jet2_eta",              
                "mva_jet2_btagDeepFlavB",       
                "mva_jet3_pt",               
                "mva_jet4_pt",               
                "mva_jet5_pt",               
                "mva_jet6_pt",               
                "mva_jet7_pt"]