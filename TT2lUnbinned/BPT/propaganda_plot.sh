#python propaganda_plot.py --model analytic_2D --nTraining 50000 --feature x

#python propaganda_plot.py --model TTLep_bTagSys_paper --version v6_for_paper --era Autumn18  --nTraining -1 --variation b --feature tr_ttbar_pt
#python propaganda_plot.py --model TTLep_bTagSys_paper --version v6_for_paper --era Autumn18  --nTraining -1 --variation b --feature tr_ttbar_dAbsEta
#python propaganda_plot.py --model TTLep_bTagSys_paper --version v6_for_paper --era Autumn18  --nTraining -1 --variation b --feature tr_ttbar_mass
#
#python propaganda_plot.py --model TTLep_bTagSys_paper --version v6_for_paper --era Autumn18  --nTraining -1 --variation l --feature tr_ttbar_pt
#python propaganda_plot.py --model TTLep_bTagSys_paper --version v6_for_paper --era Autumn18  --nTraining -1 --variation l --feature tr_ttbar_dAbsEta
#python propaganda_plot.py --model TTLep_bTagSys_paper --version v6_for_paper --era Autumn18  --nTraining -1 --variation l --feature tr_ttbar_mass


#python propaganda_plot.py --model TTLep_scale_2D_paper --version v4 --era Autumn18 --nTraining -1 --feature tr_ttbar_pt 
#python propaganda_plot.py --model TTLep_scale_2D_paper --version v4 --era Autumn18 --nTraining -1 --feature tr_ttbar_mass
#python propaganda_plot.py --model TTLep_scale_2D_paper --version v4 --era Autumn18 --nTraining -1 --feature tr_ttbar_dEta 
#python propaganda_plot.py --model TTLep_scale_2D_paper --version v4 --era Autumn18 --nTraining -1 --feature tr_ttbar_dAbsEta 
#
#python propaganda_plot.py --model TTLep_scale_2D_paper --version v4 --era Autumn18 --nTraining -1 --feature recoLep_dAbsEta
#python propaganda_plot.py --model TTLep_scale_2D_paper --version v4 --era Autumn18 --nTraining -1 --feature recoLep_dEta
#python propaganda_plot.py --model TTLep_scale_2D_paper --version v4 --era Autumn18 --nTraining -1 --feature tr_abs_delta_phi_ll_lab

#python propaganda_plot.py --modelDir delphes_models --model delphes_TTLep_MG_vs_Pow --version v4.1 --era RunII --nTraining -1 --feature tr_cos_phi_lab
#python propaganda_plot.py --modelDir delphes_models --model delphes_TTLep_MG_vs_Pow --version v4.1 --era RunII --nTraining -1 --feature tr_abs_delta_phi_ll_lab 
#python propaganda_plot.py --modelDir delphes_models --model delphes_TTLep_MG_vs_Pow --version v4.1 --era RunII --nTraining -1 --feature tr_ttbar_mass
#python propaganda_plot.py --modelDir delphes_models --model delphes_TTLep_MG_vs_Pow --version v4.1 --era RunII --nTraining -1 --feature tr_ttbar_dAbsEta 

#python  bpt_training.py --version v7 --model TTLep_JERC_linear_paper --modelDir models --feature_plots --nTraining -1 --debug --variation jesTotal

#python  propaganda_plot.py --model TTLep_JERC_linear_paper --version v7 --nTraining -1 --variation jesTotal    --feature tr_ttbar_mass 
#python  propaganda_plot.py --model TTLep_JERC_linear_paper --version v7 --nTraining -1 --variation jesTotal    --feature tr_ttbar_dAbsEta
#python  propaganda_plot.py --model TTLep_JERC_linear_paper --version v7 --nTraining -1 --variation jesTotal    --feature tr_ttbar_dEta
#python  propaganda_plot.py --model TTLep_JERC_linear_paper --version v7 --nTraining -1 --variation jesTotal    --feature tr_ttbar_pt 


#python  propaganda_plot.py --model TTLep_leptonSF --version v4.1_for_paper --era Autumn18 --nTraining -1  --feature recoLep01_pt 
#python  propaganda_plot.py --model TTLep_leptonSF --version v4.1_for_paper --era Autumn18 --nTraining -1  --feature tr_ttbar_pt 
#python  propaganda_plot.py --model TTLep_leptonSF --version v4.1_for_paper --era Autumn18 --nTraining -1  --feature tr_ttbar_dEta 
#python  propaganda_plot.py --model TTLep_leptonSF --version v4.1_for_paper --era Autumn18 --nTraining -1  --feature tr_ttbar_dAbsEta 
#python  propaganda_plot.py --model TTLep_leptonSF --version v4.1_for_paper --era Autumn18 --nTraining -1  --feature tr_ttbar_mass 

python  propaganda_plot.py --modelDir delphes_models --model delphes_TTLep_DY_red --version v4.2 --nTraining -1  --feature tr_ttbar_mass 
python  propaganda_plot.py --modelDir delphes_models --model delphes_TTLep_DY_red --version v4.2 --nTraining -1  --feature tr_ttbar_pt 
#python  propaganda_plot.py --modelDir delphes_models --model delphes_TTLep_DY_red --version v4.2 --nTraining -1  --feature tr_ttbar_dEta 
python  propaganda_plot.py --modelDir delphes_models --model delphes_TTLep_DY_red --version v4.2 --nTraining -1  --feature tr_ttbar_dAbsEta 
python  propaganda_plot.py --modelDir delphes_models --model delphes_TTLep_DY_red --version v4.2 --nTraining -1  --feature tr_abs_delta_phi_ll_lab 
python  propaganda_plot.py --modelDir delphes_models --model delphes_TTLep_DY_red --version v4.2 --nTraining -1  --feature tr_cos_phi_lab

