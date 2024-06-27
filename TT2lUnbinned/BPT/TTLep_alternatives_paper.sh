#python bpt_training.py --modelDir delphes_models --model delphes_TTLep_DY        --feature_plots --version v4.1 --nTraining -1 --debug
#python bpt_training.py --modelDir delphes_models --model  delphes_TTLep_DY_red --feature_plots --version v4.1 --nTraining -1 --debug  --learn_global_param 
#python bpt_training.py --modelDir delphes_models --model  delphes_TTLep_DY_red --feature_plots --version v4.2 --nTraining -1 --debug  --learn_global_param 
python bpt_training.py --modelDir delphes_models --model  delphes_TTLep_DY_red --feature_plots --version v4.3  --bias tr_ttbar_pt "10**(({}-200)/300)"  --nTraining -1 --debug  --learn_global_param 
#python bpt_training.py --modelDir delphes_models --model delphes_TTLep_MG_vs_Pow --feature_plots --version v4.1 --nTraining -1 --debug  --learn_global_param 

