
#python multi_bit_training.py --plot_directory multiBIT_TT01j2lCA_HT500 --prefix ctGRe --model TT01j2lCA_HT500 --modelFile models --nTraining -1 --coefficients ctGRe cQj18 cQj38 ctj8 ctj1 --loss CrossEntropy --debug --n_trees 300 --feature_plots
python multi_bit_training.py --plot_directory TT01j2lCAv2Ref_HT500_FullSim --prefix v1 --model TT01j2lCAv2Ref_HT500_FullSim --modelFile models --nTraining -1 --coefficients ctGRe cQj18 cQj38 ctj8 ctj1 --loss CrossEntropy --debug --n_trees 300 --feature_plots

python multi_bit_training.py --plot_directory TT01j2lCAv2Ref_HT500_FullSim --prefix v1 --model TT01j2lCAv2Ref_HT500_FullSim --modelFile models --nTraining -1 --coefficients ctGRe cQj18 cQj38 ctj8 ctj1 --loss CrossEntropy --n_trees 300 #SPLIT100 
