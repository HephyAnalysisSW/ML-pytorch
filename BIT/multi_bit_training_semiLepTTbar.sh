#python multi_bit_training.py --plot_directory multiBIT_semiLepTTbar_v4 --prefix ctGRe --model semiLepTTbar --modelFile models --nTraining -1 --coefficients ctGRe --loss CrossEntropy --debug --n_trees 300 --auto_clip 0.002 --feature_plots
#python multi_bit_training.py --plot_directory multiBIT_semiLepTTbar_v4 --prefix ctGIm --model semiLepTTbar --modelFile models --nTraining -1 --coefficients ctGIm --loss CrossEntropy --debug --n_trees 300 --auto_clip 0.002 --feature_plots
#python multi_bit_training.py --plot_directory multiBIT_semiLepTTbar_v4 --prefix ctWRe --model semiLepTTbar --modelFile models --nTraining -1 --coefficients ctWRe --loss CrossEntropy --debug --n_trees 300 --auto_clip 0.002 --feature_plots
#python multi_bit_training.py --plot_directory multiBIT_semiLepTTbar_v4 --prefix ctWIm --model semiLepTTbar --modelFile models --nTraining -1 --coefficients ctWIm --loss CrossEntropy --debug --n_trees 300 --auto_clip 0.002 --feature_plots

python multi_bit_training.py --plot_directory multiBIT_semiLepTTbar_v4 --prefix ctGRe --model semiLepTTbar_delphesJet --modelFile models --nTraining -1 --coefficients ctGRe --loss CrossEntropy --debug --n_trees 300 --auto_clip 0.002 --feature_plots
python multi_bit_training.py --plot_directory multiBIT_semiLepTTbar_v4 --prefix ctGIm --model semiLepTTbar_delphesJet --modelFile models --nTraining -1 --coefficients ctGIm --loss CrossEntropy --debug --n_trees 300 --auto_clip 0.002 --feature_plots
python multi_bit_training.py --plot_directory multiBIT_semiLepTTbar_v4 --prefix ctWRe --model semiLepTTbar_delphesJet --modelFile models --nTraining -1 --coefficients ctWRe --loss CrossEntropy --debug --n_trees 300 --auto_clip 0.002 --feature_plots
python multi_bit_training.py --plot_directory multiBIT_semiLepTTbar_v4 --prefix ctWIm --model semiLepTTbar_delphesJet --modelFile models --nTraining -1 --coefficients ctWIm --loss CrossEntropy --debug --n_trees 300 --auto_clip 0.002 --feature_plots

#python multi_bit_training.py --plot_directory multiBIT_semiLepTTbar_v4 --prefix v4 --model semiLepTTbar --modelFile models --nTraining -1 --coefficients ctGRe ctGIm ctWRe ctWIm --loss CrossEntropy --debug --n_trees 200 --auto_clip 0.002 --feature_plots


#python multi_bit_training.py --plot_directory multiBIT_semiLepTTbar_v4 --prefix v4 --model semiLepTTbar_delphesJet --modelFile models --nTraining -1 --coefficients ctGRe ctGIm ctWRe ctWIm --loss CrossEntropy --debug --n_trees 200 --auto_clip 0.002 --feature_plots
#python multi_bit_training.py --plot_directory multiBIT_semiLepTTbar_v3 --prefix v3 --model semiLepTTbar --modelFile models --nTraining -1 --coefficients ctGRe ctGIm ctWRe ctWIm --loss CrossEntropy --debug --n_trees 400 --auto_clip 0.002 --feature_plots
#python multi_bit_training.py --plot_directory multiBIT_semiLepTTbar_v3 --prefix v3 --model semiLepTTbar_delphesJet --modelFile models --nTraining -1 --coefficients ctGRe ctGIm ctWRe ctWIm --loss CrossEntropy --debug --n_trees 400 --auto_clip 0.002 --feature_plots
