#python choleskyNN_training.py --debug --overwrite training --nTraining 30000
#python choleskyNN_training.py --debug --overwrite training --nTraining 100000
#python choleskyNN_training.py --debug --overwrite training --nTraining 500000
#python choleskyNN_training.py --debug --overwrite training --nTraining 1000000

python choleskyNN_training.py --debug --prefix bias --bias 'pT' '10**(({}-200)/200)' --overwrite training --nTraining 30000
python choleskyNN_training.py --debug --prefix bias --bias 'pT' '10**(({}-200)/200)' --overwrite training --nTraining 100000
python choleskyNN_training.py --debug --prefix bias --bias 'pT' '10**(({}-200)/200)' --overwrite training --nTraining 500000
python choleskyNN_training.py --debug --prefix bias --bias 'pT' '10**(({}-200)/200)' --overwrite training --nTraining 1000000
