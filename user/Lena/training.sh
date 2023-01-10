##conda activate pt
##default config: ipython 2l_MVA_LSTM+db.py -- --n_epochs=500 --batches=20000 --hs1_mult=2 --hs2_add=5 (--LSTM --num_layer=4)
##variation of batch size
# ipython 2l_MVA_LSTM+db.py -- --n_epochs=500 --batches=5000 --hs1_mult=2 --hs2_add=5
# ipython 2l_MVA_LSTM+db.py -- --n_epochs=500 --batches=10000 --hs1_mult=2 --hs2_add=5
# ipython 2l_MVA_LSTM+db.py -- --n_epochs=500 --batches=20000 --hs1_mult=2 --hs2_add=5
# ipython 2l_MVA_LSTM+db.py -- --n_epochs=500 --batches=50000 --hs1_mult=2 --hs2_add=5
# ipython 2l_MVA_LSTM+db.py -- --n_epochs=500 --batches=100000 --hs1_mult=2 --hs2_add=5
 
##variation of first hidden layer size
# ipython 2l_MVA_LSTM+db.py -- --n_epochs=500 --batches=20000 --hs1_mult=1 --hs2_add=5
##ipython 2l_MVA_LSTM+db.py -- --n_epochs=500 --batches=20000 --hs1_mult=2 --hs2_add=5
# ipython 2l_MVA_LSTM+db.py -- --n_epochs=500 --batches=20000 --hs1_mult=3 --hs2_add=5
# ipython 2l_MVA_LSTM+db.py -- --n_epochs=500 --batches=20000 --hs1_mult=4 --hs2_add=5

##variation of second hidden layer size
# ipython 2l_MVA_LSTM+db.py -- --n_epochs=500 --batches=20000 --hs1_mult=2 --hs2_add=-15
# ipython 2l_MVA_LSTM+db.py -- --n_epochs=500 --batches=20000 --hs1_mult=2 --hs2_add=-5
# ipython 2l_MVA_LSTM+db.py -- --n_epochs=500 --batches=20000 --hs1_mult=2 --hs2_add=0
##ipython 2l_MVA_LSTM+db.py -- --n_epochs=500 --batches=20000 --hs1_mult=2 --hs2_add=5
# ipython 2l_MVA_LSTM+db.py -- --n_epochs=500 --batches=20000 --hs1_mult=2 --hs2_add=15

##add LSTM + vary number of layers
# ipython 2l_MVA_LSTM+db.py -- --n_epochs=500 --batches=20000 --hs1_mult=2 --hs2_add=5 --LSTM --num_layers=1
# ipython 2l_MVA_LSTM+db.py -- --n_epochs=500 --batches=20000 --hs1_mult=2 --hs2_add=5 --LSTM --num_layers=2
# ipython 2l_MVA_LSTM+db.py -- --n_epochs=500 --batches=20000 --hs1_mult=2 --hs2_add=5 --LSTM --num_layers=4
# ipython 2l_MVA_LSTM+db.py -- --n_epochs=500 --batches=20000 --hs1_mult=2 --hs2_add=5 --LSTM --num_layers=6
ipython 2l_MVA_LSTM+db.py -- --n_epochs=500 --batches=20000 --hs1_mult=2 --hs2_add=5 --LSTM --num_layers=8

##add LSTM + vary hidden layer lstm
# ipython 2l_MVA_LSTM+db.py -- --n_epochs=500 --batches=20000 --hs1_mult=2 --hs2_add=5 --LSTM --num_layers=4 --LSTM_out=5
# ipython 2l_MVA_LSTM+db.py -- --n_epochs=500 --batches=20000 --hs1_mult=2 --hs2_add=5 --LSTM --num_layers=4 --LSTM_out=15
# ipython 2l_MVA_LSTM+db.py -- --n_epochs=500 --batches=20000 --hs1_mult=2 --hs2_add=5 --LSTM --num_layers=4 --LSTM_out=25

##add LSTM + vary number of layers + DoubleB
# ipython 2l_MVA_LSTM+db.py -- --n_epochs=500 --batches=20000 --hs1_mult=2 --hs2_add=5 --LSTM --num_layers=1 --DoubleB
# ipython 2l_MVA_LSTM+db.py -- --n_epochs=500 --batches=20000 --hs1_mult=2 --hs2_add=5 --LSTM --num_layers=2 --DoubleB
# ipython 2l_MVA_LSTM+db.py -- --n_epochs=500 --batches=20000 --hs1_mult=2 --hs2_add=5 --LSTM --num_layers=4 --DoubleB
# ipython 2l_MVA_LSTM+db.py -- --n_epochs=500 --batches=20000 --hs1_mult=2 --hs2_add=5 --LSTM --num_layers=6 --DoubleB
# ipython 2l_MVA_LSTM+db.py -- --n_epochs=500 --batches=20000 --hs1_mult=2 --hs2_add=5 --LSTM --num_layers=8 --DoubleB

##add LSTM + vary hidden layer lstm + DoubleB
# ipython 2l_MVA_LSTM+db.py -- --n_epochs=500 --batches=20000 --hs1_mult=2 --hs2_add=5 --LSTM --num_layers=4 --LSTM_out=5 --DoubleB
# ipython 2l_MVA_LSTM+db.py -- --n_epochs=500 --batches=20000 --hs1_mult=2 --hs2_add=5 --LSTM --num_layers=4 --LSTM_out=15 --DoubleB
# ipython 2l_MVA_LSTM+db.py -- --n_epochs=500 --batches=20000 --hs1_mult=2 --hs2_add=5 --LSTM --num_layers=4 --LSTM_out=25 --DoubleB
                                                                                                                        
##vary number of training epochs
# ipython 2l_MVA_LSTM+db.py -- --n_epochs=750 --batches=20000 --hs1_mult=2 --hs2_add=5 --LSTM --num_layers=1 --LSTM_out=25
# ipython 2l_MVA_LSTM+db.py -- --n_epochs=750 --batches=20000 --hs1_mult=2 --hs2_add=5 --LSTM --num_layers=2 --LSTM_out=25
#ipython 2l_MVA_LSTM+db.py -- --n_epochs=2000 --batches=20000 --hs1_mult=2 --hs2_add=5 --LSTM --num_layers=4 --LSTM_out=25 
# ipython 2l_MVA_LSTM+db.py -- --n_epochs=750 --batches=20000 --hs1_mult=2 --hs2_add=5 --LSTM --num_layers=1 --LSTM_out=25 --DoubleB
# ipython 2l_MVA_LSTM+db.py -- --n_epochs=750 --batches=20000 --hs1_mult=2 --hs2_add=5 --LSTM --num_layers=2 --LSTM_out=25 --DoubleB
#ipython 2l_MVA_LSTM+db.py -- --n_epochs=2000 --batches=20000 --hs1_mult=2 --hs2_add=5 --LSTM --num_layers=4 --LSTM_out=25 --DoubleB