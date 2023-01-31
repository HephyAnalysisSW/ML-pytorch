

import torch
import uproot
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn as nn
import os
import argparse



argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--logLevel',           action='store',      default='INFO', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], help="Log level for logging")
argParser.add_argument('--sample',             action='store', type=str,   default='TTTT_MS')
argParser.add_argument('--output_directory',   action='store', type=str,   default='/groups/hephy/cms/lena.wild/tttt/models/')
argParser.add_argument('--input_directory',    action='store', type=str,   default='/eos/vbc/group/cms/lena.wild/tttt/training-ntuples-tttt/MVA-training/ttbb_2l_dilep-met30-njet4p-btag2p/')
argParser.add_argument('--batches',            action='store',      default='20000',  help="batch size", type=int)
argParser.add_argument('--n_epochs',           action='store',      default = '500',  help='number of epochs in training', type=int)
argParser.add_argument('--hs1_mult',           action='store',      default = '2',    help='hidden size 1 = #features * mult', type=int)
argParser.add_argument('--hs2_add',            action='store',      default= '5',     help='hidden size 2 = #features + add', type=int)
argParser.add_argument('--LSTM',               action='store_true', default=False,    help='add LSTM?')
argParser.add_argument('--num_layers',         action='store',      default='1',      help='number of LSTM layers', type=int)
argParser.add_argument('--LSTM_out',           action='store',      default= '1',     help='output size of LSTM', type=int)
args = argParser.parse_args()


import ttbb_2l_python3 as config

import logging
logger = logging.getLogger(__name__)


# set hyperparameters
mva_variables    = [ mva_variable[0] for mva_variable in config.mva_variables ]
sample          = args.sample
batches          = args.batches
device           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate    = 0.001
n_epochs         = args.n_epochs
input_size       = len(mva_variables) 
hidden_size      = input_size * args.hs1_mult
hidden_size2     = input_size + args.hs2_add
output_size      = 1

if (args.LSTM):
    vector_branches = ["mva_Jet_%s" % varname for varname in config.lstm_jetVarNames] #Achtung
    max_timestep = config.lstm_jets_maxN
    input_size_lstm = len(vector_branches)
    hidden_size_lstm = args.LSTM_out
    num_layers = args.num_layers

# print hyperparameters
print("\n------------Parameters for training--------------")
print("Number of epochs:                        ",n_epochs)
print("Batch size:                              ",batches)
print("Number of features,linear layer:         ",input_size)
print("Size of first hidden layer:              ",hidden_size)
print("Size of second hidden layer:             ",hidden_size2)
print("LSTM:                                    ",args.LSTM)
if (args.LSTM):
    print("          Number of LSTM layers:         ", num_layers)
    print("          Output size of LSTM:           ", hidden_size_lstm)
    print("          Number of features, LSTM:      ", len(vector_branches))
print("-------------------------------------------------",'\n')

# import training data
x = uproot.open( os.path.join( args.input_directory, sample, sample+".root")) 
x = x["Events"].arrays(mva_variables, library = "np")
x = np.array( [ x[branch] for branch in mva_variables ] ).transpose() 
y = np.zeros((len(x),1)) 


X = torch.Tensor(x)
Y = torch.Tensor(y)
V = np.zeros((len(y)))


# add lstm if needed
if (args.LSTM):
    vec_br_f  = {}
    upfile_name = os.path.join(args.input_directory, sample, sample+".root")
    with uproot.open(upfile_name) as upfile:
        for name, branch in upfile["Events"].arrays(vector_branches, library = "np").items():
            for i in range (branch.shape[0]):
                branch[i]=np.pad(branch[i][:max_timestep], (0, max_timestep - len(branch[i][:max_timestep])))
            vec_br_f[name] = branch
                
    # put columns side by side and transpose the innermost two axis
    V = np.column_stack( [np.stack(vec_br_f[name]) for name in vector_branches] ).reshape( len(Y), len(vector_branches), max_timestep).transpose((0,2,1))
    V = np.nan_to_num(V)

V = torch.Tensor(V)


# new dataset for handing data to the MVA
class NewDataset(Dataset):

    def __init__(self, X,V,Y):
        self.x = X
        self.v = V
        self.y = Y
        self.n_samples = len(Y)
            
    def __getitem__(self, index):
        return self.x[index],self.v[index], self.y[index]

    def __len__(self):
        return self.n_samples
    
    def __givey__(self):
        return self.y

dataset = NewDataset(X,V,Y)
train_loader = DataLoader(dataset=dataset,
                          batch_size=batches,
                          num_workers=0)

# set up NN
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, output_size, input_size_lstm, hidden_size_lstm, num_layers):
        super(NeuralNet, self).__init__()
        self.batchn = nn.BatchNorm1d(input_size)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size2)
        self.softm = nn.Softmax(dim = 1)
        
        if (args.LSTM):
            self.num_layers = num_layers
            self.hidden_size_lstm = hidden_size_lstm
            self.lstm = nn.LSTM(input_size_lstm, hidden_size_lstm, num_layers, batch_first=True)
            self.linear3 = nn.Linear(hidden_size2+hidden_size_lstm, output_size)
        else:
            self.linear3 = nn.Linear(hidden_size2, output_size)    
        
        
        
    def forward(self, x, y):
        # set linear layers
        x1 = self.batchn(x)
        x1 = self.linear1(x1)
        x1 = self.relu(x1)
        x1 = self.linear2(x1)
        
        # add lstm
        if (args.LSTM):
            h0 = torch.zeros(self.num_layers, y.size(0), self.hidden_size_lstm).to(device) 
            c0 = torch.zeros(self.num_layers, y.size(0), self.hidden_size_lstm).to(device) 
            x2 = self.relu(y)
            x2, _ = self.lstm(x2, (h0,c0))
            x2 = x2[:, -1, :]        
            x1 = torch.cat([x1, x2], dim=1)          
        x1 = self.relu(x1)
        x1 = self.linear3(x1)
        #x1 = self.softm(x1)        
        return x1


if (args.LSTM==False):    
    model = NeuralNet(input_size, hidden_size,hidden_size2, output_size, input_size_lstm=0, hidden_size_lstm=0, num_layers=0).to(device)    
else:
    model = NeuralNet(input_size, hidden_size, hidden_size2, output_size, input_size_lstm, hidden_size_lstm, num_layers).to(device) 
    
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
losses = []

# set up directory and model names
dir_name = 'eft_b-'+str(batches)+'_e-'+str(n_epochs)+'_hs1-'+str(hidden_size)+'_hs2-'+str(hidden_size2)
if (args.LSTM): 
    dir_name = dir_name +  '_lstm-'+str(num_layers)+'_hs-lstm-'+str(hidden_size_lstm)

results_dir = args.output_directory
if not os.path.exists( results_dir ): 
    os.makedirs( results_dir )


# train the model
for epoch in range(n_epochs):
    print("		epoch: ", epoch+1 , " of ", n_epochs)
    for i, data in enumerate(train_loader):
        inputs1,inputs2, labels = data
        z = model(inputs1, inputs2)
        loss=criterion(z,labels)
        losses.append(loss.data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
              
## plot losses 
# fig, ay = plt.subplots()        
# plt.plot(losses)
# plt.title("Losses over epoch")
# sample_file_name = str(dir_name)+"_losses.png"
# plt.savefig(sample_file_name)

# save model
with torch.no_grad():
    x = X[0,:].reshape(1,len(mva_variables))
    if (args.LSTM):
        v = V[0,:,:].reshape(1, max_timestep, len(vector_branches))
        name = str(dir_name)+".onnx"
    else: 
        v = V[0].reshape(1,1)
        name = str(dir_name)+".onnx"      
    torch.onnx.export(model,args=(x, v),f=os.path.join(results_dir, name),input_names=["input1", "input2"],output_names=["output1"]) 
  