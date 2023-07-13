import torch
import uproot
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn as nn
import os
import argparse
import  logging
from    datetime                 import datetime
import  Tools.syncer_for_gif     as syncer
import shutil 
import  torch.optim.lr_scheduler as lr_scheduler




argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--logLevel',           action='store',                   default='INFO', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], help="Log level for logging")
argParser.add_argument('--batches',       action='store',      default='20000',  help="batch size", type=int)
argParser.add_argument('--n_epochs',      action='store',      default = '500',  help='number of epochs in training', type=int)
argParser.add_argument('--hs1_mult',      action='store',      default = '2',    help='hidden size 1 = #features * mult', type=int)
argParser.add_argument('--hs2_add',       action='store',      default= '5',     help='hidden size 2 = #features + add', type=int)
argParser.add_argument('--LSTM',          action='store_true', default=False,    help='add LSTM?')
argParser.add_argument('--hs_combined',        action='store',      type=int,    default= '5',     help='hidden size of combined layer after LSTM+DNN')
argParser.add_argument('--reduce',             action='store',      type=int,    default=None,     help="Reduce training data by factor?"),
argParser.add_argument('--lr',                 action='store',      type=float,  default= 0.001,  help='learning rate')
argParser.add_argument('--dropout',            action='store',      type=float,  default= 0.5, )
argParser.add_argument('--weight_decay',       action='store',      type=float,  default= None, )
argParser.add_argument('--ReLU_slope',         action='store',      type=float,  default= 0.5, )
argParser.add_argument('--num_layers',    action='store',      default='1',      help='number of LSTM layers', type=int)
argParser.add_argument('--LSTM_out',      action='store',      default= '10',    help='output size of LSTM', type=int)
argParser.add_argument('--Delphes',       action='store_true', default=False,    help='use only reduced DELPHES-like input to LSTM')
argParser.add_argument('--output_directory',   action='store',      type=str,    default='/groups/hephy/cms/lena.wild/tttt/multiclass_study/')
argParser.add_argument('--plot_directory',     action='store',      type=str,    default='/groups/hephy/cms/lena.wild/www/tttt/plots/multiclass/')
argParser.add_argument('--scheduler',          action='store',      type=str,    nargs = 2,  default=[None, None],     help='linear+flat, decay , factor by which the final lr is reduced')
args = argParser.parse_args()


import config_multiclass as config
import  logging
logger = logging.getLogger()
logger_handler = logging.StreamHandler()
logger.addHandler(logger_handler)
logger_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
logger.setLevel(args.logLevel)

def copyIndexPHP( results_dir ):
    index_php = os.path.join( results_dir, 'index.php' )
    shutil.copyfile( os.path.join( "Tools", 'index_for_gif.php' ), index_php )

# set hyperparameters
content_list     = config.content_list
samples          = config.samples
directory        = config.directory
batches          = args.batches
ReLU_slope       = args.ReLU_slope
dropout          = args.dropout
weight_decay     = args.weight_decay if args.weight_decay is not None else 0
device           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate    = args.lr
n_epochs         = args.n_epochs
input_size       = len(content_list) 
hidden_size      = input_size * args.hs1_mult
hidden_size2     = input_size + args.hs2_add
hidden_size_comb = args.hs_combined
output_size      = len(samples)

if (args.LSTM):
    if (args.Delphes): vector_branches = ["mva_Jet_%s" % varname for varname in config.lstm_jetVarNames_delphes]
    if not (args.Delphes): vector_branches = ["mva_Jet_%s" % varname for varname in config.lstm_jetVarNames]
    max_timestep = config.lstm_jets_maxN
    input_size_lstm = len(vector_branches)
    hidden_size_lstm = args.LSTM_out
    num_layers = args.num_layers

# logging.info hyperparameters
logging.info("------------Parameters for training--------------")
logging.info("Number of epochs:                        %i",n_epochs)
logging.info("Number of features,linear layer:         %i",input_size)
logging.info("Size of first hidden layer:              %i",hidden_size)
logging.info("Size of second hidden layer:             %i",hidden_size2)
logging.info("Size of combined layer:                  %i",hidden_size_comb )
logging.info("LSTM:                                    %r",args.LSTM)
if (args.LSTM):
    logging.info("          Number of LSTM layers:         %i", num_layers)
    logging.info("          Output size of LSTM:           %i", hidden_size_lstm)
    logging.info("          Delphes-like input:            %i", args.Delphes)
    logging.info("          Number of features, LSTM:      %i", len(vector_branches))
logging.info("-------------------------------------------------\n")


if args.reduce is not None: red = args.reduce
else: red = 1
logging.info("using only small dataset of 1/%s of total events", args.reduce)

# import training data
x = { sample: uproot.concatenate( os.path.join( directory, "{sample}/{sample}.root".format(sample=sample))) for sample in samples }
x = { sample: np.array( [ getattr( array, branch ) for branch in content_list ] ).transpose() for sample, array in x.items() }


# weight wrt to largest sample
n_max_events= int(max(map( len, x.values() ))/red)
w = {sample:n_max_events/(int(len(x[sample][:,0])/red)) for sample in samples}

y = {sample:i_sample*np.ones(int(len(x[sample][:,0])/red)) for i_sample, sample in enumerate(samples)}
# Note to myself: y... "TTTT":0,0,0..., "TTbb":1,1,1... 

# make weights
samples_weight = np.concatenate([ [w[sample]]*int(len(x[sample][:,0])/red) for sample in samples]) 
sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
n_max_events= int(max(map( len, x.values() ))/red)
w = {sample:n_max_events/(int(len(x[sample][:,0])/red)) for sample in samples}

y = {sample:i_sample*np.ones(int(len(x[sample][:,0]))) for i_sample, sample in enumerate(samples)}
# Note to myself: y... "TTTT":0,0,0..., "TTbb":1,1,1... 

# make weights
samples_weight = np.concatenate([ [w[sample]]*int(len(x[sample][:,0])/red) for sample in samples]) 
sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

# concatenate
X = torch.Tensor(np.concatenate( [x[sample][:int(len(x[sample][:,0])/red),:] for sample in samples] ))
yy = torch.Tensor(np.concatenate( [y[sample][:int(len(x[sample][:,0])/red)] for sample in samples] ))
Y = torch.zeros( (len(yy), len(samples)))
for i_sample in range(len(samples)):
    Y[:,i_sample][yy.int()==i_sample]=1
V = np.zeros((len(Y)))

X_ = torch.Tensor(np.concatenate( [x[sample][int(len(x[sample][:,0])/red):2*int(len(x[sample][:,0])/red),:] for sample in samples] ))
yy = torch.Tensor(np.concatenate( [y[sample][int(len(x[sample][:,0])/red):2*int(len(x[sample][:,0])/red)] for sample in samples] ))
Y_ = torch.zeros( (len(yy), len(samples)))
for i_sample in range(len(samples)):
    Y_[:,i_sample][yy.int()==i_sample]=1
V_ = np.zeros((len(Y_)))

# add lstm if needed
if (args.LSTM):
    vec_br_f  = {}
    vec_br_f_  = {}
    for i_training_sample, training_sample in enumerate(samples):
        upfile_name = os.path.join(os.path.expandvars(directory), training_sample, training_sample+'.root')
        with uproot.open(upfile_name) as upfile:
            vec_br_f[i_training_sample]   = {}
            vec_br_f_[i_training_sample]   = {}
            #logging.info(vec_br_f)
            for name, branch in upfile["Events"].arrays(vector_branches, library = "np").items():
                for i in range (branch.shape[0]):
                    branch[i]=np.pad(branch[i][:max_timestep], (0, max_timestep - len(branch[i][:max_timestep])))
                    
                vec_br_f[i_training_sample][name] = branch[:int(len(x[training_sample][:,0])/red)]
                vec_br_f_[i_training_sample][name] = branch[int(len(x[training_sample][:,0])/red):2*int(len(x[training_sample][:,0])/red)]
                
    vec_br = {name: np.concatenate( [vec_br_f[i_training_sample][name] for i_training_sample in range(len(samples))] ) for name in vector_branches}
    vec_br_ = {name: np.concatenate( [vec_br_f_[i_training_sample][name] for i_training_sample in range(len(samples))] ) for name in vector_branches}

    # put columns side by side and transpose the innermost two axis
    V = np.column_stack( [np.stack(vec_br[name]) for name in vector_branches] ).reshape( len(Y), len(vector_branches), max_timestep).transpose((0,2,1))
    V = np.nan_to_num(V)
    V_ = np.column_stack( [np.stack(vec_br_[name]) for name in vector_branches] ).reshape( len(Y_), len(vector_branches), max_timestep).transpose((0,2,1))
    V_ = np.nan_to_num(V_)
V = torch.Tensor(V)
V_ = torch.Tensor(V_)


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
                          batch_size=len(X[:,0]),
                          sampler = sampler,
                          num_workers=0)

# set up NN
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, hidden_size_comb, output_size, input_size_lstm, hidden_size_lstm, num_layers):
        super(NeuralNet, self).__init__()
        self.batchn = nn.BatchNorm1d(input_size)
        self.linear1 = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_size),
            nn.LeakyReLU(ReLU_slope),
            nn.Dropout(dropout),
        )
        self.linear2 = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size2),
            nn.LeakyReLU(ReLU_slope),
            nn.Dropout(dropout),
        )
        
        #self.softm = nn.Softmax(dim = 1)
        
        if (args.LSTM):
            self.num_layers = num_layers
            self.hidden_size_lstm = hidden_size_lstm
            self.lstm = nn.LSTM(input_size_lstm, hidden_size_lstm, num_layers, batch_first=True)
            self.linear3 = nn.Sequential(
                nn.LeakyReLU(ReLU_slope),
                nn.Linear(in_features=hidden_size2+hidden_size_lstm, out_features=hidden_size_comb),
                nn.Dropout(dropout),
                )
        else:
            self.linear3 = nn.Sequential(
                nn.LeakyReLU(ReLU_slope),
                nn.Linear(in_features=hidden_size2, out_features=hidden_size_comb),
                nn.Dropout(dropout),
            ) 
        self.linear4 = nn.Sequential(
                nn.LeakyReLU(ReLU_slope),
                nn.Linear(in_features=hidden_size_comb, out_features=output_size),
                nn.Softmax(dim = 1),
                )
        
        
    def forward(self, x, y):
        # set linear layers
        x1 = self.batchn(x)
        x1 = self.linear1(x1)
        x1 = self.linear2(x1)
        
        # add lstm
        if (args.LSTM):
            h0 = torch.zeros(self.num_layers, y.size(0), self.hidden_size_lstm).to(device) 
            c0 = torch.zeros(self.num_layers, y.size(0), self.hidden_size_lstm).to(device) 
            x2, _ = self.lstm(y, (h0,c0))
            x2 = x2[:, -1, :]        
            x1 = torch.cat([x1, x2], dim=1)          
        x1 = self.linear3(x1)
        x1 = self.linear4(x1)

        return x1


if (args.LSTM==False):    
    model = NeuralNet(input_size, hidden_size,hidden_size2, hidden_size_comb, output_size, input_size_lstm=0, hidden_size_lstm=0, num_layers=0).to(device)    
else:
    model = NeuralNet(input_size, hidden_size, hidden_size2, hidden_size_comb, output_size, input_size_lstm, hidden_size_lstm, num_layers).to(device) 
    
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) 

if (args.scheduler[0] is not None):   
    if args.scheduler[0] == 'linear+flat': scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=float(args.scheduler[1]), total_iters=int(n_epochs*9/10)) 
    if args.scheduler[0] == 'decay': scheduler = lr_scheduler.ExponentialLR(optimizer, gamma = float(args.scheduler[1])) 
    logging.info("using %s scheduler with factor %s from initial lr rate %s", args.scheduler[0], args.scheduler[1],  args.lr)
else: scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1, total_iters=n_epochs)

losses = []
losses_ = []
logging.info(model)
# set up directory and model names
dir_name = 'multiclass'+'_e'+str(n_epochs)+'_'+str(hidden_size)+'-'+str(hidden_size2)+'-'+str(hidden_size_comb)
if (args.LSTM): 
    if (args.Delphes): dir_name = dir_name +  '_lstm'+str(num_layers)+'x'+str(hidden_size_lstm)+'_Delphes'
    else: dir_name = dir_name +  '_lstm'+str(num_layers)+'x'+str(hidden_size_lstm)
if (args.reduce is not None):                            dir_name = dir_name + '_r'+str(args.reduce)    
if (args.scheduler is not None):        dir_name = dir_name + "_s"  + str(args.scheduler[0])+str(args.scheduler[1] ) 
if (args.dropout is not None):          dir_name = dir_name + "_do"  + str(args.dropout ) 
if (args.weight_decay is not None):     dir_name = dir_name + "_wd"  + str(args.weight_decay ) 
dir_name = dir_name + "_ReLU"  + str(args.ReLU_slope ) 


results_dir = args.output_directory
if not os.path.exists( results_dir ): os.makedirs( results_dir )
plot_dir = args.plot_directory
if not os.path.exists( plot_dir ): os.makedirs( plot_dir )
model_dir = os.path.join(args.output_directory, dir_name)
if not os.path.exists( model_dir ): os.makedirs( model_dir )

# train the model
logging.info("starting training") 
logging.info("")
logger_handler.setFormatter(logging.Formatter('\x1b[80D\x1b[1A\x1b[K%(asctime)s %(message)s'))

# train the model
for epoch in range(n_epochs):
    start = datetime.now()  
    for i, data in enumerate(train_loader):
        inputs1,inputs2, labels = data
        z = model(inputs1, inputs2)
        loss=criterion(z,labels)
        losses.append(loss.data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            zz = model(X_, V_)
            loss_=criterion(zz,Y_)
            losses_.append(loss_.data)
        logging.info("		epoch: %i of %i      loss(tr/ev): %s, %s , lr=%s", epoch+1, n_epochs, losses[-1], losses_[-1],  scheduler.get_last_lr())
        if (args.n_epochs/(epoch+1)==10):    
                end = datetime.now()
                logging.info('estimate for training duration: {} \n'.format((end - start)*10))
                logging.info('training will finish appx. at {} \n'.format((end - start)*9+end))
 
logger_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))    
logging.info("done with training, plotting losses")  
# plot losses 
fig, ay = plt.subplots()        
plt.plot(losses)
plt.plot(losses_)
plt.title("Losses over epoch")
sample_file_name = str(dir_name)+"_losses.png"
plt.savefig(os.path.join(args.plot_directory,sample_file_name))

# plot the model classification histograms
with torch.no_grad():
    z = model(X, V)
    y_testpred = torch.max(z.data, 1).indices.numpy() 
    y_testtrue = torch.max(Y.data,1).indices.numpy()
    hist1 = []; hist2 = []; hist3 = []; hist4 = []
    bins = [0, 1, 2, 3, 4]
    for j in range (Y.size(0)):
        if (y_testtrue[j] == 0): hist1.append(y_testpred[j])
        if (y_testtrue[j] == 1): hist2.append(y_testpred[j])
        if (y_testtrue[j] == 2): hist3.append(y_testpred[j])
        if (y_testtrue[j] == 3): hist4.append(y_testpred[j])
    fig, az = plt.subplots(figsize = (7,7))
    plt.xticks([])
    plt.hist([hist1, hist2, hist3, hist4], bins, stacked = True,label = ["TTTT", "TTLep_bb","TTLep_cc","TTLep_other"]) 
    plt.legend()
    lab = str(dir_name)
    plt.title(lab)
    sample_file_name = str(dir_name)+".png"
    plt.savefig(os.path.join(args.plot_directory,sample_file_name))
    
with torch.no_grad():
    z = model(X_, V_)
    y_testpred = torch.max(z.data, 1).indices.numpy() 
    y_testtrue = torch.max(Y_.data,1).indices.numpy()
    hist1 = []; hist2 = []; hist3 = []; hist4 = []
    bins = [0, 1, 2, 3, 4]
    for j in range (Y_.size(0)):
        if (y_testtrue[j] == 0): hist1.append(y_testpred[j])
        if (y_testtrue[j] == 1): hist2.append(y_testpred[j])
        if (y_testtrue[j] == 2): hist3.append(y_testpred[j])
        if (y_testtrue[j] == 3): hist4.append(y_testpred[j])
    fig, az = plt.subplots(figsize = (7,7))
    plt.xticks([])
    plt.hist([hist1, hist2, hist3, hist4], bins, stacked = True,label = ["TTTT", "TTLep_bb","TTLep_cc","TTLep_other"]) 
    plt.legend()
    lab = str(dir_name)
    plt.title(lab)
    sample_file_name = str(dir_name)+"_eval.png"
    plt.savefig(os.path.join(args.plot_directory,sample_file_name))   
    logging.info("saved plots to %s", os.path.join(args.plot_directory,sample_file_name))      


    # save model
    x = X[0,:].reshape(1,len(content_list))
    if (args.LSTM):
        v = V[0,:,:].reshape(1, max_timestep, len(vector_branches))
        name = str(dir_name)+".onnx"
    else: 
        v = V[0].reshape(1,1)
        name = str(dir_name)+".onnx"      
    torch.onnx.export(model,args=(x, v),f=os.path.join(results_dir, name),input_names=["input1", "input2"],output_names=["output1"]) 
    logging.info("saved model to %s", os.path.join(results_dir, name)) 
logging.info("")
logging.info("loss plot link: %s", os.path.join('https://lwild.web.cern.ch/tttt/plots/', os.path.basename(os.path.normpath(args.plot_directory)),dir_name+"_losses.png")) 
  
copyIndexPHP(os.path.join(args.plot_directory))
syncer.sync()  


with torch.no_grad():
    z = np.concatenate((np.array(model(X_, V_)),np.array(model(X,V))))
    TTTT = z[:,0]
    TTbb = z[:,1]
    TTcc = z[:,2]
    TTOt = z[:,3]
    TTTT = np.histogram(TTTT, bins=500, range=(0,1))[0]
    TTbb = np.histogram(TTbb, bins=500, range=(0,1))[0]
    TTcc = np.histogram(TTcc, bins=500, range=(0,1))[0]
    TTOt = np.histogram(TTOt, bins=500, range=(0,1))[0]
    TTTT = TTTT/np.sum(TTTT)
    bkgsum = np.sum(TTbb)+np.sum(TTcc)+np.sum(TTOt)
    TTbb = TTbb / bkgsum 
    TTcc = TTcc / bkgsum 
    TTOt = TTOt / bkgsum 
    sig_eff = []
    bkg_eff = [] 
    for i_bin in reversed(range(0,500)):
        sig_eff.append(np.sum(TTTT[i_bin:500]))
        bkg_eff.append(np.sum(TTbb[i_bin:500])+np.sum(TTcc[i_bin:500])+np.sum(TTOt[i_bin:500]))
                #print (sig_eff[-1],bkg_eff[-1])
    fig, r = plt.subplots()
    r.plot(bkg_eff, sig_eff)
    r.set_ylim((0.6,1))
    r.set_xlim((0,0.4))
    plt.savefig(os.path.join(args.plot_directory, dir_name+"roc.png"))
copyIndexPHP(os.path.join(args.plot_directory))
syncer.sync()  