# neural net (with LSTM)



import  torch
from    torch                   import Tensor
import  uproot  
import  numpy                   as np
from    matplotlib              import pyplot as plt
from    torch.utils.data        import Dataset, DataLoader, WeightedRandomSampler
import  torch.nn                as nn
import  os
import  argparse
import  matplotlib.animation    as manimation 
from    WeightInfo              import WeightInfo #have it in my local folder
import  itertools
from    multiprocessing         import Pool
import  logging


argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--logLevel',           action='store',                   default='INFO', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], help="Log level for logging")
argParser.add_argument('--sample',             action='store',      type=str,    default='TTTT_MS')
argParser.add_argument('--output_directory',   action='store',      type=str,    default='/groups/hephy/cms/lena.wild/tttt/models/')
argParser.add_argument('--input_directory',    action='store',      type=str,    default='/eos/vbc/group/cms/lena.wild/tttt/training-ntuples-tttt_v4/MVA-training/ttbb_2l_dilep-met30-njet4p-btag2p/')
argParser.add_argument('--n_epochs',           action='store',      type=int,    default = '500',  help='number of epochs in training')
argParser.add_argument('--hs1_mult',           action='store',      type=int,    default = '2',    help='hidden size 1 = #features * mult')
argParser.add_argument('--hs2_add',            action='store',      type=int,    default= '5',     help='hidden size 2 = #features + add')
argParser.add_argument('--LSTM',               action='store_true',              default=False,    help='add LSTM?')
argParser.add_argument('--num_layers',         action='store',      type=int,    default='1',      help='number of LSTM layers')
argParser.add_argument('--LSTM_out',           action='store',      type=int,    default= '1',     help='output size of LSTM')
argParser.add_argument('--nbins',              action='store',      type=int,    default='20',     help='number of bins')
argParser.add_argument('--EFTCoefficients',    action='store',                   default='ctt',    help="Training vectors for particle net")
argParser.add_argument('--animate_step',       action='store',      type=int,    default= '10',    help="plot every n epochs")
argParser.add_argument('--animate_fps' ,       action='store',      type=int,    default= '10',    help="frames per second in animation")
argParser.add_argument('--animate' ,           action='store_true',              default= False,   help="make an animation?")
args = argParser.parse_args()


logging.basicConfig(filename=None,  format='%(asctime)s %(message)s', level=logging.INFO)


# adding hard coded reweight_pkl because of ML-pytorch
if ( args.sample == "TTTT_MS" ): reweight_pkl = "/eos/vbc/group/cms/robert.schoefbeck/gridpacks/4top/TTTT_MS_reweight_card.pkl" 
if ( args.sample == "TTbb_MS" ): reweight_pkl = "/eos/vbc/group/cms/robert.schoefbeck/gridpacks/4top/TTbb_MS_reweight_card.pkl" 


import ttbb_2l_python3 as config

# set hyperparameters
mva_variables    = [ mva_variable[0] for mva_variable in config.mva_variables ]
sample           = args.sample
device           = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
learning_rate    = 0.001
n_epochs         = args.n_epochs
input_size       = len( mva_variables ) 
hidden_size      = input_size * args.hs1_mult
hidden_size2     = input_size + args.hs2_add
output_size      = 2                        # hard coded: lin, quad

if ( args.LSTM ):
    vector_branches = ["mva_Jet_%s" % varname for varname in config.lstm_jetVarNames] 
    max_timestep = config.lstm_jets_maxN
    input_size_lstm = len( vector_branches )
    hidden_size_lstm = args.LSTM_out
    num_layers = args.num_layers


# print hyperparameter selection
logging.info( "------------Parameters for training----------------" )
logging.info( "  Number of epochs:                        %i",n_epochs )
logging.info( "  Number of features,linear layer:         %i",input_size )
logging.info( "  Size of first hidden layer:              %i",hidden_size )
logging.info( "  Size of second hidden layer:             %i",hidden_size2 )
logging.info( "  LSTM:                                    %r",args.LSTM )
if ( args.LSTM ):
    logging.info( "          Number of LSTM layers:         %i", num_layers )
    logging.info( "          Output size of LSTM:           %i", hidden_size_lstm )
    logging.info( "          Number of features, LSTM:      %i", len(vector_branches) )
logging.info( "---------------------------------------------------\n" )

# set weights 
weightInfo = WeightInfo( reweight_pkl ) 
weightInfo.set_order( 2 ) 
index_lin  = weightInfo.combinations.index( (args.EFTCoefficients,) ) 
index_quad = weightInfo.combinations.index( (args.EFTCoefficients,args.EFTCoefficients) )

# import training data
upfile_name = os.path.join( args.input_directory, sample, sample+".root" )
xx     = uproot.open( upfile_name ) 
xx     = xx["Events"].arrays( mva_variables, library = "np" )
x      = np.array( [ xx[branch] for branch in mva_variables ] ).transpose() 

weigh = {}
with uproot.open(upfile_name) as upfile:
    for name, branch in upfile["Events"].arrays( "p_C", library = "np" ).items(): 
        weigh = [ (branch[i][0], branch[i][index_lin], branch[i][index_quad]) for i in  range (branch.shape[0]) ]
        # check number of weights
        assert len( weightInfo.combinations ) == branch[0].shape[0] , "got p_C wrong: found %i weights but need %i" %( branch[0].shape[0], len( weightInfo.combinations ) )
    y = np.asarray( weigh )
    

V = np.zeros( (len(y[:,0])) )
# add lstm if needed
if ( args.LSTM ):
    vec_br_f  = {}
    upfile_name = os.path.join( args.input_directory, sample, sample+".root" )
    with uproot.open( upfile_name ) as upfile:
        for name, branch in upfile["Events"].arrays( vector_branches, library = "np" ).items():
            for i in range ( branch.shape[0] ):
                branch[i]=np.pad( branch[i][:max_timestep], (0, max_timestep - len(branch[i][:max_timestep])) )
            vec_br_f[name] = branch
    # put columns side by side and transpose the innermost two axis
    V = np.column_stack( [np.stack( vec_br_f[name] ) for name in vector_branches] ).reshape( len(y), len(vector_branches), max_timestep ).transpose((0,2,1))
    

#double check for NaNs:
assert not np.isnan( np.sum(x) ), logging.info("found NaNs in DNN input!")
assert not np.isnan( np.sum(y) ), logging.info("found NaNs in DNN truth values!")
assert not np.isnan( np.sum(V) ), logging.info("found NaNs in LSTM input!")

X = torch.Tensor( x )
Y = torch.Tensor( y )
V = torch.Tensor( V )


# Define steps for evaluation 
def eval_train ( var_evaluation ):
    z = np.array( model(X, V) )
    x_eval = np.array ( xx[var_evaluation] )
    hist, bins = np.histogram( x_eval, bins=nbins, range=(config.plot_mva_variables[var_evaluation][0][0], config.plot_mva_variables[var_evaluation][0][1]) )
    train_lin  = np.zeros( (len(bins),1) )
    train_quad = np.zeros( (len(bins),1) )
    for b in range ( 1,len(bins)-1 ):
        for ind in range ( x_eval.shape[0] ):
            val = x_eval[ind]
            if ( val > bins[b-1] and val<= bins[b] ):
                train_lin[b] += y[ind,0]*z[ind,0]
                train_quad[b]+= y[ind,0]*z[ind,1]
    plots[var_evaluation+'_lin' ].set_data( bins, train_lin[:,0]  )               
    plots[var_evaluation+'_quad'].set_data( bins, train_quad[:,0] )   

  
def eval_truth ( var_evaluation ):
    x_eval = np.array ( xx[var_evaluation] )
    hist, bins  = np.histogram( x_eval, bins=nbins, range=(config.plot_mva_variables[var_evaluation][0][0], config.plot_mva_variables[var_evaluation][0][1]) )
    truth_lin   = np.zeros( (len(bins),1) )
    truth_quad  = np.zeros( (len(bins),1) )
    for b in range ( 1,len(bins)-1 ):
        for ind in range ( x_eval.shape[0] ):
            val = x_eval[ind]
            if ( val > bins[b-1] and val<= bins[b] ):
                truth_lin[b] +=y[ind,1]
                truth_quad[b]+=y[ind,2] 
    i = plotvars.index( var_evaluation )   
    plots[var_evaluation+'truelin'],  = ax[index[i]].plot( bins,truth_lin[:,0],  drawstyle='steps', label = "lin truth",   color='orange' ,linestyle = 'dotted' )
    plots[var_evaluation+'truequad'], = ax[index[i]].plot( bins,truth_quad[:,0], drawstyle='steps', label = "quad truth",  color='red'    ,linestyle = 'dotted' )           
    plots[var_evaluation+"_lin"],     = ax[index[i]].plot(  [] , [] ,            drawstyle='steps', label = "lin train",   color='orange' )
    plots[var_evaluation+"_quad"],    = ax[index[i]].plot(  [] , [] ,            drawstyle='steps', label = "quad train",  color='red'    )
    ax[index[i]].set_xlabel( config.plot_mva_variables[var_evaluation][1] )
 

# Initialize the movie
if (args.animate):
    logging.info("setting up animation...")
    # for debug: warning: conflict with animation frame every 10 epochs <-> n_epochs < 10
    assert n_epochs >= args.animate_step, " n_epochs = %i is not sufficient for animating, required > %i " %( n_epochs, args.animate_step )
    # Define the meta data for the movie
    GifWriter = manimation.writers['pillow']
    writer = manimation.PillowWriter( fps=args.animate_fps, metadata=None )
    logging.info("       gif's duration will be %s second(s)", n_epochs / args.animate_step / args.animate_fps)

    nbins = 20
    index = list(itertools.product(list(range(0, 4)), list(range(0, 5))))
    plotvars=list(config.plot_mva_variables.keys())
    plots = {}   
    logging.info("       plotting truth for %i variables ", len(plotvars))
    fig, ax = plt.subplots(4,5, figsize=(15,12), tight_layout=True)  # plot max 20 vars
    for i in range (len(plotvars)):
        eval_truth(plotvars[i])
    logging.info("...done")

if not (args.animate):
    logging.info("skipping animation set up")


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
    
    def __givey__( self ):
        return self.y

# load data into DataLaoder
dataset = NewDataset(X,V,Y)
train_loader = DataLoader(dataset=dataset,
                          batch_size=len(dataset),
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
        return x1


class LossLikelihoodFree(torch.nn.L1Loss):
    __constants__ = ['reduction']

    def __init__(self, reduction: str = 'sum') -> None:
        super(LossLikelihoodFree, self).__init__(None, None, reduction)

        self.base_points = [1,2] # hardcoded

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        weight      = target[:,0] # this is the weight
        target_lin  = target[:,1] # this is the linear term
        target_quad = target[:,2] # this is the quadratic term

        loss = 0
        for theta_base in self.base_points: #two base-points: 1,2
            loss += weight*( theta_base*(target_lin/weight-input[:,0]) + .5*theta_base**2*(target_quad/weight-input[:,1]) )**2

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()

def get_loss( **kwargs ):
    return LossLikelihoodFree()


# define model
if ( args.LSTM == False ):    
    model = NeuralNet(input_size, hidden_size, hidden_size2, output_size, input_size_lstm=0, hidden_size_lstm=0, num_layers=0).to(device)    
else:
    model = NeuralNet(input_size, hidden_size, hidden_size2, output_size, input_size_lstm, hidden_size_lstm, num_layers).to(device) 
    
# define loss function   
criterion = get_loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
losses = []

# set up directory and model names
dir_name = str(args.sample)+'-'+str(n_epochs)+'_hs1-'+str(hidden_size)+'_hs2-'+str(hidden_size2)
if ( args.LSTM ): 
    dir_name = dir_name +  '_lstm-'+str(num_layers)+'_hs-lstm-'+str(hidden_size_lstm)

results_dir = args.output_directory
if not os.path.exists( results_dir ): 
    os.makedirs( results_dir )

# train the model
logging.info("starting training") 
if (args.animate):
    with writer.saving(fig, dir_name+'_'+"all"+".gif", 100):
        for epoch in range(n_epochs):
            logging.info("		epoch: %i of %i ", epoch+1, n_epochs)
            for i, data in enumerate(train_loader):
                inputs1,inputs2, labels = data
                z = model(inputs1, inputs2)
                loss=criterion(z,labels)
                losses.append(loss.data)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (epoch%args.animate_step==0):
                    with torch.no_grad():
                        for i in range (len(plotvars)):
                            eval_train(plotvars[i])
                        writer.grab_frame()
if not (args.animate):
    for epoch in range(n_epochs):
        logging.info("		epoch: %i of %i ", epoch+1, n_epochs)
        for i, data in enumerate(train_loader):
            inputs1,inputs2, labels = data
            z = model(inputs1, inputs2)
            loss=criterion(z,labels)
            losses.append(loss.data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()                    
           
logging.info("done with training, plotting losses") 
          
# plot losses 
fig, ay = plt.subplots()        
plt.plot(losses, color='red')
plt.title("Losses over epoch")
sample_file_name = str(dir_name)+"_losses.png"
plt.savefig(sample_file_name)


# save model
with torch.no_grad():
    x = X[0,:].reshape(1,len(mva_variables))
    if (args.LSTM):
        v = V[0,:,:].reshape(1, max_timestep, len(vector_branches))
        name = str(dir_name)+".onnx"
    else: 
        v = V[0].reshape(1,1)
        name = str(dir_name)+".onnx" 
    torch.save(model.state_dict(), os.path.join(results_dir, str(dir_name)+'.pth'))    
    torch.onnx.export(model,args=(x, v),f=os.path.join(results_dir, name),input_names=["input1", "input2"],output_names=["output1"]) 
    logging.info("Saved model to %s", os.path.join(results_dir, name)) 
