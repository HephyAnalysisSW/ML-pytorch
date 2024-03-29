# neural net (with LSTM)



import  torch
from    torch                   import Tensor
import  uproot  
import  numpy                   as np
from    matplotlib              import pyplot as plt
from    torch.utils.data        import Dataset, DataLoader, WeightedRandomSampler
import  torch.nn                as nn
import  os, shutil
import  argparse
import  matplotlib.animation    as manimation 
from    Tools.WeightInfo        import WeightInfo #have it in my local folder because it's ML-pytorch
import  itertools
from    multiprocessing         import Pool
import  logging
from    datetime                 import datetime
import  Tools.syncer_for_gif     as syncer 
import  torch.optim.lr_scheduler as lr_scheduler

argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--logLevel',           action='store',                   default='INFO', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], help="Log level for logging")
argParser.add_argument('--sample',             action='store',      type=str,    default='TTTT_MS')
argParser.add_argument('--output_directory',   action='store',      type=str,    default='/groups/hephy/cms/lena.wild/tttt/models_DNN/')
argParser.add_argument('--plot_directory',     action='store',      type=str,    default='/groups/hephy/cms/lena.wild/www/tttt/plots/DNN_all')
argParser.add_argument('--input_directory',    action='store',      type=str,    default='/eos/vbc/group/cms/lena.wild/tttt/training-ntuples-tttt_v6_1/MVA-training/PN_ttbb_2l_dilep2-bjet_delphes-met30-njet4p-btag2p/')
argParser.add_argument('--scheduler',          action='store',      type=str,    nargs = 2,  default=[None, None],     help='linear+flat, decay , factor by which the final lr is reduced')
argParser.add_argument('--n_epochs',           action='store',      type=int,    default= '500',   help='number of epochs in training')
argParser.add_argument('--hs1_mult',           action='store',      type=int,    default= '2',     help='hidden size 1 = #features * mult')
argParser.add_argument('--hs2_add',            action='store',      type=int,    default= '5',     help='hidden size 2 = #features + add')
argParser.add_argument('--hs_combined',        action='store',      type=int,    default= '5',     help='hidden size of combined layer after LSTM+DNN')
argParser.add_argument('--LSTM',               action='store_true',              default=False,    help='add LSTM?')
argParser.add_argument('--num_layers',         action='store',      type=int,    default= '1',     help='number of LSTM layers')
argParser.add_argument('--LSTM_out',           action='store',      type=int,    default= '1',     help='output size of LSTM')
argParser.add_argument('--nbins',              action='store',      type=int,    default='20',     help='number of bins')
argParser.add_argument('--EFTCoefficients',    action='store',                   default='ctt',    help="Training vectors for particle net")
argParser.add_argument('--animate_step',       action='store',      type=int,    default= '10',    help="plot every n epochs")
argParser.add_argument('--animate_fps' ,       action='store',      type=int,    default= '10',    help="frames per second in animation")
argParser.add_argument('--animate' ,           action='store_true',              default= False,   help="make an animation?")
argParser.add_argument('--reduce',             action='store',      type=int,    default=None,     help="Reduce training data by factor?"),
argParser.add_argument('--lr',                 action='store',      type=float,  default= '0.001',  help='learning rate')
argParser.add_argument('--dropout',            action='store',      type=float,  default= 0.5, )
argParser.add_argument('--weight_decay',       action='store',      type=float,  default= None, )
argParser.add_argument('--ReLU_slope',         action='store',      type=float,  default= 0.5, )
argParser.add_argument('--load_model' ,        action='store',      type=str,    default= None,    help="load model and continue training?")


args = argParser.parse_args()

torch.manual_seed(0)

import  logging
logger = logging.getLogger()
logger_handler = logging.StreamHandler()
logger.addHandler(logger_handler)
logger_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
logger.setLevel(args.logLevel)

def copyIndexPHP( results_dir ):
    index_php = os.path.join( results_dir, 'index.php' )
    shutil.copyfile( os.path.join( "Tools", 'index_for_gif.php' ), index_php )

# adding hard coded reweight_pkl because of ML-pytorch
if ( args.sample == "TTTT_MS" ): reweight_pkl = "/eos/vbc/group/cms/robert.schoefbeck/gridpacks/4top/TTTT_MS_reweight_card.pkl" 
if ( args.sample == "TTbb_MS" ): reweight_pkl = "/eos/vbc/group/cms/robert.schoefbeck/gridpacks/4top/TTbb_MS_reweight_card.pkl" 

if args.reduce is not None: reduce = True 
else: reduce = False


import ttbb_2l_python3 as config

# set hyperparameters
mva_variables    = [ mva_variable[0] for mva_variable in config.mva_variables ]
sample           = args.sample
device           = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
learning_rate    = args.lr
dropout          = args.dropout if args.dropout is not None else 0
weight_decay     = args.weight_decay if args.weight_decay is not None else 0
ReLU_slope       = args.ReLU_slope
n_epochs         = args.n_epochs
input_size       = len( mva_variables ) 
hidden_size      = input_size * args.hs1_mult
hidden_size2     = input_size + args.hs2_add
hidden_size_comb = args.hs_combined
output_size      = 2                        # hard coded: lin, quad

# save checkpoint
def save_ckp(state, checkpoint_path):
    torch.save(state, checkpoint_path)

# load checkpoint        
def load_ckp(checkpoint_fpath, model, optimizer):
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    valid_loss_min = checkpoint['valid_loss_min']
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item() 

if ( args.LSTM ):
    vector_branches = ["mva_Jet_%s" % varname for varname in config.lstm_jetVarNames] 
    max_timestep = config.lstm_jets_maxN
    input_size_lstm = len( vector_branches )
    hidden_size_lstm = args.LSTM_out
    num_layers = args.num_layers


# print hyperparameter selection
logging.info( "------------Parameters for training----------------" )
logging.info( "  EFT Coefficient:                         %s",args.EFTCoefficients )
logging.info( "  Number of epochs:                        %i",n_epochs )
logging.info( "  Number of features,linear layer:         %i",input_size )
logging.info( "  Size of first hidden layer:              %i",hidden_size )
logging.info( "  Size of second hidden layer:             %i",hidden_size2 )
logging.info( "  Size of combined layer:                  %i",hidden_size_comb )
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
    

v = np.zeros( (len(y[:,0]),1,1) )
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
    v = np.column_stack( [np.stack( vec_br_f[name] ) for name in vector_branches] ).reshape( len(y), len(vector_branches), max_timestep ).transpose((0,2,1))

if args.reduce is not None: red = int(len(x[:,0])/args.reduce) 
else: red = len(x[:,0])   
logging.info("using only small dataset of 1/%s of total events -> %s events for signal %s", args.reduce, red, args.sample)
    
#split train and eval data
half = int(red*0.8)
v_v = v[half:red, :, :]
x_v = x[half:red, :]
y_v = y[half:red, :]
x = x[0:half, :]
v = v[0:half, :, :]    
y = y[0:half, :]    
    
#double check for NaNs:
assert not np.isnan( np.sum(x) ), logging.info("found NaNs in DNN input!")
assert not np.isnan( np.sum(y) ), logging.info("found NaNs in DNN truth values!")
assert not np.isnan( np.sum(v) ), logging.info("found NaNs in LSTM input!")

X = torch.Tensor( x )
Y = torch.Tensor( y )
V = torch.Tensor( v )
X_v = torch.Tensor( x_v )
Y_v = torch.Tensor( y_v )
V_v = torch.Tensor( v_v )


# Define steps for evaluation, evaluate on eval data set
def eval_train ( var_evaluation, zz ):
    zz = np.array( zz )
    x_eval = np.array(xx[var_evaluation])[half:red]
    nbins = 8 if var_evaluation=="nrecoJet" else 20
    hist_lin,  bins = np.histogram( x_eval, bins=nbins, range=(config.plot_mva_variables[var_evaluation][0][0], config.plot_mva_variables[var_evaluation][0][1]), weights=y_v[:,0]*zz[:,0])
    hist_quad, bins = np.histogram( x_eval, bins=nbins, range=(config.plot_mva_variables[var_evaluation][0][0], config.plot_mva_variables[var_evaluation][0][1]), weights=y_v[:,0]*zz[:,1])                               

    plots[var_evaluation+'_lin' ].set_data( bins, np.hstack((0,hist_lin))   )            
    plots[var_evaluation+'_quad'].set_data( bins, np.hstack((0,hist_quad)) )

  
def eval_truth ( var_evaluation ):
    x_eval = np.array(xx[var_evaluation])[half:red]
    nbins = 8 if var_evaluation=="nrecoJet" else 20
    hist_truelin,      bins  = np.histogram( x_eval, bins=nbins, range=(config.plot_mva_variables[var_evaluation][0][0], config.plot_mva_variables[var_evaluation][0][1]), weights= y_v[:,1] )
    hist_truequad,     bins  = np.histogram( x_eval, bins=nbins, range=(config.plot_mva_variables[var_evaluation][0][0], config.plot_mva_variables[var_evaluation][0][1]), weights= y_v[:,2] )

    i = plotvars.index( var_evaluation )   
       
    plots[var_evaluation+'truelin'],  = ax[index[i]].plot( bins,np.hstack((0,hist_truelin)),  drawstyle='steps', label = "lin truth",   color='blue' ,linestyle = 'dotted' )
    plots[var_evaluation+'truequad'], = ax[index[i]].plot( bins,np.hstack((0,hist_truequad)), drawstyle='steps', label = "quad truth",  color='red'    ,linestyle = 'dotted' )           
    plots[var_evaluation+"_lin"],     = ax[index[i]].plot(  [] , [] ,            drawstyle='steps', label = "lin train",   color='blue' )
    plots[var_evaluation+"_quad"],    = ax[index[i]].plot(  [] , [] ,            drawstyle='steps', label = "quad train",  color='red'    )
    ax[index[i]].set_xlabel( config.plot_mva_variables[var_evaluation][1] )
 

# Initialize the movie
if (args.animate):
    logging.info("setting up animation...")
    # for debug: warning: conflict with animation frame every 10 epochs <-> n_epochs < 10
    assert n_epochs >= args.animate_step, " n_epochs = %i is not sufficient for animating, required > %i " %( n_epochs, args.animate_step )
    # Define the meta data for the moviel
    GifWriter = manimation.writers['pillow']
    writer = manimation.PillowWriter( fps=args.animate_fps, metadata=None )
    logging.info("       gif's duration will be %s second(s)", n_epochs / args.animate_step / args.animate_fps)

    nbins = 20
    index = list(itertools.product(list(range(0, 4)), list(range(0, 10))))
    plotvars=list(config.plot_mva_variables.keys())
    plots = {}   
    logging.info("       plotting truth for %i variables ", len(plotvars))
    fig, ax = plt.subplots(4,10, figsize=(28,12), tight_layout=True)  # plot max 20 vars
    for i in range (len(plotvars)):
        eval_truth(plotvars[i])

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
        
        if (args.LSTM):
            self.num_layers = num_layers
            self.hidden_size_lstm = hidden_size_lstm
            self.lstm = nn.LSTM(input_size_lstm, hidden_size_lstm, num_layers, batch_first=True)
            self.after_lstm = nn.Sequential(
                nn.LeakyReLU(ReLU_slope),
                nn.Dropout(dropout),
            )
            self.linear3 = nn.Sequential(
                nn.Linear(in_features=hidden_size2+hidden_size_lstm, out_features=hidden_size_comb),
                nn.LeakyReLU(ReLU_slope),
                nn.Dropout(dropout),
            )
            
        else:
            self.linear3 = nn.Sequential(
                nn.Linear(in_features=hidden_size2, out_features=hidden_size_comb),
                nn.LeakyReLU(ReLU_slope),
                nn.Dropout(dropout)
            )
        self.linear4 = nn.Sequential(
                nn.Linear(in_features=hidden_size_comb, out_features=output_size),
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
            x2, _ = self.lstm(y)
            x2 = x2[:, -1, :] 
            x2 = self.after_lstm(x2)
            x1 = torch.cat([x1, x2], dim=1)          
                 
        x1 = self.linear3(x1)   
        x1 = self.linear4(x1)   
        return x1


class LossLikelihoodFree(torch.nn.L1Loss):
    __constants__ = ['reduction']

    def __init__(self, reduction: str = 'mean') -> None:
        super(LossLikelihoodFree, self).__init__(None, None, reduction)

        self.base_points = [1,2] # hardcoded

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        weight      = target[:,0] # this is the weight
        target_lin  = target[:,1] # this is the linear term
        target_quad = target[:,2] # this is the quadratic term

        loss = 0
        for theta_base in self.base_points: #two base-points: 1,2
            loss += weight*( theta_base*(target_lin/weight-input[:,0]) + theta_base**2*(target_quad/weight-input[:,1]) )**2

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
    model = NeuralNet(input_size, hidden_size, hidden_size2, hidden_size_comb, output_size, input_size_lstm=0, hidden_size_lstm=0, num_layers=0).to(device)    
else:
    model = NeuralNet(input_size, hidden_size, hidden_size2, hidden_size_comb, output_size, input_size_lstm, hidden_size_lstm, num_layers).to(device) 

logging.info("")
logging.info(model) 

if args.load_model is not None:
    model.load_state_dict(torch.load(args.load_model))
# define loss function   
criterion = get_loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay= weight_decay) 
losses = []
losses_v = []

if (args.scheduler[0] is not None):   
    if args.scheduler[0] == 'linear+flat': scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=float(args.scheduler[1]), total_iters=int(n_epochs*9/10)) 
    if args.scheduler[0] == 'decay': scheduler = lr_scheduler.ExponentialLR(optimizer, gamma = float(args.scheduler[1])) 
    logging.info("using %s scheduler with factor %s from initial lr rate %s", args.scheduler[0], args.scheduler[1],  args.lr)
else: scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1, total_iters=n_epochs) #==no scheduler


# set up directory and model names
dir_name = str(args.sample)+'_'+str(args.EFTCoefficients)+'_lr'+str(args.lr)+'_e'+str(n_epochs)+'_'+str(hidden_size)+'-'+str(hidden_size2)+'-'+str(hidden_size_comb)
if ( args.LSTM ):                       dir_name = dir_name +  '_lstm'+str(num_layers)+'x'+str(hidden_size_lstm)
if (reduce):                            dir_name = dir_name + '_r'+str(args.reduce)    
if (args.scheduler is not None):        dir_name = dir_name + "_s"  + str(args.scheduler[0])+str(args.scheduler[1] ) 
if (args.dropout is not None):          dir_name = dir_name + "_do"  + str(args.dropout ) 
if (args.weight_decay is not None):     dir_name = dir_name + "_wd"  + str(args.weight_decay ) 
if (args.load_model is not None):       dir_name = dir_name + "_loaded"
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

dummyx = X[0,:].reshape(1,len(mva_variables))
if (args.LSTM):
    dummyv = V[0,:,:].reshape(1, max_timestep, len(vector_branches))
else: 
    dummyv = V[0].reshape(1,1)
    
if (args.animate):
    with writer.saving(fig, os.path.join(args.plot_directory, dir_name+"_.gif"), 100):
        start = datetime.now()
        for epoch in range(n_epochs):
            model.train()
            for i, data in enumerate(train_loader):
                inputs1,inputs2, labels = data
                z = model(inputs1, inputs2)
                loss=criterion(z,labels)
                losses.append(loss.data)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
            # save epoch
            #if (epoch%10==0):
                #print(z[:10,0])
                #assert (z[0,0] != z[1,0]), "z is all the same" 
                # with torch.no_grad():
                    # torch.save(model.state_dict(), os.path.join(model_dir, str(dir_name)+"_epoch_"+str(epoch)+'.pth'))    
                    # torch.onnx.export(model,args=(dummyx, dummyv),f=os.path.join(model_dir, str(dir_name)+"_epoch_"+str(epoch)+'.onnx'),input_names=["input1", "input2"],output_names=["output1"])    
                                
            model.eval()
            # zz = model(X_v, V_v)
            with torch.no_grad():
                zz = model(X_v, V_v)
                loss_v=criterion(zz,Y_v)
                losses_v.append(loss_v.data)
                
            try:
                l = u'\u2191' if (losses[-2]<=losses[-1]) else u'\u2193'
                l_v = u'\u2191' if (losses_v[-2]<=losses_v[-1]) else u'\u2193'
                
            except:
                l = "-"
                l_v = "-"
                
            logging.info("		epoch: %i of %i      loss(tr/ev): %s, %s (%s, %s)        lr: %s", epoch+1, n_epochs, losses[-1], losses_v[-1], l, l_v, scheduler.get_last_lr())
            if (epoch%args.animate_step==args.animate_step-1):
                with torch.no_grad():
                    for i in range (len(plotvars)):
                        eval_train(plotvars[i], zz)
                    writer.grab_frame()
            if (args.n_epochs/(epoch+1)==args.n_epochs/args.animate_step):    
                    end = datetime.now()
                    logging.info('estimate for training duration: {} \n'.format((end - start)*(args.n_epochs/args.animate_step-1)))
                    logging.info('training will finish appx. at {} \n'.format((end - start)*(args.n_epochs/args.animate_step-1)+end))
            
                
            
if not (args.animate):
    start = datetime.now()  
    for epoch in range(n_epochs):
        model.train()
        for i, data in enumerate(train_loader):
            inputs1,inputs2, labels = data
            z = model(inputs1, inputs2)
            loss=criterion(z,labels)
            losses.append(loss.data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
            if (n_epochs/(epoch+1)==10):    
                end = datetime.now()
                logging.info('estimate for training duration: {} \n'.format((end - start)*9))
                logging.info('training will finish appx. at {} \n'.format((end - start)*9+end))
            scheduler.step()
        # save epoch
        #if (epoch%10==0):
            #assert (z[0,0] != z[1,0]), "z is all the same" 
            #with torch.no_grad():
                #torch.save(model.state_dict(), os.path.join(model_dir, str(dir_name)+"_epoch_"+str(epoch)+'.pth'))    
                #torch.onnx.export(model,args=(dummyx, dummyv),f=os.path.join(model_dir, str(dir_name)+"_epoch_"+str(epoch)+'.onnx'),input_names=["input1", "input2"],output_names=["output1"]) 
                            
        model.eval()
        # zz = model(X_v, V_v)
        with torch.no_grad(): 
            zz = model(X_v, V_v)
            loss_v=criterion(zz,Y_v)
            losses_v.append(loss_v.data)
            
            logging.info("		epoch: %i of %i        train loss: %s , validation loss: %s", epoch+1, n_epochs, losses[-1], losses_v[-1])
                
      
logger_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))    
logging.info("done with training, plotting losses") 
with torch.no_grad():
    name = str(dir_name)+".onnx" 
    torch.save(model.state_dict(), os.path.join(results_dir, str(dir_name)+'.pth'))    
    torch.onnx.export(model,args=(dummyx, dummyv),f=os.path.join(results_dir, name),input_names=["input1", "input2"],output_names=["output1"]) 
    logging.info("saved model to %s", os.path.join(results_dir, name)) 
if args.animate:      
    shutil.copyfile(os.path.join(args.plot_directory, dir_name+"_.gif"), os.path.join(args.plot_directory,dir_name+".gif"))
    #os.remove(os.path.join(args.plot_directory, dir_name+"_.gif")) 
    
# plot losses 
fig, ay = plt.subplots()        
plt.plot(losses[10:], color='red', label='training loss')
plt.plot(losses_v[10:], color='blue', label='valiadation loss')
plt.title("Losses over epoch")
plt.legend()
sample_file_name = str(dir_name)+".png"
plt.yscale("log")  
plt.savefig(os.path.join(args.plot_directory,sample_file_name))
logging.info("saved plots to %s", os.path.join(args.plot_directory,sample_file_name))      


copyIndexPHP(os.path.join(args.plot_directory))


logging.info("")
logging.info("loss plot link: %s", os.path.join('https://lwild.web.cern.ch/tttt/plots/', os.path.basename(os.path.normpath(args.plot_directory)),dir_name+".png")) #HARDCODED
if args.animate: logging.info("gif link: %s", os.path.join('https://lwild.web.cern.ch/tttt/plots/', os.path.basename(os.path.normpath(args.plot_directory)), dir_name+".gif")) 

        
syncer.sync()    