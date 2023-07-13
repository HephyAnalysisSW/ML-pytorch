# neural net (with LSTM)
# option to add TT_2l background


import  torch
from    torch                    import Tensor
import  uproot                   
import  numpy                    as np
from    matplotlib               import pyplot as plt
from    torch.utils.data         import Dataset, DataLoader, WeightedRandomSampler
import  torch.nn                 as nn
import  os, shutil                       
import  argparse                 
import  matplotlib.animation     as manimation 
from    Tools.WeightInfo         import WeightInfo #have it in my local folder
import  itertools                
from    multiprocessing          import Pool
from    datetime                 import datetime
import  torch.optim              as optim
import  torch.optim.lr_scheduler as lr_scheduler
import  Tools.syncer_for_gif     as syncer 

argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--logLevel',           action='store',                   default='INFO', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], help="Log level for logging")
argParser.add_argument('--sample',             action='store',      type=str,    default='TTTT_MS'),
argParser.add_argument('--full_bkg',           action='store_true',              default=False,    help="add full TT_2L sample as bkg"),
argParser.add_argument('--lumi',               action='store',      type=float,  default=300.0)
argParser.add_argument('--reduce',             action='store',      type=int,    default=None,     help="Reduce training data by factor?"),
argParser.add_argument('--output_directory',   action='store',      type=str,    default='/groups/hephy/cms/lena.wild/tttt/models_LSTM/')
argParser.add_argument('--plot_directory',     action='store',      type=str,    default='/groups/hephy/cms/lena.wild/www/tttt/plots/train_sig+bkg_red10_v3/')
argParser.add_argument('--input_directory',    action='store',      type=str,    default='/eos/vbc/group/cms/lena.wild/tttt/training-ntuples-tttt_v7/MVA-training/ttbb_2l_dilep2-bjet_delphes-ht500-njet6p-btag1p/')
argParser.add_argument('--n_epochs',           action='store',      type=int,    default= '500',   help='number of epochs in training')
argParser.add_argument('--lr',                 action='store',      type=float,  default= '0.01',  help='learning rate')
argParser.add_argument('--hs1_mult',           action='store',      type=int,    default= '2',     help='hidden size 1 = #features * mult')
argParser.add_argument('--hs2_add',            action='store',      type=int,    default= '5',     help='hidden size 2 = #features + add')
argParser.add_argument('--hs_combined',        action='store',      type=int,    default= '5',     help='hidden size of combined layer after LSTM+DNN')
argParser.add_argument('--LSTM',               action='store_true',              default=False,    help='add LSTM?')
argParser.add_argument('--num_layers',         action='store',      type=int,    default= '1',     help='number of LSTM layers')
argParser.add_argument('--LSTM_out',           action='store',      type=int,    default= '1',     help='output size of LSTM')
argParser.add_argument('--nbins',              action='store',      type=int,    default='20',     help='number of bins')
argParser.add_argument('--add_bkg',            action='store_true',              default=False,    help='add bkg TT_2L?')
argParser.add_argument('--scheduler',          action='store',      type=str,    nargs = 2,  default=[None, None],     help='linear+flat, decay , factor by which the final lr is reduced')
argParser.add_argument('--EFTCoefficients',    action='store',                   default='ctt',    help="Training vectors for particle net")
argParser.add_argument('--animate' ,           action='store_true', default=False,                 help="make an animation of data?")
argParser.add_argument('--animate_step',       action='store',      type=int,    default= '10',    help="plot every n epochs")
argParser.add_argument('--animate_fps' ,       action='store',      type=int,    default= '10',    help="frames per second in animation")
argParser.add_argument('--dropout',            action='store',      type=float,  default= None, )
argParser.add_argument('--weight_decay',       action='store',      type=float,  default= None, )
argParser.add_argument('--ReLU_slope',         action='store',      type=float,  default= 0.1, )
argParser.add_argument('--load_model' ,        action='store',      type=str,    default= None,    help="load model and continue training?")

# for LLR eval
argParser.add_argument('--LLR_eval',           action='store_true', default=False     )
argParser.add_argument('--theta_range',        action='store',      type=float,    default=10     )
argParser.add_argument('--sample_weight',      action='store',      type=float,  default=100)
argParser.add_argument('--shape_effects_only', action='store_true', default=True, help="Normalize sm *and* bsm weights to sample_weight number of events")
args = argParser.parse_args()

def copyIndexPHP( results_dir ):
    index_php = os.path.join( results_dir, 'index.php' )
    shutil.copyfile( os.path.join( "Tools", 'index_for_gif.php' ), index_php )

torch.manual_seed(0)

import  logging
logger = logging.getLogger()
logger_handler = logging.StreamHandler()
logger.addHandler(logger_handler)
logger_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
logger.setLevel(args.logLevel)

assert not (args.add_bkg==False and (args.animate == ("sig+bkg") or args.animate == ("sig-bkg"))), "cannot animate signal and bkg if --add_bkg is not set!"

# adding hard coded reweight_pkl because of ML-pytorch
if ( args.sample == "TTTT_MS" ): reweight_pkl = "/eos/vbc/group/cms/robert.schoefbeck/gridpacks/4top/TTTT_MS_reweight_card.pkl" 
if ( args.sample == "TTbb_MS" ): reweight_pkl = "/eos/vbc/group/cms/robert.schoefbeck/gridpacks/4top/TTbb_MS_reweight_card.pkl" 


import ttbb_2l_python3 as config
# set hyperparameters
mva_variables    = [ mva_variable[0] for mva_variable in config.mva_variables ]
sample           = args.sample
bkg              = "TT_2L" if args.full_bkg==False else "TT_2L_full"
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
lumi             = args.lumi

if args.reduce is not None: reduce = True 
else: reduce = False

if ( args.LSTM ):
    vector_branches = ["mva_Jet_%s" % varname for varname in config.lstm_jetVarNames] 
    max_timestep = config.lstm_jets_maxN
    input_size_lstm = len( vector_branches )
    hidden_size_lstm = args.LSTM_out
    num_layers = args.num_layers


# print hyperparameter selection
logging.info( "------------parameters for training----------------" )
logging.info( "  EFT Coefficient:                         %s",args.EFTCoefficients )
logging.info( "  number of epochs:                        %i",n_epochs )
logging.info( "  learning rate:                           %s",learning_rate )
logging.info( "  number of features,linear layer:         %i",input_size )
logging.info( "  size of first hidden layer:              %i",hidden_size )
logging.info( "  size of third hidden layer:              %i",hidden_size2 )
logging.info( "  size of combined layer:                  %i",hidden_size_comb )
logging.info( "  LSTM:                                    %r",args.LSTM )
if ( args.LSTM ):
    logging.info( "          number of LSTM layers:         %i", num_layers )
    logging.info( "          output size of LSTM:           %i", hidden_size_lstm )
    logging.info( "          number of features, LSTM:      %i", len(vector_branches) )
if ( args.add_bkg ):
    logging.info( "  adding background %s", bkg )       
logging.info( "---------------------------------------------------\n" )

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
        SM_ref      = target[:,3] # this is the new sm weight after reweighting to the integrated luminosity
        loss = 0
        for theta_base in self.base_points: #two base-points: 1,2
            loss += SM_ref*( theta_base*(target_lin/weight-input[:,0]) + theta_base**2*(target_quad/weight-input[:,1]) )**2

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()

def get_loss( **kwargs ):
    return LossLikelihoodFree()


# define model
if ( args.LSTM == False ): model = NeuralNet(input_size, hidden_size, hidden_size2, hidden_size_comb, output_size, input_size_lstm=0, hidden_size_lstm=0, num_layers=0).to(device)    
else:                      model = NeuralNet(input_size, hidden_size, hidden_size2, hidden_size_comb, output_size, input_size_lstm, hidden_size_lstm, num_layers).to(device) 

logging.info("")
logging.info(model) 
 
# define loss function   
criterion = get_loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay= weight_decay)

# load pretrained model if necessary
if (args.load_model is not None):
    ckp_path = args.load_model
    model, optimizer, start_epoch, valid_loss_min = load_ckp(ckp_path, model, optimizer)
    logging.info("loaded model %s, continue training", ckp_path)
    
# set up directory and model names
n_epochs_ = n_epochs if args.load_model is None else n_epochs+start_epoch
dir_name = str(args.sample)+"_"+str(args.EFTCoefficients)+'_lr'+str(args.lr)+'_l'+str(int(lumi))+'_e'+str(n_epochs_)+'_'+str(hidden_size)+'-'+str(hidden_size2)+'-'+str(hidden_size_comb)
if ( args.LSTM ):              dir_name = dir_name +  '_lstm'+str(num_layers)+'x'+str(hidden_size_lstm)
if (reduce):                   dir_name = dir_name + '_r'+str(args.reduce)    
if (args.add_bkg):             dir_name = dir_name + '+bkg'  if args.full_bkg==False else dir_name + '+fullbkg'
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


dummyx = torch.Tensor(np.ones((1,len(mva_variables))))
if (args.LSTM):
    dummyv = torch.Tensor(np.ones((1, max_timestep, len(vector_branches))))
else: 
    dummyv = torch.Tensor(np.ones((1,1)))

      
logger_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))           

# save model
with torch.no_grad():
    name = str(dir_name)+".onnx" 
    torch.save(model.state_dict(), os.path.join(results_dir, str(dir_name)+'.pth'))    
    torch.onnx.export(model,args=(dummyx, dummyv),f=os.path.join(results_dir, name),input_names=["input1", "input2"],output_names=["output1"]) 
    logging.info("saved model to %s", os.path.join(results_dir, name)) 

syncer.sync()    