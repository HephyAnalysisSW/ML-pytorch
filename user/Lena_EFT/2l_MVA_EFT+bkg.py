# neural net (with LSTM)
# option to add TT_2l background


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
from    datetime                import datetime


argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--logLevel',           action='store',                   default='INFO', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], help="Log level for logging")
argParser.add_argument('--sample',             action='store',      type=str,    default='TTTT_MS'),
argParser.add_argument('--full_bkg',           action='store_true',              default=False,    help="add full TT_2L sample as bkg"),
argParser.add_argument('--lumi',               action='store',      type=float,  default=300.0)
argParser.add_argument('--reduce',             action='store',      type=int,    default=None,     help="Reduce training data by factor?"),
argParser.add_argument('--output_directory',   action='store',      type=str,    default='/groups/hephy/cms/lena.wild/tttt/models/')
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
argParser.add_argument('--add_bkg',            action='store_true',              default=False,     help='add bkg TT_2L?')
argParser.add_argument('--EFTCoefficients',    action='store',                   default='ctt',    help="Training vectors for particle net")
argParser.add_argument('--animate' ,           action='store',                   default=None,     choices=["FULL", "sig-bkg", "sig+bkg"], help="make an animation of full / sig or separated sig+bkg data?")
argParser.add_argument('--animate_step',       action='store',      type=int,    default= '10',    help="plot every n epochs")
argParser.add_argument('--animate_fps' ,       action='store',      type=int,    default= '10',    help="frames per second in animation")
args = argParser.parse_args()


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
logging.info( "  number of epochs:                        %i",n_epochs )
logging.info( "  learning rate:                           %s",learning_rate )
logging.info( "  number of features,linear layer:         %i",input_size )
logging.info( "  size of first hidden layer:              %i",hidden_size )
logging.info( "  size of second hidden layer:             %i",hidden_size2 )
logging.info( "  size of combined layer:                  %i",hidden_size_comb )
logging.info( "  LSTM:                                    %r",args.LSTM )
if ( args.LSTM ):
    logging.info( "          number of LSTM layers:         %i", num_layers )
    logging.info( "          output size of LSTM:           %i", hidden_size_lstm )
    logging.info( "          number of features, LSTM:      %i", len(vector_branches) )
if ( args.add_bkg ):
    logging.info( "  adding background %s", bkg )       
logging.info( "---------------------------------------------------\n" )

# set weights 
weightInfo = WeightInfo( reweight_pkl ) 
weightInfo.set_order( 2 ) 
index_lin  = weightInfo.combinations.index( (args.EFTCoefficients,) ) 
index_quad = weightInfo.combinations.index( (args.EFTCoefficients,args.EFTCoefficients) )

# import training data
logging.info("import training data for sample %s", sample)
upfile_name = os.path.join( args.input_directory, sample, sample+".root" )
xx     = uproot.open( upfile_name ) 
xx     = xx["Events"].arrays( mva_variables, library = "np" )
x      = np.array( [ xx[branch] for branch in mva_variables ] ).transpose() 

weigh = {}
with uproot.open(upfile_name) as upfile:
    for name, branch in upfile["Events"].arrays( "p_C", library = "np" ).items(): 
        weigh = [ (branch[i][0], branch[i][index_lin], branch[i][index_quad]) for i in  range (branch.shape[0]) ]
        assert len( weightInfo.combinations ) == branch[0].shape[0] , "got p_C wrong: found %i weights but need %i" %( branch[0].shape[0], len( weightInfo.combinations ) ) # check number of weights
    y = np.asarray( weigh )


# add lstm if needed
v = np.zeros( (len(y[:,0]),1, 1) )
if ( args.LSTM ):
    vec_br_f  = {}    
    with uproot.open( upfile_name ) as upfile:
        for name, branch in upfile["Events"].arrays( vector_branches, library = "np" ).items():
            for i in range ( branch.shape[0] ):
                branch[i]=np.pad( branch[i][:max_timestep], (0, max_timestep - len(branch[i][:max_timestep])) )
            vec_br_f[name] = branch
    # put columns side by side and transpose the innermost two axis
    v = np.column_stack( [np.stack( vec_br_f[name] ) for name in vector_branches] ).reshape( len(y), len(vector_branches), max_timestep ).transpose((0,2,1))
    
# import bkg training data 
if args.add_bkg: 
    logging.info("import training data for bkg %s", bkg)
    upfile_name = os.path.join( args.input_directory, bkg, bkg+".root" )
    xx_bkg = uproot.open( upfile_name ) 
    xx_bkg = xx_bkg["Events"].arrays( mva_variables, library = "np" )
    x_bkg  = np.array( [ xx_bkg[branch] for branch in mva_variables ] ).transpose() 

    yy_bkg = uproot.open( upfile_name )
    yy_bkg = yy_bkg["Events"].arrays( "genWeight", library = "np" )
    yy_bkg = np.array( [ yy_bkg[branch] for branch in ['genWeight'] ] ).transpose() 
    y_bkg  = np.zeros((len(x_bkg[:,0]), 3))
    y_bkg[:,0] = np.reshape(yy_bkg, (len(yy_bkg), ))

    # add lstm if needed
    v_bkg = np.zeros( (len(y_bkg[:,0]), 1, 1) )
    if ( args.LSTM ):
        vec_br_f  = {}
        with uproot.open( upfile_name ) as upfile:
            for name, branch in upfile["Events"].arrays( vector_branches, library = "np" ).items():
                for i in range ( branch.shape[0] ):
                    branch[i]=np.pad( branch[i][:max_timestep], (0, max_timestep - len(branch[i][:max_timestep])) )
                vec_br_f[name] = branch
        v_bkg = np.column_stack( [np.stack( vec_br_f[name] ) for name in vector_branches] ).reshape( len(y_bkg[:,0]), len(vector_branches), max_timestep ).transpose((0,2,1))

# the target vectors y (and y_bkg) now contain the SM and EFT-weights: y = (w_0, w_1, w_2)      and y_bkg = (genWeight,0,0)
# normalizing to integrated luminosity W_0 gives new target weights:   w = (w_0, w_1, w_2, W_0) and w_bkg = (genWeight,0,0,W_0)
logging.info("calculating new weights, reweighting to integrated luminosity")
W_0     = lumi * config.xsec[sample] * 1000 * y[:,0]/ config.total_genWeight[sample]
# W_0     = lumi * config.xsec[sample] * 1000 * y[:,0]/ np.sum(y[:,0])
if args.add_bkg:
    W_0_bkg = lumi * config.xsec[bkg] * 1000 * y_bkg[:,0]/ config.total_genWeight[bkg]
    # W_0_bkg = lumi * config.xsec[bkg] * 1000 * y_bkg[:,0]/ np.sum(y_bkg[:,0])

# store new weights W_0 as a 4th position in y and y_bkg -> w and w_bkg
w = np.zeros((len(y[:,0]),4))
w[:,:-1] = y 
w[:,-1]  = W_0

if args.add_bkg:
    w_bkg = np.zeros((len(y_bkg[:,0]),4))
    w_bkg[:,:-1] = y_bkg 
    w_bkg[:,-1]  = np.array([aa if aa >= 0 else 0 for aa in list(W_0_bkg)]) #remove negative weights 


if args.reduce is not None: red = int(len(x[:,0])/args.reduce)
red_bkg = 0
if args.add_bkg and args.reduce is not None: red_bkg = int(len(x_bkg[:,0])/args.reduce)
if (reduce):
    x = x[0:red, :]
    y = y[0:red, :]
    v = v[0:red, :, :]
    W_0 = W_0[0:red]
    w = w[0:red, :]
    if args.add_bkg:
        x_bkg = x_bkg[0:red_bkg :]
        y_bkg = y_bkg[0:red_bkg, :]
        v_bkg = v_bkg[0:red_bkg, :, :]
        W_0_bkg = W_0_bkg[0:red_bkg]
        w_bkg = w_bkg[0:red_bkg, :]
    logging.info("using only small dataset of 1/%s of total events -> %s events for signal %s and %s events for bkg %s", args.reduce, red, args.sample, red_bkg, bkg)

if args.add_bkg:
    X_np = np.concatenate(( x, x_bkg ))
    W_np = np.concatenate(( w, w_bkg ))
    V_np = np.concatenate(( v, v_bkg ))
else:     
    X_np = x
    W_np = w
    V_np = v
    
#double check for NaNs:
assert not np.isnan( np.sum( X_np ) ), logging.info("found NaNs in DNN input!")
assert not np.isnan( np.sum( W_np ) ), logging.info("found NaNs in DNN truth values!")
assert not np.isnan( np.sum( V_np ) ), logging.info("found NaNs in LSTM input!")

X = torch.Tensor( X_np )
W = torch.Tensor( W_np )
V = torch.Tensor( V_np )

logging.debug("shape of the DNN input is %s", X.shape)
logging.debug("shape of the truth is %s", W.shape)
logging.debug("shape of the LSTM input is %s", V.shape)

#Define steps for evaluation 
def eval_train ( var_evaluation ):
    total_weight =  np.sum(W_np[:,3])
    if (args.animate == "FULL"):
        z = np.array( model(X,V) )
        x_eval = np.concatenate((np.array(xx[var_evaluation]),np.array(xx_bkg[var_evaluation]))) if args.add_bkg else np.array(xx[var_evaluation]) 
        if reduce: x_eval = np.concatenate((np.array(xx[var_evaluation][0:red]),np.array(xx_bkg[var_evaluation][0:red_bkg]))) if args.add_bkg else np.array(xx[var_evaluation][0:red])
        hist, bins = np.histogram( x_eval, bins=nbins, range=(config.plot_mva_variables[var_evaluation][0][0], config.plot_mva_variables[var_evaluation][0][1]) )
        train_lin  = np.zeros( (len(bins),1) )
        train_quad = np.zeros( (len(bins),1) )
        for b in range ( 1,len(bins)-1 ):
            for ind in range ( x_eval.shape[0] ):
                val = x_eval[ind]
                if ( val > bins[b-1] and val<= bins[b] ):
                    train_lin[b] += z[ind,0] * W_np[ind,3] /total_weight
                    train_quad[b]+= z[ind,1] * W_np[ind,3] /total_weight                              
        name_var = var_evaluation          
        plots[name_var+'_lin' ].set_data( bins, train_lin[:,0]  )               
        plots[name_var+'_quad'].set_data( bins, train_quad[:,0] )
    
    if  (args.animate=="sig+bkg" or args.animate=="sig-bkg"):
        z = np.array( model(torch.Tensor(x),torch.Tensor(v)) )
        x_eval = np.array(xx[var_evaluation])
        if reduce: x_eval = np.array(xx[var_evaluation])[0:red]
        hist, bins  = np.histogram( x_eval, bins=nbins, range=(config.plot_mva_variables[var_evaluation][0][0], config.plot_mva_variables[var_evaluation][0][1]) )
        train_lin  = np.zeros( (len(bins),1) )
        train_quad = np.zeros( (len(bins),1) )
        for b in range ( 1,len(bins)-1 ):
            for ind in range ( x_eval.shape[0] ):
                val = x_eval[ind]
                if ( val > bins[b-1] and val<= bins[b] ):
                    train_lin[b] += z[ind,0] * w[ind,3]/total_weight
                    train_quad[b]+= z[ind,1] * w[ind,3]/total_weight                        
        name_var = var_evaluation          
        plots[name_var+'_lin' ].set_data( bins, train_lin[:,0]  )               
        plots[name_var+'_quad'].set_data( bins, train_quad[:,0] )  
        
        z_bkg = np.array( model(torch.Tensor(x_bkg),torch.Tensor(v_bkg)) )
        x_eval_bkg = np.array(xx_bkg[var_evaluation])
        if reduce: x_eval_bkg = np.array(xx_bkg[var_evaluation])[0:red_bkg]
        hist_bkg, bins  = np.histogram( x_eval_bkg, bins=bins, range=(config.plot_mva_variables[var_evaluation][0][0], config.plot_mva_variables[var_evaluation][0][1]) )
        train_lin  = np.zeros( (len(bins),1) )
        train_quad = np.zeros( (len(bins),1) )
        for b in range ( 1,len(bins)-1 ):
            for ind in range ( x_eval_bkg.shape[0] ):
                val = x_eval_bkg[ind]
                if ( val > bins[b-1] and val<= bins[b] ):
                    train_lin[b] += z_bkg[ind,0]*w_bkg[ind,3]/total_weight  
                    train_quad[b]+= z_bkg[ind,1]*w_bkg[ind,3]/total_weight                                   
        if (args.animate == "sig+bkg"):
            name_var = var_evaluation+'_bkg'          
            plots[name_var+'_lin' ].set_data( bins, train_lin[:,0]  )               
            plots[name_var+'_quad'].set_data( bins, train_quad[:,0] )  
        if (args.animate == "sig-bkg"):
            name_var = var_evaluation+'_bkg_'          
            plots[name_var+'_lin' ].set_data( bins, train_lin[:,0]  )               
            plots[name_var+'_quad'].set_data( bins, train_quad[:,0] )  

def eval_truth ( var_evaluation ):
    total_weight =  np.sum(W_np[:,3])
    if (args.animate == "FULL"):
        x_eval = np.concatenate((np.array(xx[var_evaluation]),np.array(xx_bkg[var_evaluation]))) if args.add_bkg else np.array(xx[var_evaluation]) 
        if reduce: x_eval = np.concatenate((np.array(xx[var_evaluation][0:red]),np.array(xx_bkg[var_evaluation][0:red_bkg]))) if args.add_bkg else np.array(xx[var_evaluation][0:red])
        hist, bins  = np.histogram( x_eval, bins=nbins, range=(config.plot_mva_variables[var_evaluation][0][0], config.plot_mva_variables[var_evaluation][0][1]) )
        truth_lin   = np.zeros( (len(bins),1) )
        truth_quad  = np.zeros( (len(bins),1) )
        for b in range ( 1,len(bins)-1 ):
            for ind in range ( x_eval.shape[0] ):
                val = x_eval[ind]
                if ( val > bins[b-1] and val<= bins[b] ):
                    truth_lin[b] +=W_np[ind,1] / W_np[ind,0]*W_np[ind,3]   / total_weight
                    truth_quad[b]+=W_np[ind,2] / W_np[ind,0]*W_np[ind,3]   / total_weight
        i = plotvars.index( var_evaluation )
        name_var = var_evaluation
        plots[name_var+'truelin'],  = ax[index[i]].plot( bins,truth_lin[:,0],  drawstyle='steps', label = "lin truth",   color='orange' ,linestyle = 'dotted' )
        plots[name_var+'truequad'], = ax[index[i]].plot( bins,truth_quad[:,0], drawstyle='steps', label = "quad truth",  color='red'    ,linestyle = 'dotted' )           
        plots[name_var+"_lin"],     = ax[index[i]].plot(  [] , [] ,            drawstyle='steps', label = "lin train",   color='orange' )
        plots[name_var+"_quad"],    = ax[index[i]].plot(  [] , [] ,            drawstyle='steps', label = "quad train",  color='red'    )
        max_  = np.max(hist)
        max__ = np.max(truth_quad[:,0])
        hist_ = np.zeros((len(hist)+1))
        hist_[1:] = hist
        plots["hist"]               = ax[index[i]].plot( bins, hist_ * max__/max_      ,drawstyle='steps',   label = "yield",       color='gray',   linestyle = 'dotted' )
        #plots[name_var+"_quad"],    = ax[index[i]].plot(  [] , [] ,            drawstyle='steps', label = "quad train",  color='red'    )
        ax[index[i]].set_xlabel( config.plot_mva_variables[var_evaluation][1] )    
    
    if  (args.animate=="sig+bkg" or args.animate=="sig-bkg"):
        x_eval = np.array(xx[var_evaluation])
        if reduce: x_eval = np.array(xx[var_evaluation])[0:red]
        hist, bins  = np.histogram( x_eval, bins=nbins, range=(config.plot_mva_variables[var_evaluation][0][0], config.plot_mva_variables[var_evaluation][0][1]) )
        truth_lin   = np.zeros( (len(bins),1) )
        truth_quad  = np.zeros( (len(bins),1) )
        for b in range ( 1,len(bins)-1 ):
            for ind in range ( x_eval.shape[0] ):
                val = x_eval[ind]
                if ( val > bins[b-1] and val<= bins[b] ):
                    truth_lin[b] +=w[ind,1]/w[ind,0]*w[ind,3] / total_weight 
                    truth_quad[b]+=w[ind,2]/w[ind,0]*w[ind,3] / total_weight            
        i = plotvars.index( var_evaluation )
        name_var = var_evaluation
        plots[name_var+'truelin'],  = ax[index[i]].plot( bins,truth_lin[:,0],  drawstyle='steps', label = "lin truth",   color='orange' ,linestyle = 'dotted' )
        plots[name_var+'truequad'], = ax[index[i]].plot( bins,truth_quad[:,0], drawstyle='steps', label = "quad truth",  color='red'    ,linestyle = 'dotted' )           
        plots[name_var+"_lin"],     = ax[index[i]].plot(  [] , [] ,            drawstyle='steps', label = "lin train",   color='orange' )
        plots[name_var+"_quad"],    = ax[index[i]].plot(  [] , [] ,            drawstyle='steps', label = "quad train",  color='red'    )
        max_  = np.max(hist)
        max__ = np.max(truth_quad[:,0])
        hist_ = np.zeros((len(hist)+1))
        hist_[1:] = hist
        plots["hist"]               = ax[index[i]].plot( bins, hist_ * max__/max_      ,drawstyle='steps',   label = "yield",       color='gray',   linestyle = 'dotted' )
        
        x_eval_bkg = np.array(xx_bkg[var_evaluation])
        if reduce: x_eval_bkg = np.array(xx_bkg[var_evaluation])[0:red_bkg]
        hist_bkg, bins  = np.histogram( x_eval_bkg, bins=bins, range=(config.plot_mva_variables[var_evaluation][0][0], config.plot_mva_variables[var_evaluation][0][1]) )
        truth_lin   = np.zeros( (len(bins),1) )
        truth_quad  = np.zeros( (len(bins),1) )
        for b in range ( 1,len(bins)-1 ):
            for ind in range ( x_eval_bkg.shape[0] ):
                val = x_eval_bkg[ind]
                if ( val > bins[b-1] and val<= bins[b] ):
                    truth_lin[b] +=w_bkg[ind,1] /w_bkg[ind,0]*w_bkg[ind,3] / total_weight
                    truth_quad[b]+=w_bkg[ind,2] /w_bkg[ind,0]*w_bkg[ind,3] / total_weight        
        name_var = var_evaluation+"_bkg"
        
        hist_ = np.zeros((len(hist_bkg)+1))
        hist_[1:] = hist_bkg
        plots["hist"]               = ay[index[i]].plot( bins, hist_ ,drawstyle='steps',   label = "yield",       color='black',   linestyle = 'dotted' )
        
        if args.animate=="sig+bkg":
            plots[name_var+'truelin'],  = ax[index[i]].plot( bins,truth_lin[:,0],  drawstyle='steps', label = "lin truth",   color='blue' ,linestyle = 'dotted' )
            plots[name_var+'truequad'], = ax[index[i]].plot( bins,truth_quad[:,0], drawstyle='steps', label = "quad truth",  color='green'    ,linestyle = 'dotted' )           
            plots[name_var+"_lin"],     = ax[index[i]].plot(  [] , [] ,            drawstyle='steps', label = "lin train",   color='blue' )
            plots[name_var+"_quad"],    = ax[index[i]].plot(  [] , [] ,            drawstyle='steps', label = "quad train",  color='green'    )
        if args.animate=="sig-bkg":
            name_var = var_evaluation+"_bkg_"
            plots[name_var+'truelin'],  = ay[index[i]].plot( bins,truth_lin[:,0],  drawstyle='steps', label = "lin truth",   color='blue' ,linestyle = 'dotted' )
            plots[name_var+'truequad'], = ay[index[i]].plot( bins,truth_quad[:,0], drawstyle='steps', label = "quad truth",  color='green'    ,linestyle = 'dotted' )           
            plots[name_var+"_lin"],     = ay[index[i]].plot(  [] , [] ,            drawstyle='steps', label = "lin train",   color='blue' )
            plots[name_var+"_quad"],    = ay[index[i]].plot(  [] , [] ,            drawstyle='steps', label = "quad train",  color='green'    )    
        ax[index[i]].set_xlabel( config.plot_mva_variables[var_evaluation][1] )  
        ay[index[i]].set_xlabel( config.plot_mva_variables[var_evaluation][1] )          
    
# Initialize the gif
if (args.animate):
    logging.info("setting up animation...")
    # for debug: warning: conflict with animation frame every 10 epochs <-> n_epochs < 10
    assert n_epochs >= args.animate_step, " n_epochs = %i is not sufficient for animating, required > %i " %( n_epochs, args.animate_step )
    # Define the meta data for the movie
    GifWriter = manimation.writers['pillow']
    writer = manimation.PillowWriter( fps=args.animate_fps, metadata=None )
    writer1 = manimation.PillowWriter( fps=args.animate_fps, metadata=None )
    
    logging.info("       gif's duration will be %s second(s)", n_epochs / args.animate_step / args.animate_fps)

    nbins = 20
    index = list(itertools.product(list(range(0, 4)), list(range(0, 5))))
    plotvars=list(config.plot_mva_variables.keys())
    plots = {}   
    logging.info("       plotting truth for %i variables ", len(plotvars))
    fig,  ax = plt.subplots(4,5, figsize=(15,12), tight_layout=True)  # plot max 20 vars
    fig1, ay = plt.subplots(4,5, figsize=(15,12), tight_layout=True)  # plot max 20 vars
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
dataset = NewDataset(X,V,W)
train_loader = DataLoader(dataset=dataset, shuffle = True,
                          batch_size=len(dataset),
                          num_workers=0)

# set up NN
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, hidden_size_comb, output_size, input_size_lstm, hidden_size_lstm, num_layers):
        super(NeuralNet, self).__init__()
        self.batchn = nn.BatchNorm1d(input_size)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.LeakyReLU(0.1) #seems to work slightly better than just a ReLU
        self.linear2 = nn.Linear(hidden_size, hidden_size2)
        
        if (args.LSTM):
            self.num_layers = num_layers
            self.hidden_size_lstm = hidden_size_lstm
            self.lstm = nn.LSTM(input_size_lstm, hidden_size_lstm, num_layers, batch_first=True)
            self.linear3 = nn.Linear(hidden_size2+hidden_size_lstm, hidden_size_comb)
        else:
            self.linear3 = nn.Linear(hidden_size2, hidden_size_comb) 
        self.linear4 = nn.Linear(hidden_size_comb, output_size)    
        
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
        x1 = self.relu(x1)  
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
    
# define loss function   
criterion = get_loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
losses = []

# set up directory and model names
dir_name = "NEW_"+str(args.sample)+'_mean_'+str(args.EFTCoefficients)+'_lr-'+str(args.lr)[2:]+'_lumi-'+str(int(lumi))+'_epx-'+str(n_epochs)+'_hs1-'+str(hidden_size)+'_hs2-'+str(hidden_size2)+'_hsc-'+str(hidden_size_comb)
if ( args.LSTM ):              dir_name = dir_name +  '_lstm-'+str(num_layers)+'_hs-lstm-'+str(hidden_size_lstm)
if (reduce):                   dir_name = dir_name + '_red-'+str(args.reduce)    
if (args.add_bkg):             dir_name = dir_name + '+bkg'  if args.full_bkg==False else dir_name + '+fullbkg'
if (args.animate is not None): dir_name = dir_name + "_" + args.animate    

results_dir = args.output_directory
if not os.path.exists( results_dir ): os.makedirs( results_dir )

# train the model
logging.info("starting training") 
logging.info("")


logger_handler.setFormatter(logging.Formatter('\x1b[80D\x1b[1A\x1b[K%(asctime)s %(message)s'))
if (args.animate):
    with writer.saving(fig, dir_name+".gif", 100):
        with writer1.saving(fig1, dir_name+"_BKG.gif", 100):
            start = datetime.now()
            for epoch in range(n_epochs):
                logging.info("		epoch: %i of %i ", epoch+1, n_epochs)
                for i, data in enumerate(train_loader):
                    inputs1,inputs2, labels = data
                    z = model(inputs1, inputs2)
                    #print(z)
                    loss=criterion(z,labels)
                    assert (loss.data>=0), "Loss haut schun wido o dio knedl"
                    losses.append(loss.data)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if (epoch%args.animate_step==args.animate_step-1):
                        with torch.no_grad():
                            for i in range (len(plotvars)):
                                eval_train(plotvars[i])
                            writer.grab_frame()
                            writer1.grab_frame()
                    if (args.n_epochs/(epoch+1)==args.n_epochs/args.animate_step):    
                        end = datetime.now()
                        logging.info('estimate for training duration: {} \n'.format((end - start)*(args.n_epochs/args.animate_step-1)))
                        logging.info('training will finish appx. at {} \n'.format((end - start)*(args.n_epochs/args.animate_step-1)+end))
if not (args.animate):
    start = datetime.now()  
    for epoch in range(n_epochs):
        logging.info("		epoch: %i of %i ", epoch+1, n_epochs)
        for i, data in enumerate(train_loader):
            inputs1,inputs2, labels = data
            z = model(inputs1, inputs2)
            loss=criterion(z,labels)
            assert (loss.data>=0), "Loss haut schun wido o dio knedl"
            losses.append(loss.data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()   
            if (args.n_epochs/(epoch+1)==10):    
                end = datetime.now()
                logging.info('estimate for training duration: {} \n'.format((end - start)*9))
                logging.info('training will finish appx. at {} \n'.format((end - start)*9+end))

logger_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))           
logging.info("done with training, plotting losses") 
if args.animate != "sig-bkg": os.remove(dir_name+"_BKG.gif")  #remove empty bkg animation         

# plot losses 
fig, ay = plt.subplots()        
plt.plot(losses, color='red')
plt.title("Losses over epoch")
sample_file_name = str(dir_name)+"_losses.png"
plt.yscale("log")  
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
