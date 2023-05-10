import torch
import numpy as np
import math 
import array
import sys, os
import glob
sys.path.insert(0, '..')
import tools.syncer as syncer
import tools.user as user
import tools.helpers as helpers
import pickle

import ROOT
dir_path = os.path.dirname(os.path.realpath(__file__))

# reproducibility
torch.manual_seed(1)
import numpy as np
np.random.seed(0)

delay  = 50/5

logZ = True
#training_name  = "onlyR_try2"
#config_name    = "onlyR"
#xmin, xmax = 0, 1.5
#varName = "R"
#index_truth = 0
#index_out   = 0
#every = 2
#def func( out ):
#    return out[:, 0] 

#def func( out ):
#    return out[:, 0] 
#training_name  = "onlyGamma_try2"
#config_name    = "onlyGamma"
#xmin, xmax = -math.pi, math.pi
#varName = "#gamma"
#index_truth = 1
#index_out   = 0
#every = 2
#def func( out ):
#    return out[:, 0] 

#training_name  = "onlyGamma_oneProng_sinCosSep"
#config_name    = "onlyGamma_oneProng"
#xmin, xmax = -math.pi, math.pi
#varName = "#gamma"
#index_truth = 1
#index_out   = 0
#every = 2
#def func( out ):
#    return torch.atan2( out[:,0], out[:,1] )

logZ=False
training_name  = "onlyGamma_twoProng_sinCosShark"
config_name    = "onlyGamma_twoProng"
xmin, xmax = -math.pi, math.pi
varName = "#gamma"
index_truth = 1
index_out   = 0
every = 10
#def func( out ):
#    return torch.atan2( out[:,0], out[:,1] )
def func( out ):
    return out[:,0]

exec("import configs.regressJet_%s as config"%config_name)

pts, angles, truth = config.data_model.getEvents(5000)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

plot_directory_ = os.path.join( user.plot_directory, 'SMEFTNet', "regression", training_name)
os.makedirs( plot_directory_, exist_ok=True)

ROOT.gStyle.SetPalette(ROOT.kBird)
ROOT.gROOT.LoadMacro(os.path.join( dir_path, "../tools/scripts/tdrstyle.C"))
ROOT.setTDRStyle()

torch.autograd.set_grad_enabled(False)

model_directory = "/groups/hephy/cms/robert.schoefbeck/NN/models/SMEFTNet/regressJet_%s/%s/"%(config_name,training_name)
files = glob.glob( os.path.join( model_directory, 'epoch-*_state.pt') )

# Text on the plots
tex = ROOT.TLatex()
tex.SetNDC()
tex.SetTextSize(0.04)

c1 = ROOT.TCanvas()
for filename in files[0::every]:
    load_epoch = int(filename.split('-')[-1].split('_')[0])
    print('At %s' % filename)
    model_state = torch.load(filename, map_location=device)
    config.model.load_state_dict(model_state)
    config.model.cfg_dict = pickle.load( open( filename.replace('_state.pt', '_cfg_dict.pkl'), 'rb') )
    out = config.model( pts, angles )
    h = ROOT.TH2F( "R", "R", 20,xmin, xmax,20,xmin,xmax)
    for R_pred, R_true in zip(func(out).numpy(), truth[:,index_truth].numpy()):
        h.Fill( R_true, R_pred)

    h.Draw("COLZ")
    h.GetZaxis().SetRangeUser(0,100)
    h.GetYaxis().SetTitle(varName+"_{pred}")
    h.GetXaxis().SetTitle(varName+"_{truth}")
    line1 = ( 0.16, 0.96, "Epoch %i"%load_epoch )
    o = tex.DrawLatex(*line1) 
    o.Draw()
    c1.SetRightMargin(0.15)
    c1.SetLogz(logZ)
    c1.Print(os.path.join( plot_directory_, training_name+'_%s.png'%(str(load_epoch).zfill(5))) )

syncer.sync()
syncer.makeRemoteGif(plot_directory_, pattern=training_name+"_*.png", name=training_name, delay=delay)
helpers.copyIndexPHP( plot_directory_ )
