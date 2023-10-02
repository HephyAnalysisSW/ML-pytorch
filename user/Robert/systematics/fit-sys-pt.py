#!/usr/bin/env python
import ROOT, os
ROOT.gStyle.SetOptStat(0)
dir_path = os.path.dirname(os.path.realpath(__file__))
ROOT.gROOT.LoadMacro(os.path.join( dir_path, "../../../tools/scripts/tdrstyle.C"))
ROOT.setTDRStyle()
import numpy as np
if __name__=="__main__":
    import sys
    sys.path.append('../../..')

import tools.syncer as syncer
import tools.user as user 
import tools.helpers as helpers 

import torch
device        = 'cuda' if torch.cuda.is_available() else 'cpu'

c1 = ROOT.TCanvas()
c1.Draw()
c1.Print('/tmp/delete.png')

import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--prefix',             action='store', type=str,   default='v1', help="Name of the training")
argParser.add_argument('--data_model',         action='store', type=str,   default='TTLep_pow_sys', help="Which data model?")
argParser.add_argument('--epochs',             action='store', type=int,   default=200, help="Number of epochs")
argParser.add_argument('--every',              action='store', type=int,   default=5, help="Plot at every n epochs?")
argParser.add_argument('--small',              action='store_true', help="Small?")
argParser.add_argument('--overwrite',          action='store_true', help="Overwrite?")
argParser.add_argument('--output_directory',   action='store', type=str,   default='/eos/vbc/group/cms/robert.schoefbeck/tt-jec/models/')
args = argParser.parse_args()

# fix random seed for reproducibility
np.random.seed(1)

#########################################################################################
# Training data 

exec( "import data_models.%s as data_model"%args.data_model )

levels    = [-2.0, -1.5, -1, -0.5, 0.5, 1.0, 1.5, 2.0]
quadratic = True

generator = data_model.DataGenerator(maxN=200000 if args.small else None, levels = levels)

# directories
plot_directory   = os.path.join( user.plot_directory, 'systematics', args.data_model, 'training', args.prefix )
output_directory = os.path.join( args.output_directory, args.prefix+'_small' if args.small else args.prefix) 

# flat layers
n_var_flat = len(data_model.feature_names)

##def make_NN( hidden_layers  = [ 2*n_var_flat, n_var_flat+5] ):
##    model_nn = [torch.nn.BatchNorm1d(n_var_flat), torch.nn.Linear(n_var_flat, hidden_layers[0]), torch.nn.ReLU()]
##    for i_layer, layer in enumerate(hidden_layers):
##
##        model_nn.append(torch.nn.Linear(hidden_layers[i_layer], hidden_layers[i_layer+1] if i_layer+1<len(hidden_layers) else (2 if quadratic else 1)))
##        if i_layer+1<len(hidden_layers):
##            model_nn.append( torch.nn.ReLU() )
##
##    return torch.nn.Sequential(*model_nn)

class SysNetwork(torch.nn.Module):
     
    def __init__( self, hidden_layers  = [ 2*n_var_flat, n_var_flat+5] ):
        super().__init__()

        self.bias = 0.

        self.my_modules = torch.nn.ModuleList([torch.nn.BatchNorm1d(n_var_flat), torch.nn.Linear(n_var_flat, hidden_layers[0]), torch.nn.LeakyReLU()])

        #self.my_modules.extend( [torch.nn.BatchNorm1d(n_var_flat), torch.nn.Linear(n_var_flat, hidden_layers[0]), torch.nn.LeakyReLU()])
        self.my_modules[-1].negative_slope = 0.1

        for i_layer, layer in enumerate(hidden_layers):

            self.my_modules.append(torch.nn.Linear(hidden_layers[i_layer], hidden_layers[i_layer+1] if i_layer+1<len(hidden_layers) else (2 if quadratic else 1)))
            if i_layer+1<len(hidden_layers):
                self.my_modules.append( torch.nn.LeakyReLU() )
                self.my_modules[-1].negative_slope = 0.1

    def set_bias( self, bias):
        self.bias = bias

    def forward( self, x):

        for m in self.my_modules: 
            x = m(x)
        return self.bias+x
        return x

t_sp = torch.nn.Softplus()
def loss(truth, prediction):

    prediction_nominal  = prediction[truth[:,0]==0]

    #print( "truth", truth.shape, truth)
    #print( "prediction", prediction.shape, prediction)

    loss = torch.tensor(0., requires_grad=True)
    for level in generator.levels:
        #print ("Adding loss level", level)
        if level==0: continue #sanity

        prediction_level = prediction[truth[:,0]==level]

        #print( "prediction", prediction.shape, prediction)

        nom = level*prediction_nominal[:,0]
        lev = level*prediction_level[:,0]
        if quadratic:
            nom += level**2*prediction_nominal[:,1]
            lev += level**2*prediction_level[:,1]
        #print( "nom", nom.shape, nom)

        loss = loss + (t_sp(nom)).sum()
        loss = loss + (t_sp(-lev)).sum()

    return loss 

from sklearn.model_selection import train_test_split

features, variations = generator[0]
features_train, features_test, variations_train, variations_test = train_test_split(
      torch. nan_to_num(torch.tensor(features).to(device)), 
      torch. nan_to_num(torch.tensor(variations).to(device)) )

#predictions = np.ones((len(variations_train),1))

network   = SysNetwork()#make_NN()
optimizer = torch.optim.Adam(network.parameters(), lr=0.003)

print (network)

count_0 = torch.count_nonzero(variations_train==0)
count_1 = torch.count_nonzero(variations_train==1)
print ("Training with", count_0,count_1, "expect", np.log(count_1/count_0), "rel stat error", np.sqrt(1./count_0+1./count_1)) 

with torch.no_grad(): 
    network.set_bias( np.log(count_1/count_0) - network(features_train.float()).mean())

output_file = os.path.join(output_directory, 'multiclass_model.h5')
if not os.path.exists(output_file) or args.overwrite:

    network.train()
    losses = []
    for epoch in range(args.epochs):

        # Compute and print loss.

        predictions =  network(features_train.float())
        loss_ = loss(variations_train, predictions )
        losses.append(loss_.item())

        optimizer.zero_grad()
        loss_.backward()

        optimizer.step()
        #scheduler.step()
        mean_pred = predictions.mean().item()
        print ("loss %5.2f" %loss_.item(), "mean prediction %5.5f"% mean_pred, "diff %5.5f", mean_pred-np.log(count_1/count_0))

        stuff = []
        if epoch%args.every==0 or epoch==args.epochs-1: 
          with torch.no_grad():

            _prediction  = network(torch.tensor(features[variations[:,0]==0]).float()).detach().numpy() 

            print ("Mean prediction", _prediction.mean())
            def prediction( nu ):
                return np.exp( nu*_prediction[:,0] + ( nu**2*_prediction[:,1] if quadratic else 0))

            color = {2:ROOT.kRed, 1.5: ROOT.kOrange+10, 1:ROOT.kMagenta+2, 0.5: ROOT.kBlue+1, -0.5: ROOT.kCyan+2, -1.:ROOT.kGreen+2, -1.5: ROOT.kYellow+2, -2:ROOT.kAzure-8}

            n_pads = len(data_model.feature_names)+1
            n_col  = int(np.sqrt(n_pads))
            n_rows = n_pads//n_col
            if n_rows*n_col<n_pads: n_rows+=1

            c1 = ROOT.TCanvas("c1","multipads",500*n_col,500*n_rows);
            c1.Divide(n_col,n_rows)

            c2 = ROOT.TCanvas("c2","ratiopads",500*n_col,500*n_rows);
            c2.Divide(n_col,n_rows)

            l = ROOT.TLegend(0.2,0.8,0.9,0.95)
            l.SetNColumns(3)
            l.SetFillStyle(0)
            l.SetShadowColor(ROOT.kWhite)
            l.SetBorderSize(0)
 
            for i_feature, feature in enumerate(data_model.feature_names):

                binning   = data_model.plot_options[feature]['binning'] 
                np_binning= np.linspace(binning[1], binning[2], 1+binning[0])

                h_nominal  = helpers.make_TH1F(np.histogram(features[variations[:,0]==0][:,i_feature], np_binning ))
                stuff.append( h_nominal )
                h_nominal  .SetLineColor(ROOT.kGray+2)
                h_nominal  .SetMarkerColor(ROOT.kGray+2)
                h_nominal  .SetMarkerStyle(0)

                c1.cd(i_feature+1)
                ROOT.gStyle.SetOptStat(0)

                h_nominal.Draw()
                h_nominal.SetTitle("")
                h_nominal.GetXaxis().SetTitle(data_model.plot_options[feature]['tex'])
                c1.SetLogy()

                if i_feature==0: l.AddEntry(h_nominal, "nominal")
                h_level_truth = {}
                h_level_pred  = {}
                for level in generator.levels:
                        
                    h_level_truth[level]  = helpers.make_TH1F(np.histogram(features[variations[:,0]==level][:,i_feature], np_binning))
                    stuff.append( h_level_truth[level] )
                    h_level_truth[level] .SetLineColor(color[level])
                    h_level_truth[level].SetLineStyle(ROOT.kDashed)
                    h_level_truth[level]      .SetMarkerColor(color[level])
                    h_level_truth[level]      .SetMarkerStyle(0)
                    h_level_pred[level]  = helpers.make_TH1F(np.histogram(features[variations[:,0]==0][:,i_feature], np_binning, weights=prediction(level)))
                    stuff.append( h_level_pred[level] )
                    h_level_pred[level] .SetLineColor(color[level])
                    h_level_pred[level]      .SetMarkerColor(color[level])
                    h_level_pred[level]      .SetMarkerStyle(0)

                    #l.AddEntry(h_level_truth[level]   , "#sigma = %2.1f (truth)"%level )
                    if i_feature==0: l.AddEntry(h_level_pred[level]   , "#sigma = %2.1f"%level )

                    h_level_truth[level].Draw("same")
                    h_level_pred[level].Draw("same")

                h_nominal.Draw("same")

                #c1.Print(os.path.join( plot_directory, feature+'.png'))

                c2.cd(i_feature+1)
                ROOT.gStyle.SetOptStat(0)

                h_nominal_ = h_nominal.Clone()
                ref = h_nominal_.Clone()
                stuff.append( ref )
                stuff.append( h_nominal_ )

                h_nominal_.Divide(ref)
                h_nominal_.Draw()
                c2.SetLogy(0)
                h_nominal.GetYaxis().SetRangeUser(0.8,1.4)
                for level in generator.levels:
                    h_level_truth_ = h_level_truth[level].Clone()
                    stuff.append( h_level_truth_ )
                    h_level_truth_.Divide(ref)

                    h_level_pred_ = h_level_pred[level].Clone()
                    stuff.append( h_level_pred_ )
                    h_level_pred_.Divide(ref)

                    h_level_truth_.Draw("same")
                    h_level_pred_.Draw("same")

            c1.cd(len(data_model.feature_names)+1)
            l.Draw()
            c1.Print(os.path.join( plot_directory, "features_epoch%04i"%epoch+'.png'))
            c2.cd(len(data_model.feature_names)+1)
            l.Draw()
            c2.Print(os.path.join( plot_directory, "variation_epoch%04i"%epoch+'.png'))

            syncer.makeRemoteGif(plot_directory, pattern="features_epoch*", name="features_epoch" )
            syncer.makeRemoteGif(plot_directory, pattern="variation_epoch*", name="variation_epoch" )
            
            helpers.copyIndexPHP( plot_directory )
            syncer.sync()

    # saving
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if os.path.exists(output_file):
        os.remove( output_file)
    #network.save(output_file)
    #print("Written network to: %s"% output_file)
else:
    #network.load_weights(output_file)
    #print("Loaded network from  %s"% output_file)
    pass

