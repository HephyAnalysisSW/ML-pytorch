#This script is a neural network 

#Settings
Plot_loss = 1
Plot_y_pred_hist=0
Plot_weighted=0


#!/usr/bin/env python
import ROOT, os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT.gStyle.SetOptStat(0)
dir_path = os.path.dirname(os.path.realpath(__file__))
ROOT.gROOT.LoadMacro(os.path.join( dir_path, "../../tools/scripts/tdrstyle.C"))
ROOT.setTDRStyle()


if __name__=="__main__":
    import sys
    sys.path.append('..')
    sys.path.append('../..')

sys.path.insert(0,os.path.expanduser("~/ML-pytorch/"))

import tools.user as user 
import tools.helpers as helpers 
import TT2lUnbinned.data_models.TTLep_pow_sys as data_model

import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--prefix',             action='store', type=str,   default='v1', help="Name of the training")
argParser.add_argument('--data_model',         action='store', type=str,   default='TTLep_pow_sys', help="Which data model?")
argParser.add_argument('--epochs',             action='store', type=int,   default=10, help="Number of epochs")
argParser.add_argument('--small',              action='store_true', help="Small?")
argParser.add_argument('--quadratic',              action='store_true', help="quadratic?" , default=False)
argParser.add_argument('--overwrite',          action='store_true', help="Overwrite?")
argParser.add_argument('--output_directory',   action='store', type=str,   default=os.path.join(user.model_directory,'tt-jec/models/') )
args = argParser.parse_args()

feature_names = [ "nJetGood", "ht", "jet0_pt", "jet1_pt", "jet2_pt", "jet3_pt", "jet0_eta", "jet1_eta", "jet2_eta", "jet3_eta" ]

plot_options =  {
    "met": {'binning': [20, 100, 500], 'logY': True, 'tex': r'$E_{T}^{\mathrm{miss}}$'},
    "ht": {'binning': [20, 500, 1500], 'logY': True, 'tex': r'$H_{T}$'},
    "nJetGood": {'binning': [7, 3, 10], 'logY': True, 'tex': r'$N_{\mathrm{jet}}$'},
    "jet0_pt": {'binning': [30, 0, 1000], 'logY': True, 'tex': r'$p_{T}(\mathrm{jet\ 0})$'},
    "jet1_pt": {'binning': [30, 0, 1000], 'logY': True, 'tex': r'$p_{T}(\mathrm{jet\ 1})$'},
    "jet2_pt": {'binning': [30, 0, 500], 'logY': True, 'tex': r'$p_{T}(\mathrm{jet\ 2})$'},
    "jet3_pt": {'binning': [30, 0, 500], 'logY': True, 'tex': r'$p_{T}(\mathrm{jet\ 3})$'},
    "jet4_pt": {'binning': [30, 0, 500], 'logY': True, 'tex': r'$p_{T}(\mathrm{jet\ 4})$'},
    "jet0_eta": {'binning': [30, -4, 4], 'logY': False, 'tex': r'$\eta(\mathrm{jet\ 0})$'},
    "jet1_eta": {'binning': [30, -4, 4], 'logY': False, 'tex': r'$\eta(\mathrm{jet\ 1})$'},
    "jet2_eta": {'binning': [30, -4, 4], 'logY': False, 'tex': r'$\eta(\mathrm{jet\ 2})$'},
    "jet3_eta": {'binning': [30, -4, 4], 'logY': False, 'tex': r'$\eta(\mathrm{jet\ 3})$'},
    "jet4_eta": {'binning': [30, -4, 4], 'logY': False, 'tex': r'$\eta(\mathrm{jet\ 4})$'},
}


# directories
plot_directory   = os.path.join( user.plot_directory, 'tt-jec', args.data_model, args.prefix+('_small' if args.small else "") + ("_quadratic" if args.quadratic else ""), 'training')
output_directory = os.path.join( args.output_directory, args.data_model, args.prefix+('_small' if args.small else "") + ("_quadratic" if args.quadratic else "")) 

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

n_var_flat = len(data_model.feature_names)

# Define neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, n_var_flat, quadratic):
        super(NeuralNetwork, self).__init__()   #That nn.Module is usable
        self.quadratic = quadratic
        self.batch_norm = nn.BatchNorm1d(n_var_flat)    #Normalization over 2D/3D input
        self.fc1 = nn.Linear(n_var_flat, n_var_flat*2)  #Linear trafo, input values are dimension of Matrix A^T
        self.act1=nn.Sigmoid()
        self.fc2 = nn.Linear(n_var_flat*2, n_var_flat+5)    #Trafo: y=xA^T +b, A and b are adjusted during training, Initial numbers of A and b random (adjustable)
        self.act2=nn.Sigmoid()
        self.output_layer = nn.Linear(n_var_flat+5, 2 if quadratic else 1)
        self.act_output=nn.Sigmoid()

    def forward(self, x):   #is executed every time model(...) is called
        x = self.batch_norm(x)
        x = self.act1(self.fc1(x))  #sigmoid transforms the input data to data between 0 and 1
        x = self.act2(self.fc2(x))
        x = self.act_output(self.output_layer(x))
        return x    #returns value between 0 and 1 
    
#Define Loss Function
def loss(x,y):
    loss = - (y * torch.log(x) + (1 - y) * torch.log(1 - x))    #Binary Cross Entropy
    loss = torch.mean(loss)
    return loss


generator = data_model.DataGenerator(200000)  # Adjust maxN as needed
features, variations = generator[0]
all_features=features
all_variations=variations

epochs = 1000
sigma_range = np.arange(-2, 2.5, 0.5)
loss_matrix = np.zeros((len(sigma_range), epochs))
ratio_array=[]

print("Training starts")

for i,sigma in enumerate(sigma_range):
    np.random.seed(1)
    torch.manual_seed(1)
    print("Now training with sigma= " + str(sigma))
    model = NeuralNetwork(n_var_flat, args.quadratic)   #Reset
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #sigma=2 #Value filtered
    features=all_features       #Reset
    variations=all_variations
    features = features[(variations[:, 0] == sigma) | (variations[:, 0] == 0)] #filter the needed variations and the nominal data
    variations = variations[(variations[:, 0] == sigma) | (variations[:, 0] == 0)]
    features_train, features_test, variations_train, variations_test = train_test_split(features, variations) #Seperates the data into a training and a test set

    # Training loop
    features_train_tensor   = torch.tensor(features_train, dtype=torch.float32)
    variations_train_tensor = torch.tensor(variations_train, dtype=torch.float32)
    features_test_tensor    = torch.tensor(features_test, dtype=torch.float32)
    variations_test_tensor  = torch.tensor(variations_test, dtype=torch.float32)
    features_train_tensor    = torch.where(torch.isnan(features_train_tensor), torch.tensor(0.0), features_train_tensor)        #Replaces NaN values in the tensors with 0
    variations_train_tensor  = torch.where(torch.isnan(variations_train_tensor), torch.tensor(0.0), variations_train_tensor)
    features_test_tensor     = torch.where(torch.isnan(features_test_tensor), torch.tensor(0.0), features_test_tensor)
    variations_test_tensor   = torch.where(torch.isnan(variations_test_tensor), torch.tensor(0.0), variations_test_tensor)
    variations_train_tensor  = torch.where(variations_train_tensor==sigma, torch.tensor(1.0), variations_train_tensor)  #Replace sigmas with 1 to ensure correct Cross entropy
    variations_test_tensor   = torch.where(variations_test_tensor==sigma, torch.tensor(1.0), variations_test_tensor)

    loss_array=[]

    for epoch in range(epochs):
        # Forward pass
        predictions = model(features_train_tensor)
        #print(predictions)
        # Compute the loss
        loss_value = loss(predictions, variations_train_tensor)
        loss_array.append(loss_value.item())
        #Zero the gradients
        optimizer.zero_grad()
        # Backward pass
        loss_value.backward()
        # Update weights
        optimizer.step()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss_value.item()}')


    y_pred=model(features_test_tensor)
    #print(y_pred)
    ratio=1/y_pred-1
    ratio_array.append(ratio)
    #print(ratio)
    #print(variations_test_tensor)
    num_1= torch.sum(variations_test_tensor==1)
    print(num_1)
    num_0= torch.sum(variations_test_tensor==0)
    print(num_0)
    accuracy = (y_pred.round() == variations_test_tensor).float().mean() #Converts to float and does mean
    print("Accuracy: " + str(accuracy.item()))
    loss_matrix[i, :] = loss_array      #Add the loss array to the matrix

print("Training finished")

### Plotting
output_folder = 'Plots'
os.makedirs(output_folder, exist_ok=True)

# Create Histogramm for y_pred values
if Plot_y_pred_hist:
    plt.hist(y_pred.detach().numpy(), bins=10)
    plt.title('Y Pred Values')
    plt.xlabel('Values')
    plt.ylabel('Number')
    output_path1 = os.path.join(output_folder, 'Histo.png')
    plt.savefig(output_path1)
    plt.clf()

#Create loss plots
if Plot_loss:
    for i,sigma in enumerate(sigma_range):
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.title('Loss Function for $\sigma$= ' + str(sigma))
        plt.plot(loss_matrix[i,:])
        output_path2= os.path.join(output_folder, str(sigma) + ' Loss.png')
        plt.savefig(output_path2)
        plt.clf()

# Create weighted Histogramms
if Plot_weighted:
    column_arrays=np.split(all_features,all_features.shape[1],axis=1)
    unique_variations=np.unique(all_variations)
    for i_feature, feature in enumerate(feature_names):
        binning_info = plot_options[feature]['binning']
        bins = np.linspace(binning_info[1], binning_info[2], binning_info[0] + 1)
        nominal_data = column_arrays[i_feature][all_variations == 0]
        n_nominal, _ = np.histogram(nominal_data, bins=bins)
        n_nominal_safe = np.where(n_nominal == 0, 1, n_nominal)     #avoid division by 0
        for variation_value in unique_variations:
            selected_data = column_arrays[i_feature][all_variations == variation_value]
            n, _ = np.histogram(selected_data, bins=bins)
            normalized_histogram = n / n_nominal_safe
            label = f'$\sigma = {variation_value}$'
            color = plt.cm.viridis(variation_value / np.max(unique_variations))
            plt.stairs(normalized_histogram, bins, color = color, label=label)

        plt.xlabel(plot_options[feature]['tex'])
        plt.ylabel('Normalized events weighted')
        scale_info = plot_options[feature]['logY']
        if scale_info == True:
            plt.yscale('log')
        else:
            plt.yscale('linear')
        output_path3 = os.path.join(output_folder, 'weightedvariation_'+ feature + '.png')
        plt.legend(loc='best')
        plt.savefig(output_path3)
        plt.clf()


"""
# Save the model
output_file = os.path.join(args.output_directory, 'multiclass_model_pytorch.pth')
torch.save(model.state_dict(), output_file)
print('Model saved to {output_file}')
"""