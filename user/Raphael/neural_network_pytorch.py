#This script for training with a neural network

#Settings
epochs = 100


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
import models.TTLep_pow_sys as data_model

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


generator = data_model.DataGenerator()  # Adjust maxN as needed
features, variations = generator[0]
all_features=features
all_variations=variations
"""
count1=(variations==-2).sum()
print(count1)
count2=(variations==-1.5).sum()
print(count2)
count3=(variations==-1).sum()
print(count3)
count4=(variations==-0.5).sum()
print(count4)
count5=(variations==0).sum()
print(count5)
count6=(variations==0.5).sum()
print(count6)
count7=(variations==1).sum()
print(count7)
count8=(variations==1.5).sum()
print(count8)
count9=(variations==2).sum()
print(count9)
"""

sigma_range = np.arange(-2, 2.5, 0.5)
loss_matrix = np.zeros((len(sigma_range), epochs))
ratio_array=[]
delta_array=[]
y_pred_array=[]

print("Training starts")

for i,sigma in enumerate(sigma_range):
    np.random.seed(1)
    torch.manual_seed(1)
    print("Now training with sigma= " + str(sigma))
    model = NeuralNetwork(n_var_flat, args.quadratic)   #Reset
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    features=all_features       #Reset
    variations=all_variations
    features = features[(variations[:, 0] == sigma) | (variations[:, 0] == 0)] #filter the needed variations and the nominal data
    variations = variations[(variations[:, 0] == sigma) | (variations[:, 0] == 0)]
    features_train, features_test, variations_train, variations_test = train_test_split(features, variations) #Seperates the data into a training and a test set


    # Training loop
    features_tensor   = torch.tensor(features, dtype=torch.float32)
    variations_tensor = torch.tensor(variations, dtype=torch.float32)
    features_tensor =   torch.where(torch.isnan(features_tensor), torch.tensor(0.0), features_tensor) 
    variations_tensor  = torch.where(torch.isnan(variations_tensor), torch.tensor(0.0), variations_tensor)
    variations_tensor   = torch.where(variations_tensor==sigma, torch.tensor(1.0), variations_tensor)   #Replace sigmas with 1 to ensure correct Cross entropy


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
        predictions = model(features_tensor)
        #print(predictions)
        # Compute the loss
        loss_value = loss(predictions, variations_tensor)
        loss_array.append(loss_value.item())
        #Zero the gradients
        optimizer.zero_grad()
        # Backward pass
        loss_value.backward()
        # Update weights
        optimizer.step()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss_value.item()}')


    y_pred=model(features_tensor)
    y_pred_0=y_pred[variations_tensor==0].detach().numpy()
    y_pred_array.append(y_pred_0)
    ratio=(1-y_pred)/(y_pred)
    #accuracy = (y_pred.round() == variations_tensor).float().mean() #Converts to float and does mean
    #print("Accuracy: " + str(accuracy.item()))
    loss_matrix[i, :] = loss_array      #Add the loss array to the matrix
    ratio_0 = ratio[variations_tensor == 0].detach().numpy()    #Only add the ratios corresponding to the nominal data
    ratio_array.append(ratio_0)
    delta=1/sigma*np.log(ratio_0)
    delta_array.append(delta)
print("Training finished")

#Save the data
np.savez('training_data.npz', loss_matrix=loss_matrix, ratio_array=ratio_array, delta_array=delta_array, y_pred_array=y_pred_array, sigma_range=sigma_range, all_features=all_features, all_variations=all_variations)

"""
# Save the model
output_file = os.path.join(args.output_directory, 'multiclass_model_pytorch.pth')
torch.save(model.state_dict(), output_file)
print('Model saved to {output_file}')
"""
