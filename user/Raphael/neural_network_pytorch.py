#This script for training with a neural network

#Settings
epochs = 30


#!/usr/bin/env python
import ROOT, os
import sys
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

    def forward(self, x, nu):   #is executed every time model(...) is called
        x = self.batch_norm(x)
        x = self.act1(self.fc1(x))  #sigmoid transforms the input data to data between 0 and 1
        x = self.act2(self.fc2(x))
        x = self.act_output(self.output_layer(x))   #returns value between 0 and 1, is equal to 1/(1+r_v)
        x = 1/nu *torch.log(1/x -1)
        return x     #Returns Delta Value! 
    
#Define Loss Function
def loss(Delta,nu):
    soft=torch.nn.Softplus(nu)
    loss = nu*(soft(Delta)+soft(-Delta))
    loss = torch.sum(loss)
    return loss

def train():
    generator = data_model.DataGenerator()  # Adjust maxN as needed
    features, variations = generator[0]

    sigma_range = np.arange(-2, 2.5, 0.5)
    loss_matrix = np.zeros((len(sigma_range), epochs))
    loss_array=[]
    best_loss=float('inf')

    print("Training starts")
    np.random.seed(1)
    torch.manual_seed(1)
    model = NeuralNetwork(n_var_flat, args.quadratic)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(model)

    for epoch in range(epochs):
        # Training loop
        Loss = 0
        # Forward pass
        for i,sigma in enumerate(sigma_range):
            if sigma !=0:
                selected_features = features[(variations[:, 0] == sigma) | (variations[:, 0] == 0)] #filter the needed variations and the nominal data
                selected_variations = variations[(variations[:, 0] == sigma) | (variations[:, 0] == 0)]
                selected_features_tensor   = torch.tensor(selected_features, dtype=torch.float32)
                selected_variations_tensor = torch.tensor(selected_variations, dtype=torch.float32)
                selected_features_tensor =   torch.where(torch.isnan(selected_features_tensor), torch.tensor(0.0), selected_features_tensor)   #Replace missing values with 0
                predictions = model(selected_features_tensor,sigma)
                #print(predictions)
                # Compute the loss
                loss_value = loss(predictions, sigma)
                Loss+=loss_value.item()
        if Loss < best_loss:
            best_loss = Loss
            best_model = NeuralNetwork(n_var_flat, args.quadratic)
            best_model.load_state_dict(model.state_dict())
            best_epoch=epoch       
        #Zero the gradients
        optimizer.zero_grad()
        # Backward pass
        loss_value.backward()
        # Update weights
        optimizer.step()
        
        loss_array.append(Loss)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {Loss}')
    print("Training finished")

    #Save the data
    np.savez('training_data.npz', loss_array=loss_array)
    output_file = 'best_model.pth'
    torch.save(best_model.state_dict(), output_file)
    print(f'Best model saved. Best Loss: {best_loss} in Epoch: {best_epoch}')

if __name__ == "__main__":
    train()