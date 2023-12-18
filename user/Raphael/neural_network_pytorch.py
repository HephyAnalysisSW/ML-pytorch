#This script is a neural network 

#!/usr/bin/env python
import ROOT, os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split

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
    

def loss(x,y):
    loss = - (y * torch.log(x) + (1 - y) * torch.log(1 - x))    #Binary Cross Entropy
    loss = torch.mean(loss)
    return loss

#loss=nn.BCELoss()
np.random.seed(1)
torch.manual_seed(1)


sigma=2
generator = data_model.DataGenerator(maxN=200000)  # Adjust maxN as needed
features, variations = generator[0]
features = features[(variations[:, 0] == sigma) | (variations[:, 0] == 0)] #filter the needed variations
variations = variations[(variations[:, 0] == sigma) | (variations[:, 0] == 0)]
features_train, features_test, variations_train, variations_test = train_test_split(features, variations) #Seperates the data into a training and a test set


n_var_flat = len(data_model.feature_names)
model = NeuralNetwork(len(data_model.feature_names), args.quadratic)
optimizer = optim.Adam(model.parameters(), lr=0.001)
print(model)

# Training loop
epochs = 1000
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

for epoch in range(epochs):
 
    # Forward pass
    predictions = model(features_train_tensor)
    #print(predictions)
    # Compute the loss
    loss_value = loss(predictions, variations_train_tensor)
    #Zero the gradients
    optimizer.zero_grad()
    # Backward pass
    loss_value.backward()
    # Update weights
    optimizer.step()

    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss_value.item()}')



y_pred=model(features_test_tensor)
print(y_pred)
print(variations_test_tensor)
num_1= torch.sum(variations_test_tensor==1)
print(num_1)
num_0= torch.sum(variations_test_tensor==0)
print(num_0)
accuracy = (y_pred.round() == variations_test_tensor).float().mean()
print("Accuracy: " + str(accuracy.item()))

"""
# Save the model
output_file = os.path.join(args.output_directory, 'multiclass_model_pytorch.pth')
torch.save(model.state_dict(), output_file)
print('Model saved to {output_file}')
"""
