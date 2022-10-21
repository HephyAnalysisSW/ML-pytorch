# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 10:37:43 2022

@author: Lena
"""

import torch
import math
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
#import os

#import sys
#sys.path.append('..')

#############################################################
batches = 80000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate = 0.001



class NewDataset(Dataset):

    def __init__(self, number):
        # Initialize data, download, etc.
        # read with numpy or pandas
        path = 'C:/Users/Lena/Documents/DNN/localdata'+str(number)+'.txt'
        xy = np.loadtxt(path, delimiter=None, dtype=np.float32, skiprows=0)
        self.n_samples = xy.shape[0]

        # here the first column is the class label, the rest are the features
        self.x_data = torch.from_numpy(xy[:, :]) # size [n_samples, n_features]
        self.y_data = torch.from_numpy(xy[:, [0]])# size [n_samples, 4]
        for i in range (self.n_samples):
            self.y_data[i] = int(number-1)
            
    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
    
    # we can call len(dataset) to return the size
    def __givey__(self):
        return self.y_data



# create dataset
dataset0 = NewDataset(1)
dataset1 = NewDataset(2)
dataset2 = NewDataset(3)
dataset3 = NewDataset(4)
print(len(dataset0)+len(dataset1)+len(dataset2)+len(dataset3))

x0, y0 = dataset0[:]
x1, y1 = dataset1[:]
x2, y2 = dataset2[:]
x3, y3 = dataset3[:]


y_test = torch.cat([y0, y1, y2, y3], dim=0, out=None)
x_test = torch.cat([x0, x1, x2, x3], dim=0, out=None)

fig, ax = plt.subplots()
y_testplot = y_test.numpy()
ax.hist(y_testplot, bins=4)

datadef = torch.utils.data.ConcatDataset([dataset0,dataset1,dataset2,dataset3])


# Load whole dataset with DataLoader
# shuffle: shuffle data, good for training
# num_workers: faster loading with multiple subprocesses
# !!! IF YOU GET AN ERROR DURING LOADING, SET num_workers TO 0 !!!
train_loader = DataLoader(dataset=datadef,
                          batch_size=batches,
                          shuffle=True,
                          num_workers=0)


input_size = 35
hidden_size = 20
output_size = 4

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out
    

model = NeuralNet(input_size, hidden_size, output_size).to(device)    

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
losses = []
n_epochs=2
print('Training set has {} instances'.format(len(train_loader)))
for epoch in range(n_epochs):
    print("epoch: ", epoch )
    for i, data in enumerate(train_loader):
        inputs, labels = data
        print(i)
        #make a prediction 
        z=model(inputs)
        # calculate loss, da Cross Entropy benutzt wird muss ich in den loss Klassen vorhersagen, 
        # also Wahrscheinlichkeit pro Klasse. Das mach torch.max(y,1)[1])
        loss=criterion(z,labels)
        # calculate gradients of parameters
        optimizer.zero_grad()
        loss.backward()
        # update parameters 
        optimizer.step()
        
        losses.append(loss.data)
    
    
fig, ay = plt.subplots()        
ay.plot(losses)


with torch.no_grad():
    z = model(x_test)
    yhat = torch.max(z.data, 1)    
    y_train_predplot = yhat.indices.numpy()
    fig, az = plt.subplots()
    az.hist(y_train_predplot, bins=4)  