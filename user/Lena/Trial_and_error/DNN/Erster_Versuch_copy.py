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




#######################################################################
batches = 80000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate = 0.001
n_epochs=10
input_size = 35
hidden_size = 20
output_size = 4
#######################################################################





class NewDataset(Dataset):

    def __init__(self, number):
        path = 'C:/Users/Lena/Documents/DNN/localdata'+str(number)+'.txt' #!!!!!CHANGE PATH HERE
        xy = np.loadtxt(path, delimiter=None, dtype=np.float32, skiprows=0)
        self.n_samples = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :]) 
        self.y_data = torch.from_numpy(xy[:, [0,0,0,0]]) #size [n_samples, 4]
        for i in range (self.n_samples):
            self.y_data[i][:] = 0
            self.y_data[i][number-1] = int(1)
            
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples
    
    def __givey__(self):
        return self.y_data




############################### 1. DATA PREPARATION ###############################

# create datasets
dataset0 = NewDataset(1)
dataset1 = NewDataset(2)
dataset2 = NewDataset(3)
dataset3 = NewDataset(4)
print("Number of samples: ", len(dataset0)+len(dataset1)+len(dataset2)+len(dataset3))

# access individual samples
x0, y0 = dataset0[:]
x1, y1 = dataset1[:]
x2, y2 = dataset2[:]
x3, y3 = dataset3[:]

# concatenate
y_full = torch.cat([y0, y1, y2, y3], dim=0, out=None)
x_full = torch.cat([x0, x1, x2, x3], dim=0, out=None)


# plot data for checking afterwards
fig, ax = plt.subplots()
y_testplot = torch.max(y_full,1)
y_testplot = y_testplot.indices.numpy()
plt.xticks([])
ax.hist(y_testplot, bins=4)


# split data in train and test
train_size = int(0.8 * len(y_full))
test_size = len(y_full) - train_size
datadef = torch.utils.data.ConcatDataset([dataset0,dataset1,dataset2,dataset3])
train_dataset, test_dataset = torch.utils.data.random_split(datadef, [train_size, test_size])


# load whole datasets with DataLoader
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batches,
                          shuffle=True,
                          num_workers=0)
test_loader = DataLoader(dataset=test_dataset,
                          batch_size=test_size,
                          shuffle=True,
                          num_workers=0)




######################## 2. SET UP NN ##########################################

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out
    

model = NeuralNet(input_size, hidden_size, output_size).to(device)    
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
losses = []
print('Training set has {} instances'.format(len(train_loader)))



############################## 3. TRAINING THE MODEL #############################

for epoch in range(n_epochs):
    print("epoch: ", epoch )
    for i, data in enumerate(train_loader):
        inputs, labels = data
        z=model(inputs)
        loss=criterion(z,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.data)
    
fig, ay = plt.subplots()        
ay.plot(losses)





############################### 4. TEST MODEL #####################################

with torch.no_grad():
    for i, data in enumerate(test_loader):
        x_test, y_test = data
        z = model(x_test)
        test_loss = criterion(z, y_test)
        print(test_loss)
        
        yhat = torch.max(z.data, 1)    
        y_testpred = yhat.indices.numpy()
        fig, az = plt.subplots(2, 1)
        plt.xticks([])
        az[0].hist(y_testpred, bins=4) 
    
        y_testtrue = torch.max(y_test,1)
        y_testtrue = y_testtrue.indices.numpy()
        plt.xticks([])
        az[1].hist(y_testtrue, bins=4)
         
############################## 5. SAVE MODEL #####################################

PATH = PATH = 'C:/Users/Lena/Documents/DNN/dnn.pth'
torch.save(model.state_dict(), PATH)    