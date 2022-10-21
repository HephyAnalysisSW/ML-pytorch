# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 10:37:43 2022

@author: Lena
########################################
#GENERATE DATA CALLED LOCALDATA WITH JUPYTER NOTEBOOK Load_data_to_local_file.ipynb and Event_properties.txt
########################################

import torch
import math
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn



#######################################################################
#batches = 80000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate = 0.001
n_epochs= 1600
input_size = 35
hidden_size = 70
hidden_size2 = 40
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


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size,hidden_size2, output_size):
        super(NeuralNet, self).__init__()
        self.batchn = nn.BatchNorm1d(input_size)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, output_size)
        self.softm = nn.Softmax(dim =1)
    
    def forward(self, x):
        out = self.batchn(x)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.relu(out)
        out = self.softm(out)
        return out
    
    
    

############################### 1. DATA PREPARATION ###############################

# create datasets
dataset0 = NewDataset(1)
dataset1 = NewDataset(2)
dataset2 = NewDataset(3)
dataset3 = NewDataset(4)
print("Number of samples: ", len(dataset0)+len(dataset1)+len(dataset2)+len(dataset3))

# access individual samples
x0, y0 = dataset0[0:175933]
x1, y1 = dataset1[0:175933]
x2, y2 = dataset2[0:175933]
x3, y3 = dataset3[0:175933]

# concatenate
y_full = torch.cat([y0, y1, y2, y3], dim=0, out=None)
x_full = torch.cat([x0, x1, x2, x3], dim=0, out=None)

fig, ax = plt.subplots()
y_testplot = torch.max(y_full,1)
y_testplot = y_testplot.indices.numpy()
plt.xticks([])
ax.hist(y_testplot, bins=4)


print("hallo, ich bin mit dem Dateneinlesen fertig, du Arschkappl! ")



######################## 2. SET UP NN ##########################################
  

model = NeuralNet(input_size, hidden_size, hidden_size2, output_size).to(device)    
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
losses = []
print('Training set has {} instances'.format(len(y_full)))


############################## 3. TRAINING THE MODEL #############################

for epoch in range(n_epochs):
    print("epoch: ", epoch ) 
    z=model(x_full)
    #print(z)
    loss=criterion(z,y_full)
   
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.data)
    if (epoch % 100)==99:
       with torch.no_grad():
            yhat = torch.max(z.data, 1) 
            y_testtrue = torch.max(y_full,1)
            y_testpred = yhat.indices.numpy()
            y_testtrue = y_testtrue.indices.numpy()
            histodata = np.zeros((4,4))
            bins = [0, 1, 2, 3]
            for j in range (len(y_full)):
                histodata[y_testtrue[j]][y_testpred[j]]+=1
            fig, az = plt.subplots(3, 1, figsize = (10,10))
            plt.xticks([])
            az[0].bar(bins, histodata[0,:], label = "Cat 1 = TTTT") 
            az[0].bar(bins, histodata[1,:], bottom = histodata[0,:], label = "Cat 2 = TTLep_bb") 
            az[0].bar(bins, histodata[2,:], bottom = histodata[1,:]+histodata[0,:], label = "Cat 3 = TTLep_cc") 
            az[0].bar(bins, histodata[3,:], bottom = histodata[1,:]+histodata[0,:]+histodata[2,:], label = "Cat 4 = TTLep_other") 
            az[0].legend()
            plt.xticks([])
            az[1].hist(y_testpred, bins=4)  
            az[2].hist(y_testtrue, bins=4)  
        
        
        
fig, ay = plt.subplots()        
ay.plot(losses)


############################### 4. TEST MODEL #####################################
with torch.no_grad():
    z = model(x_full)
    test_loss = criterion(z, y_full)
    print(test_loss)        
    yhat = torch.max(z.data, 1)    
    y_testpred = yhat.indices.numpy()
    fig, az = plt.subplots(2, 1)
    plt.xticks([])
    az[0].hist(y_testpred, bins=4) 
    y_testtrue = torch.max(y_full,1)
    y_testtrue = y_testtrue.indices.numpy()
    plt.xticks([])
    az[1].hist(y_testtrue, bins=4)
         
         
############################## 5. SAVE MODEL #####################################

PATH = PATH = 'C:/Users/Lena/Documents/DNN/dnn.pth'
torch.save(model.state_dict(), PATH)    
