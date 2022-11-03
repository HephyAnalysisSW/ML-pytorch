
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler




class NewDataset(Dataset):

    def __init__(self, number):
        path = 'C:/Users/Lena/Documents/DNN/Funktionierendes_Model/localdata'+str(number)+'.txt' #!!!!!CHANGE PATH HERE
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
l0 = len(dataset0)
l1 = len(dataset1)
l2 = len(dataset2)
l3 = len(dataset3)

f = open("C:/Users/Lena/Documents/DNN/Funktionierendes_Model/Event_properties.txt", "r")
lis = f.read()
lis = lis.split()
print(lis[0])

# access individual samples
x0, y0 = dataset0[:]
x1, y1 = dataset1[:]
x2, y2 = dataset2[:]
x3, y3 = dataset3[:]

for i in range (len(lis)):
    data1 = np.array(x0[:,i])
    data2 = np.array(x1[:,i])
    data3 = np.array(x2[:,i])
    data4 = np.array(x3[:,i])
    datafull = np.concatenate((data1,data2,data3,data4))
    print(i, np.max(datafull),  np.min(datafull))
    fig, az = plt.subplots(figsize = (7, 5))
    bins = 7
    plt.hist([data1,data2, data3,data4], bins=bins, stacked=True, label = ["TTTT","TTLep_bb", "TTLep_cc","TTLep_other"]) 
    plt.title(lis[i])
    plt.legend()
    plt.show()