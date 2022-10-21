import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn as nn




#######################################################################
batches = 500
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate = 0.001
n_epochs= 10
input_size = 35
hidden_size = 2*input_size
hidden_size2 = 4*input_size
hidden_size3 = input_size+5
output_size = 4
#######################################################################





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
num = l0+l1+l2+l3
print("Number of samples: ", num)


x0, y0 = dataset0[:]
x1, y1 = dataset1[:]
x2, y2 = dataset2[:]
x3, y3 = dataset3[:]

datadef = torch.utils.data.ConcatDataset([dataset0,dataset1,dataset2,dataset3])
y_full = torch.cat([y0, y1, y2, y3], dim=0, out=None)
x_full = torch.cat([x0, x1, x2, x3], dim=0, out=None)

weight = []
for i in range (l0):
    weight.append(1/l0)
for i in range (l1):
    weight.append(1/l1)
for i in range (l2):
    weight.append(1/l2)
for i in range (l3):
    weight.append(1/l3)    
weight = np.array(weight)

samples_weight = torch.from_numpy(weight)
sampler = WeightedRandomSampler(samples_weight, len(samples_weight))


train_loader = DataLoader(dataset=datadef,
                          batch_size=batches,
                          sampler=sampler,
                          num_workers=0)


######################## 2. SET UP NN ##########################################

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size,hidden_size2, hidden_size3, output_size):
        super(NeuralNet, self).__init__()
        self.batchn = nn.BatchNorm1d(input_size)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, hidden_size3)
        self.linear4 = nn.Linear(hidden_size3, output_size)
        self.softm = nn.Softmax(dim = 1)
    
    def forward(self, x):
        out = self.batchn(x)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.relu(out)
        out = self.linear4(out)
        out = self.softm(out)
        return out
    

model = NeuralNet(input_size, hidden_size, hidden_size2, hidden_size3,output_size).to(device)    
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
        
        if (i % 300)==0:
               with torch.no_grad():
                    z=model(x_full)
                    yhat = torch.max(z.data, 1) 
                    y_testtrue = torch.max(y_full,1)
                    y_testpred = yhat.indices.numpy()
                    y_testtrue = y_testtrue.indices.numpy()
                    histodata = np.zeros((4,4))
                    bins = [0, 1, 2, 3]
                    for j in range (len(y_full)):
                        histodata[y_testtrue[j]][y_testpred[j]]+=1
                    fig, az = plt.subplots(figsize = (7,7))
                    plt.xticks([])
                    plt.bar(bins, histodata[0,:], label = "TTTT") 
                    plt.bar(bins, histodata[1,:], bottom = histodata[0,:], label = "TTLep_bb") 
                    plt.bar(bins, histodata[2,:], bottom = histodata[1,:]+histodata[0,:], label = "TTLep_cc") 
                    plt.bar(bins, histodata[3,:], bottom = histodata[1,:]+histodata[0,:]+histodata[2,:], label = "TTLep_other") 
                    plt.legend()
                    plt.xticks([])
                    lab = "epoch = "+str(epoch)+"   batch = " + str(i)
                    plt.title(lab)
                    plt.show()
                    
        
        
        
fig, ay = plt.subplots()        
ay.plot(losses)
plt.title("Losses over epoch")


############################### 4. TEST MODEL #####################################

with torch.no_grad():
    z=model(x_full)
    yhat = torch.max(z.data, 1) 
    y_testtrue = torch.max(y_full,1)
    y_testpred = yhat.indices.numpy()
    y_testtrue = y_testtrue.indices.numpy()
    histodata = np.zeros((4,4))
    histotrue = np.zeros((4,4))
    bins = [0, 1, 2, 3]
    for j in range (len(y_full)):
        histodata[y_testtrue[j]][y_testpred[j]]+=1
        histotrue[y_testtrue[j]][y_testtrue[j]]+=1
    fig, az = plt.subplots(2, 1, figsize = (7,7))
    plt.xticks([])
    plt.title("Model vs. Test")
    az[0].bar(bins, histodata[0,:], label = "TTTT") 
    az[0].bar(bins, histodata[1,:], bottom = histodata[0,:], label = "TTLep_bb") 
    az[0].bar(bins, histodata[2,:], bottom = histodata[1,:]+histodata[0,:], label = "TTLep_cc") 
    az[0].bar(bins, histodata[3,:], bottom = histodata[1,:]+histodata[0,:]+histodata[2,:], label = "TTLep_other") 
    az[0].legend()
    plt.xticks([])
    az[1].bar(bins, histotrue[0,:], label = "TTTT") 
    az[1].bar(bins, histotrue[1,:], bottom = histotrue[0,:], label = "TTLep_bb") 
    az[1].bar(bins, histotrue[2,:], bottom = histotrue[1,:]+histotrue[0,:], label = "TTLep_cc") 
    az[1].bar(bins, histotrue[3,:], bottom = histotrue[1,:]+histotrue[0,:]+histotrue[2,:], label = "TTLep_other") 
    az[1].legend()
    plt.show()

PATH = PATH = 'C:/Users/Lena/Documents/DNN/dnn.pth'
torch.save(model.state_dict(), PATH)           
        