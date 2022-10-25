import torch
import uproot
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn as nn
import os


######################### import training data ########################
my_file = open("Event_properties.txt", "r")
content = my_file.read()
content_list = content.split('\n')
my_file.close()

print("I. content list read")

array1 = uproot.concatenate( ['/eos/vbc/group/cms/robert.schoefbeck/TMB/training-ntuples-tttt-v2/MVA-training/tttt_2l/TTTT/TTTT.root'], content_list )
array2 = uproot.concatenate( ['/eos/vbc/group/cms/robert.schoefbeck/TMB/training-ntuples-tttt-v2/MVA-training/tttt_2l/TTLep_bb/TTLep_bb.root'], content_list )
array3 = uproot.concatenate( ['/eos/vbc/group/cms/robert.schoefbeck/TMB/training-ntuples-tttt-v2/MVA-training/tttt_2l/TTLep_cc/TTLep_cc.root'], content_list )
array4 = uproot.concatenate( ['/eos/vbc/group/cms/robert.schoefbeck/TMB/training-ntuples-tttt-v2/MVA-training/tttt_2l/TTLep_other/TTLep_other.root'], content_list )

print("II. data read")

data1 = array1.to_numpy()
data2 = array2.to_numpy()
data3 = array3.to_numpy()
data4 = array4.to_numpy()


np.savetxt("localdata1.txt", data1)
np.savetxt("localdata2.txt", data2)
np.savetxt("localdata3.txt", data3)
np.savetxt("localdata4.txt", data4)

print("III. data converted")
#######################################################################





###################### general parameters #############################
batches = 10000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate = 0.001
n_epochs= 10
input_size = 35
hidden_size = 2*input_size
#hidden_size2 = 4*input_size
#hidden_size3 = input_size+5
output_size = 4
#######################################################################





class NewDataset(Dataset):

    def __init__(self, number):
        path = './localdata'+str(number)+'.txt' 
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

print("IV. number of training events: ", num)

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
    weight.append(2.5/l2)
for i in range (l3):
    weight.append(4/l3)    
weight = np.array(weight)

samples_weight = torch.from_numpy(weight)
sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

train_loader = DataLoader(dataset=datadef,
                          batch_size=batches,
                          sampler=sampler,
                          num_workers=0)
print("V. sampled shuffled train loader ready")


######################## 2. SET UP NN ##########################################
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.batchn = nn.BatchNorm1d(input_size)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
        #self.linear3 = nn.Linear(hidden_size2, hidden_size3)
        #self.linear4 = nn.Linear(hidden_size3, output_size)
        self.softm = nn.Softmax(dim = 1)
    
    def forward(self, x):
        out = self.batchn(x)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        #out = self.relu(out)
        #out = self.linear3(out)
        #out = self.relu(out)
        #out = self.linear4(out)
        out = self.softm(out)
        return out
    
model = NeuralNet(input_size, hidden_size, output_size).to(device)    
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
losses = []
print('VI. model ready, training set has {} instances'.format(len(train_loader)))

script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'Results/')
print('VII. plot directory ready, starting training')


############################## 3. TRAINING THE MODEL #############################
for epoch in range(n_epochs):
    print("		epoch: ", epoch+1 , " of ", n_epochs)
    for i, data in enumerate(train_loader):
        inputs, labels = data
        z=model(inputs)
        loss=criterion(z,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.data)
        
        if (i % (len(train_loader)-1))==0:
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
                    sample_file_name = "epoch="+str(epoch+1)+".png"
                    plt.xticks([])
                    plt.bar(bins, histodata[0,:], label = "TTTT") 
                    plt.bar(bins, histodata[1,:], bottom = histodata[0,:], label = "TTLep_bb") 
                    plt.bar(bins, histodata[2,:], bottom = histodata[1,:]+histodata[0,:], label = "TTLep_cc") 
                    plt.bar(bins, histodata[3,:], bottom = histodata[1,:]+histodata[0,:]+histodata[2,:], label = "TTLep_other") 
                    plt.legend()
                    plt.xticks([])
                    lab = "epoch = "+str(epoch+1)
                    plt.title(lab)
		    #plt.show(block=False)
                    plt.savefig(results_dir + sample_file_name)
                    #plt.pause(.1)
        
        
        
fig, ay = plt.subplots()        
plt.plot(losses)
plt.title("Losses over epoch")
sample_file_name = "losses.png"
plt.savefig(results_dir + sample_file_name)
print("VIII. training done")

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
    sample_file_name = "final_dnn_output.png"
    plt.savefig(results_dir + sample_file_name)
    plt.show()

print("IX. evaluation done")
#PATH = PATH = 'C:/Users/Lena/Documents/DNN/Funktionierendes_Model/model.pth'
#torch.save(model, PATH)                  
#loaded_model = torch.load(PATH)
#loaded_model.eval()
#for param in loaded_model.parameters():
#    print(param)

torch.onnx.export(model, x_full, "./model.onnx")   
    
print("X. model saved")    
