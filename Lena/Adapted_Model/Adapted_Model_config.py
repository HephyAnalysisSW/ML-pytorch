import torch
import uproot
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn as nn
import os

###################### general parameters #############################
import tttt as config
content_list = config.content_list
samples = config.samples
directory = config.directory
batches = 10000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate = 0.001
n_epochs= 1
input_size = len(content_list) 
hidden_size = 2*input_size
output_size = len(samples)



######################### import training data ########################
# load events in dictionary of np.array (the 2nd line reduces memory consumption ... not dramatically though) 
x = { sample: uproot.concatenate( os.path.join( directory, "{sample}/{sample}.root".format(sample=sample))) for sample in samples }
x = { sample: np.array( [ getattr( array, branch ) for branch in content_list ] ).transpose() for sample, array in x.items() }

# weight wrt to largest sample
n_max_events= max(map( len, x.values() ))
w = {sample:n_max_events/len(x[sample]) for sample in samples}

y = {sample:i_sample*np.ones(len(x[sample])) for i_sample, sample in enumerate(samples)}
# Note to myself: y... "TTTT":0,0,0..., "TTbb":1,1,1... 

# make weights. This wastes some memory, but OK.
samples_weight = np.concatenate([ [w[sample]]*len(x[sample]) for sample in samples]) 
sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

#It is time to concatenate
x = torch.Tensor(np.concatenate( [x[sample] for sample in samples] ))
y = torch.Tensor(np.concatenate( [y[sample] for sample in samples] ))
train_loader = DataLoader(np.hstack((x,y.reshape(-1,1))), # we make the label the last column
                          batch_size=batches,
                          sampler=sampler,
                          num_workers=0)
# Note to myself: Data in tensor (n_samples x (n_features + label = 36))


######################## set up nn ##########################################
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.batchn = nn.BatchNorm1d(input_size)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.softm = nn.Softmax(dim = 1)
    
    def forward(self, x):
        out = self.batchn(x)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.softm(out)
        return out
    
model = NeuralNet(input_size, hidden_size, output_size).to(device)    
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
losses = []

results_dir = './Results/'
if not os.path.exists( results_dir ): 
    os.makedirs( results_dir )


############################## training the model #############################
for epoch in range(n_epochs):
    print("		epoch: ", epoch+1 , " of ", n_epochs)
    for i, data in enumerate(train_loader):
        inputs = data[:,:-1] # N-1 columns
        labels = data[:,-1].int()  # last column
        training_labels = torch.zeros( (len(inputs), len(samples)))
        for i_sample in range(len(samples)):
            training_labels[:,i_sample][labels==i_sample]=1
        z=model(inputs)
        loss=criterion(z,training_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.data)
        
        if (i % (len(train_loader)))==0:
               with torch.no_grad():
                    z = model(x)
                    y_testpred = torch.max(z.data, 1).indices.numpy() 
                    y_testtrue = y.int()
                    hist1 = []; hist2 = []; hist3 = []; hist4 = []
                    bins = [0, 1, 2, 3, 4]
                    for j in range (len(y)):
                        if (y_testtrue[j] == 0): hist1.append(y_testpred[j])
                        if (y_testtrue[j] == 1): hist2.append(y_testpred[j])
                        if (y_testtrue[j] == 2): hist3.append(y_testpred[j])
                        if (y_testtrue[j] == 3): hist4.append(y_testpred[j])
                    fig, az = plt.subplots(figsize = (7,7))
                    plt.xticks([])
                    plt.hist([hist1, hist2, hist3, hist4], bins, stacked = True,label = ["TTTT", "TTLep_bb","TTLep_cc","TTLep_other"]) 
                    plt.legend()
                    lab = "epoch = "+str(epoch)+"   batch = " + str(i)
                    plt.title(lab)
                    sample_file_name = "epoch="+str(epoch+1)+".png"
                    plt.savefig(results_dir + sample_file_name)
        
        
fig, ay = plt.subplots()        
plt.plot(losses)
plt.title("Losses over epoch")
sample_file_name = "losses.png"
plt.savefig(results_dir + sample_file_name)

############################### test the model #####################################
with torch.no_grad():
    #z = model(x)
    #y_testpred = torch.max(z.data, 1).indices.numpy() 
    #y_testtrue = y.int()
    #hist1 = []; hist2 = []; hist3 = []; hist4 = []
    #bins = [0, 1, 2, 3, 4]
    #for j in range (len(y)):
     #   if (y_testtrue[j] == 0): hist1.append(y_testpred[j])
      #  if (y_testtrue[j] == 1): hist2.append(y_testpred[j])
       # if (y_testtrue[j] == 2): hist3.append(y_testpred[j])
        #if (y_testtrue[j] == 3): hist4.append(y_testpred[j])
    #fig, az = plt.subplots(2, 1, figsize = (7,7))
    #plt.xticks([])
    #plt.title("Model vs. Test")
    #az[0].hist([hist1, hist2, hist3, hist4], bins, stacked = True,label = ["TTTT", "TTLep_bb","TTLep_cc","TTLep_other"]) 
    #plt.xticks([])
    #az[1].hist(y_testtrue, bins) 
    #sample_file_name = "final_dnn_output.png"
    #plt.savefig(results_dir + sample_file_name)
    #plt.show()
    dummyx = x[0,:].reshape(1,len(content_list))
    torch.onnx.export(model, dummyx, "./model.onnx")   
