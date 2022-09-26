import torch
from torch import nn
import math
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
sys.path.append('..')
from tools import syncer 
from tools import user

learning_rate = 1e-3
device        = 'cuda' if torch.cuda.is_available() else 'cpu'
n_epoch       = 200
plot_every    = 10

# Data Generation
n_data = 10000
n_features = 10
x = 10 * np.random.random_sample((n_data, n_features))+20
y = np.sum(x, axis=1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

x_train = torch.from_numpy(x).float().to(device)
y_train = torch.from_numpy(y).float().to(device)



###  set up model
######################

# subclass the torch.nn.Module class
class MyModule(nn.Module, input_nfeatures=1, output_nfeatures=1):
    # an __init__ function is needed, which has to call (with the super() function) the torch.nn.Module __init__ function
    # here we define Modules we want to use, eg. linear layers, activation functions, or even some complicated module
    def __init__(self):
        super().__init__()
        hidden1 = 50
        hidden2 = 30
        l1 = nn.Linear(n_features, hidden1)
        a1 = nn.ReLU()
        l2 = nn.Linear(hidden1, hidden2)
        a2 = nn.ReLU()
        self.module_list = nn.ModuleList([l1,a1,l2,a2])

    # then we need to implement a forward method that specifies the network structure
    def forward(self, x):
        # we just connect the layers and activation functions sequentially
        for f in self.module_list:
            x=f(x)
        return x




# hidden  = 50
# hidden2 = 30

# model = torch.nn.Sequential(
#   torch.nn.Linear(n_features, hidden),
#   torch.nn.ReLU(),
#   torch.nn.Linear(hidden, hidden2),
#   torch.nn.ReLU(),
#   torch.nn.Linear(hidden2,1),
  
# ).to(device)

loss_fn = torch.nn.MSELoss(reduction='sum')

#optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

losses = []

# variables for ploting results
model.train()
for epoch in range(n_epoch):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x_train)

    # Compute and print loss.
    loss = loss_fn(y_pred, y_train)
    losses.append(loss.item())
    if epoch % plot_every == 0:
        print(epoch, loss.item())

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()

        
plt.plot(losses)
plt.show(block=False)
plt.savefig(os.path.join( user.plot_directory, "loss.png" ) )
plt.pause(2)
#plt.close()

with torch.no_grad():
    model.eval()
    y_train_pred = model(x_train)
