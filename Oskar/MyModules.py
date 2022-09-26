import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
import math
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
sys.path.append('..')
from tools import syncer 
from tools import user

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data Generation
def data_generation(n_data=1000, input_nfeatures = 10, data_range=None, target_function=lambda x : x):

    lower, upper = 0,1
    if data_range: lower, upper = data_range[0], data_range[1]

    x = (upper - lower) * np.random.random_sample((n_data, input_nfeatures)) + lower
    y = target_function(x)
    target_nfeatures = y[0].size

    x_train = torch.from_numpy(x).float().to(device)
    y_train = torch.from_numpy(y).float().to(device)

    return x_train, y_train, input_nfeatures, target_nfeatures



###  set up model
######################

# subclass the torch.nn.Module class
class MyModule(nn.Module):
    # an __init__ function is needed, which has to call (with the super() function) the torch.nn.Module __init__ function
    # input parameters belong in the __init__ method
    # here we define Modules we want to use, eg. linear layers, activation functions, or even some complicated module
    def __init__(self, input_nfeatures=1, target_nfeatures=1, hidden1=1, hidden2=1):
        super().__init__()
        l1 = nn.Linear(input_nfeatures, hidden1, bias=True)
        a1 = nn.ReLU()
        l2 = nn.Linear(hidden1, hidden2, bias=True)
        a2 = nn.ReLU()
        l3 = nn.Linear(hidden2, target_nfeatures, bias=True)
        self.module_list = nn.ModuleList([l1,a1,l2,a2,l3])

    # then we need to implement a forward method that specifies the network structure
    def forward(self, x):
        # we just connect the layers and activation functions sequentially
        for f in self.module_list:
            x=f(x)
        return x



def learn(model, x_train, y_train, n_epoch=100, learning_rate=1e-3, plot_every=10, scheduler=False):
    loss_fn = torch.nn.MSELoss(reduction='sum')

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if scheduler:
        scheduler = StepLR(optimizer=optimizer, step_size=plot_every, gamma=0.9, verbose=False)

    losses = []

    model.train()

    for epoch in range(n_epoch):
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(x_train)

        # Compute and print loss.
        loss = loss_fn(y_pred, y_train)
        losses.append(loss.item())
        if epoch % plot_every == 0:
            print(epoch, loss.item(),scheduler.get_last_lr())

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

        if scheduler:
            scheduler.step()
        #scheduler.get_last_lr()

            
    plt.plot(losses)
    plt.savefig(os.path.join( user.plot_directory, "loss.png" ) )
    plt.show(block=False)
    #plt.pause(2)
    plt.close()

    with torch.no_grad():
        model.eval()
        y_train_pred = model(x_train)
