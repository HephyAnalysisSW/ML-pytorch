import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils import Dataset
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
    print(target_nfeatures)
    y = y.reshape(n_data, target_nfeatures)    #####not having the correct shape broke the models

    x_train = torch.from_numpy(x).float().to(device)
    y_train = torch.from_numpy(y).float().to(device)

    return x_train, y_train, input_nfeatures, target_nfeatures

# join the data and the labels in a jointdataset
class JointDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]




###  set up model
######################

# subclass the torch.nn.Module class
class MyModule(nn.Module):
    # an __init__ function is needed, which has to call (with the super() function) the torch.nn.Module __init__ function
    # input parameters belong in the __init__ method
    # here we define Modules we want to use, eg. linear layers, activation functions, or even some complicated module
    def __init__(self, input_nfeatures=1, target_nfeatures=1, hidden1=1, hidden2=1, activation=nn.ReLU()):
        super().__init__()
        l1 = nn.Linear(input_nfeatures, hidden1, bias=False)
        a1 = activation
        l2 = nn.Linear(hidden1, hidden2, bias=False)
        a2 = activation
        l3 = nn.Linear(hidden2, target_nfeatures, bias=False)
        self.module_list = nn.ModuleList([l1,a1,l2,a2,l3])


    # then we need to implement a forward method that specifies the network structure
    def forward(self, x):
        # we just connect the layers and activation functions sequentially
        for f in self.module_list:
            x=f(x)
        return x


    # predict method
    def predict(self, data):
        with torch.no_grad():
            self.eval()
            return self(data)



# end of class ####################

def learn(model, x_train, y_train, n_epoch=100, learning_rate=1e-3, plot_every=10, scheduler=None):

    def loss_fn(input, target):
        return ((input-target)**2).mean()

    # loss_fn_2 = torch.nn.MSELoss(reduction='mean')

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


    if scheduler is not None:
        scheduler = StepLR(optimizer=optimizer, step_size=plot_every, gamma=0.9, verbose=False)

    losses = []
    losses_2 = []
    # gradients = []
    # learning_rates = []

    model.train()

    for epoch in range(n_epoch):
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(x_train)

        # Compute loss.
        loss = loss_fn(y_train, y_pred)
        # loss_2 = loss_fn_2(y_train, y_pred)
        losses.append(loss.item())
        # losses_2.append(loss_2.item())

        loss_norm = loss_fn(y_train,0)*len(y_train)
        

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model). This is because by default, gradients are
        # accumulated in buffers( i.e, not overwritten) whenever .backward()
        # is called. Checkout docs of torch.autograd.backward for more details.
        # optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()

        with torch.no_grad():
            for param in model.parameters():
                param -= param.grad * learning_rate #/ torch.linalg.norm(param.grad,ord=float('inf'))

            if epoch % plot_every == 0:
                # for name, param in model.named_parameters():
                    # if param.requires_grad==True:
                print(f'epoch: {epoch}, Loss: {loss.item()}, lr: {learning_rate}')
                        # print(f'{name}: {param.data}')
                        # print(f'grad: {param.grad}')

            for param in model.parameters():
                param.grad.zero_()

        # Calling the step function on an Optimizer makes an update to its parameters
        # optimizer.step()

        if scheduler is not None:
            scheduler.step()
            scheduler.get_last_lr()

    # for l in losses:
        # print(l)

    rel_loss_decrease = np.array(losses[1:])/np.array(losses[:-1])
    print(rel_loss_decrease.shape)

    fig, axs = plt.subplots(1,2)
    axs[0].plot(losses)
    axs[0].set_title('loss')
    axs[0].set_yscale('log')
    axs[1].plot(rel_loss_decrease)
    axs[1].set_yscale('log')
    axs[1].set_title('relative decrease of loss')

    # plt.plot(losses)
    # plt.savefig(os.path.join( user.plot_directory, "loss.png" ) )
    plt.show(block=False)
    #plt.pause(2)
    plt.close()


def plot_gradients(model, x_train, target_function):

    def loss_fn(input, target):
        return ((input-target)**2).mean()

    gradients = []
    x_plot = np.linspace(-1,3,20) # a grid for plotting

    for x in x_plot:
        for param in model.parameters():
            param.data.fill_(x)

        y_pred = model(x_train)
        loss = loss_fn(target_function(x_train), y_pred)
        loss.backward()

        for param in model.parameters():
            if param.requires_grad==True:
                gradients.append(param.grad.item())
                print(f'x: {x}, loss: {loss}, grad: {param.grad}')
                param.grad.zero_()

    print(np.array(gradients).flatten())

    plt.plot(x_plot, np.array(gradients).flatten())
    plt.show()



def test_kwargs(**kwargs):
    name = kwargs['name']
    number = kwargs['number']

    for k,v in kwargs.items():
        print(f'{k} = {v}')