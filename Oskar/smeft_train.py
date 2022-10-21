import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset


import math
import numpy as np
from matplotlib import pyplot as plt

import uproot

from tqdm import trange, tqdm

import os
import sys

# from tools import syncer 
# from tools import user

sys.path.append('..')
device = 'cuda' if torch.cuda.is_available() else 'cpu'



# create a MLP class
class MLP(nn.Module):
    def __init__(self, input_nfeatures=1, num_classes=1, hidden_layers=(), **kwargs):
        super().__init__(**kwargs)
        channels = [input_nfeatures] + list(hidden_layers) + [num_classes]
        layers = []
        for c, c_next in zip(channels[:-1], channels[1:]):
            layers.append(nn.BatchNorm1d(c))
            layers.append(nn.Linear(c, c_next, bias=True))
            # layers.append(nn.BatchNorm1d(c_next))
            layers.append(nn.ReLU())
        del layers[-1:]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)




def learn_lin_weight(model, data_loader, n_epoch=10, learning_rate=1e-3, print_every=1, scheduler=None):

    def loss_fn(pred, w0, w1_0):
        return (w0 * (w1_0 - pred)**2).sum()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    if scheduler is not None:
        scheduler = StepLR(optimizer=optimizer, step_size=plot_every, gamma=0.9, verbose=False)

    losses = []
    # accuracies = []
    batch_size = data_loader.batch_size
    n_samples = len(data_loader.dataset)
    # gradients = []
    # learning_rates = []

    model.train()

    for epoch in range(n_epoch):
        epoch_loss = 0
        for x_batch, w0_batch, w1_0_batch in tqdm(data_loader, desc=f'epoch: {epoch}', ncols=100, leave=False):
            # forward pass
            pred = model(x_batch)

            # Compute loss
            loss = loss_fn(pred, w0_batch, w1_0_batch)

            #reset the gradients to zero
            optimizer.zero_grad()

            # Backward pass
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its parameters
            optimizer.step()

            # add batch loss to current epoch loss
            epoch_loss += loss.item()

        losses.append(epoch_loss)

        if epoch % print_every == 0:
            print(f'################ epoch: {epoch}, loss: {epoch_loss}')

        if scheduler is not None:
            scheduler.step()
            scheduler.get_last_lr()

    # plot loss and accuracy over epochs
    plt.plot(losses)
    plt.title('loss')
    # plt.savefig(os.path.join( user.plot_directory, "loss.png" ) )
    plt.show(block=False)
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