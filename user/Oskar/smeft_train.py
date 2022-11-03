import os
import sys
sys.path.append('..')
sys.path.insert(0, '../..')

import datetime

import pickle

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import math
import numpy as np
from matplotlib import pyplot as plt

import uproot

from tqdm import trange, tqdm

# from tools import syncer 
from tools import user




# parameters for training if module is run as a script
TRAIN_FILE_RANGE = (0,20)
HIDDEN_LAYERS = (300,100,100)
N_EPOCH = 10
LEARNING_RATE = 1e-5
PRINT_EVERY = 1
SAVE_EVERY = 1
MODEL_DIRECTORY = None





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

        self.losses=[]
        self.epoch=0

    def forward(self, x):
        return self.mlp(x)



# train function
def learn_lin_weight(model, data_loader, n_epoch=10, learning_rate=1e-3, print_every=1, save_every=None, model_directory='', scheduler=None):

    def loss_fn(pred, w0, w1_0):
        return (w0 * (w1_0 - pred)**2).sum()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    if scheduler is not None:
        scheduler = StepLR(optimizer=optimizer, step_size=plot_every, gamma=0.9, verbose=False)

    #losses = []
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

        model.losses.append(epoch_loss)
        model.epoch+=1

        if print_every is not None and (epoch % print_every == 0 or epoch+1 == n_epoch):
            print(f'################ epoch: {model.epoch}, loss: {epoch_loss}')

        if save_every is not None and (epoch % save_every == 0 or epoch+1 == n_epoch):
            file_name = f'epoch_{model.epoch}.pkl'
            os.makedirs(os.path.join(user.model_directory, model_directory), exist_ok=True)
            file_path = os.path.join(user.model_directory, model_directory, file_name)
            with open(file_path, 'wb') as f:
                pickle.dump(model, f)

        if scheduler is not None:
            scheduler.step()
            scheduler.get_last_lr()

    # plot loss and accuracy over epochs
    # plt.plot(model.losses)
    # plt.title('loss')
    # # plt.savefig(os.path.join( user.plot_directory, "loss.png" ) )
    # plt.show(block=False)
    # plt.close()


if __name__ == '__main__':
    from smeft_data import *
    import datetime

    print('get train file names')
    train_file_names = [
        f'/scratch-cbe/users/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_{n_file}.root:Events'
        for n_file in range(*TRAIN_FILE_RANGE)]

    print('get branch names')
    scalar_branches, vector_branches = get_branch_names()

    print('load train files')
    train_scalar_events, train_vector_events, train_weights = load_data(file_names=train_file_names)

    print('get weights')
    train_w0 = train_weights[:,0,np.newaxis]
    train_w1_0 = train_weights[:,1,np.newaxis]/train_w0

    print('create dataset and loader')
    train_dataset = JointDataset(x=train_scalar_events, y=(train_w0,train_w1_0))
    train_data_loader = DataLoader(train_dataset, batch_size=1000)

    print('initialize model')
    model = MLP(input_nfeatures=len(scalar_branches), num_classes=1, hidden_layers=(300,100,100))
    print(model)

    if MODEL_DIRECTORY is None:
        MODEL_DIRECTORY = datetime.datetime.now().strftime('%d_%m_%Y_%H-%M-%S')

    print('start training')
    learn_lin_weight(
        model=model,
        data_loader=train_data_loader,
        n_epoch=N_EPOCH,
        learning_rate=LEARNING_RATE,
        print_every=PRINT_EVERY,
        save_every=SAVE_EVERY,
        model_directory=MODEL_DIRECTORY)

    
