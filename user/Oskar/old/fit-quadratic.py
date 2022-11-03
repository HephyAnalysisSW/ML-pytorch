import torch
import math
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
sys.path.append('..')
#from tools import syncer 
from tools import user

frequency     = 1.
learning_rate = 1e-3
device        = 'cuda' if torch.cuda.is_available() else 'cpu'
n_epoch       = 8000
plot_every    = 100

# Data Generation
data_range = 15
x = data_range*(np.random.rand(data_range*100, 1))-data_range/2
y = x**2 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

x_train = torch.from_numpy(x).float().to(device)
y_train = torch.from_numpy(y).float().to(device)

hidden  = 3
hidden2 = 3

model = torch.nn.Sequential(
  torch.nn.Linear(1, hidden),
  torch.nn.ReLU(),
  torch.nn.Linear(hidden, hidden2),
  torch.nn.ReLU(),
  torch.nn.Linear(hidden2,1),
  
).to(device)

loss_fn = torch.nn.MSELoss(reduction='sum')

#optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

losses = []

# variables for ploting results
res = 10
x_axis = (np.arange(data_range*res)-data_range/2*res).reshape(data_range*res,1)/res
x_axis_torch = torch.from_numpy(x_axis).float().to(device)
model.train()
for epoch in range(n_epoch):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x_train)

    # Compute and print loss.
    loss = loss_fn(y_pred, y_train)
    losses.append(loss.item())
    if epoch % 100 == 99:
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
    if (epoch % plot_every)==0:
        pred  = model(x_axis_torch).cpu().detach().numpy()
        truth = ( (x_axis_torch.cpu())**2) .detach().numpy()
        plt.clf()
        plt.plot(pred)
        plt.plot(truth)
        plt.show(block=False)
        plt.savefig(os.path.join( user.plot_directory, "plt_epoch_%i.png"%epoch ) )
        plt.pause(.1)
        #plt.close()

        
plt.plot(losses)
plt.show(block=False)
plt.savefig(os.path.join( user.plot_directory, "loss.png" ) )
plt.pause(2)
#plt.close()

with torch.no_grad():
    model.eval()
    y_train_pred = model(x_train)
