import torch
import math
import numpy as np
from matplotlib import pyplot as plt
import syncer 
import user
import os

import ZH_Nakamura as model

n_events      = 10000

learning_rate = 5e-2
device        = 'cuda' if torch.cuda.is_available() else 'cpu'
n_epoch       = 2000
plot_every    = 100

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# training data
features   = model.getEvents(n_events)[:,0:6]
n_features = len(features[0]) 
weights    = model.getWeights(features, model.make_eft() )
WC = 'cHW'
features_train = torch.from_numpy(features).float().to(device)
w0_train       = torch.from_numpy(weights[()]).float().to(device)
wp_train       = torch.from_numpy(weights[(WC,)]).float().to(device)
wpp_train      = torch.from_numpy(weights[(WC,WC)]).float().to(device)

#features_train = torch.ones(n_events).unsqueeze(-1)
#n_features     = len(features_train[0]) 
#w0_train       = torch.ones(n_events)
#wp_train       = torch.ones(n_events)
#wpp_train      = torch.ones(n_events)

# s and t network
hidden  = 5
hidden2 = 1

model_t = torch.nn.Sequential(
  torch.nn.BatchNorm1d(n_features),
  torch.nn.Linear(n_features, hidden),
  torch.nn.ReLU(),
  torch.nn.Linear(hidden, hidden2),
  torch.nn.ReLU(),
  torch.nn.Linear(hidden2,1),
  
).to(device)

model_s = torch.nn.Sequential(
  torch.nn.BatchNorm1d(n_features),
  torch.nn.Linear(n_features, hidden),
  torch.nn.ReLU(),
  torch.nn.Linear(hidden, hidden2),
  torch.nn.ReLU(),
  torch.nn.Linear(hidden2,1),
  
).to(device)

# loss functional
def f_loss(w0_input, wp_input, wpp_input, t_output, s_output):
    base_points = [1., 2.]
    loss = -0.5*w0_input.sum()
    for theta in base_points:
        fhat  = 1./(1. + ( 1. + theta*t_output)**2 + (theta*s_output)**2 )
        #print ("fhat",  theta, fhat )
        #print ("fhat2", ((1. + wp_input*theta +.5*wpp_input*theta**2)*fhat**2).sum() )
        loss += ( w0_input*( (1. + wp_input/w0_input*theta +.5*wpp_input/w0_input*theta**2)*fhat**2 + (1-fhat)**2 ) ).sum()
    return loss

#optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(list(model_t.parameters())+list(model_s.parameters()), lr=learning_rate)

losses = []

# variables for ploting results
model_s.train()
model_t.train()
for epoch in range(n_epoch):
    # Forward pass: compute predicted y by passing x to the model.
    pred_t = model_t(features_train).squeeze()
    pred_s = model_s(features_train).squeeze()

    # Compute and print loss.
    loss = f_loss(w0_train, wp_train ,wpp_train, pred_t, pred_s)
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

    #print ("t", pred_t.mean().item(), pred_s.mean().item() )

    #print ("model_t", list(model_t.parameters()))
    #print ("model_s", list(model_s.parameters()))

#    print ()
    if (epoch % plot_every)==0:
        with torch.no_grad():
            print (loss.item())
#            pred_t = model_t(features_train).squeeze().cpu().detach().numpy()
#            pred_s = model_s(features_train).squeeze().cpu().detach().numpy()
#            truth = ( np.sin(frequency*x_axis_torch.cpu())) .detach().numpy()
#            plt.clf()
#            plt.plot(pred)
#            plt.plot(truth)
#            plt.show()
#            plt.savefig(os.path.join( user.plot_directory, "plt_epoch_%i.png"%epoch ) )
        
#plt.clf()
#plt.plot(losses)
#plt.show()
#plt.savefig(os.path.join( user.plot_directory, "loss.png" ) )
#
#with torch.no_grad():
#    model.eval()
#    y_train_pred = model(x_train)
