import torch
import math
import numpy as np


frequency     = 1.
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_epoch = 2000

x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(frequency*x)

x_train = torch.from_numpy(x).float().to(device)
y_train = torch.from_numpy(y).float().to(device)

hidden  = 200
hidden2 = 200

model = torch.nn.Sequential(
  torch.nn.Linear(1, hidden),
  torch.nn.ReLU(),
  torch.nn.Linear(hidden, hidden2),
  torch.nn.ReLU(),
  torch.nn.Linear(hidden2,1),
  
).to(device)

loss_fn = torch.nn.MSELoss(reduction='sum')

assert False, ""
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

losses = []

for epoch in range(n_epoch):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(xx)

    # Compute and print loss.
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

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


linear_layer = model[0]
print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')
