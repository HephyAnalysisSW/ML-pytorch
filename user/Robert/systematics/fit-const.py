import torch
import math
import numpy as np
import ROOT
ROOT.TH1.SetDefaultSumw2()
from   matplotlib import pyplot as plt
import os
from math import sqrt, log
from scipy.stats import ncx2
import sys
sys.path.append('..')
from Tools import tdrstyle
#from Tools import syncer
from Tools import user
from Tools import helpers

import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--name',               action='store', type=str,   default='default', help="Name of the training")
#argParser.add_argument('--output_directory',   action='store', type=str,   default='/mnt/hephy/cms/robert.schoefbeck/TMB/models/')
argParser.add_argument('--plot_directory',    action='store', type=str,   default=os.path.expandvars("v2") )
#argParser.add_argument('--small',              action='store_true', help="small?")
args = argParser.parse_args()

try:
    import uproot
except:
    import uproot3 as uproot

import awkward
import numpy as np
import pandas as pd


losses = []

tex = ROOT.TLatex()
tex.SetNDC()
tex.SetTextSize(0.04)

## variables for ploting results
#for network in networks.values():
#    network.train()

n_epoch       = 5000
c = torch.tensor([.3],requires_grad=True) 
#optimizer = torch.optim.SGD([c],  lr = 1e-2)
optimizer = torch.optim.Adam([c], lr = 1e-2)

#const_A, const_B = 4., 5.
const_A, const_B = 6759, 6898 
truth_r = const_A/float(const_B)
truth_c = 1./(1+truth_r) 

for epoch in range(n_epoch):

    loss = (const_A*c**2 + const_B*(1-c)**2)/(const_A+const_B)
    #loss = 6759.*c**2 + 6898.*(1-c)**2 

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    with torch.no_grad():
        print ("loss",loss.item(),"c")
        print ("r(truth)=%5.4f c(truth)=%5.4f epoch=%i: r=%5.4f c=%5.4f (grad-c: %5.4f)" % (truth_r, truth_c, epoch, c, 1/c-1, c.grad) ) 

        print()

