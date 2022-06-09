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
from Tools import syncer 
from Tools import user
from Tools import helpers

delta_alpha= 1 # for differentiation
N_events   = 1000
N_toys     = 10000

class Exp1D:
    def __init__(self, alpha_0):
        self.alpha_0 = alpha_0

    def getEvents( self, N_events=N_events ):
        return np.random.exponential(scale=self.alpha_0, size=N_events)

    @classmethod
    def estimate( cls, x):
        return np.mean(x)

    def t(self, alpha, x ):
        #print ("alpha",alpha, np.sum(-np.log(alpha)-x/alpha), self.estimate(x), np.sum(np.log(self.estimate(x)) + x/self.estimate(x)), -2*np.sum( -np.log(alpha)-x/alpha + np.log(self.estimate(x)) + x/self.estimate(x) ) )
        return -2*np.sum( -np.log(alpha)-x/alpha + log(self.estimate(x)) + x/self.estimate(x) )

    def sigma( self, toys = None):
        #return 4.742446342616055
        if toys is None:
            toys = np.array( [self.getEvents(N_events) for i_toy in range(N_toys) ] )
        logL_exp    = np.mean([ np.sum( -np.log(self.alpha_0) - toy/self.alpha_0 ) for toy in toys ])
        alpha_p     = self.alpha_0 + delta_alpha
        logL_exp_p  = np.mean([ np.sum( -np.log(alpha_p) - toy/alpha_p ) for toy in toys ]) 
        alpha_m     = self.alpha_0 - delta_alpha
        logL_exp_m  = np.mean([ np.sum( -np.log(alpha_m) - toy/alpha_m ) for toy in toys ])

        #print (logL_exp_m, logL_exp, logL_exp_p)

        fisher_I = -(logL_exp_p+logL_exp_m-2*logL_exp)/delta_alpha**2 

        return sqrt(1./fisher_I)
        #print (logL_exp, (logL_exp_p-logL_exp)/delta_alpha, (logL_exp_p+logL_exp_m-2*logL_exp)/delta_alpha**2 )

# Truth 
alpha_0    = 100.
model      = Exp1D(alpha_0)

# Simulation
toys  = np.array( [model.getEvents(N_events) for i_toy in range(N_toys)] )
#sigma = model.sigma(toys)
sigma = model.sigma(toys)

# Test
alpha_test = 100
model2      = Exp1D(alpha_test)
toys2  = np.array( [model2.getEvents(N_events) for i_toy in range(N_toys)] )
t_toys     = list(map( lambda toy: model.t(alpha_test,toy), toys ))

# Validation
Gaussian_t_fluctuations = ((alpha_test - np.array(list(map(model.estimate, toys))))/sigma)**2
Lambda_nonCentrality    = ((alpha_0-alpha_test)/sigma)**2

np_binning = [0,max([np.quantile(Gaussian_t_fluctuations,0.995), np.quantile(t_toys,0.995)]),50]

#f_chi2_nonCentral = ROOT.TF1("dist_t", "{norm}*0.5/sqrt(x*2*pi)*( exp(-0.5*(sqrt(x)+sqrt({Lambda}))**2) + exp(-0.5*(sqrt(x)-sqrt({Lambda})**2)))".format(norm = len(toys), Lambda=Lambda_nonCentrality),np_binning[0],np_binning[1])

h_chi2_nonCentral = ROOT.TH1D("chi2", "chi2", np_binning[2],np_binning[0],np_binning[1])
for i_bin in range(1,h_chi2_nonCentral.GetNbinsX()+1):
    h_chi2_nonCentral.SetBinContent(i_bin, 
        ncx2.cdf(h_chi2_nonCentral.GetBinLowEdge(i_bin)+h_chi2_nonCentral.GetBinWidth(i_bin),df=1,nc=Lambda_nonCentrality) - ncx2.cdf(h_chi2_nonCentral.GetBinLowEdge(i_bin),df=1,nc=Lambda_nonCentrality))
    h_chi2_nonCentral.SetBinError(i_bin, 0.)
h_chi2_nonCentral.SetLineColor(ROOT.kBlack)
h_chi2_nonCentral.SetMarkerColor(ROOT.kBlack)
h_chi2_nonCentral.SetMarkerStyle(0)
h_chi2_nonCentral.Scale(len(toys))

h_t_toys = helpers.make_TH1F(np.histogram(t_toys,np.linspace(*np_binning)))
h_t_toys.SetLineColor(ROOT.kBlue)
h_t_toys.SetMarkerColor(ROOT.kBlack)
h_t_toys.SetMarkerStyle(0)
h_Gaussian_t_fluctuations = helpers.make_TH1F(np.histogram(Gaussian_t_fluctuations,np.linspace(*np_binning)))
h_Gaussian_t_fluctuations.SetLineColor(ROOT.kRed)
h_Gaussian_t_fluctuations.SetMarkerColor(ROOT.kRed)
h_Gaussian_t_fluctuations.SetMarkerStyle(0)

max_ = max( [h_Gaussian_t_fluctuations.GetMaximum(), h_t_toys.GetMaximum(), h_chi2_nonCentral.GetMaximum()] )

l = ROOT.TLegend(0.15,0.8,0.9,0.95)
l.SetNColumns(2)
l.SetFillStyle(0)
l.SetShadowColor(ROOT.kWhite)
l.SetBorderSize(0)
l.AddEntry( h_Gaussian_t_fluctuations, "((#alpha-#hat{#alpha})/#sigma)^2" )
l.AddEntry( h_chi2_nonCentral, "#chi^{2}_{1}(#sqrt{#Lambda}=|#alpha-#alpha'|/#sigma = %3.2f)"%sqrt(Lambda_nonCentrality) )
l.AddEntry( h_t_toys,          "-2Log#lambda(#alpha=%3i) for #alpha'=%3i"%(alpha_test, alpha_0) )

c1 = ROOT.TCanvas()
h_t_toys.Draw("h")
h_t_toys.GetYaxis().SetRangeUser(0.1,10*max_)
c1.SetLogy(1)
h_Gaussian_t_fluctuations.Draw("hsame")
h_chi2_nonCentral.Draw("same")
l.Draw()
c1.Print(os.path.join(user.plot_directory, 'sys_plots', 'chi2.png'))
