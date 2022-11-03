import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

import os

from tools import user

import pickle

import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib import colors

from smeft_train import MLP
from smeft_data import *


MODEL_PATH = 'model_24_10_22'
MODEL_NAME = 'epoch_1.pkl'

OUT_PATH = 'test'

TEST_FILE_RANGE = (50,51)

HIST_BINS = 25
HIST_THETA = (-0.5, 0.5)

TR_V_PR_BINS = 100

ESTIMATOR_HIST_BINS = 20
ESTIMATOR_HIST_THETA = 0.1

LLR_PLOT_THETA_RANGE = (-0.5,0.5)
LLR_PLOT_BINS = 20
N_THETA_PLOT = 1000



# loss over epochs
def plot_losses(model, out_path=None, out_file='loss.png'):
    '''
    :model: instance of MLP class or path to pickled model
    :out_path: subdirectory to create in user.plot_directoy
    :out_file: output file name
    '''
    if not isinstance(model, MLP):
        with open(model, 'rb') as m:
            model=pickle.load(m)

    plt.plot(model.losses)
    plt.title('loss')
    if out_path is not None:
        os.makedirs(os.path.join(user.plot_directory, out_path), exist_ok=True)
        plt.savefig(os.path.join(user.plot_directory, out_path, out_file))
        plt.close()



# histograms for branches
def plot_eft_hists(data, branch_list, weights, bins=50, theta=(-1.0,1.0), out_path=None, out_file='eft_hists.png'):
    n_plots = int(np.ceil(np.sqrt(len(branch_list))))
    plt.subplots(figsize=[10*n_plots,10*n_plots])
    for n, branch in enumerate(branch_list):
        plt.subplot(n_plots,n_plots,n+1)
        plt.hist(data[:,n], bins=bins, weights=weights[:,0], histtype='step', color='black', label='ctW=0', density=True)
        for th in theta:
            plt.hist(data[:,n], bins=bins, weights=weight_theta(weights, th), histtype='step', label=f'ctW={th}', density=True)
        plt.title(branch)
        plt.legend()
    if out_path is not None:
        os.makedirs(os.path.join(user.plot_directory, out_path), exist_ok=True)
        plt.savefig(os.path.join(user.plot_directory, out_path, out_file))
        plt.close()

# hist2d truth vs predictionf
def plot_truth_vs_prediction(model_path, data, w1_0, bins=100, out_path=None, out_file='truth_vs_pred.png'):

    model_files = [file for file in os.listdir(os.path.join(user.model_directory, model_path))]

    n_plots = int(np.ceil(np.sqrt(len(model_files))))
    plt.subplots(figsize=[12*n_plots,10*n_plots])
    for n, m_f in enumerate(model_files):
        with open(os.path.join(user.model_directory, model_path, m_f), 'rb') as f:
            model = pickle.load(f)
        pred=model(torch.tensor(data)).detach().numpy()
    
        plt.subplot(n_plots, n_plots, n+1)
        plt.hist2d(w1_0.flatten(), pred.flatten(), range=[[-1.,1.],[-1.,1.]], bins=bins, norm=colors.SymLogNorm(1))
        plt.plot([-1,1],[-1,1])
        plt.colorbar()
        plt.title(m_f)

    if out_path is not None:
        os.makedirs(os.path.join(user.plot_directory, out_path), exist_ok=True)
        plt.savefig(os.path.join(user.plot_directory, out_path, out_file))
        plt.close()



# nn weight estimator histogram
def plot_estimator_hist(model, data, weights, theta, bins=20, out_path=None, out_file='estimator_hist.png'):
    if not isinstance(model, MLP):
        with open(model, 'rb') as m:
            model=pickle.load(m)
    pred = model(torch.tensor(data)).detach().numpy()
    quantiles = weighted_quantile(values=pred.flatten(), quantiles=np.linspace(0,1,bins+1), sample_weight=weight_theta(weights,theta=0))
    plt.figure(figsize=(10,10))
    bin_counts_sm, _, _ = plt.hist(pred.flatten(), bins=quantiles, weights=weight_theta(weights,theta=0), histtype='step')
    bin_counts_eft, _, _ = plt.hist(pred.flatten(), bins=quantiles, weights=weight_theta(weights,theta=theta), histtype='step')
    plt.xlim(quantiles[1]-0.1, quantiles[-2]+0.1)
    plt.ylim(bottom=min(min(bin_counts_sm), min(bin_counts_eft))*0.95,top=max(max(bin_counts_sm), max(bin_counts_eft))*1.05)

    if out_path is not None:
        os.makedirs(os.path.join(user.plot_directory, out_path), exist_ok=True)
        plt.savefig(os.path.join(user.plot_directory, out_path, out_file))
        plt.close()

# LLR plot
def LLR(pred, weights, theta, bins=20):
    quantiles = weighted_quantile(values=pred.flatten(), quantiles=np.linspace(0,1,bins+1), sample_weight=weight_theta(weights,theta=0))
    n_hat_0 = np.histogram(pred.flatten(), bins=quantiles, weights=weight_theta(weights, theta=0))[0]

    if isinstance(theta, float):
        n_hat_theta = np.histogram(pred.flatten(), bins=quantiles, weights=weight_theta(weights, theta=theta))[0]
        return -(n_hat_0 * np.log(n_hat_theta) - n_hat_theta).sum()
    elif isinstance(theta, np.ndarray):
        LLR=[]
        for th in theta:
            n_hat_theta = np.histogram(pred.flatten(), bins=quantiles, weights=weight_theta(weights, theta=th))[0]
            LLR.append(-(n_hat_0 * np.log(n_hat_theta) - n_hat_theta).sum())
        return np.array(LLR)
    else:
        print('theta must be float or 1d array')


def plot_LLR(model, data, weights, theta_range=(-1.,1.), bins=20, n_theta_plot=100, out_path=None, out_file='LLR.png'):
    if not isinstance(model, MLP):
        with open(model, 'rb') as m:
            model=pickle.load(m)
    pred = model(torch.tensor(data)).detach().numpy()
    w0 = weights[:,0,np.newaxis]
    w1_0 = weights[:,1,np.newaxis]/w0

    theta_plot = np.linspace(*theta_range, n_theta_plot)
    print('compute LLRS for estimator')
    LLR_plot = LLR(pred=pred, weights=weights, theta=theta_plot, bins=bins)
    print('compute LLRS for true w1_0')
    LLR_true_plot = LLR(pred=w1_0, weights=weights, theta=theta_plot, bins=bins)

    plt.figure(figsize=(10,10))
    plt.plot(theta_plot, LLR_plot, label='nn')
    plt.plot(theta_plot, LLR_true_plot, label='truth')
    plt.legend()

    if out_path is not None:
        os.makedirs(os.path.join(user.plot_directory, out_path), exist_ok=True)
        plt.savefig(os.path.join(user.plot_directory, out_path, out_file))
        plt.close()






if __name__ == '__main__':

    model_file_path=os.path.join(user.model_directory, MODEL_PATH, MODEL_NAME)

    print('plot model loss history')
    plot_losses(model=model_file_path, out_path=OUT_PATH)

    test_file_names = [
        f'/scratch-cbe/users/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_{n_file}.root:Events'
        for n_file in range(*TEST_FILE_RANGE)]


    print('load hist files, branch names, weights')
    scalar_branches, vector_branches = get_branch_names()
    test_scalar_events, test_vector_events, test_weights = load_data(file_names=test_file_names)
    test_w0 = test_weights[:,0,np.newaxis]
    test_w1_0 = test_weights[:,1,np.newaxis]/test_w0

    print('plot_eft_hists')
    plot_eft_hists(
        data=test_scalar_events,
        branch_list=scalar_branches,
        weights=test_weights,
        bins=HIST_BINS,
        out_path=OUT_PATH)

    print('plot truth vs prediction')
    plot_truth_vs_prediction(
        model_path=MODEL_PATH,
        data=test_scalar_events,
        w1_0=test_w1_0,
        bins=TR_V_PR_BINS,
        out_path=OUT_PATH)

    print('plot estimator histogram')
    plot_estimator_hist(
        model=model_file_path,
        data=test_scalar_events,
        weights=test_weights,
        theta=ESTIMATOR_HIST_THETA,
        bins=ESTIMATOR_HIST_BINS,
        out_path=OUT_PATH)

    print('plot LLR')
    plot_LLR(
        model=model_file_path,
        data=test_scalar_events,
        weights=test_weights,
        theta_range=LLR_PLOT_THETA_RANGE,
        bins=LLR_PLOT_BINS,
        n_theta_plot=N_THETA_PLOT,
        out_path=OUT_PATH)









