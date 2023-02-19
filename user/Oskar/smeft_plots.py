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

# ignore some matplotlib warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


MODEL_PATH = 'test_no_top_mass'
MODEL_NAME = 'epoch_200.pkl'

EPOCH_LIST = [1, 11, 31, 71, 200]

OUT_PATH = 'test_no_top_mass'

TEST_FILE_RANGE = (80,100)

HIST_BINS = 25
HIST_THETA = (-0.5, 0.5)

TR_V_PR_BINS = 100

ESTIMATOR_HIST_BINS = 20
ESTIMATOR_HIST_THETA = 1.

LLR_PLOT_THETA_RANGE = (-0.5, 0.5)
LLR_PLOT_BINS = 20
N_THETA_PLOT = 1000
NORM_TO_NEVENTS = 10000

VAR_NAME = 'gen_theta'
REWEIGHT_VAR = 'gen_theta'

PREP_DATA = True



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


def normalize_hist(hist: np.ndarray, norm_to_nevents: int) -> None:
    '''
    normalize a histogram to a number of events
    '''
    norm = float(hist.sum())
    hist *= norm_to_nevents/norm
    return None


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


# histogram comparison of a variable with eft estimated weights vs true weights
def plot_var_hist_pred_v_truth(var_name, var_data, truth_weights, pred_weights, theta, bins=20, eq_width=True, norm_to_nevents=1000, out_path=None, out_file='_eft_hist_pred_v_truth'):
    out_file = var_name + out_file
    
    if eq_width is not True:
        bins = weighted_quantile(values=var_data, quantiles=np.linspace(0,1,bins+1), sample_weight=weight_theta(truth_weights,theta=0))
        out_file += '_weighted_quantiles'

    hist_truth, bin_edges_truth = np.histogram(var_data, bins=bins, weights=weight_theta(truth_weights, theta))
    hist_pred, bin_edges_pred = np.histogram(var_data, bins=bins, weights=weight_theta(pred_weights, theta))
    hist_sm, bin_edges_sm = np.histogram(var_data, bins=bins, weights=weight_theta(truth_weights, 0))
    # normalize hists
    norm_truth, norm_pred, norm_sm = hist_truth.sum(), hist_pred.sum(), hist_sm.sum()
    hist_truth *= norm_to_nevents/norm_truth
    hist_pred *= norm_to_nevents/norm_pred
    hist_sm *= norm_to_nevents/norm_sm

    plt.figure(figsize=(10,10))
    plt.stairs(hist_truth, bin_edges_truth, label='eft_hist_true_weights')
    plt.stairs(hist_pred, bin_edges_pred, label='eft_hist_pred_weights')
    plt.stairs(hist_sm, bin_edges_sm, label='sm')
    plt.xlabel(var_name, fontsize=20)
    plt.ylabel('counts', fontsize=20)
    plt.legend()

    if out_path is not None:
        os.makedirs(os.path.join(user.plot_directory, out_path), exist_ok=True)
        plt.savefig(os.path.join(user.plot_directory, out_path, out_file+'.png'))
        plt.close()


# histogram showing eft effet on a variable using true weights
def plot_var_hist(var_name, var_data, weights, theta, bins=20, eq_width=True, ratio_hist=False, norm_to_nevents=1000, out_path=None, out_file='_hist'):
    out_file = var_name + out_file
    if eq_width is not True:
        bins = weighted_quantile(var_data.flatten(), np.linspace(0,1,bins+1), sample_weight=weight_theta(weights,0))
        out_file += '_weighted_quantiles'
    hist_sm, bin_edges_sm = np.histogram(var_data.flatten(), bins, weights=weight_theta(weights,0))
    hist_eft, bin_edges_eft = np.histogram(var_data.flatten(), bins, weights=weight_theta(weights,theta))
    norm_sm, norm_eft = hist_sm.sum(), hist_eft.sum()

    hist_sm = hist_sm*(norm_to_nevents/norm_sm)
    hist_eft = hist_eft*(norm_to_nevents/norm_eft)
    
    plt.figure(figsize=(10,10))
    if ratio_hist:
        out_file += '_ratio'
        ratio = hist_sm/hist_eft
        hist_sm = hist_sm/hist_sm
        plt.stairs(ratio, bin_edges_sm, label=f'eft_ratio (theta={theta})')
        plt.stairs(hist_sm, bin_edges_sm, label='sm_ratio')
    else:
        plt.stairs(hist_sm, bin_edges_sm, label='sm')
        plt.stairs(hist_eft, bin_edges_eft, label=f'eft (theta={theta})')
    plt.xlabel(var_name)
    plt.ylabel('counts')
    plt.legend()
   
    if out_path is not None:
        os.makedirs(os.path.join(user.plot_directory, out_path), exist_ok=True)
        plt.savefig(os.path.join(user.plot_directory, out_path, out_file+'.png'))
        plt.close()



def plot_reweighted_hist(var_name, var_data, reweight_var, reweight_data, weights, theta=1., bins=10, eq_width=True, norm_to_nevents=1000, out_path=None, out_file='_reweight_hist'):

    out_file = f'{var_name}_reweight_with_{reweight_var}_hist'
   

    w_sm = weight_theta(weights, 0)
    w_eft = weight_theta(weights, theta)

    rw_hist_sm, rw_bin_edges = np.histogram(reweight_data.flatten(), bins, weights=w_sm)
    rw_hist_eft, _ = np.histogram(reweight_data.flatten(), bins, weights=w_eft)

    normalize_hist(rw_hist_sm, norm_to_nevents)
    normalize_hist(rw_hist_eft, norm_to_nevents)

    ratio = rw_hist_sm/rw_hist_eft

    inds = np.digitize(reweight_data.flatten(), rw_bin_edges, right=True)
    inds[inds!=0]-=1 # np.digitize puts the lowest value in a separate 'bin' i.e the lowest value gets the index 0 then the lowest full bin the index 1 

    reweight = w_eft.copy()
    for ind in range(bins):
        reweight[inds==ind] *= ratio[ind]

    # reweight_eft_hist = [reweight[inds==ind].sum() for ind in range(bins)]
    # dig_hist = [w_sm[inds==ind].sum() for ind in range(bins)]
    # print('reweight_hist:', *reweight_eft_hist)
    # print('dig_hist:', *dig_hist)
    # print('np hist:', *rw_hist_sm)

    # print(f'# bins in hist: {len(rw_hist_sm)}')
    # print(f'inds: {min(inds)}, {max(inds)}')

    if eq_width is not True:
        bins = weighted_quantile(var_data.flatten(), np.linspace(0,1,bins+1), sample_weight=w_sm)
        out_file += '_weighted_quantiles'

    reweighted_eft_hist, reweighted_bin_edges = np.histogram(var_data.flatten(), bins, weights=reweight)
    eft_hist, _ = np.histogram(var_data.flatten(), bins, weights=w_eft)
    sm_hist, _ = np.histogram(var_data.flatten(), bins, weights=w_sm)

    normalize_hist(reweighted_eft_hist, norm_to_nevents)
    normalize_hist(eft_hist, norm_to_nevents)
    normalize_hist(sm_hist, norm_to_nevents)



    plt.figure(figsize=(10,10))
    plt.stairs(reweighted_eft_hist, reweighted_bin_edges, label='reweighted_eft')
    plt.stairs(eft_hist, reweighted_bin_edges, label='eft')
    plt.stairs(sm_hist, reweighted_bin_edges, label='sm')
    plt.xlabel(var_name, fontsize=20)
    plt.ylabel('counts', fontsize=20)
    plt.legend()
    

    if out_path is not None:
            os.makedirs(os.path.join(user.plot_directory, out_path), exist_ok=True)
            plt.savefig(os.path.join(user.plot_directory, out_path, out_file+'.png'))
            plt.close()

# hist2d truth vs prediction
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


# LLR plot
def LLR(variable, weights, theta, bins=20, norm_to_nevents=1000):
    quantiles = weighted_quantile(values=variable.flatten(), quantiles=np.linspace(0,1,bins+1), sample_weight=weight_theta(weights,theta=0))
    n_hat_0, _ = np.histogram(variable.flatten(), bins=quantiles, weights=weight_theta(weights, theta=0))
    norm_0 = n_hat_0.sum()
    n_hat_0 *= (norm_to_nevents/norm_0)

    if isinstance(theta, float):
        n_hat_theta, _ = np.histogram(variable.flatten(), bins=quantiles, weights=weight_theta(weights, theta=theta))
        norm_theta = n_hat_theta.sum()
        n_hat_theta *= (norm_to_nevents/norm_theta) # is this the correct norm? is the LLR dominated by total yield? try norming here at bsm yields
        return -2*(n_hat_0 * np.log(n_hat_theta) - n_hat_theta).sum()
    elif isinstance(theta, np.ndarray):
        LLR=[]
        for th in theta:
            n_hat_theta, _ = np.histogram(variable.flatten(), bins=quantiles, weights=weight_theta(weights, theta=th))
            norm_theta = n_hat_theta.sum()
            n_hat_theta *= (norm_to_nevents/norm_theta)
            LLR.append(-2*(n_hat_0 * np.log(n_hat_theta) - n_hat_theta).sum())
        return np.array(LLR)
    else:
        print('theta must be float or 1d array')


def plot_nn_LLR(model, data, weights, theta_range=(-1.,1.), bins=20, n_theta_plot=100, norm_to_nevents=1000, out_path=None, out_file='nn_LLR.png'):
    if not isinstance(model, MLP):
        with open(model, 'rb') as m:
            model=pickle.load(m)
    pred = model(torch.tensor(data)).detach().numpy()

    theta_plot = np.linspace(*theta_range, n_theta_plot)
    print('compute LLRS for estimator')
    LLR_plot = LLR(variable=pred, weights=weights, theta=theta_plot, bins=bins, norm_to_nevents=norm_to_nevents)
    print('compute LLRS for true w1_0')
    w0 = weights[:,0,np.newaxis]
    w1_0 = weights[:,1,np.newaxis]/w0
    LLR_true_plot = LLR(variable=w1_0, weights=weights, theta=theta_plot, bins=bins, norm_to_nevents=norm_to_nevents)

    LLR_min = np.min(LLR_plot)
    idx = np.argwhere(np.diff(np.sign(LLR_plot - LLR_min - 1))).flatten()

    plt.figure(figsize=(10,10))
    plt.plot(theta_plot, LLR_plot, label='nn')
    plt.plot(theta_plot, LLR_true_plot, label='truth')
    plt.xlabel(r'$\theta$', fontsize=20)
    plt.ylabel('LLR', fontsize=20)

    # plt.plot(theta_plot, np.full_like(theta_plot, LLR_min), ':')
    plt.plot(theta_plot, np.full_like(theta_plot, LLR_min+1), ':')
    plt.plot(theta_plot[idx], LLR_plot[idx], 'or')

    plt.legend()

    if out_path is not None:
        os.makedirs(os.path.join(user.plot_directory, out_path), exist_ok=True)
        plt.savefig(os.path.join(user.plot_directory, out_path, out_file))
        plt.close()


def plot_nn_LLR_over_epochs(model_path, epoch_list, data, weights, theta_range=(-1.,1.), bins=20, n_theta_plot=100, norm_to_nevents=1000, out_path=None, out_file='nn_LLR_over_epochs.png'):
    model_files = [os.path.join(user.model_directory, model_path, f'epoch_{epoch}.pkl') for epoch in epoch_list]

    theta_plot = np.linspace(*theta_range, n_theta_plot)
    
    plt.figure(figsize=(10,10))
    plt.xlabel(r'$\theta$', fontsize=20)
    plt.ylabel('LLR', fontsize=20)
    
    print('compute LLRS for true w1_0')
    w0 = weights[:,0,np.newaxis]
    w1_0 = weights[:,1,np.newaxis]/w0
    LLR_true_plot = LLR(variable=w1_0, weights=weights, theta=theta_plot, bins=bins, norm_to_nevents=norm_to_nevents)
    LLR_min = np.min(LLR_true_plot)
    plt.plot(theta_plot, np.full_like(theta_plot, LLR_min+1), ':')
    plt.plot(theta_plot, LLR_true_plot, label='truth')
    
    for epoch, m_f in zip(epoch_list, model_files):
        with open(m_f, 'rb') as m:
            model=pickle.load(m)
            pred = model(torch.tensor(data)).detach().numpy().flatten()

        print('compute LLRS for estimator')
        LLR_plot = LLR(variable=pred, weights=weights, theta=theta_plot, bins=bins, norm_to_nevents=norm_to_nevents)
        plt.plot(theta_plot, LLR_plot, label=f'nn_epoch_{epoch}')

        idx = np.argwhere(np.diff(np.sign(LLR_plot - LLR_min - 1))).flatten()
        plt.plot(theta_plot[idx], LLR_plot[idx], 'or')

        # plt.plot(theta_plot, np.full_like(theta_plot, LLR_min), ':')

    plt.legend()


    if out_path is not None:
        os.makedirs(os.path.join(user.plot_directory, out_path), exist_ok=True)
        plt.savefig(os.path.join(user.plot_directory, out_path, out_file))
        plt.close()


def plot_var_LLR(var_name, data, weights, theta_range=(-1,1), bins=20, n_theta_plot=100, norm_to_nevents=1000, out_path=None, out_file='_LLR.png'):
    out_file = var_name + out_file

    theta_plot = np.linspace(*theta_range, n_theta_plot)
    LLR_plot = LLR(variable=data, weights=weights, theta=theta_plot, bins=bins, norm_to_nevents=norm_to_nevents)

    w0 = weights[:,0,np.newaxis]
    w1_0 = weights[:,1,np.newaxis]/w0
    LLR_true_plot = LLR(variable=w1_0, weights=weights, theta=theta_plot, bins=bins, norm_to_nevents=norm_to_nevents)

    LLR_min = np.min(LLR_plot)
    idx = np.argwhere(np.diff(np.sign(LLR_plot - LLR_min - 1))).flatten()

    plt.figure(figsize=(10,10))
    plt.plot(theta_plot, LLR_plot, label=var_name)
    plt.plot(theta_plot, LLR_true_plot, label='truth')
    plt.xlabel(r'$\theta$', fontsize=20)
    plt.ylabel('LLR', fontsize=20)

    # plt.plot(theta_plot, np.full_like(theta_plot, LLR_min), ':')
    plt.plot(theta_plot, np.full_like(theta_plot, LLR_min+1), ':')
    plt.plot(theta_plot[idx], LLR_plot[idx], 'or')

    plt.legend()

    if out_path is not None:
        os.makedirs(os.path.join(user.plot_directory, out_path), exist_ok=True)
        plt.savefig(os.path.join(user.plot_directory, out_path, out_file))
        plt.close()





if __name__ == '__main__':

    if PREP_DATA is True:

        model_file_path=os.path.join(user.model_directory, MODEL_PATH, MODEL_NAME)
        print(f'model file path: {model_file_path}')
        print(f'plot directory: {os.path.join(user.plot_directory, OUT_PATH)}')

        test_file_names = [
            f'/scratch-cbe/users/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_{n_file}.root:Events'
            for n_file in range(*TEST_FILE_RANGE)]

        print('load hist files, branch names, weights')
        scalar_branches, vector_branches = get_branch_names()
        test_scalar_events, test_vector_events, test_weights = load_data(file_names=test_file_names)
        test_w0 = test_weights[:,0,np.newaxis]
        test_w1_0 = test_weights[:,1,np.newaxis]/test_w0

        with open(model_file_path, 'rb') as m:
            model=pickle.load(m)

        pred_w1_0 = model(torch.tensor(test_scalar_events)).detach().numpy()
        pred_weights = np.concatenate([test_w0, pred_w1_0*test_w0], axis=-1)

        var_data = uproot.concatenate(test_file_names, cut=selection, branches=VAR_NAME, library='np')[VAR_NAME]

        reweight_data = uproot.concatenate(test_file_names, cut=selection, branches=VAR_NAME, library='np')[REWEIGHT_VAR]

    print('plot model loss history')
    plot_losses(model=model_file_path, out_path=OUT_PATH)

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
    for eq_width, ratio_hist in zip((True, True, False, False),(True, False, True, False)):
        plot_var_hist(
            var_name='nn',
            var_data=pred_w1_0,
            weights=test_weights,
            theta=ESTIMATOR_HIST_THETA,
            bins=ESTIMATOR_HIST_BINS,
            eq_width=eq_width,
            ratio_hist=ratio_hist,
            norm_to_nevents=NORM_TO_NEVENTS,
            out_path=OUT_PATH)

    print('plot gen_theta histogram')
    for eq_width, ratio_hist in zip((True, True, False, False),(True, False, True, False)):
        plot_var_hist(
            var_name=VAR_NAME,
            var_data=var_data,
            weights=test_weights,
            theta=ESTIMATOR_HIST_THETA,
            bins=ESTIMATOR_HIST_BINS,
            eq_width=eq_width,
            norm_to_nevents=NORM_TO_NEVENTS,
            ratio_hist=ratio_hist,
            out_path=OUT_PATH)

    print('reweight gem_theta with gen_theta plot')
    for eq_width in (True,False):
        plot_reweighted_hist(
            var_name=VAR_NAME,
            var_data=var_data,
            reweight_var=REWEIGHT_VAR,
            reweight_data=reweight_data,
            theta=ESTIMATOR_HIST_THETA,
            bins=ESTIMATOR_HIST_BINS,
            eq_width=eq_width,
            norm_to_nevents=NORM_TO_NEVENTS,
            weights=test_weights,
            out_path=OUT_PATH)

    print('reweight nn with gen_theta plot')
    for eq_width in (True,False):
        plot_reweighted_hist(
            var_name='nn',
            var_data=pred_w1_0,
            reweight_var=REWEIGHT_VAR,
            reweight_data=reweight_data,
            theta=ESTIMATOR_HIST_THETA,
            bins=ESTIMATOR_HIST_BINS,
            eq_width=eq_width,
            norm_to_nevents=NORM_TO_NEVENTS,
            weights=test_weights,
            out_path=OUT_PATH)
    

    print('plot gen_theta for eft pred v truth weights')
    plot_var_hist_pred_v_truth(
        var_name=VAR_NAME,
        var_data=var_data,
        truth_weights=test_weights,
        pred_weights=pred_weights,
        theta=ESTIMATOR_HIST_THETA,
        bins=ESTIMATOR_HIST_BINS,
        eq_width=False,
        out_path=OUT_PATH)    

    plot_var_hist_pred_v_truth(
        var_name=VAR_NAME,
        var_data=var_data,
        truth_weights=test_weights,
        pred_weights=pred_weights,
        theta=ESTIMATOR_HIST_THETA,
        bins=ESTIMATOR_HIST_BINS,
        eq_width=True,
        out_path=OUT_PATH)

    print('plot nn_LLR')
    plot_nn_LLR(
        model=model_file_path,
        data=test_scalar_events,
        weights=test_weights,
        theta_range=LLR_PLOT_THETA_RANGE,
        bins=LLR_PLOT_BINS,
        n_theta_plot=N_THETA_PLOT,
        norm_to_nevents=NORM_TO_NEVENTS,
        out_path=OUT_PATH)

    print('plot nn LLR over epochs')
    plot_nn_LLR_over_epochs(
        model_path=MODEL_PATH,
        epoch_list=EPOCH_LIST,
        data=test_scalar_events,
        weights=test_weights,
        theta_range=LLR_PLOT_THETA_RANGE,
        bins=LLR_PLOT_BINS,
        n_theta_plot=N_THETA_PLOT,
        norm_to_nevents=NORM_TO_NEVENTS,
        out_path=OUT_PATH)

    print('plot theta angle LLR')
    plot_var_LLR(
        var_name=VAR_NAME,
        data=var_data,
        weights=test_weights,
        theta_range=LLR_PLOT_THETA_RANGE,
        bins=LLR_PLOT_BINS,
        n_theta_plot=N_THETA_PLOT,
        norm_to_nevents=NORM_TO_NEVENTS,
        out_path=OUT_PATH)






