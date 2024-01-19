# This script is for plotting the results of the Training

import matplotlib.pyplot as plt
import numpy as np
import ROOT, os
import torch
import torch.nn as nn
import neural_network_pytorch as nnp
import classifier as cf 
import classifier_weighted as wcf

#Settings 
shuffle=0   #creates Plots with shuffle in name (check training if shuffle is on!) Shuffeling does not make a difference in result!

Plot_loss_regressor = 0
Plot_loss_classifier=1
Plot_loss_weighted_classifier=1

Plot_weighted_regressor=0   #Plots validation of training

Plot_classifier_distribution=1
Plot_weighted_classifier_distribution=1

Plot_y_pred_hist=0
Plot_ratio_hist=0


plot_options =  {
    "met": {'binning': [20, 100, 500], 'logY': True, 'tex': r'$E_{T}^{\mathrm{miss}}$'},
    "ht": {'binning': [20, 500, 1500], 'logY': True, 'tex': r'$H_{T}$'},
    "nJetGood": {'binning': [7, 3, 10], 'logY': False, 'tex': r'$N_{\mathrm{jet}}$'},
    "jet0_pt": {'binning': [30, 0, 1000], 'logY': True, 'tex': r'$p_{T}(\mathrm{jet\ 0})$'},
    "jet1_pt": {'binning': [30, 0, 1000], 'logY': True, 'tex': r'$p_{T}(\mathrm{jet\ 1})$'},
    "jet2_pt": {'binning': [30, 0, 500], 'logY': True, 'tex': r'$p_{T}(\mathrm{jet\ 2})$'},
    "jet3_pt": {'binning': [30, 0, 500], 'logY': True, 'tex': r'$p_{T}(\mathrm{jet\ 3})$'},
    "jet4_pt": {'binning': [30, 0, 500], 'logY': True, 'tex': r'$p_{T}(\mathrm{jet\ 4})$'},
    "jet0_eta": {'binning': [30, -4, 4], 'logY': False, 'tex': r'$\eta(\mathrm{jet\ 0})$'},
    "jet1_eta": {'binning': [30, -4, 4], 'logY': False, 'tex': r'$\eta(\mathrm{jet\ 1})$'},
    "jet2_eta": {'binning': [30, -4, 4], 'logY': False, 'tex': r'$\eta(\mathrm{jet\ 2})$'},
    "jet3_eta": {'binning': [30, -4, 4], 'logY': False, 'tex': r'$\eta(\mathrm{jet\ 3})$'},
    "jet4_eta": {'binning': [30, -4, 4], 'logY': False, 'tex': r'$\eta(\mathrm{jet\ 4})$'},
}

feature_names = [ "nJetGood", "ht", "jet0_pt", "jet1_pt", "jet2_pt", "jet3_pt", "jet0_eta", "jet1_eta", "jet2_eta", "jet3_eta" ]

#Load Regressor
loaded_loss_regressor= np.load('loss_data_regressor.npz', allow_pickle=True)
loss_array_regressor=loaded_loss_regressor['loss_array']
loaded_regressor = nnp.NeuralNetwork(nnp.n_var_flat, nnp.args.quadratic)
loaded_regressor.load_state_dict(torch.load('best_regressor_model.pth'))
loaded_regressor.eval()

#Load Classifier for \nu =1 and \nu =0
loaded_loss_classifier=np.load('loss_data_classifier.npz',allow_pickle=True)
loss_array_classifier=loaded_loss_classifier['loss_array']
loaded_classifier=cf.NeuralNetwork(cf.n_var_flat, cf.args.quadratic)
loaded_classifier.load_state_dict(torch.load('best_classifier_model.pth'))
loaded_classifier.eval()

#Load Classifier for weighted Data
loaded_loss_weighted_classifier=np.load('loss_data_weighted_classifier.npz',allow_pickle=True)
loss_array_weighted_classifier=loaded_loss_weighted_classifier['loss_array']
loaded_weighted_classifier=wcf.NeuralNetwork(wcf.n_var_flat, wcf.args.quadratic)
loaded_weighted_classifier.load_state_dict(torch.load('best_weighted_classifier_model.pth'))
loaded_weighted_classifier.eval()


generator = nnp.data_model.DataGenerator()  # Adjust maxN as needed
features, variations = generator[0]

############################################################## Plotting ######################################################
output_folder = 'Plots'
os.makedirs(output_folder, exist_ok=True)

#Create Histogramm for y_pred values
if Plot_y_pred_hist:
    plt.hist(y_pred.detach().numpy(), bins=10)
    plt.title('Y Pred Values')
    plt.xlabel('Values')
    plt.ylabel('Number')
    output_path = os.path.join(output_folder, 'Histo_Pred.png')
    plt.savefig(output_path)
    plt.clf()

#Create Histogramm for Ratio values
if Plot_ratio_hist:
    plt.hist(ratio.detach().numpy(), bins=10)
    plt.title('Ratio Values')
    plt.xlabel('Values')
    plt.ylabel('Number')
    output_path = os.path.join(output_folder, 'Histo_Ratio.png')
    plt.savefig(output_path)
    plt.clf()


#Create loss plot for Regressor
if Plot_loss_regressor:
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Loss Function Regressor')
    plt.plot(loss_array_regressor)
    output_path= os.path.join(output_folder, 'Loss_regressor.png')
    plt.savefig(output_path)
    plt.clf()

#Create Loss for Classifier
if Plot_loss_classifier:
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Loss Function Classifier')
    plt.plot(loss_array_classifier)
    if shuffle:
        output_path= os.path.join(output_folder, 'Loss_classifier_shuffeld.png')
    else:
        output_path= os.path.join(output_folder, 'Loss_classifier.png')
    plt.savefig(output_path)
    plt.clf()

#Create Loss for weighted Classifier
if Plot_loss_weighted_classifier:
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Loss Function weighted Classifier')
    plt.plot(loss_array_weighted_classifier)
    if shuffle:
        output_path= os.path.join(output_folder, 'Loss_weighted_classifier_shuffeld.png')
    else:
        output_path= os.path.join(output_folder, 'Loss_weighted_classifier.png')
    plt.savefig(output_path)
    plt.clf()

#Creat Plot for Classifier Distribution
if Plot_classifier_distribution:
    features_0=features[(variations[:,0]==0)]
    features_1=features[(variations[:,0]==1)]
    features_tensor_0 =  torch.tensor(features_0, dtype=torch.float32)
    features_tensor_0 =  torch.where(torch.isnan(features_tensor_0), torch.tensor(0.0), features_tensor_0)  
    features_tensor_1 =  torch.tensor(features_1, dtype=torch.float32)
    features_tensor_1 =  torch.where(torch.isnan(features_tensor_1), torch.tensor(0.0), features_tensor_1)  
    delta_0=loaded_classifier(features_tensor_0,1)
    delta_1=loaded_classifier(features_tensor_1,1)
    probability_0=1/(1+torch.exp(delta_0))
    probability_0=probability_0.detach().numpy()
    probability_1=1/(1+torch.exp(delta_1))
    probability_1=probability_1.detach().numpy()

    plt.xlabel('Probability')
    plt.ylabel('Number of Events')
    plt.title('Probability Distribution for Classifier')
    bins = np.linspace(0.4, 0.6, 100 + 1)
    n_0,_ = np.histogram(probability_0, bins=bins)
    n_1,_= np.histogram(probability_1, bins=bins)
    n_0_safe= np.where(n_0 == 0, 1, n_0)
    probabilty_ratio=n_1/n_0_safe
    plt.plot(bins[1:],n_0, drawstyle='steps',color='blue', label = 'D_0')
    plt.plot(bins[1:],n_1, drawstyle='steps',color='red', label = 'D_1')
    if shuffle:
        output_path = os.path.join(output_folder, 'classifier_distribution_shuffeld.png')
    else:
        output_path = os.path.join(output_folder, 'classifier_distribution.png')
    plt.legend(loc='best')
    plt.savefig(output_path)
    plt.clf()

    plt.xlabel('Probability')
    plt.ylabel('Ratio D1/D0')
    plt.title('Ratio for Classifier')
    plt.plot(bins[1:],probabilty_ratio,drawstyle='steps', color='black', label='D_1/D_0')
    if shuffle:
        output_path = os.path.join(output_folder, 'classifier_distribution_ratio_shuffeld.png')
    else:
        output_path = os.path.join(output_folder, 'classifier_distribution_ratio.png')
    plt.legend(loc='best')
    plt.savefig(output_path)
    plt.clf()

#Create Plots for weighted Classifier Distribution
if Plot_weighted_classifier_distribution:
    features_0=features[(variations[:,0]==0)]
    features_1=features[(variations[:,0]==1)]
    features_tensor_0 =  torch.tensor(features_0, dtype=torch.float32)
    features_tensor_0 =  torch.where(torch.isnan(features_tensor_0), torch.tensor(0.0), features_tensor_0)  
    features_tensor_1 =  torch.tensor(features_1, dtype=torch.float32)
    features_tensor_1 =  torch.where(torch.isnan(features_tensor_1), torch.tensor(0.0), features_tensor_1)  
    delta=loaded_regressor(features_tensor_0,1)
    weights=torch.exp(delta)
    weights=weights.detach().numpy()
    features_weighted_1= features_0 * weights
    features_tensor_weighted_1 =  torch.tensor(features_weighted_1, dtype=torch.float32)
    features_tensor_weighted_1 =  torch.where(torch.isnan(features_tensor_weighted_1), torch.tensor(0.0), features_tensor_weighted_1)  

    delta_0=loaded_weighted_classifier(features_tensor_weighted_1,1)
    delta_1=loaded_weighted_classifier(features_tensor_1,1)
    probability_0=1/(1+torch.exp(delta_0))
    probability_0=probability_0.detach().numpy()
    probability_1=1/(1+torch.exp(delta_1))
    probability_1=probability_1.detach().numpy()

    plt.xlabel('Probability')
    plt.ylabel('Number of Events')
    plt.title('Probability Distribution for weighted Classifier')
    bins = np.linspace(0.4, 0.6, 100 + 1)
    n_0,_ = np.histogram(probability_0, bins=bins)
    n_1,_= np.histogram(probability_1, bins=bins)
    n_0_safe= np.where(n_0 == 0, 1, n_0)
    probabilty_ratio=n_1/n_0_safe
    plt.plot(bins[1:],n_0,drawstyle='steps', color='blue', label = 'D_0') 
    plt.plot(bins[1:],n_1,drawstyle='steps', color='red', label = 'D_1')
    if shuffle:
        output_path = os.path.join(output_folder, 'weighted_classifier_distribution_shuffeld.png')
    else:
        output_path = os.path.join(output_folder, 'weighted_classifier_distribution.png')
    plt.legend(loc='best')
    plt.savefig(output_path)
    plt.clf()

    plt.xlabel('Probability')
    plt.ylabel('Ratio D1/D0')
    plt.title('Ratio for weighted Classifier')
    plt.plot(bins[1:],probabilty_ratio,drawstyle='steps', color='black', label='D_1/D_0')
    if shuffle:
        output_path = os.path.join(output_folder, 'weighted_classifier_distribution_ratio_shuffeld.png')
    else:
        output_path = os.path.join(output_folder, 'weighted_classifier_distribution_ratio.png')
    plt.legend(loc='best')
    plt.savefig(output_path)
    plt.clf()


#Create weighted Histogramms Regressor (validation of training)
if Plot_weighted_regressor:
    ratio_array=[]
    unique_variations=np.unique(variations)
    nominal_features= features[(variations[:, 0] == 0)] #filter the nominal data
    nominal_features_tensor=   torch.tensor(nominal_features, dtype=torch.float32)
    nominal_features_tensor =  torch.where(torch.isnan(nominal_features_tensor), torch.tensor(0.0), nominal_features_tensor)  
    for i_variation_value, variation_value in enumerate(unique_variations):
        if variation_value ==0:
            ratio_array.append(0)
        if variation_value != 0:
            delta=loaded_regressor(nominal_features_tensor,variation_value)
            ratio=torch.exp(variation_value*delta)
            ratio_array.append(ratio)
    column_arrays=np.split(features,features.shape[1],axis=1)
    for i_feature, feature in enumerate(feature_names):
        binning_info = plot_options[feature]['binning']
        bins = np.linspace(binning_info[1], binning_info[2], binning_info[0] + 1)
        nominal_data = column_arrays[i_feature][variations == 0]
        n_nominal, _ = np.histogram(nominal_data, bins=bins)
        n_nominal_safe = np.where(n_nominal == 0, 1, n_nominal)     #avoid division by 0
        for i_variation_value, variation_value in enumerate(unique_variations):
            selected_data = column_arrays[i_feature][variations == variation_value]
            n, _ = np.histogram(selected_data, bins=bins) 
            normalized_histogram= n /n_nominal_safe  #Truth  
            if i_variation_value != 4:
                weights=ratio_array[i_variation_value]
                weights=weights.detach().numpy()
                weights=np.squeeze(weights, axis=1)
                n_nominal_w, _ = np.histogram(nominal_data,bins=bins, weights=weights)
                n_nominal_w_safe = np.where(n_nominal_w == 0, 1, n_nominal_w)
                normalized_histogram_w = n / n_nominal_w_safe   #Prediction
            label = f'$\sigma = {variation_value}$ (Pred)'
            color = plt.cm.viridis(i_variation_value/len(unique_variations))
            if i_variation_value != 4:
                plt.plot(bins[1:],normalized_histogram_w, drawstyle='steps', color = color, linestyle='dashed', label=label) #Prediction
            plt.plot(bins[1:],normalized_histogram, drawstyle='steps', color = color)  #Truth

        plt.xlabel(plot_options[feature]['tex'])
        plt.ylabel('Normalized events weighted')
        scale_info = plot_options[feature]['logY']
        if scale_info == True:
            plt.yscale('log')
        else:
            plt.yscale('linear')
        output_path = os.path.join(output_folder, 'weightedvariation_'+ feature + '.png')
        plt.legend(loc='best')
        plt.savefig(output_path)
        plt.clf()

