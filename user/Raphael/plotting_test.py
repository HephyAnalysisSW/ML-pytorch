#This file is for plotting
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt


if __name__=="__main__":
    import sys
    sys.path.append('..')


sys.path.insert(0,os.path.expanduser("~/ML-pytorch/"))

import data_models.TTLep_pow_sys as data_model

generator = data_model.DataGenerator() #maybe add a maxN argument
features, variations = generator[0]
column_arrays=np.split(features,features.shape[1],axis=1)

feature_names = [ "nJetGood", "ht", "jet0_pt", "jet1_pt", "jet2_pt", "jet3_pt", "jet0_eta", "jet1_eta", "jet2_eta", "jet3_eta" ]

plot_options =  {
    "met": {'binning': [20, 100, 500], 'logY': True, 'tex': r'$E_{T}^{\mathrm{miss}}$'},
    "ht": {'binning': [20, 500, 1500], 'logY': True, 'tex': r'$H_{T}$'},
    "nJetGood": {'binning': [7, 3, 10], 'logY': True, 'tex': r'$N_{\mathrm{jet}}$'},
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

output_folder = 'Plots'
os.makedirs(output_folder, exist_ok=True)

for i in range(len(feature_names)):
    binning_info = plot_options[feature_names[i]]['binning']
    bins = np.linspace(binning_info[1], binning_info[2], binning_info[0] + 1)
    n, bins = np.histogram(column_arrays[i], bins=bins)
    plt.hlines(n, bins[:-1], bins[1:], color='black')
    plt.xlabel(plot_options[feature_names[i]]['tex'])
    plt.ylabel('Number of events')
    scale_info = plot_options[feature_names[i]]['logY']
    if scale_info == True:
        plt.yscale('log')
    else:
        plt.yscale('linear')
    #plt.ylim(10**3,10**7)
    output_path = os.path.join(output_folder, feature_names[i] + '.png')
    plt.savefig(output_path)
    plt.clf()

