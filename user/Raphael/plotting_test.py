#This file is for plotting
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

if __name__=="__main__":
    import sys
    sys.path.append('..')
    sys.path.append('../..')

sys.path.insert(0,os.path.expanduser("~/ML-pytorch/"))

import TT2lUnbinned.data_models.TTLep_pow_sys as data_model
import tools.helpers as helpers

generator = data_model.DataGenerator() #maybe add a maxN argument
features, variations = generator[0]
column_arrays=np.split(features,features.shape[1],axis=1)
unique_variations=np.unique(variations)

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

for i_feature, feature in enumerate(feature_names):
    binning_info = plot_options[feature]['binning']
    bins = np.linspace(binning_info[1], binning_info[2], binning_info[0] + 1)
    nominal_data = column_arrays[i_feature][variations == 0]
    n_nominal, _ = np.histogram(nominal_data, bins=bins)
    for variation_value in unique_variations:
        selected_data=column_arrays[i_feature][variations==variation_value]
        color = plt.cm.viridis(variation_value / np.max(unique_variations))
        label = f'$\sigma = {variation_value}$'
        plt.hist(selected_data, bins=bins, histtype='step', stacked=True, fill=False, color=color, label=label)

    plt.xlabel(plot_options[feature]['tex'])
    plt.ylabel('Number of events')
    scale_info = plot_options[feature]['logY']
    if scale_info == True:
        plt.yscale('log')
    else:
        plt.yscale('linear')
    output_path1 = os.path.join(output_folder, feature + '.png')
    plt.legend(loc='best')
    plt.savefig(output_path1)
    plt.clf()

    for variation_value in unique_variations:
        selected_data = column_arrays[i_feature][variations == variation_value]
        n, _ = np.histogram(selected_data, bins=bins)
        normalized_histogram = n / n_nominal
        label = f'$\sigma = {variation_value}$'
        color = plt.cm.viridis(variation_value / np.max(unique_variations))
        #plt.hlines(normalized_histogram, bins[:-1], bins[1:], label=label, color=color, linestyle='-')
        plt.stairs(normalized_histogram, bins, color = color, label=label)

    plt.xlabel(plot_options[feature]['tex'])
    plt.ylabel('Normalized events')
    scale_info = plot_options[feature]['logY']
    if scale_info == True:
        plt.yscale('log')
    else:
        plt.yscale('linear')
    output_path2 = os.path.join(output_folder, 'variation_'+ feature + '.png')
    plt.legend(loc='best')
    plt.savefig(output_path2)
    plt.clf()
