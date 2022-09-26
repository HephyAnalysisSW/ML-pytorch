import sys
sys.path.insert( 0, '..')
from BIT.BoostedInformationTree import BoostedInformationTree

import numpy as np

N_events = 10000
N_channels = 21

expected_peak_yield   = 100
expected_flat_yield    = 0

sigma_signal= 0.02
sigma_bkg   = 0.1
kernel_size = 5

import Normal1D as model
from groups import U1
G = U1(N_channels)

signal_events       = model.getEvents( N_events//2,                 N_channels, expected_flat_yield=expected_flat_yield, norm_peak = True, expected_peak_yield=expected_peak_yield, sigma=sigma_signal)
background_events   = model.getEvents( N_events-len(signal_events), N_channels, expected_flat_yield=expected_flat_yield, norm_peak = True, expected_peak_yield=expected_peak_yield, sigma=sigma_bkg) 

signal_labels       = np.ones(len(signal_events))
background_labels   = -1*np.ones(len(signal_events))

training_events     = np.concatenate( [ signal_events, background_events ] )
training_weights    = np.concatenate( [ signal_labels, background_labels ] )

shuffle = np.array(range(len(signal_events)+len(background_events)))
np.random.shuffle(shuffle)

training_events  =   training_events[shuffle]
training_weights =   training_weights[shuffle]

def kernel_projector( features, kernel_size=kernel_size):
    return features[:,:kernel_size]

bit = BoostedInformationTree(
        training_features       = kernel_projector(np.matmul( G.elements, training_events.transpose()).transpose().reshape(-1,N_channels)),
        training_weights        = np.ones(G.N_elements*len(training_weights)), 
        training_diff_weights   = np.repeat(training_weights, G.N_elements), 
        learning_rate = 0.2, 
        n_trees   = 200,
        max_depth = 4 ,
        min_size  = 100,
        )

bit.boost()

def predict_eq( events ):
    return bit.vectorized_predict(kernel_projector(np.matmul( G.elements, events.transpose()).transpose().reshape(-1,N_channels))).reshape(-1,N_channels) 

sig_predictions = predict_eq( signal_events ) 
bkg_predictions = predict_eq( background_events )
