#Comment
import itertools
import torch
import pickle
import numpy as np
import subprocess
import os

from Tools import syncer 
from Tools import user
from Tools import helpers
import ROOT
import random
class R:
    def __init__(self,nfeatures,coefficient_names):
        self.nfeatures         = nfeatures
        self.coefficient_names = coefficient_names
        self.combination_list=list(itertools.chain.from_iterable(itertools.combinations_with_replacement(self.coefficient_names, i) for i in np.arange(0,3)))
        self.n_hat = {combination: self.make_NN() for combination in self.combination_list}
        
    def make_NN(self, hidden_layers  = [32, 32, 32, 32]):
        '''
        Creates the Neural Network Architecture
        '''
        model_nn = [torch.nn.BatchNorm1d(self.nfeatures), torch.nn.ReLU(), torch.nn.Linear(self.nfeatures, hidden_layers[0])]
        for i_layer, layer in enumerate(hidden_layers):
            model_nn.append(torch.nn.Linear(hidden_layers[i_layer], hidden_layers[i_layer+1] if i_layer+1<len(hidden_layers) else 1))
            if i_layer+1<len(hidden_layers):
                model_nn.append( torch.nn.ReLU() )
        return torch.nn.Sequential(*model_nn)

    def evaluate_NN(self, features):
        '''Evaluate Neural Network: The zeroth dimension of features is the number of data points and and the first dimension
        is the number of features(variables). Returns the output of the NNs of dimensions: (noutput,ndatapoints)
        '''
        noutputs=len(self.combination_list)
        ndatapoints=features.shape[0]
        
        output=torch.zeros((noutputs,ndatapoints))
        for i in range(noutputs):
            x=self.n_hat[self.combination_list[i]](features)
            if i==0:
                output[i,:]=1
            else:
                output[i,:]=torch.flatten(x)            
        return output
        
    
    def predict_r_hat(self, features, base_points):
        '''
        Evaluate positive xsec ratio for given theta and.
        First it fills the coefficients of the matrix containing the upper cholesky decomposition.
        Then it computes the multiplication of this matrix with the basepoints and squares it and sums it.
        '''
        ndatapoints=features.shape[0]
        output_NN = self.evaluate_NN(features)
        n_terms=len(self.coefficient_names)
        row,column=np.triu_indices(n_terms+1)
        Omega=torch.zeros((ndatapoints,n_terms+1,n_terms+1))
        for i in range(0, len(row)):
            Omega[:][row[i]][column[i]]=output_NN[i,:]
        out=torch.matmul(Omega_swapped,torch.transpose(base_points,0,1))
        return torch.linalg.norm(out, 2, 1)
    
    def predict_r_hat2(self, features, predicions, basepoints):
        predictions=self.evaluate_NN(features)
        linear_terms=torch.sum( torch.stack( [(1. + predictions[ci,:]*basepoints[c])**2 for ci,c in enumerate(coefficients) ]), dim=0)
        quadratic_terms= torch.sum( torch.stack( [torch.sum( torch.stack( [ predictions[(c_1,c_2)]*eft[c_2] for c_2 in coefficients[i_c_1:] ]), dim=0)**2 for i_c_1, c_1 in enumerate(coefficients) ] ), dim=0 )
        return torch.add(linear_terms,quadratic_terms)      

    
    def save(self,fileName):
        outfile = open(fileName,'wb')
        pickle.dump(self, outfile)
        outfile.close()
        
    @classmethod
    def load(self, fileName):
        infile = open(fileName,'rb')
        print(fileName)
        new_dict = pickle.load(infile)
        infile.close()
        return new_dict