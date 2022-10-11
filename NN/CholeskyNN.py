import torch
import math
import numpy as np
import os
import itertools
import time
import copy
import sys
sys.path.append('..')
from tools import helpers
        
device  = 'cuda' if torch.cuda.is_available() else 'cpu'

default_cfg = {
    "hidden_layers" : [32, 32, 32, 32],
}

class CholeskyNN:
    def __init__( self, coefficients, n_features, **kwargs):

        self.n_features = n_features

        # make cfg and node_cfg from the kwargs keys known by the Node
        self.cfg = default_cfg
        for (key, val) in kwargs.items():
            if key in default_cfg.keys():
                self.cfg[key] = val
            else:
                raise RuntimeError( "Got unexpected keyword arg: %s:%r" %( key, val ) )

        for (key, val) in self.cfg.items():
            setattr( self, key, val )

        # Coefficients
        self.coefficients  = tuple(sorted(coefficients))
        self.lin_combinations  = list(itertools.combinations_with_replacement(self.coefficients, 1))
        self.quad_combinations = list(itertools.combinations_with_replacement(self.coefficients, 2))
        self.combinations = self.lin_combinations + self.quad_combinations

        # Setup NN 
        self.n_hat = {} 
        for combination in self.combinations:
            model_nn = [torch.nn.BatchNorm1d(self.n_features), torch.nn.ReLU(), torch.nn.Linear(self.n_features, self.hidden_layers[0])]
            for i_layer, layer in enumerate(self.hidden_layers):

                model_nn.append(torch.nn.Linear(self.hidden_layers[i_layer], self.hidden_layers[i_layer+1] if i_layer+1<len(self.hidden_layers) else 1))
                if i_layer+1<len(self.hidden_layers):
                    model_nn.append( torch.nn.ReLU() )

            self.n_hat[combination] = torch.nn.Sequential(*model_nn)

    def print_model( self ):
        for combination in self.combinations:
            print ("n_hat( %s ) = \n"% ", ".join(combination), self.n_hat[combination])

    def r_hat( self, predictions, eft ):
        return torch.add( 
            torch.sum( torch.stack( [(1. + predictions[(c,)]*eft[c])**2 for c in self.coefficients ]), dim=0),
            torch.sum( torch.stack( [torch.sum( torch.stack( [ predictions[tuple(sorted((c_1,c_2)))]*eft[c_2] for c_2 in self.coefficients[i_c_1:] ]), dim=0)**2 for i_c_1, c_1 in enumerate(self.coefficients) ] ), dim=0 ) )

    def make_weight_ratio( self, weights, **kwargs ):
        ''' Computes 1 + (theta-theta_0)_a w_a + 0.5 * (theta-theta_0)_a (theta-theta_0)_b w_ab from input events
        '''
        eft      = kwargs
        result = torch.ones(len(weights[()])) 
        for combination in self.combinations:
            if len(combination)==1:
                result += eft[combination[0]]*weights[combination]/weights[()]
            elif len(combination)==2:# add up without the factor 1/2 because off diagonals are only summed in upper triangle 
                result += (0.5 if len(set(combination))==1 else 1.)*eft[combination[0]]*eft[combination[1]]*weights[combination]/weights[()]
        return result

    def predict( self, features ):
        if type(features) == np.ndarray:
            return {combination:self.n_hat[combination](torch.from_numpy(features).float().to(device)).squeeze().numpy() for combination in self.combinations} 
        else:
            return {combination:self.n_hat[combination](features).squeeze() for combination in self.combinations} 

    def vectorized_predict( self, features):
        return self.predict( features )

    # n_hat -> [ [d_lin_1, ..., d_quad_N] ] according to the "combinations" 
    def dict_to_derivatives( self, dict_ ):
        lin  = 2*np.array([dict_[c] for c in self.lin_combinations])
        quad = 2*np.array([dict_[(c[0],)]*dict_[(c[1],)] + np.sum( [dict_[tuple(sorted((c2, c[0])))]*dict_[tuple(sorted((c2, c[1])))] for c2 in self.coefficients if (self.coefficients.index(c2)>=self.coefficients.index(c[0]) and self.coefficients.index(c2)>=self.coefficients.index(c[1])) ],axis=0)  for c in self.quad_combinations ])

        return np.concatenate( (lin, quad), axis=0).transpose()

    # main training method
    def train( self, base_points, weights, features, test_weights = None, test_features = None, monitor_epoch = None, snapshots = None, n_epoch = 100, learning_rate = 1e-3):

        for net in list(self.n_hat.values()):
            net.train(True)

        torch_features = torch.from_numpy(features).float().to(device)

        self.monitoring = []

        optimizer               = torch.optim.Adam(sum([list(nn.parameters()) for nn in self.n_hat.values()],[]), lr=learning_rate)
        base_point_weight_ratios= list( map( lambda base_point: self.make_weight_ratio( weights, **base_point ), base_points ) )
        training_time           = 0 
        self.snapshots          = {} 

        if test_features is not None:
            with torch.no_grad():
                torch_test_features = torch.from_numpy(test_features).float().to(device)
                base_point_test_weight_ratios= list( map( lambda base_point: self.make_weight_ratio( test_weights, **base_point ), base_points ) )

        for epoch in range(n_epoch):

            monitoring  = {'epoch':epoch}
            start_time  = time.process_time()
            predictions = self.predict( torch_features) #{combination:self.n_hat[combination](torch_features).squeeze() for combination in self.combinations}

            # remove const piece
            loss = -0.5*weights[()].sum()
            for i_base_point, base_point in enumerate(base_points):
                fhat  = 1./(1. + self.r_hat(predictions, base_point) )
                loss += ( torch.tensor(weights[()])*( -0.25 + base_point_weight_ratios[i_base_point]*fhat**2 + (1-fhat)**2 ) ).sum()
                #loss += - ( torch.tensor(weights[()])*( base_point_weight_ratios[i_base_point]*torch.log(1-fhat) + torch.log(fhat) ) ).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_time         =  time.process_time() - start_time
            training_time      += total_time 
            monitoring['training_loss'] = loss.item()
            monitoring['training_time'] = training_time 
            self.monitoring.append( monitoring ) 
            print ( "Epoch %i time %3.2f Loss %7.5f"%( epoch, time.process_time()-start_time, loss))

            if snapshots is not None and epoch in snapshots:
                #self.snapshots[epoch] = copy.deepcopy( self ) #{ key:net.state_dict() for key, net in self.n_hat.items() }
                self.snapshots[epoch] = copy.deepcopy( { key:net.state_dict() for key, net in self.n_hat.items() } )
                #print(self.snapshots[epoch])
            # Compute test loss
            if test_features is not None:
                if monitor_epoch is None or epoch in monitor_epoch: 
                    with torch.no_grad():
                        test_predictions = {combination:self.n_hat[combination](torch_test_features).squeeze() for combination in self.combinations}                         
                        test_loss        = -0.5*test_weights[()].sum()
                        for i_base_point, base_point in enumerate(base_points):
                            test_fhat  = 1./(1. + self.r_hat(test_predictions, base_point) )
                            test_loss += ( torch.tensor(test_weights[()])*( -0.25 + base_point_test_weight_ratios[i_base_point]*test_fhat**2 + (1-test_fhat)**2 ) ).sum()
                    monitoring['test_loss'] = test_loss.item() 

        for net in list(self.n_hat.values()):
            net.train(False)

    # Wrappers just to have the same interface as BIT
    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as _file:
            self_ = torch.load(_file)
            for net in list(self_.n_hat.values()):
                net.train(False)
            return self_ 

    # sett the NN cfg to the snapshot
    def load_snapshot( self, snapshot ):
        for key, dict_ in snapshot.items():
            self.n_hat[key].load_state_dict( dict_ )
        for net in list(self.n_hat.values()):
            net.train(False)
        return self

    def save(self, filename):
        _path = os.path.dirname(filename)
        if not os.path.exists(_path):
            os.makedirs(_path)
        with open(filename, 'wb') as _file:
            torch.save( self, _file )

if __name__ == '__main__':

    # training data
    import models.ZH_Nakamura as model

    nEvents = 30000 

    model.feature_names = model.feature_names[0:6] # restrict features
    features       = model.getEvents(nEvents)[:,0:6]
    feature_names  = model.feature_names
    plot_options   = model.plot_options
    plot_vars      = model.feature_names

    mask       = (features[:,feature_names.index('pT')]<900) & (features[:,feature_names.index('sqrt_s_hat')]<1800) 
    features = features[mask]

    weights    = model.getWeights(features, model.make_eft() )

    # select coefficients
    WC = 'cHW'

    coefficients   = ('cHW', ) 
    #coefficients   =  ( 'cHW', 'cHWtil', 'cHQ3') 

    # Initialize model
    nn = CholeskyNN( coefficients, len(features[0])) 

    base_points = [ {'cHW':value} for value in [-1.5, -.8, -.4, -.2, .2, .4, .8, 1.5] ]
    #base_points = [ {'cHW':value1, 'cHWtil':value2} for value1 in [-1.5, -.8, -.2, 0., .2, .8, 1.5]  for value2 in [-1.5, -.8, -.2, 0, .2, .8, 1.5]]
    #base_points = list(filter( (lambda point: all([ coeff in args.coefficients or (not (coeff in point.keys() and point[coeff]!=0)) for coeff in point.keys()]) and any(map(bool, point.values()))), base_points)) 

    #coefficients = tuple(filter( lambda coeff: coeff in args.coefficients, list(coefficients))) 

    #base_points    = [ { 'cHW':-1.5 }, {'cHW':-.8}, {'cHW':-.4}, {'cHW':-.2}, {'cHW':.2}, {'cHW':.4}, {'cHW':.8}, {'cHW':1.5} ]
    #base_points   += [ { 'cHWtil':-1.5 }, {'cHWtil':-.8}, {'cHWtil':-.4}, {'cHWtil':-.2}, {'cHWtil':.2}, {'cHWtil':.4}, {'cHWtil':.8}, {'cHWtil':1.5} ]
    #base_points   += [ { 'cHQ3':-.15 }, {'cHQ3':-.08}, {'cHQ3':-.04}, {'cHQ3':-.02}, {'cHQ3':.02}, {'cHQ3':.04}, {'cHQ3':.08}, {'cHQ3':0.15} ]

    base_points    = list(map( lambda b:model.make_eft(**b), base_points ))

    nn.train( base_points, weights, features )
