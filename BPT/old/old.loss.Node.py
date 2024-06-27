#!/usr/bin/env python

import numpy as np
import operator 
from math import sqrt
import itertools
import sys
sys.path.insert(0,'..')
sys.path.insert(0,'.')

import functools

default_cfg = {
    "max_depth":        4,
    "min_size" :        50,
    #"max_n_split":      -1,     # similar to TMVA: test only n_split values in the node-split search. 
    "loss" : "CrossEntropy",    # or "CrossEntropy" # MSE for toys is fine, in real life CrossEntropy is a bit more stable against outliers
}

class Node:
    def __init__( self, features, weights, enumeration, _depth=0, **kwargs):

        ## basic BDT configuration + kwargs
        self.cfg = default_cfg
        self.cfg.update( kwargs )
        for attr, val in self.cfg.items():
            setattr( self, attr, val )

        self.features    = features
        self.weights     = weights
        self.enumeration = enumeration
        self.sizes       = np.unique(self.enumeration,axis=0,return_counts=True)[1]

        #print ("Constructor features, weights, enumeration", self.features.shape, self.weights.shape, self.enumeration.shape)
        #print ("Constructor enum", np.unique(enumeration,return_counts=True))

        if self.cfg['loss'] not in ["CrossEntropy"]:
            raise RuntimeError( "Unknown loss. Should be 'CrossEntropy'." ) 

        self.requires = [ 'MkA', 'Mkkp', 'n_base_points', 'nominal_base_point_index']

        for r in self.requires:    
            assert (r in kwargs) and kwargs[r] is not None, "Must provide %s"%r
            setattr( self, r, kwargs[r])
            self.cfg[r] = kwargs[r]

        # remove the nominal base point from all the base points with this mask 
        self.nu_mask = np.ones(self.n_base_points, bool)
        self.nu_mask[self.nominal_base_point_index] = 0

        # keep track of recursion depth
        self._depth = _depth

        self.split(_depth=_depth)
        self.prune()

        # Let's not leak the dataset.
        del self.features
        del self.weights 
        del self.enumeration
        del self.split_left_group 

    def get_split_vectorized( self ):
        ''' determine where to split the features, first vectorized version of FI maximization
        '''

        # loop over the features ... assume the features consists of rows with [x1, x2, ...., xN]
        self.split_i_feature, self.split_value, self.split_loss, self.split_left_group = 0, -float('inf'), float('inf'), None
        self.split_loss_left, self.split_loss_right = float('inf'), float('inf')
        # for a valid binary split, we need at least twice the mean size
        assert all(self.sizes >= 2*self.min_size), "Too few elements for at least one variation!"

        # loop over features
        #print "len(self.features[0]))",len(self.features[0])

        for i_feature in range(len(self.features[0])):
            #print ("At feature: %i/%i"%( i_feature, len(self.feature_names)), self.feature_names[i_feature], "depth", self._depth)

            # sorting all the data
            feature_values         = self.features[:,i_feature]
            feature_sorted_indices = np.argsort(feature_values)
            sorted_features        = feature_values[feature_sorted_indices]
            sorted_weights         = self.weights[feature_sorted_indices]
            sorted_enumeration     = self.enumeration[feature_sorted_indices]
            # cumulative weighted sum
            sorted_weight_sums_left = np.cumsum(
                np.multiply( sorted_weights[:,np.newaxis], np.eye(self.n_base_points)[sorted_enumeration]), axis=0 )

            total_weight_sum         = sorted_weight_sums_left[-1]
            sorted_weight_sums_left  = sorted_weight_sums_left[0:-1]
            sorted_weight_sums_right = total_weight_sum-sorted_weight_sums_left

            sorted_weight_sums_nominal_left  = sorted_weight_sums_left[:,  self.nominal_base_point_index]
            sorted_weight_sums_nominal_right = sorted_weight_sums_right[:, self.nominal_base_point_index]

            # cumulative unweighted sum (for min size!)
            sorted_count_sums_left  = np.cumsum( np.eye(self.n_base_points)[sorted_enumeration], axis=0) #We might train with samples with weight=0
            #sorted_count_sums_left = np.cumsum(
            #    np.multiply( (sorted_weights[:,np.newaxis]!=0).astype('int'), np.eye(self.n_base_points)[sorted_enumeration]), axis=0 )
            total_count_sum         = sorted_count_sums_left[-1]
            sorted_count_sums_left  = sorted_count_sums_left[0:-1]
            sorted_count_sums_right = total_count_sum-sorted_count_sums_left

            #print ("Test",self.weights[self.enumeration==0].sum())
            #print ("Test",sorted_weights[sorted_enumeration==0].sum())
            #print ("Test",self.weights[self.enumeration==1].sum())
            #print ("Test",sorted_weights[sorted_enumeration==1].sum())
            #print ("Test",self.weights[self.enumeration==2].sum())
            #print ("Test",sorted_weights[sorted_enumeration==2].sum())

            if i_feature==0:
                self.Delta = np.dot(self.MkA, np.log(total_weight_sum[self.nu_mask]/total_weight_sum[self.nominal_base_point_index]))
                #print ("At first feature. Depth",self._depth )
                #print ("At first feature: total_weight_sum", total_weight_sum )
                #print ("At first feature: total_count_sum", total_count_sum )
                #print ("At first feature: sorted_weight_sums_left", sorted_weight_sums_left )
                #print ("At first feature: sorted_count_sums_left", sorted_count_sums_left )
                #print ("At first feature: weights", self.weights)
                #print ("At first feature: self.enumeration", self.enumeration)
                #print ("At first feature: feature_sorted_indices", feature_sorted_indices)
                #print ("At first feature: total_weight_sum", total_weight_sum)
                #print ("At first feature: nominal_base_point_index, self.n_base_points", nominal_base_point_index, self.n_base_points )
                #print ("At first feature: Delta:", self.Delta, "sizes", self.sizes, "depth", self._depth)

            sorted_weight_sums_left  = sorted_weight_sums_left[:, self.nu_mask]
            sorted_weight_sums_right = sorted_weight_sums_right[:, self.nu_mask]

            plateau_and_split_range_mask  = ((np.all(sorted_count_sums_left>=default_cfg['min_size'], axis=1)) & (np.all(sorted_count_sums_right>=default_cfg['min_size'], axis=1))).astype('bool')
            
            if self.cfg['loss'] == 'CrossEntropy':
                with np.errstate(divide='ignore', invalid='ignore'):
                    exponent_left  = np.dot( self.Mkkp, np.log(sorted_weight_sums_left/sorted_weight_sums_nominal_left[:,None]).transpose())
                    exponent_right = np.dot( self.Mkkp, np.log(sorted_weight_sums_right/sorted_weight_sums_nominal_right[:,None]).transpose())

                    l_left  = sorted_weight_sums_nominal_left    *(-np.math.log(2) + np.log1p( np.exp(  exponent_left )))\
                             +sorted_weight_sums_left.transpose()*(-np.math.log(2) + np.log1p( np.exp( -exponent_left )))
                    l_right = sorted_weight_sums_nominal_right    *(-np.math.log(2) + np.log1p( np.exp(  exponent_right )))\
                             +sorted_weight_sums_right.transpose()*(-np.math.log(2) + np.log1p( np.exp( -exponent_right )))

            l_left_summed  = l_left.sum(axis=0)
            l_right_summed = l_right.sum(axis=0)
            loss = l_left_summed + l_right_summed 

            ## respect min size for split
            #if self.max_n_split<2:
            #    plateau_and_split_range_mask = np.ones(len(feature_values)-1, dtype=np.dtype('bool'))
            #else:
            #    min_, max_ = min(feature_values), max(feature_values) 
            #    #print "_depth",self._depth, "len(feature_values)",len(feature_values), "min_, max_", min_, max_
            #    plateau_and_split_range_mask  = np.digitize(feature_values[feature_sorted_indices], np.arange (min_, max_, (max_-min_)/(self.max_n_split+1)))
            #    #print len(plateau_and_split_range_mask), plateau_and_split_range_mask
            #    plateau_and_split_range_mask = plateau_and_split_range_mask[1:]-plateau_and_split_range_mask[:-1]
            #    plateau_and_split_range_mask = np.insert( plateau_and_split_range_mask, 0, 0).astype('bool')[:-1]
            #    #print "plateau_and_split_range_mask", plateau_and_split_range_mask
            #    #print "CUTS", feature_values[feature_sorted_indices][:-1][plateau_and_split_range_mask] 

            #if self.min_size > 1:
            #    plateau_and_split_range_mask[0:self.min_size-1] = False
            #    plateau_and_split_range_mask[-self.min_size+1:] = False

            plateau_and_split_range_mask &= (np.diff(sorted_features) != 0)
            
            #plateau_and_split_range_mask = plateau_and_split_range_mask.astype(int) #FIXME

            loss_masked = loss #= np.nan_to_num(loss)#*plateau_and_split_range_mask
            loss_masked[~plateau_and_split_range_mask] = float('nan') #FIXME
            try:
                argmin_split = np.nanargmin(loss_masked)
            except ValueError:
                argmin_split = None

            if argmin_split:
                loss_value  = loss_masked[argmin_split]
                value       = feature_values[feature_sorted_indices[argmin_split]]
                loss_left   = l_left_summed[feature_sorted_indices[argmin_split]] 
                loss_right  = l_right_summed[feature_sorted_indices[argmin_split]] 
            else:
                loss_value = float('inf')    
                value      =-float('inf')
                loss_left  = float('inf') 
                loss_right = float('inf') 

            #Delta_left  = np.dot( self.MkA, np.log( sorted_weight_sums_left[argmin_split]/sorted_weight_sums_nominal_left[argmin_split] ))
            #Delta_right = np.dot( self.MkA, np.log( sorted_weight_sums_right[argmin_split]/sorted_weight_sums_nominal_right[argmin_split] ))
            #print ("Node depth", self._depth)
            #print ("Split at position ",argmin_split, "left (nominal/var)", sorted_weight_sums_nominal_left[argmin_split], sorted_weight_sums_left[argmin_split],
            #       "right (nominal/var)", sorted_weight_sums_nominal_right[argmin_split], sorted_weight_sums_right[argmin_split])
            #print ("Delta_left",Delta_left,"Delta_right",Delta_right)

            #print ("Old loss value", self.split_loss, "new one", loss_value, "feature", self.feature_names[i_feature],"<",value)
            if loss_value < self.split_loss: 
                #print ("Found better split!")
                self.split_i_feature = i_feature
                self.split_value     = value
                self.split_loss      = loss_value
                self.split_loss_left  = loss_left
                self.split_loss_right = loss_right

            if np.count_nonzero(self.features[:,self.split_i_feature]<=self.split_value) == 1: 
                print ("Warning! Single-entry node!!")

        assert not np.isnan(self.split_value)


        self.split_left_group = self.features[:,self.split_i_feature]<=self.split_value if not  np.isnan(self.split_value) else np.ones(self.size, dtype='bool')

        if np.count_nonzero(self.split_left_group)>0 and np.count_nonzero(self.split_left_group)<self.min_size:
            print ("Mask:",plateau_and_split_range_mask[argmin_split])
        
            self.split_sorted_count_sums_left = sorted_count_sums_left[argmin_split]
            self.split_sorted_count_sums_right = sorted_count_sums_right[argmin_split]

            print ("End of vectorized split. We have found", self.feature_names[self.split_i_feature], "<", self.split_value, "loss", self.split_loss)
            #print ("End of vectorized split. Prediction", self.Delta,"left/right", Delta_left, Delta_right)
            #print ("End of vectorized split. Delta came from ", np.log(total_weight_sum[nu_mask]/total_weight_sum[nominal_base_point_index]), total_weight_sum[nu_mask], total_weight_sum[nominal_base_point_index] )
            print ("split_sorted_count_sums_left", self.split_sorted_count_sums_left)
            print ("split_sorted_count_sums_right", self.split_sorted_count_sums_right)

            print (np.unique(self.enumeration[self.split_left_group],return_counts=True))
            print (np.unique(self.enumeration[~self.split_left_group],return_counts=True))
            raise RuntimeError

    # everything we want to store in the terminal nodes
    def __store( self, group ):
        
        if np.count_nonzero(group)>0 and np.count_nonzero(group)<self.min_size:
            print ("Warning! Too small group.")
        return {
            'size': np.count_nonzero(group),
            }

    # Create child splits for a node or make terminal
    def split(self, _depth=0):

        # Find the best split
        #tic = time.time()

        self.get_split_vectorized()

        # check for max depth or a 'no' split
        if  self.max_depth <= _depth+1 or (not any(self.split_left_group)) or all(self.split_left_group): # Jason Brownlee starts counting depth at 1, we start counting at 0, hence the +1
            #print ("Choice2 (left) at depth ", _depth, self.Delta, "(all goes left, right is empty)")
            # The split was good, but we stop splitting further. Put everything in the left node! 
            self.split_value = float('inf')
            self.left        = ResultNode(Delta=self.Delta, loss_value=self.split_loss_left, **self.__store(np.ones(self.sizes.sum(),dtype=bool)), **self.cfg)
            self.right       = ResultNode(Delta=self.Delta, loss_value=self.split_loss_right, **self.__store(np.zeros(self.sizes.sum(),dtype=bool)), **self.cfg)
            return

        # process left child
        if any( np.unique(self.enumeration[self.split_left_group],return_counts=True)[1] < 2*self.min_size):
            #print ("Choice3 (left) at depth ", _depth,  "too few counts ", self.Delta )
            # Too few events in the left box. We stop.
            self.left             = ResultNode(Delta=self.Delta, loss_value=self.split_loss_left, **self.__store(self.split_left_group), **self.cfg)
        else:
            #print ("Choice4", _depth )
            # Continue splitting left box.
            self.left             = Node(self.features[self.split_left_group], weights=self.weights[self.split_left_group], enumeration=self.enumeration[self.split_left_group],  _depth=self._depth+1, **self.cfg)
        # process right child
        if any( np.unique(self.enumeration[~self.split_left_group],return_counts=True)[1] < 2*self.min_size):
            #print ("Choice5 (right) at depth", _depth, "too few counts", self.Delta )
            # Too few events in the right box. We stop.
            self.right            = ResultNode(Delta=self.Delta, loss_value=self.split_loss_right, **self.__store(~self.split_left_group), **self.cfg)
        else:
            #print ("Choice6", _depth  )
            # Continue splitting right box. 
            self.right            = Node(self.features[~self.split_left_group], weights=self.weights[~self.split_left_group], enumeration=self.enumeration[~self.split_left_group], _depth=self._depth+1, **self.cfg)

    # Prediction    
    def predict( self, features):
        ''' obtain the result by recursively descending down the tree
        '''
        node = self.left if features[self.split_i_feature]<=self.split_value else self.right
        if isinstance(node, ResultNode):
            return node.Delta 
        else:
            return node.predict(features)

    @property
    def loss_value(self):
        ''' obtain the loss by recursively descending down the tree
        '''
        if isinstance(self, ResultNode):
            return self.loss_value
        else:
            return self.left.loss_value + self.right.loss_value

    def vectorized_predict(self, feature_matrix):
        """Create numpy logical expressions from all paths to results nodes, associate with prediction defined by key, and return predictions for given feature matrix
           Should be faster for shallow trees due to numpy being implemented in C, despite going over feature vectors multiple times."""

        emmitted_expressions_with_predictions = []

        def emit_expressions_with_predictions(node, logical_expression):
            if isinstance(node, ResultNode):
                emmitted_expressions_with_predictions.append((logical_expression, node.Delta))
            else:
                if node == self:
                    prepend = ""
                else:
                    prepend = " & "
                if np.isinf(node.split_value):
                    split_value_str = 'np.inf'
                else:
                    split_value_str = format(node.split_value, '.32f')
                emit_expressions_with_predictions(node.left, logical_expression + "%s(feature_matrix[:,%d] <= %s)" % (prepend, node.split_i_feature, split_value_str))
                emit_expressions_with_predictions(node.right, logical_expression + "%s(feature_matrix[:,%d] > %s)" % (prepend, node.split_i_feature, split_value_str))

        emit_expressions_with_predictions(self, "")
        predictions = np.zeros((len(feature_matrix), len(self.Delta)))

        for expression, prediction in emmitted_expressions_with_predictions:
            predictions[eval(expression)] = prediction

        return predictions

    # remove the 'inf' splits
    def prune( self ):
        if not isinstance(self.left, ResultNode) and self.left.split_value==float('+inf'):
            self.left = self.left.left
        elif not isinstance(self.left, ResultNode):
            self.left.prune()
        if not isinstance(self.right, ResultNode) and self.right.split_value==float('+inf'):
            self.right = self.right.left
        elif not isinstance(self.right, ResultNode):
            self.right.prune()

    # Print a decision tree
    def print_tree(self, _depth=0):
        print('%s[%s <= %.3f]' % ((self._depth*' ', "nu%d"%self.split_i_feature if self.feature_names is None else self.feature_names[self.split_i_feature], self.split_value)))
        for node in [self.left, self.right]:
            node.print_tree(_depth = _depth+1)

    def get_list(self):
        ''' recursively obtain all thresholds '''
        return [ (self.split_i_feature, self.split_value), self.left.get_list(), self.right.get_list() ] 

class ResultNode:
    ''' Simple helper class to store result value.
    '''
    def __init__( self, Delta=None, loss_value=None, **kwargs):
        for k, v in kwargs.items():
            setattr( self, k, v)
        self.Delta     = Delta
        self.loss_value= loss_value

    def print_tree(self, _depth=0):
        poly_str = "".join(["*".join(["{:+.2e}".format(self.Delta[i_comb])] + list(comb) ) for i_comb, comb in enumerate(self.combinations)])

        print_string = '%s(%6i) log(r) = %s' % ((_depth)*' ', self.size, poly_str)
        print(print_string)

    def get_list(self):
        ''' recursively obtain all thresholds (bottom of recursion)'''
        return self.Delta 

if __name__=='__main__':

    import argparse
    argParser = argparse.ArgumentParser(description = "Argument parser")
    #argParser.add_argument('--data_model',         action='store', type=str,   default='TTLep_pow_sys', help="Which data model?")
    argParser.add_argument('--small',              action='store_true', help="Small?")
    argParser.add_argument('--toy',                action='store_true', help="Toy?")
    args = argParser.parse_args()

    # Interface: MOVE TO BPT TRAINING 
    if args.toy:
        import analytic as data_model
        parameters         = data_model.parameters
        combinations       = data_model.combinations
        nominal_base_point = data_model.nominal_base_point

        training_data = data_model.getEvents( 10000 if args.small else 1000000, weighted = True)
    else:
        parameters    = [ "nu" ]
        combinations  = [ ("nu",), ("nu", "nu") ]
        nominal_base_point = (0.,) 
    
        import models.TTLep_pow_sys as data_model
        generator = data_model.DataGenerator(maxN=200000 if args.small else None)#, levels = levels)
        features, variations = generator[0]

        # find all the base points
        _base_points   = np.unique(variations,axis=0)

        training_data = { tuple(base_point):{'features': features[np.where(np.all(base_point==variations,axis=1))]} for base_point in _base_points }

    # Precompute all the ingredients MOVE TO TREE INIT

    # Make sure of the format 
    base_points        = np.array( sorted(list(training_data.keys())), dtype='float')
    nominal_base_point = np.array( nominal_base_point, dtype='float')

    # Dissect inputs into nominal sample and variied
    nominal_base_point_index = np.where(np.all(base_points==nominal_base_point,axis=1))[0]
    assert len(nominal_base_point_index)>0, "Could not find nominal base %r point in training data keys %r"%( nominal_base_point, base_points)
    nominal_base_point_index = nominal_base_point_index[0]
    nominal_base_point_key   = tuple(nominal_base_point)

    nu_mask = np.ones(len(base_points), bool)
    nu_mask[nominal_base_point_index] = 0

    # remove the nominal from the list of all the base_points
    masked_base_points = base_points[nu_mask]

    # computing base-point matrix
    C  = np.zeros( [len(combinations), len(combinations) ], dtype='float64')
    for i_base_point, base_point in enumerate(masked_base_points):
        for i_comb1, comb1 in enumerate(combinations):
            for i_comb2, comb2 in enumerate(combinations):
                C[i_comb1][i_comb2] += functools.reduce(operator.mul, [base_point[parameters.index(c)] for c in list(comb1)+list(comb2)], 1)

    assert np.linalg.matrix_rank(C)==C.shape[0], "Base point matrix does not have full rank. Check base points & combinations."

    CInv = np.linalg.inv(C)

    # Compute matrix Mkk from non-nominal base_points
    VKA = np.zeros( (len(masked_base_points), len(combinations)) )
    for i_base_point, base_point in enumerate(masked_base_points):
        for i_combination, combination in enumerate(combinations):
            res=1
            for var in combination:
                res*=base_point[parameters.index(var)]

            VKA[i_base_point, i_combination ] = res

    MkA  = np.dot(VKA, CInv).transpose()
    Mkkp = np.dot(VKA, MkA )

    # Complement training data
    for k, v in training_data.items():
        if "features" not in v and "weights" not in v:
            raise RuntimeError( "Key %r has neither features nor weights" %k  )
        if k == nominal_base_point_key:
            if 'features' not in v:
                raise RuntimeError( "Nominal base point does not have features!" )
        else:
            if (not 'weights' in training_data[nominal_base_point_key].keys()) and 'weights' in v:
                raise RuntimeError( "Found no weights for nominal base point, but for a variation. This is not allowed" )
        
            if not 'weights' in v:
                v['weights'] = np.ones(v['features'].shape[0])
            elif not 'features' in v:
                v['features'] = training_data[nominal_base_point_key]['features']
            else:
                if len(v['weights'])!=len(v['features']):
                    raise RuntimeError("Key %r has unequal length of weights and features: %i != %i" % (k, len(v['weights']), len(v['features'])) )
    if 'weights' not in training_data[nominal_base_point_key]:
        training_data[nominal_base_point_key]['weights'] = np.ones(v['features'].shape[0]) 

    n_base_points = len(base_points)
    enumeration = np.concatenate( [ np.array( [i_base_point for _ in training_data[tuple(base_point)]['features']]) for i_base_point, base_point in enumerate( base_points)] , axis=0)
    features    = np.concatenate( [ training_data[tuple(base_point)]['features'] for i_base_point, base_point in enumerate( base_points)] , axis=0)
    weights     = np.concatenate( [ training_data[tuple(base_point)]['weights'] for i_base_point, base_point in enumerate( base_points)] , axis=0)

    ### BELOW IN THE LOOP  

    feature_values = features[:,0]

    feature_sorted_indices  = np.argsort(feature_values)
    sorted_weight_sums_left = np.cumsum(
        np.multiply( weights[feature_sorted_indices][:,np.newaxis], np.eye(n_base_points)[enumeration[feature_sorted_indices]]), axis=0 )
    total_weight_sum         = sorted_weight_sums_left[-1]
    sorted_weight_sums_left  = sorted_weight_sums_left[0:-1]
    sorted_weight_sums_right = total_weight_sum-sorted_weight_sums_left 

    sorted_weight_sums_nominal_left  = sorted_weight_sums_left[:, nominal_base_point_index]
    sorted_weight_sums_nominal_right = sorted_weight_sums_right[:, nominal_base_point_index]

    sorted_weight_sums_left  = sorted_weight_sums_left[:, nu_mask] 
    sorted_weight_sums_right = sorted_weight_sums_right[:, nu_mask] 

    with np.errstate(divide='ignore', invalid='ignore'):
        exponent_left  = np.dot( Mkkp, np.log(sorted_weight_sums_left/sorted_weight_sums_nominal_left[:,None]).transpose())
        exponent_right = np.dot( Mkkp, np.log(sorted_weight_sums_right/sorted_weight_sums_nominal_right[:,None]).transpose())

        l_left  = sorted_weight_sums_nominal_left    *(-np.math.log(2) + np.log1p( np.exp(  exponent_left )))\
                 +sorted_weight_sums_left.transpose()*(-np.math.log(2) + np.log1p( np.exp( -exponent_left )))
        l_right = sorted_weight_sums_nominal_right    *(-np.math.log(2) + np.log1p( np.exp(  exponent_right )))\
                 +sorted_weight_sums_right.transpose()*(-np.math.log(2) + np.log1p( np.exp( -exponent_right )))


    loss = (l_left+l_right).sum(axis=0)

    # cumulative unweighted sum (for min size!)
    #sorted_count_sums_left = np.cumsum( np.eye(n_base_points)[enumeration[feature_sorted_indices]], axis=0)
    sorted_count_sums_left = np.cumsum(
        np.multiply( (weights[feature_sorted_indices][:,np.newaxis]!=0).astype('int'), np.eye(n_base_points)[enumeration[feature_sorted_indices]]), axis=0 )
    total_count_sum         = sorted_count_sums_left[-1]
    sorted_count_sums_left  = sorted_count_sums_left[0:-1]
    sorted_count_sums_right = total_count_sum-sorted_count_sums_left

    plateau_and_split_range_mask  = ((np.all(sorted_count_sums_left>=default_cfg['min_size'], axis=1)) & (np.all(sorted_count_sums_right>=default_cfg['min_size'], axis=1))).astype('bool')

    plateau_and_split_range_mask &= (np.diff(feature_values[feature_sorted_indices]) != 0)

    argmin_split = np.nanargmin(loss*plateau_and_split_range_mask)
    Delta_left  = np.dot( MkA, np.log( sorted_weight_sums_left[argmin_split]/sorted_weight_sums_nominal_left[argmin_split] ))
    Delta_right = np.dot( MkA, np.log( sorted_weight_sums_right[argmin_split]/sorted_weight_sums_nominal_right[argmin_split] ))

    print ("Split at position ",argmin_split, "left (nominal/var)", sorted_weight_sums_nominal_left[argmin_split], sorted_weight_sums_left[argmin_split], 
           "right (nominal/var)", sorted_weight_sums_nominal_right[argmin_split], sorted_weight_sums_right[argmin_split])
    print ("Delta_left",Delta_left,"Delta_right",Delta_right)

    ## Now doe the SAME with the node    

    #node = Node( features     = features,
    #             weights      = weights,
    #             enumeration  = enumeration,
    #             max_depth    = 4,
    #             Mkkp         = Mkkp,
    #             MkA          = MkA,
    #             n_base_points=n_base_points,
    #             nominal_base_point_index=nominal_base_point_index,
    #             feature_names= data_model.feature_names if hasattr( data_model, "feature_names") else None,
    #             combinations = combinations,
    #             )

    #base_point_indices = {tuple(v):i_v for i_v,v in enumerate(base_points)}
    #def vector_cumsum( weights, variations ): 

    #    def cumsum(it):
    #        total = np.zeros(base_points.shape[0]) 
    #        for x in it:
    #            total[base_point_indices[tuple(x[1])]] += x[0]
    #            yield list(total)

    #    return np.array(list(cumsum(zip(weights, variations ))))

    #feature_values = features[:,3]
    #feature_sorted_indices = np.argsort(feature_values)
    #sorted_weight_sums = vector_cumsum( weights[feature_sorted_indices], variations[feature_sorted_indices] )
