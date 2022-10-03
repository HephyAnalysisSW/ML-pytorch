from sklearn import linear_model

from BasePoints import BasePoints
import numpy as np

default_cfg = {
    "min_size" : 50,
}

if not __name__=="__main__":
    import logging
    logger = logging.getLogger('ML')

import DecisionNode
from NodeBase     import NodeBase

class LeafNode(NodeBase):

    @classmethod
    def root( cls, features, weights, base_points, save_history=False, **kwargs):
        root = cls(indices=None, features=features, weights=weights, base_points=base_points, save_history=save_history, **kwargs)
        root.depth = 0
        return root

    def __del__(self):
        NodeBase.remove_instance(self)

    def __init__( self, indices=None, features=None, weights=None, base_points=None, save_history=False, **kwargs):

        NodeBase.add_instance(self)

        self.save_history=save_history

        # set features for ALL instances of ALL classes deriving from NodeBase (!)
        # this is the root node
        if features is not None or weights is not None or base_points is not None:

            if features is None or weights is None or base_points is None:
                raise RuntimeError( "Need features AND weights AND base_points OR indices!")

            # make a BasePoint class that stores the linalg matrices
            coefficients = list(set(sum(map(list, weights.keys()),[])))
            coefficients.sort()
            base_points = base_points if type(base_points) == BasePoints else BasePoints(coefficients, base_points)

            # We store the training data as a class attribute of a common super class (NodeBase) of all the Node classes. 
            super().set_class_attrs(
                features     = features, 
                weights      = weights, 
                base_points  = base_points, 
                coefficients = coefficients
                )
            
            self.indices = np.array( range(len(self.features)) )
            if indices is not None:
                raise RuntimeError( "Do not know what to do with indices in root node" )

        # this is not the root node
        elif indices is not None:
            self.indices = indices 

        else:
            raise RuntimeError( "Need features AND weights AND base_points OR indices!")

        self.left, self.right = None, None 
        self.cfg = default_cfg
        self.cfg.update( kwargs )
        for (key, val) in kwargs.items():
            if key in default_cfg.keys():
                self.cfg[key]      = val
            else:
                raise RuntimeError( "Got unexpected keyword arg: %s:%r" %( key, val ) )

    # randomized split
    def split_even( self, i_feature = None):
        if i_feature is None:
            i_feature = np.random.randint(len(self.features[0]))

        # only split when we have 2x min_node_size
        if len(self.indices)>2*self.cfg['min_size']:

            parent = DecisionNode.DecisionNode(save_history=self.save_history)

            # get threshold
            threshold = np.quantile( self.features[self.indices][:, i_feature], .5) 
            mask      = self.features[self.indices][:, i_feature] > threshold

            parent.right = LeafNode( indices = self.indices[mask],  save_history=self.save_history, **self.cfg)
            parent.left  = LeafNode( indices = self.indices[~mask], save_history=self.save_history, **self.cfg)

            parent.right.parent = parent
            parent.left.parent  = parent
 
            logger.debug( "Split node %r (will disappear). The new parent is %r -> (%r %r) with %i-> (%i %i) events." % 
                ( self, parent, parent.left.parent, parent.right.parent, len(self.indices), len(parent.left.indices), len(parent.right.indices)))

            return parent
        else:
            logger.warning("Warning! Do not split because node is too small.")

    def fit( self, lasso):
        # actual lasso fit in Leaf node
        lasso = lasso.fit( 
            self.features[self.indices], 
            self.base_points.L.dot( np.array( [ self.weights[c][self.indices]/self.weights[()][self.indices] for c in self.base_points.combinations ] ) ).transpose(), 
            sample_weight = self.weights[()][self.indices] 
            ) 

        # recall parameters
        self.w0 = self.base_points.Linv.dot(lasso.intercept_.reshape(-1,1))
        self.w1 = self.base_points.Linv.dot(lasso.coef_)
        
        # recall history
        if self.save_history:
            if not hasattr(self, "history"):
                self.history = [(self.w0, self.w1)]
            else:
                self.history.append((self.w0, self.w1))

    # recall an earlier state
    def set_history( self, history=-1 ):
        self.w0, self.w1 = self.history[history]

    def predict( self, features ):
        return features.dot(self.w1.transpose()) + self.w0.transpose()
        #return self.base_points.Linv.dot(self.lasso.predict(features).transpose()).transpose()

    # the leaf node of the leaf node is the ... leaf node
    def get_leaf_node( self, features ):
        return self

    # compute loss
    def loss( self, indices ):
        return  self.weights[()][indices]*np.sum( self.base_points.L.dot( np.array( [ self.weights[c][indices]/self.weights[()][indices] for c in self.base_points.combinations ]  - self.predict(self.features[indices]).transpose() ) )**2, axis=0)

    # print this node
    def print_tree(self, _depth=0):
        if hasattr(self, "w0"):
            fit_str = "w0: %s w1: %s" %( str(np.round(self.w0.tolist(),4)).replace("\n", ""), str(np.round(self.w1.tolist(),4)).replace("\n", ""))
        else:
            fit_str = " (not fit)"
        if hasattr(self, "indices"):
            fit_str = "nEvents: %i "% len(self.indices) + fit_str
        if hasattr(self, "depth"):
            fit_str = " depth: %i "%self.depth + fit_str
        else:
            fit_str = " (no depth) "+fit_str

        print('%sLeafNode %s' % (_depth* ' ', fit_str ))

if __name__ == "__main__":
    import itertools

    import sys
    sys.path.append('..')

    import toy_models.quadratic as model
    N_events_requested=10000

    coefficients = model.wilson_coefficients
    base_points_ = []
    for comb in list(itertools.combinations_with_replacement(coefficients,1))+list(itertools.combinations_with_replacement(coefficients,2)):
        base_points_.append( {c:comb.count(c) for c in coefficients} )

    base_points = BasePoints( coefficients, base_points_ )

    features = model.getEvents(N_events_requested)
    weights  = model.getWeights(features)

    l = LeafNode.root( features, weights, base_points_)

    lasso = linear_model.Lasso(alpha=0.01, fit_intercept=True)

    l.fit(lasso)
    l.predict(features)

    #LTy = base_points.L.dot( np.array( [ weights[c] for c in base_points.combinations ] ) ).transpose() 
    #l.lasso.fit( features, LTy, sample_weight = weights[()])
 
    #print (base_points.Linv.dot(l.lasso.intercept_.reshape(-1,1)))
    #print (base_points.Linv.dot(l.lasso.coef_))

    #from sklearn.datasets import load_iris
    #X, y = load_iris(return_X_y=True)
    #y[y==2]=1

    #lr = linear_model.LogisticRegression(penalty='l1', C=100., solver='liblinear').fit(X,y)
    #lr2 = linear_model.LogisticRegression(penalty='l1', C=100., solver='liblinear')
    #lr2.classes_ = lr.classes_
    #lr2.coef_ = 0.5*np.ones_like(lr.coef_)
    #lr2.intercept_ = lr.intercept_
