from sklearn import linear_model

import functools
import numpy as np
import operator

default_cfg = {
    "n_trees" : 100,
    "learning_rate" : 0.2,
}

from LeafNode import LeafNode

class DecisionTree:

    def __init__( self, features, weights,  base_points, **kwargs):
        self.cfg = default_cfg
        self.cfg.update( kwargs )
        for (key, val) in kwargs.items():
            if key in default_cfg.keys():
                self.cfg[key]      = val
            else:
                raise RuntimeError( "Got unexpected keyword arg: %s:%r" %( key, val ) )

        self.LogisticRegression = LogisticRegression(penalty='l1', solver='liblinear')
        


if __name__ == "__main__":
    import itertools

    import sys
    sys.path.append('..')

    import toy_models.sine as model
    N_events_requested=10000

    coefficients = ['theta1']
    base_points_ = []
    for comb in list(itertools.combinations_with_replacement(coefficients,1))+list(itertools.combinations_with_replacement(coefficients,2)):
        base_points_.append( {c:comb.count(c) for c in coefficients} )

    base_points = BasePoints( base_points_ )

    features = model.getEvents(N_events_requested)
    weights  = model.getWeights(features)

    #l = LeafNode( features, weights, base_points_)
    #LTy = base_points.L.dot( np.array( [ weights[c] for c in base_points.combinations ] ) ).transpose() 
    #l.lasso.fit( features, LTy, sample_weight = weights[()])
 
    #print (base_points.Linv.dot(l.lasso.intercept_.reshape(-1,1)))
    #print (base_points.Linv.dot(l.lasso.coef_))

    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    y[y==2]=1

    lr = linear_model.LogisticRegression(penalty='l1', C=100., solver='liblinear').fit(X,y)
    lr2 = linear_model.LogisticRegression(penalty='l1', C=100., solver='liblinear')
    lr2.classes_ = lr.classes_
    lr2.coef_ = 0.5*np.ones_like(lr.coef_)
    lr2.intercept_ = lr.intercept_
