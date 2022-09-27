from sklearn import linear_model
from BasePoints import BasePoints
import numpy as np

default_cfg = {
    "alpha" : 0.000001,
}

class LeafNode:

    def __init__( self, features, weights, base_points, **kwargs):

        self.cfg = default_cfg
        self.cfg.update( kwargs )
        for (key, val) in kwargs.items():
            if key in default_cfg.keys():
                self.cfg[key]      = val
            else:
                raise RuntimeError( "Got unexpected keyword arg: %s:%r" %( key, val ) )

        self.coefficients = list(set(sum( map( list, weights.keys() ), [] )))
        self.coefficients.sort()

        self.features    = features
        self.weights     = weights
        self.base_points = base_points if type(base_points) == BasePoints else BasePoints(coefficients, base_points)

        self.lasso    = linear_model.Lasso(alpha=self.cfg['alpha'], fit_intercept=True)

    def fit( self ):
        self.lasso.fit( self.features, self.base_points.L.dot( np.array( [ weights[c] for c in self.base_points.combinations ] ) ).transpose(), sample_weight = self.weights[()] ) 

        self.w0    = self.base_points.Linv.dot( self.lasso.intercept_.reshape(-1,1) )
        self.w1    = self.base_points.Linv.dot(self.lasso.coef_)

        #print ("Const: %r linear: %r" %(self.w0, self.w1))

    def predict( self, features ):
        return self.base_points.Linv.dot(self.lasso.predict(features).transpose()).transpose()

if __name__ == "__main__":
    import itertools

    import sys
    sys.path.append('..')

    import toy_models.quadratic as model
    N_events_requested=10000

    coefficients = ['theta1']
    base_points_ = []
    for comb in list(itertools.combinations_with_replacement(coefficients,1))+list(itertools.combinations_with_replacement(coefficients,2)):
        base_points_.append( {c:comb.count(c) for c in coefficients} )

    base_points = BasePoints( coefficients, base_points_ )

    features = model.getEvents(N_events_requested)
    weights  = model.getWeights(features)

    l = LeafNode( features, weights, base_points_)
    l.fit()
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
