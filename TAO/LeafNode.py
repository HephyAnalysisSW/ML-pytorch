from sklearn import linear_model
import functools
import numpy as np
import operator

class BasePoints:

    def __init__( self, base_points ):

        # dictionary with base points
        self.base_points = base_points

        self.coefficients = list( set( sum([ list(b.keys()) for b in self.base_points],[])))

        self.coefficients.sort()

        self.combinations = list(itertools.combinations_with_replacement(coefficients,1))+list(itertools.combinations_with_replacement(coefficients,2))

        # sum_base-points ( theta_comb1 * theta_comb2 ) where comb1, comb2 are the matrix indices
        self.base_point_matrix = np.sum( [ np.array( [[ 
                functools.reduce(operator.mul, [bp[c] for c in comb1])*functools.reduce(operator.mul, [bp[c] for c in comb2])/(len(comb1)*len(comb2)) 
                for comb1 in self.combinations] for comb2 in self.combinations] ) for bp in self.base_points ], axis=0 )

        # the matrix in  
        self.L = np.linalg.cholesky(self.base_point_matrix).transpose()

class LeafNode:

    def __init__( self ):

        pass


if __name__ == "__main__":
    import itertools

    import sys
    sys.path.append('..')

    import toy_models.analytic as model
    N_events_requested=10000

    coefficients = ['theta1']
    base_points = []
    for comb in list(itertools.combinations_with_replacement(coefficients,1))+list(itertools.combinations_with_replacement(coefficients,2)):
        base_points.append( {c:comb.count(c) for c in coefficients} )

    b = BasePoints( base_points )

    features = model.getEvents(N_events_requested)
    weights  = model.getWeights(features)
