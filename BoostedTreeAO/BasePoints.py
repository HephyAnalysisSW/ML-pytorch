import functools
import numpy as np
import operator
import itertools

class BasePoints:

    def __init__( self, coefficients, base_points, max_order = 2):

        # dictionary with base points
        self.base_points = base_points

        self.coefficients = coefficients 
        self.coefficients.sort()

        # Assume missing coefficients are actually zero
        for base_point in self.base_points:
            for coeff in self.coefficients:
                if not coeff in base_point:
                    base_point[coeff] = 0.

        self.combinations = sum( [ list(itertools.combinations_with_replacement(coefficients,order)) for order in range(1, max_order+1) ], [] )

        # sum_base-points ( theta_comb1 * theta_comb2 ) where comb1, comb2 are the matrix indices
        self.base_point_matrix = np.sum( [ np.array( [[ 
                functools.reduce(operator.mul, [bp[c] for c in comb1])*functools.reduce(operator.mul, [bp[c] for c in comb2])/(len(comb1)*len(comb2)) 
                for comb1 in self.combinations] for comb2 in self.combinations] ) for bp in self.base_points ], axis=0 )

        # Cholesky decomposition of bas point matrix 
        self.L    = np.linalg.cholesky(self.base_point_matrix)
        self.Linv = np.linalg.inv(self.L)
