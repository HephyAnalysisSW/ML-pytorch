from sklearn.utils.extmath import safe_sparse_dot
import numpy as np

class DecisionNode:

    def __init__( self, indices ):
        self.indices = indices
        self.left, self.right = None, None 

    def print_tree(self, _depth=0):
        print('%sDecisionNode' % (_depth* ' ') ) 
        for node in [self.left, self.right]:
            node.print_tree(_depth = _depth+1)

    def fit( self, log_reg ):
        self._indices = np.concatenate((self.left._indices, self.right._indices))
        self.features = np.concatenate((self.left.features, self.right.features))
        self.weights  = {k:np.concatenate((self.left.weights[k], self.right.weights[k])) for k in self.left.weights.keys()}

        delta_loss  = self.left.loss( self.features, weights = self.weights) - self.right.loss( self.features, weights = self.weights)
        # best loss

        y_target_new = np.sign(delta_loss).astype('int')
        y_weight     = np.abs( delta_loss )

        res = log_reg.fit( self.features, y_target_new, sample_weight = y_weight )

        self.coef_      = res.coef_.T
        self.intercept_ = res.intercept_
        self.classes_   = res.classes_

    def loss( self, features, weights):
        pass    

    def predict( self, features ):
        # predict
        scores = safe_sparse_dot(self.features, self.coef_, dense_output=True) + self.intercept_
        indices = (scores > 0).astype(int)
        return self.classes_[indices]
