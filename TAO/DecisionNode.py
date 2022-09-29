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

    # pass down 
    def loss( self, features, weights):
        pred = self.predict_node( features ).reshape(-1)
        loss = np.zeros_like(pred).astype('float')
        loss[pred==-1] = self.left.loss(features[pred==-1], {k:weights[k][pred==-1] for k in weights.keys()})
        loss[pred==+1] = self.left.loss(features[pred==+1], {k:weights[k][pred==+1] for k in weights.keys()})

        return loss

    def predict( self, features ):
        pred = self.predict_node( features ).reshape(-1)
        res_left  = self.left.predict(features[pred==-1])
        res_right = self.left.predict(features[pred==+1])

        if len(res_left)==0 and not len(res_right)==0:
            return res_right
        elif len(res_right)==0 and not len(res_left)==0:
            return res_left
        elif len(res_right)==len(res_left)==0:
            return np.array([]).astype('float')
        
        res = np.zeros((len(features), len(res_left[0]))).astype('float')
        res[pred==-1]=res_left
        res[pred==+1]=res_right

        return res

    def predict_node( self, features ):
        # predict
        scores = safe_sparse_dot(features, self.coef_, dense_output=True) + self.intercept_
        indices = (scores > 0).astype(int)
        return self.classes_[indices]
