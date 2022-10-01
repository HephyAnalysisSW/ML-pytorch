from sklearn.utils.extmath import safe_sparse_dot
import numpy as np

from NodeBase import NodeBase
import LeafNode
import helpers

class DecisionNode(NodeBase):

    def __del__(self):
        NodeBase.remove_instance(self)

    def __init__( self ):
        NodeBase.add_instance(self)

    def print_tree(self, _depth=0):
        if hasattr( self, "coef_"):
            fit_str = "w=%s, intercept=%s"%( str( np.round( self.coef_,2).tolist()).replace('\n',''), str( np.round( self.intercept_,2).tolist()).replace('\n','') )
        else:
            fit_str = "(not fit)"
        if hasattr(self, "indices"):
            fit_str = "nEvents: %i "%len(self.indices) + fit_str
        if hasattr(self, "depth"):
            fit_str = " depth: %i "%self.depth + fit_str
        else:
            fit_str = " (no depth) "+fit_str
        print('%sDecisionNode %s' % (_depth* ' ', fit_str) )
        for node in [self.left, self.right]:
            node.print_tree(_depth = _depth+1)

    def fit( self, log_reg ):
        self.indices = np.concatenate((self.left.indices, self.right.indices))

        weights     = helpers.sel( self.weights, self.indices)
        features    = self.features[self.indices]
        delta_loss  = self.left.loss( self.indices ) - self.right.loss( self.indices )
        # best loss

        y_target_new = np.sign(delta_loss).astype('int')
        y_weight     = np.abs( delta_loss )

        res = log_reg.fit( features, y_target_new, sample_weight = y_weight )

        self.coef_      = res.coef_.T
        self.intercept_ = res.intercept_
        self.classes_   = res.classes_
        self.is_updated = True

    # returns the decision on where the events should go (left or right)
    def predict_daughter( self, indices = None, features = None):
        # predict
        if indices is not None and features is None:
            scores = safe_sparse_dot(self.features[indices], self.coef_, dense_output=True) + self.intercept_
            _indices = (scores > 0).astype(int)
            return self.classes_[_indices]
        elif indices is None and features is not None:
            scores = safe_sparse_dot(features, self.coef_, dense_output=True) + self.intercept_
            _indices = (scores > 0).astype(int)
            return self.classes_[_indices]
        else:
            raise RuntimeError ("Need either indices or features but not both (or none of them)" )

    # obtain the value of the loss function by asking the daughters 
    def loss( self, indices):
        pred = self.predict_daughter( indices=indices ).reshape(-1)
        loss = np.zeros_like(pred).astype('float')
        loss[pred==-1] = self.left.loss (indices[pred==-1])
        loss[pred==+1] = self.right.loss(indices[pred==+1])
        return loss

    def predict( self, indices = None, features = None):
        # predict
        if indices is not None and features is None:
            pred = self.predict_daughter( indices=indices ).reshape(-1)
            res_left  = self.left.predict(  indices=indices[pred==-1] )
            res_right = self.right.predict( indices=indices[pred==+1] )
        elif indices is None and features is not None:
            pred = self.predict_daughter( features=features ).reshape(-1)
            res_left  = self.left.predict(  features=features[pred==-1] )
            res_right = self.right.predict( features=features[pred==+1] )
        else:
            raise RuntimeError ("Need either indices or features but not both (or none of them)" )


        if len(res_left)==0 and not len(res_right)==0:
            return res_right
        elif len(res_right)==0 and not len(res_left)==0:
            return res_left
        elif len(res_right)==len(res_left)==0:
            return np.array([]).astype('float')
        else: 
            res = np.zeros((len(features), len(res_left[0]))).astype('float')
            res[pred==-1]=res_left
            res[pred==+1]=res_right
            return res

    def get_leaf_node( self, features ):
        pred = self.predict_daughter( features=features ).reshape(-1)
        res_left  = self.left.get_leaf_node ( features=features[pred==-1] )
        res_right = self.right.get_leaf_node( features=features[pred==+1] )
        if type(res_left)==LeafNode.LeafNode:
            res_left  = [res_left for i in range(np.count_nonzero(pred==-1))]
        if type(res_right)==LeafNode.LeafNode:
            res_right = [res_right for i in range(np.count_nonzero(pred==+1))]

        if len(res_left)==0 and not len(res_right)==0:
            return res_right
        elif len(res_right)==0 and not len(res_left)==0:
            return res_left
        elif len(res_right)==len(res_left)==0:
            return [] 
        else: 
            i_left, i_right = iter(res_left), iter(res_right)
            return [next(i_left) if m==-1 else next(i_right) for m in pred]
