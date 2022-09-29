from sklearn import linear_model

import functools
import numpy as np
import operator

default_cfg = {
    "max_depth":     3,
    "C":             0.01,     # LogisticRegression
    "alpha":         0.0001,    # Lasso
   #"initialization": "axisaligned", # nothing else atm
    "strategy":     "rBFS", # or "BFS"
    "max_iter":     500,
}

import LeafNode
import DecisionNode
import BasePoints 

class DecisionTree:

    def __init__( self, features, weights,  base_points, **kwargs):

        self.cfg = default_cfg
        self.leaf_node_cfg = LeafNode.default_cfg
        for (key, val) in kwargs.items():
            if key in default_cfg.keys():
                self.cfg[key]      = val
            elif key in self.leaf_node_cfg:
                self.leaf_node_cfg[key] = val 
            else:
                raise RuntimeError( "Got unexpected keyword arg: %s:%r" %( key, val ) )

        # classifier for DecisionNode
        self.log_reg = linear_model.LogisticRegression(penalty='l1', solver='liblinear', C=self.cfg['C'], max_iter=self.cfg['max_iter'])
        # regressor for LeafNode
        self.lasso   = linear_model.Lasso(alpha=self.cfg['alpha'], fit_intercept=True)

        # root node    
        self.nodes = [LeafNode.LeafNode(features, weights, base_points, **self.leaf_node_cfg)]
        self.nodes[-1].depth=0
   
        # Initialization: split all nodes and replace with parent and right/left children 
        #if self.cfg["init"]=="axisaligned":
        for depth in range(0, self.cfg['max_depth'] ):
            new_elements = []
            discard = []
            for node in self.nodes:
                if type(node)==LeafNode.LeafNode:
                    parent       = node.split_even()
                    if parent is not None:
                        if node.depth>0:
                            if node.parent.left == node:
                                node.parent.left = parent
                            if node.parent.right == node:
                                node.parent.right = parent

                        parent.depth = node.depth

                        parent.right.depth=node.depth+1
                        parent.left.depth=node.depth+1

                        new_elements += [parent, parent.left, parent.right]
                        discard.append( node )

            for node in discard:
                self.nodes.remove(node)
                del node

            self.nodes += new_elements

        self.root = self.nodes[0]

    @property
    def leafnodes(self):
        return [ n for n in self.nodes if type(n)==LeafNode.LeafNode]

    def print_tree( self ):
        self.root.print_tree() 

    def fit_iteration( self ):

        if self.cfg["strategy"] == "rBFS":       
        
            # fit all leaf nodes
            print ("Fitting all leaf nodes...")
            n_leaf_nodes=0
            for node in self.nodes:
                if type( node ) == LeafNode.LeafNode:
                    node.fit( self.lasso )
                    n_leaf_nodes+=1
            print ("Fitted %i all leaf nodes." % n_leaf_nodes)

            # fit all nodes
            for depth in reversed(range(self.cfg['max_depth'])):
                for node in filter( lambda node_: node_.depth==depth and type(node_)==DecisionNode.DecisionNode, self.nodes ):
                    print ("At depth %i: Working on Node %r"%(depth, node))
                    node.fit( self.log_reg )

    def predict( self, features ):
        return self.root.predict( features )
 
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

    base_points = BasePoints.BasePoints( coefficients, base_points_ )

    features = model.getEvents(N_events_requested)
    weights  = model.getWeights(features)

    d = DecisionTree( features, weights, base_points )

    d.fit_iteration()
    # try to fit this node
    node = d.nodes[-3]

    _indices = np.concatenate((node.left._indices, node.right._indices))
    features = np.concatenate((node.left.features, node.right.features))
    weights  = {k:np.concatenate((node.left.weights[k], node.right.weights[k])) for k in node.left.weights.keys()}

    delta_loss  = node.left.loss( features, weights = weights) - node.right.loss( features, weights = weights)
    # best loss

    y_target_new = np.sign(delta_loss).astype('int')
    y_weight     = np.abs( delta_loss )
    
    res = d.log_reg.fit( features, y_target_new, sample_weight = y_weight ) 

    # predict
    from sklearn.utils.extmath import safe_sparse_dot
    scores = safe_sparse_dot(features, res.coef_.T, dense_output=True) + res.intercept_
    indices = (scores > 0).astype(int)
    pred = res.classes_[indices]

    #loss_terms_left =

    #for C in [0.001, 0.01, 0.1, 1, 10, 100, 100]:
    #    for i in range(100):
    #        print (i, "C",C)
    #        lr = linear_model.LogisticRegression(penalty='l1', solver='liblinear', C=C, max_iter=500)
    #        node.fit(lr)
    #        print (node.coef_, node.intercept_) 
