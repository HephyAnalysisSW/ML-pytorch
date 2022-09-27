from sklearn import linear_model

import functools
import numpy as np
import operator

default_cfg = {
    "C":             1000.,
    #"init":          "axisaligned",
    "max_depth":     3,
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

        self.LogisticRegression = linear_model.LogisticRegression(penalty='l1', solver='liblinear', C=self.cfg['C'])

        # root node    
        self.nodes = [LeafNode.LeafNode(features, weights, base_points, **self.leaf_node_cfg)]
        self.nodes[-1].depth=0
   
        #if self.cfg["init"]=="axisaligned":
        # split all nodes and replace with parent and right/left children 
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

    def print_tree( self ):
        self.root.print_tree() 

    def fit_iteration( self ):
       
        for node in self.nodes:
            if type( node ) == LeafNode.LeafNode:
                node.fit() 
        
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

    #from sklearn.datasets import load_iris
    #X, y = load_iris(return_X_y=True)
    #y[y==2]=1

    #lr = linear_model.LogisticRegression(penalty='l1', C=100., solver='liblinear').fit(X,y)
    #lr2 = linear_model.LogisticRegression(penalty='l1', C=100., solver='liblinear')
    #lr2.classes_ = lr.classes_
    #lr2.coef_ = 0.5*np.ones_like(lr.coef_)
    #lr2.intercept_ = lr.intercept_
