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

import sys
sys.path.append('..')

import LeafNode
import DecisionNode
import BasePoints 
import NodeBase
import helpers

import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--logLevel',           action='store',      default='INFO',          nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], help="Log level for logging")
args = argParser.parse_args()

#Logger
import tools.logger as logger
logger = logger.get_logger(args.logLevel, logFile = None )

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
        self.nodes = [ LeafNode.LeafNode.root(features, weights, base_points, **self.leaf_node_cfg)]
   
        # Initialization: split all nodes and replace with parent and right/left children 
        #if self.cfg["init"]=="axisaligned":
        for depth in range(0, self.cfg['max_depth'] ):
            new_elements = []
            discard = []
            for node in self.nodes:
                if type(node)==LeafNode.LeafNode:
                    parent       = node.split_even()
                    if parent is not None:
                        if hasattr(node, "parent"):
                            parent.parent=node.parent
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

    @property
    def root( self ):
        return self.nodes[0]

    @property
    def leafnodes(self):
        return [ n for n in self.nodes if type(n)==LeafNode.LeafNode]

    def print_tree( self ):
        self.root.print_tree() 

    def fitall( self ):
        if self.cfg["strategy"] == "rBFS":       
        
            # fit all leaf nodes
            logger.info ("Fitting all leaf nodes...")
            n_leaf_nodes=0
            for node in self.nodes:
                if type( node ) == LeafNode.LeafNode:
                    node.fit( self.lasso )
                    n_leaf_nodes+=1
                    logger.debug ("Fit LeafNode %r: w0: %s w1: %s", node, str(np.round(node.w0.tolist(),2)).replace("\n", ""), str(np.round(node.w1.tolist(),2)).replace("\n", ""))
            logger.info ("Fitted %i all leaf nodes." % n_leaf_nodes)

            # fit all nodes
            for depth in reversed(range(self.cfg['max_depth'])):
                for node in filter( lambda node_: node_.depth==depth and type(node_)==DecisionNode.DecisionNode, self.nodes ):
                    logger.info ("At depth %i: Working on Node %r"%(depth, node))
                    node.fit( self.log_reg )
        else:
            raise NotImplementedError
        

    def refill( self ):
        for node in self.nodes:
            node.indices = []
        for index, node in enumerate( self.root.get_leaf_node(self.root.features) ):
            node.indices.append(index)

        # prune
    def prune (self ):
        while True:
            changed         = False
            to_be_skipped   = []
            for node in self.nodes:
                # find DecisionNodes with empty dauther-Leafnodes
                if type(node)==DecisionNode.DecisionNode:
                    if  type(node.left)==LeafNode.LeafNode and len(node.left.indices)==0:
                        changed = True
                        # two nodes have to be removed: the empty leaf (we recall that) and the current node 
                        to_be_skipped.append( node.left )
                        to_be_skipped.append( node )
                        helpers.reduce_depth( node.right )                

                        # determine where the current node comes from
                        print (node)
                        if hasattr( node.parent, "left") and node.parent.left==node:
                            node.parent.left  = node.right
                            node.right.parent = node.parent
                        if hasattr( node.parent, "right") and node.parent.right==node:
                            node.parent.right = node.right
                            node.right.parent = node.parent
 
                    elif type(node.right)==LeafNode.LeafNode and len(node.right.indices)==0:
                        changed = True
                        # two nodes have to be removed: the empty leaf (we recall that) and the current node 
                        to_be_skipped.append( node.right )
                        to_be_skipped.append( node )
                        helpers.reduce_depth( node.left )                
                            
                        # determine where the current node comes from
                        print (node)
                        if hasattr( node.parent, "left") and node.parent.left==node:
                            node.parent.left  = node.left
                            node.left.parent = node.parent
                        if hasattr( node.parent, "right") and node.parent.right==node:
                            node.parent.right = node.left
                            node.left.parent = node.parent

            if not changed: break
            self.nodes = [ n for n in self.nodes if n not in to_be_skipped ]

    def predict( self, features ):
        return self.root.predict( features=features )
 
if __name__ == "__main__":
    import itertools
    import training_plot
    import sys
    sys.path.append('..')

    import toy_models.sine as model
    N_events_requested=10000

    coefficients = ['theta1']
    base_points_ = []
    for comb in list(itertools.combinations_with_replacement(coefficients,1))+list(itertools.combinations_with_replacement(coefficients,2)):
        base_points_.append( {c:comb.count(c) for c in coefficients} )

    base_points = BasePoints.BasePoints( coefficients, base_points_ )

    training_features = model.getEvents(N_events_requested)
    training_weights  = model.getWeights(training_features)

    test_features = model.getEvents(N_events_requested)
    test_weights  = model.getWeights(test_features)

    d = DecisionTree( training_features, training_weights, base_points )

    iteration = 0
    while True:
        print ("At iteration %i" % iteration)
        logger.info("Before fit")
        d.print_tree()
        logger.info("Now fitting.")
        d.fitall()
        logger.info("After fit.")
        d.print_tree()
        logger.info("Now filling." )
        d.refill()
        logger.info("After fill.")
        d.print_tree()
        logger.info("Now pruning")
        d.prune()
        logger.info("After pruning = before the fit")
        iteration+=1
        break
       
    training_plot( model, plot_directory, training_features, training_weights, test_features, test_weights, label = None) 
