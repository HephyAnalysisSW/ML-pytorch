from sklearn import linear_model
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

import functools
import numpy as np
import operator

# alpha: L1 regularization strength
# alpha is 1/(2C) in other regression models: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso.score

alpha_default = 0.01

default_cfg = {
    "max_depth":     3,
   #"initialization": "axisaligned", # nothing else atm
    "strategy":     "rBFS", # or "BFS"
}

import sys
sys.path.insert(0, '..')

if __name__=="__main__":
    import argparse
    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--logLevel',           action='store',      default='INFO',          nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], help="Log level for logging")
    argParser.add_argument("--prefix",             action="store",      default="",              type=str,  help="prefix")
    argParser.add_argument("--model",              action="store",      default="sine",                 help="Which model?")
    argParser.add_argument("--alpha",              action="store",      default=0.01,       type=float,       help="Regularization alpha")
    argParser.add_argument("--max_iter",           action="store",      default=500,       type=int,       help="Max iterations for logistic regression")
    argParser.add_argument("--modelFile",          action="store",      default="toy_models",                 help="Which model directory?")
    args, extra = argParser.parse_known_args(sys.argv[1:])

    def parse_value( s ):
        try:
            r = int( s )
        except ValueError:
            try:
                r = float(s)
            except ValueError:
                r = s
        return r

    extra_args = {}
    key        = None
    for arg in extra:
        if arg.startswith('--'):
            # previous no value? -> Interpret as flag
            #if key is not None and extra_args[key] is None:
            #    extra_args[key]=True
            key = arg.lstrip('-')
            extra_args[key] = True # without values, interpret as flag
            continue
        else:
            if type(extra_args[key])==type([]):
                extra_args[key].append( parse_value(arg) )
            else:
                extra_args[key] = [parse_value(arg)]
    for key, val in extra_args.items():
        if type(val)==type([]) and len(val)==1:
            extra_args[key]=val[0]

    #Logger
    import tools.logger as logger_
    logger = logger_.get_logger(args.logLevel, logFile = None )
else:
    import logging
    logger = logging.getLogger('ML')
    #print ("importing logger!! decisiontree", __name__,"level", logger.level)

import tools.syncer as syncer
import LeafNode
import DecisionNode
import BasePoints 
import NodeBase
import BoostedTreeAO.helpers as helpers

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

    def fit_nodes( self, lasso, log_reg):
        if self.cfg["strategy"] == "rBFS":       
        
            # fit all leaf nodes
            logger.debug ("Fitting all leaf nodes...")
            n_leaf_nodes=0
            for node in self.nodes:
                if type( node ) == LeafNode.LeafNode:
                    node.fit( lasso )
                    n_leaf_nodes+=1
                    logger.debug ("Fit LeafNode %r: w0: %s w1: %s", node, str(np.round(node.w0.tolist(),4)).replace("\n", ""), str(np.round(node.w1.tolist(),4)).replace("\n", ""))
            logger.debug ("Fitted %i all leaf nodes." % n_leaf_nodes)

            # fit all nodes
            for depth in reversed(range(self.cfg['max_depth'])):
                for node in filter( lambda node_: node_.depth==depth and type(node_)==DecisionNode.DecisionNode, self.nodes ):
                    logger.debug ("At depth %i: Working on Node %r"%(depth, node))
                    node.fit(log_reg )
        else:
            raise NotImplementedError
        
    def refill( self ):
        for node in self.nodes:
            node.indices = []

        if type(self.root)==DecisionNode.DecisionNode:
            # filling leaf nodes
            for index, node in enumerate( self.root.get_leaf_node(self.root.features) ):
                node.indices.append(index)
        # this is the case where the root note is a leaf node, so all events are in there
        elif type(self.root)==LeafNode.LeafNode:
            node.indices = np.array( range(len(node.features)) ) 
            
        # FIXME: We could fill the indices of the decision nodes


    # pruning the tree
    def prune (self ):
        pruning_iteration = 0
        while True:
            logger.debug( "Pruning iteration %i", pruning_iteration )

            self.print_tree()
            logger.info("Now start to prune")

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
                        try:
                            if hasattr( node, "parent"):
                                if hasattr( node.parent, "left") and node.parent.left==node:
                                    node.parent.left  = node.right
                                    node.right.parent = node.parent
                                if hasattr( node.parent, "right") and node.parent.right==node:
                                    node.parent.right = node.right
                                    node.right.parent = node.parent
                            else:
                                # the right daughter is the new root node
                                del node.right.parent
                        except AttributeError:
                            self.print_tree()
                            print ("problematic node:", node)
                            raise AttributeError
 
                    elif type(node.right)==LeafNode.LeafNode and len(node.right.indices)==0:
                        changed = True
                        # two nodes have to be removed: the empty leaf (we recall that) and the current node 
                        to_be_skipped.append( node.right )
                        to_be_skipped.append( node )
                        helpers.reduce_depth( node.left )                
                            
                        # determine where the current node comes from
                        try:
                            if hasattr( node, "parent" ):
                                if hasattr( node.parent, "left") and node.parent.left==node:
                                    node.parent.left  = node.left
                                    node.left.parent = node.parent
                                if hasattr( node.parent, "right") and node.parent.right==node:
                                    node.parent.right = node.left
                                    node.left.parent = node.parent
                            else:
                                # the right daughter is the new root node
                                del node.left.parent
                        except AttributeError:
                            self.print_tree()
                            print ("problematic node:", node)
                            raise AttributeError

            if not changed: break
            self.nodes = [ n for n in self.nodes if n not in to_be_skipped ]

    # remove decorator for debugging
    @ignore_warnings(category=ConvergenceWarning)
    def fit( self, lasso, log_reg, max_iteration = 5):
        iteration = 0
        while iteration<max_iteration:
            logger.debug("At iteration %i" % iteration)
            logger.debug("Before fit")
            if logger.level<=logging.DEBUG: self.print_tree()
            logger.debug("Now fitting.")
            self.fit_nodes(lasso, log_reg)
            logger.debug("After fit.")
            if logger.level<=logging.DEBUG: self.print_tree()
            logger.debug("Now filling." )
            self.refill()
            logger.debug("After fill.")
            if logger.level<=logging.DEBUG: self.print_tree()
            logger.debug("Now pruning")
            self.prune()
            logger.debug("After pruning = before the fit")
            iteration+=1

    def predict( self, features ):
        return self.root.predict( features=features )
 
if __name__ == "__main__":

    import itertools
    import training_plot

    import user
    import os

    exec('import %s.%s as model'%(args.modelFile, args.model))

    N_events_requested = 50000

    coefficients = model.wilson_coefficients
 
    base_points_ = []
    for comb in list(itertools.combinations_with_replacement(coefficients,1))+list(itertools.combinations_with_replacement(coefficients,2)):
        base_points_.append( {c:comb.count(c) for c in coefficients} )

    base_points = BasePoints.BasePoints( coefficients, base_points_ )

    training_features = model.getEvents(N_events_requested)
    training_weights  = model.getWeights(training_features)

    test_features = model.getEvents(N_events_requested)
    test_weights  = model.getWeights(test_features)

    d = DecisionTree( training_features, training_weights, base_points, **extra_args)

    # classifier for DecisionNode
    log_reg = linear_model.LogisticRegression(penalty='l1', solver='liblinear', C=2./args.alpha, max_iter=args.max_iter)
    # regressor for LeafNode
    lasso   = linear_model.Lasso(alpha=args.alpha, fit_intercept=True)

    max_iteration=25
    iteration = 0
    while True:
        print ("At iteration %i" % iteration)
        logger.info("Before fit")
        d.print_tree()
        logger.info("Now fitting.")
        d.fit_nodes(lasso, log_reg)
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

        plot_directory = os.path.join( user.plot_directory, args.prefix, args.model, "alpha_%3.2E"%args.alpha )
        training_plot.training_plot( model, plot_directory, 
            features        = test_features, 
            weights         = test_weights, 
            predictions     = d.predict(test_features), 
            label = "iter_%05i"%iteration,
            )

        if iteration>max_iteration:
            break 

    syncer.sync()
