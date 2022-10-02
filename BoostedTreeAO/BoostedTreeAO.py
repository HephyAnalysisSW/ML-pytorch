from sklearn import linear_model

import functools
import numpy as np
import operator
import copy
import time

# alpha: L1 regularization strength
# alpha is 1/(2C) in other regression models: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso.score
alpha_def = 0.01

default_cfg = {
    "learning_rate": 0.8,
    "max_depth":     3,
    "n_trees":      10, # number of trees in boosting
    "C":             2./alpha_def,     # LogisticRegression
    "alpha":         alpha_def,    # Lasso
   #"initialization": "axisaligned", # nothing else atm
    "strategy":     "rBFS", # or "BFS" (not implemented yet)
    "max_iter":     500, # for logistic regression
    "min_size" :    50,

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
    argParser.add_argument("--n_trees",            action="store",      default=default_cfg['n_trees'],       type=int,       help="Maximum number of trees")
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

import tools.syncer as syncer

import LeafNode
import DecisionNode
import DecisionTree
import BasePoints 
import NodeBase
import BoostedTreeAO.helpers as helpers

class BoostedTreeAO:

    def __init__( self, features, weights,  base_points, **kwargs):

        self.cfg = default_cfg
        self.tree_cfg = DecisionTree.default_cfg
        for (key, val) in kwargs.items():
            if key in default_cfg.keys():
                self.cfg[key]      = val
            elif key in self.tree_cfg or key in LeafNode.default_cfg:
                self.tree_cfg[key] = val 
            else:
                raise RuntimeError( "Got unexpected keyword arg: %s:%r" %( key, val ) )

        # classifier for DecisionNode
        self.log_reg = linear_model.LogisticRegression(penalty='l1', solver='liblinear', C=self.cfg['C'], max_iter=self.cfg['max_iter'])
        # regressor for LeafNode
        self.lasso   = linear_model.Lasso(alpha=self.cfg['alpha'], fit_intercept=True)

        # Will hold the trees
        self.trees    = []

        self.weights     = copy.deepcopy(weights)
        self.features    = features
        self.base_points = base_points

    @classmethod
    def load(cls, filename):
        with open(filename,'rb') as file_:
            old_instance = pickle.load(file_)
            new_instance = cls( None, None,
                    n_trees = old_instance.n_trees,
                    learning_rate = old_instance.learning_rate,
                    )
            new_instance.trees = old_instance.trees

            return new_instance

    def __setstate__(self, state):
        self.__dict__ = state

    # save class, but don't save the sklearn objects
    def save(self, filename):
        tmp_log_reg = self.log_reg
        tmp_lasso   = self.lasso
        del self.log_reg
        del self.lasso

        with open(filename,'wb') as file_:
            pickle.dump( self, file_ )

        self.lasso   = tmp_lasso
        self.log_reg = tmp_log_reg

    def boost( self ):

        toolbar_width = min(20, self.cfg['n_trees'])

        # setup toolbar
        sys.stdout.write("[%s]" % (" " * toolbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['

        weak_learner_time = 0.0
        update_time = 0.0
        for n_tree in range(self.cfg['n_trees']):

            training_time = 0

            # fit to data
            time1 = time.process_time()
            tree = DecisionTree.DecisionTree(
                        features    = self.features,
                        weights     = self.weights,
                        base_points = self.base_points,
                            **self.tree_cfg
                        )
            tree.fit(self.lasso, self.log_reg)

            time2 = time.process_time()
            weak_learner_time += time2 - time1
            training_time      = time2 - time1

            self.trees.append( tree )

            # Recall current tree
            time1 = time.process_time()

            prediction   = tree.predict(self.features)
            #len_         = len(prediction)
            #print (self.weights)
            #print (prediction)
            delta_weight = self.weights[tuple()].reshape(-1,1)*prediction
            for i_der, der in enumerate(self.base_points.combinations):
                self.weights[der] += -self.cfg['learning_rate']*delta_weight[:,i_der]

            time2 = time.process_time()
            update_time   += time2 - time1
            training_time += time2 - time1

            self.trees[-1].training_time = training_time

            # update the bar
            if self.cfg['n_trees']>=toolbar_width:
                if n_tree % (self.cfg['n_trees']/toolbar_width)==0:   sys.stdout.write("-")
            sys.stdout.flush()

        sys.stdout.write("]\n") # this ends the progress bar
        logger.info ("weak learner time: %.2f" % weak_learner_time)
        logger.info ("update time: %.2f" % update_time)

        # purge training data
        del self.weights
        del self.features

    ## BELOW is the non-vectorized predict function. Maybe not needed.
    #def predict( self, feature_array, max_n_tree = None, summed = True, last_tree_counts_full = False):
    #    # list learning rates
    #    learning_rates = self.cfg['learning_rate']*np.ones(max_n_tree if max_n_tree is not None else self.cfg['n_trees'])
    #    # keep the last tree?
    #    if last_tree_counts_full and (max_n_tree is None or max_n_tree==self.cfg['n_trees']):
    #        learning_rates[-1] = 1

    #    predictions = np.array([ tree.predict( feature_array ) for tree in self.trees[:max_n_tree] ])
    #    predictions = predictions[:,1:]/predictions[:,0].reshape(-1,1)
    #    if summed:
    #        return np.dot(learning_rates, predictions)
    #    else:
    #        return learning_rates.reshape(-1, 1)*predictions

    def predict( self, feature_array, max_n_tree = None, summed = True, last_tree_counts_full = False):
        # list learning rates
        learning_rates = self.cfg['learning_rate']*np.ones(max_n_tree if max_n_tree is not None else self.cfg['n_trees'])
        # keep the last tree?
        if last_tree_counts_full and (max_n_tree is None or max_n_tree==self.cfg['n_trees']):
            learning_rates[-1] = 1

        predictions = np.array([ tree.predict( feature_array ) for tree in self.trees[:max_n_tree] ])
        predictions = predictions[:,:,1:]/np.expand_dims(predictions[:,:,0], -1)
        if summed:
            return np.sum(learning_rates.reshape(-1,1,1)*predictions, axis=0)
        else:
            return learning_rates.reshape(-1,1,1)*predictions


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

    b = BoostedTreeAO( training_features, training_weights, base_points, 
        n_trees  = args.n_trees, 
        max_iter = args.max_iter,
        alpha    = args.alpha,
        **extra_args)

    b.boost()

    # debug
    #tree=DecisionTree.DecisionTree(b.features, b.weights, b.base_points)

    #    plot_directory = os.path.join( user.plot_directory, args.prefix, args.model, "alpha_%3.2E"%d.cfg['alpha'] )
    #    training_plot.training_plot( model, plot_directory, 
    #        features        = test_features, 
    #        weights         = test_weights, 
    #        predictions     = d.predict(test_features), 
    #        label = "iter_%05i"%iteration,
    #        )

    #    if iteration>max_iteration:
    #        break 

    #syncer.sync()
