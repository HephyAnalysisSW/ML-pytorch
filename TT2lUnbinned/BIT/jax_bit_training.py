#!/usr/bin/env python

# Standard imports
import ROOT
import numpy as np
import random
import cProfile
import time
import os, sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
from math import log, exp, sin, cos, sqrt, pi
import copy
import pickle
import itertools
from tqdm import tqdm

dir_path = os.path.dirname(os.path.realpath(__file__))
ROOT.gROOT.LoadMacro(os.path.join( dir_path, "../../tools/scripts/tdrstyle.C"))
ROOT.setTDRStyle()

from   tools import helpers
import tools.syncer as syncer

# Old BIT (only for legacy load detection)
from BIT.MultiBoostedInformationTree import MultiBoostedInformationTree

# --- JAX BIT core ---
sys.path.insert(0, os.path.join(dir_path, "../../JAXBIT"))
import jax
import jax.numpy as jnp
import JAXBIT as jb

# User
import tools.user as user

# Parser
import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument("--plot_directory",     action="store",      default="JAXBIT_v1",                 help="plot sub-directory")
argParser.add_argument("--prefix",             action="store",      default=None, type=str,  help="prefix")
argParser.add_argument("--model",              action="store",      default="TT2l_PDF", type=str,  help="model?")
argParser.add_argument("--nTraining",          action="store",      default=-1,        type=int,  help="number of training events")
argParser.add_argument("--plot_iterations",    action="store",      default=None,          nargs="*", type=int, help="Certain iterations to plot? If first iteration is -1, plot only list provided.")
argParser.add_argument('--overwrite',          action='store',      default=None, choices = [None, "training", "data", "all"],  help="Overwrite output?")
argParser.add_argument('--bias',               action='store',      default=None, nargs = "*",  help="Bias training? Example:  --bias 'pT' '10**(({}-200)/200) ")
argParser.add_argument('--debug',              action='store_true', help="Make debug plots?")
argParser.add_argument('--feature_plots',      action='store_true', help="Feature plots?")
argParser.add_argument('--auto_clip',          action='store',      default=None, type=float, help="Remove quantiles of the training variable?")

argParser.add_argument('--top_kinematics',     action='store_true')
argParser.add_argument('--lepton_kinematics',  action='store_true')
argParser.add_argument('--asymmetry',          action='store_true')
argParser.add_argument('--spin_correlation',   action='store_true')

argParser.add_argument("--red",         action="store",      default=-1,        type=int,  help="Reduction facto")
argParser.add_argument('--nJobs',       action='store',         nargs='?',  type=int, default=0,                                    help="Bootstrapping total number" )
argParser.add_argument('--job',         action='store',                     type=int, default=0,                                    help="Bootstrepping iteration" )

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
        key = arg.lstrip('-')
        extra_args[key] = True
        continue
    else:
        if type(extra_args[key])==type([]):
            extra_args[key].append( parse_value(arg) )
        else:
            extra_args[key] = [parse_value(arg)]
for key, val in extra_args.items():
    if type(val)==type([]) and len(val)==1:
        extra_args[key]=val[0]

exec("import data_models.%s as model"%args.model)
from data_models.plot_options import plot_options

model.jax_bit_cfg.update( extra_args )
data_model = model.DataModel(
        top_kinematics      =   args.top_kinematics, 
        lepton_kinematics   =   args.lepton_kinematics, 
        asymmetry           =   args.asymmetry, 
        spin_correlation    =   args.spin_correlation
    )

# directory for plots
plot_directory = os.path.join( user.plot_directory, args.plot_directory, args.model )
os.makedirs( plot_directory, exist_ok=True)

training_data_filename = os.path.join(user.data_directory, args.model, data_model.name, "training_%i"%args.nTraining)+'.pkl'
if args.overwrite in ["all", "data"] or not os.path.exists(training_data_filename):
    training_features, training_weights, training_observers = data_model.getEvents(args.nTraining, return_observers=True)
    print ("Created data set of size %i" % len(training_features) )
    if not os.path.exists(os.path.dirname(training_data_filename)):
        os.makedirs(os.path.dirname(training_data_filename))
    with open( training_data_filename, 'wb' ) as _file:
        pickle.dump( [training_features, training_weights, training_observers], _file )
        print ("Written training data to", training_data_filename)
else:
    with open( training_data_filename, 'rb') as _file:
        training_features, training_weights, training_observers= pickle.load( _file )
        print ("Loaded training data from ", training_data_filename, "with size", len(training_features))

if args.auto_clip is not None:
    len_before = len(training_features)
    training_features, training_weights = helpers.clip_quantile(training_features, args.auto_clip, training_weights )
    print ("Auto clip efficiency (training) %4.3f is %4.3f"%( args.auto_clip, len(training_features)/len_before ) )

# Resample for bootstrapping
if args.nJobs>0:
    from sklearn.utils import resample
    rs_mask = resample(range(training_features.shape[0]))
    training_features = training_features[rs_mask]
    training_weights = {key:val[rs_mask] for key, val in training_weights.items()}
    print("Bootstrapping training data for job %i/%i"%( args.job, args.nJobs) )

# reduce training data 
if args.red>0:
    oldlen_ = training_features.shape[0] 
    len_ = int(training_features.shape[0]/args.red)
    training_features = training_features[:len_]
    training_weights = {key:val[:len_] for key, val in training_weights.items()}
    print("Reducing training from %i to %i"%( oldlen_, len_) )

# Text on the plots
def drawObjects( offset=0 ):
    tex1 = ROOT.TLatex()
    tex1.SetNDC()
    tex1.SetTextSize(0.05)
    tex1.SetTextAlign(11) # align right

    tex2 = ROOT.TLatex()
    tex2.SetNDC()
    tex2.SetTextSize(0.04)
    tex2.SetTextAlign(11) # align right

    line1 = ( 0.15+offset, 0.95, "Boosted Info Trees" )
    return [ tex1.DrawLatex(*line1) ]

###############
## Plot Model #
###############

stuff = []
if args.feature_plots and hasattr( model, "pdf_plot_points"):
    h    = {}
    h_obs= {}
    for i_pdf, pdf_plot_point in enumerate(model.pdf_plot_points):
        pdf = pdf_plot_point['pdf']

        if i_pdf == 0:
            pdf_sm     = pdf

        name = ''
        name= '_'.join( [ (coeff+'_%3.2f'%pdf[coeff]).replace('.','p').replace('-','m') for coeff in model.pdf.variables if coeff in model.pdf.variables ])
        tex_name = pdf_plot_point['tex'] 

        if i_pdf==0: name='SM'

        h[name]     = {}
        h_obs[name] = {}

        pdf['name'] = name
        
        for i_feature, feature in enumerate(data_model.feature_names):
            h[name][feature]        = ROOT.TH1F(name+'_'+feature+'_nom',    name+'_'+feature, *plot_options[feature]['binning'] )
        for i_observer, observer in enumerate(model.observers):
            h_obs[name][observer]    = ROOT.TH1F(name+'_'+observer+'_nom_obs',name+'_'+observer+'_obs', *plot_options[observer]['binning'] )

        # make reweights for x-check
        reweight     = copy.deepcopy(training_weights[()])
        # linear term
        for param1 in model.pdf.variables:
            reweight += (pdf[param1]-pdf_sm[param1])*training_weights[(param1,)] 
        reweight_lin  = copy.deepcopy( reweight )
        # quadratic term
        for param1 in model.pdf.variables:
            if pdf[param1]-pdf_sm[param1] ==0: continue
            for param2 in model.pdf.variables:
                if pdf[param2]-pdf_sm[param2] ==0: continue
                reweight += (.5 if param1!=param2 else 1)*(pdf[param1]-pdf_sm[param1])*(pdf[param2]-pdf_sm[param2])*training_weights[tuple(sorted((param1,param2)))]

        for i_feature, feature in enumerate(data_model.feature_names):
            binning = plot_options[feature]['binning']

            h[name][feature] = helpers.make_TH1F( np.histogram(training_features[:,i_feature], np.linspace(binning[1], binning[2], binning[0]+1), weights=reweight) )

            h[name][feature].SetLineWidth(2)
            h[name][feature].SetLineColor( pdf_plot_point['color'] )
            h[name][feature].SetMarkerStyle(0)
            h[name][feature].SetMarkerColor(pdf_plot_point['color'])
            h[name][feature].legendText = tex_name

        for i_observer, observer in enumerate(model.observers):
            binning = plot_options[observer]['binning']

            h_obs[name][observer] = helpers.make_TH1F( np.histogram(training_observers[:,i_observer], np.linspace(binning[1], binning[2], binning[0]+1), weights=reweight) )

            h_obs[name][observer].SetLineWidth(2)
            h_obs[name][observer].SetLineColor( pdf_plot_point['color'] )
            h_obs[name][observer].SetMarkerStyle(0)
            h_obs[name][observer].SetMarkerColor(pdf_plot_point['color'])
            h_obs[name][observer].legendText = tex_name

    for _h, feature_names, ratio_y, in [  [h_obs, model.observers, (0.8, 1.3)], [h, data_model.feature_names, (0.94, 1.1)] ]:
   
        ratio_y_low, ratio_y_high = ratio_y 
        for i_feature, feature in enumerate(feature_names):

            norm = _h[model.pdf_plot_points[0]['pdf']['name']][feature].Integral()
            if norm>0:
                for pdf_plot_point in model.pdf_plot_points:
                    _h[pdf_plot_point['pdf']['name']][feature].Scale(1./norm) 

            histos = [_h[pdf_plot_point['pdf']['name']][feature] for pdf_plot_point in model.pdf_plot_points]
            max_   = max( map( lambda h__:h__.GetMaximum(), histos ))

            for logY in [True, False]:

                c1 = ROOT.TCanvas("c1");
                l = ROOT.TLegend(0.2,0.68,0.9,0.91)
                l.SetNColumns(2)
                l.SetFillStyle(0)
                l.SetShadowColor(ROOT.kWhite)
                l.SetBorderSize(0)
                for i_histo, histo in enumerate(reversed(histos)):
                    histo.GetXaxis().SetTitle(plot_options[feature]['tex'])
                    histo.GetYaxis().SetTitle("1/#sigma_{SM}d#sigma/d%s"%plot_options[feature]['tex'])
                    if i_histo == 0:
                        histo.Draw('hist')
                        histo.GetYaxis().SetRangeUser( (0.001 if logY else 0), (10*max_ if logY else 1.3*max_))
                        histo.Draw('hist')
                    else:
                        histo.Draw('histsame')
                    l.AddEntry(histo, histo.legendText)
                    c1.SetLogy(logY)
                l.Draw()

                plot_directory_ = os.path.join( plot_directory, "feature_plots", "nTraining_%i"%args.nTraining, "log" if logY else "lin" )
                helpers.copyIndexPHP( plot_directory_ )
                c1.Print( os.path.join( plot_directory_, feature+'.png' ))
                c1.Print( os.path.join( plot_directory_, feature+'.pdf' ))

            # Norm all shapes to 1
            for i_histo, histo in enumerate(histos):
                norm = histo.Integral()
                if norm>0:
                    histo.Scale(1./histo.Integral())

            # Divide all shapes by the SM
            ref = histos[0].Clone()
            for i_histo, histo in enumerate(histos):
                histo.Divide(ref)

            # Now plot shape differences
            for logY in [True, False]:
                c1 = ROOT.TCanvas("c1");
                l = ROOT.TLegend(0.2,0.78,0.9,0.91)
                l.SetNColumns(2)
                l.SetFillStyle(0)
                l.SetShadowColor(ROOT.kWhite)
                l.SetBorderSize(0)

                c1.SetLogy(logY)
                for i_histo, histo in enumerate(reversed(histos)):
                    histo.GetXaxis().SetTitle(plot_options[feature]['tex'])
                    histo.GetYaxis().SetTitle("shape wrt. SM")
                    if i_histo == 0:
                        histo.Draw('hist')
                        histo.GetYaxis().SetRangeUser( (0.01 if logY else ratio_y_low), (10 if logY else ratio_y_high))
                        histo.Draw('hist')
                    else:
                        histo.Draw('histsame')
                    l.AddEntry(histo, histo.legendText)
                    c1.SetLogy(logY)
                l.Draw()

                plot_directory_ = os.path.join( plot_directory, "shape_plots", "nTraining_%i"%args.nTraining, "log" if logY else "lin" )
                helpers.copyIndexPHP( plot_directory_ )
                c1.Print( os.path.join( plot_directory_, feature+'.png' ))

print ("Done with plots")
syncer.sync()

postfix = ""
if args.nJobs>0:
    postfix += "_resample%05i"%args.job
if args.red>0:
    postfix += "_red%00i"%args.red

base_points = []
for comb in list(itertools.combinations_with_replacement(model.pdf.variables,1))+list(itertools.combinations_with_replacement(model.pdf.variables,2)):
    base_points.append( {c:comb.count(c) for c in model.pdf.variables} )
if args.prefix == None:
    bit_name = "JAXBIT_%s_%s_coeffs_%s_nTraining_%i_nTrees_%i"%(args.model+postfix, data_model.name, "_".join(model.pdf.variables), args.nTraining, model.multi_bit_cfg["n_trees"])
else:
    bit_name = "JAXBIT_%s_%s_%s_coeffs_%s_nTraining_%i_nTrees_%i"%(args.model+postfix, data_model.name, args.prefix, "_".join(model.pdf.variables), args.nTraining, model.multi_bit_cfg["n_trees"])

filename = os.path.join(user.model_directory, bit_name)+'.pkl'

# --------------------------------------------------------------------
# JAX BIT TRAINING + RESUME (compatible with --overwrite training)
# --------------------------------------------------------------------

# Build config for JAXBIT
cfg = dict(
    n_trees            = model.jax_bit_cfg["n_trees"],
    learning_rate      = model.jax_bit_cfg["learning_rate"],
    max_depth          = model.jax_bit_cfg["max_depth"],
    min_size           = model.jax_bit_cfg["min_size"],
    positive           = model.jax_bit_cfg.get("positive", False),
    loss               = model.jax_bit_cfg.get("loss", "MSE"),  # 'MSE' or 'CrossEntropy'
    learn_global_score = model.jax_bit_cfg.get("learn_global_score", False),
    max_n_split        = model.jax_bit_cfg.get("max_n_split", -1),  # -1 full search; >=2 -> subsample

)

# Ensure weight keys are normalized (sorted tuples)
training_weights = {tuple(sorted(k)) if isinstance(k, tuple) else k: v for k, v in training_weights.items()}

# Derivatives come from model.pdf.variables (not from what happens to be present in weights)
derivatives = tuple(jb.build_derivatives(list(model.pdf.variables)))

# Base-point matrices
base_mat_jax     = jb.base_point_matrix(base_points, list(derivatives))
base_mat_pos_jax = jb.base_point_matrix_for_pos(base_mat_jax)

# Helper: pack dict of weights into (N, M) matrix matching `derivatives`
def pack_W(Wdict, derivatives):
    w0 = Wdict[()].reshape(-1, 1)
    cols = [w0]
    for der in derivatives[1:]:
        arr = Wdict.get(der, Wdict.get(tuple(reversed(der))))
        cols.append(arr.reshape(-1, 1))
    return jnp.concatenate(cols, axis=1)

# Thin facade so the plotting section can call vectorized_predict like before
class BITState:
    def __init__(self, trees, derivatives, cfg):
        self.trees = trees
        self.derivatives = derivatives[1:]
        self.n_trees = len(trees)
        self.learning_rate = cfg["learning_rate"]
        self.cfg = cfg

    def vectorized_predict(self, X, max_n_tree=None, summed=True, last_tree_counts_full=False):
        n_used = self.n_trees if max_n_tree is None else min(max_n_tree, self.n_trees)
        if n_used == 0:
            return np.zeros((X.shape[0], len(self.derivatives)), dtype=float)
        lr = np.full(n_used, self.learning_rate, dtype=float)
        if last_tree_counts_full and (max_n_tree is None or max_n_tree == self.n_trees):
            lr[-1] = 1.0
        if self.cfg.get("learn_global_score", False) and n_used>0:
            lr[0] = 1.0
        preds = []
        Xj = jnp.asarray(X)
        for t in self.trees[:n_used]:
            p = jb.predict_tree(t, Xj)
            p = np.asarray(p)
            p = p[:, 1:] / p[:, [0]]
            preds.append(p)
        pred_stack = np.stack(preds, axis=0)
        if summed:
            return np.tensordot(lr, pred_stack, axes=(0, 0))
        else:
            return (lr[:, None, None] * pred_stack)

# Checkpoint helpers (use same filename)
def save_checkpoint(path, trees, derivatives, cfg, n_trained):
    trees_np = []
    for t in trees:
        trees_np.append({
            "split_feat":  np.asarray(t.split_feat),
            "threshold":   np.asarray(t.threshold),
            "left":        np.asarray(t.left),
            "right":       np.asarray(t.right),
            "is_leaf":     np.asarray(t.is_leaf),
            "leaf_value":  np.asarray(t.leaf_value),
        })
    ckpt = dict(
        trees       = trees_np,
        derivatives = list(derivatives),  # list of tuples
        cfg         = cfg,
        n_trained   = n_trained,
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(ckpt, f)
    print("Saved checkpoint:", path)

def load_checkpoint(path):
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    # If legacy object, force retrain
    if isinstance(ckpt, MultiBoostedInformationTree):
        raise ValueError("Legacy MultiBoostedInformationTree pickle found; please run with --overwrite training to retrain JAX trees.")
    trees = []
    for t in ckpt["trees"]:
        trees.append(jb.Tree(
            split_feat = jnp.asarray(t["split_feat"]),
            threshold  = jnp.asarray(t["threshold"]),
            left       = jnp.asarray(t["left"]),
            right      = jnp.asarray(t["right"]),
            is_leaf    = jnp.asarray(t["is_leaf"]),
            leaf_value = jnp.asarray(t["leaf_value"]),
            derivatives= tuple(tuple(x) for x in ckpt["derivatives"]),
        ))
    return dict(
        trees       = trees,
        derivatives = tuple(tuple(x) for x in ckpt["derivatives"]),
        cfg         = ckpt["cfg"],
        n_trained   = ckpt["n_trained"],
    )

# Overwrite training -> start fresh (delete any existing checkpoint)
if args.overwrite in ["all", "training"]:
    if os.path.exists(filename):
        try:
            os.remove(filename)
            print("Removed existing checkpoint:", filename)
        except Exception as e:
            print("Warning: could not remove existing checkpoint:", e)

# Try to resume
trees = []
start_tree = 0
if os.path.exists(filename) and args.overwrite not in ["all", "training"]:
    try:
        print("Resuming from:", filename)
        ckpt = load_checkpoint(filename)
        if tuple(ckpt["derivatives"]) == tuple(derivatives):
            trees = ckpt["trees"]
            start_tree = len(trees)
            print(f"Loaded {start_tree} trees.")
        else:
            print("Derivative set changed; starting fresh.")
    except Exception as e:
        print("No compatible checkpoint found; starting fresh. Reason:", e)

# Working copy of weights (dict of arrays)
W = {k: v.copy() for k, v in training_weights.items()}

# If resuming, replay updates to reconstruct W
if start_tree > 0:
    print(f"Replaying updates for {start_tree} trees to reconstruct weights…")
    lr = cfg["learning_rate"]
    learn_global = cfg.get("learn_global_score", False)
    Xj = jnp.asarray(training_features)
    for i_tree, t in enumerate(trees):
        pred = np.asarray(jb.predict_tree(t, Xj))  # (N, 1+M)
        N = pred.shape[0]
        delta = (W[()].reshape(N, 1) * (pred[:, 1:] / pred[:, [0]]))
        eta = 1.0 if (learn_global and i_tree == 0) else lr
        for i_der, der in enumerate(derivatives[1:]):
            W[der] = W[der] - eta * delta[:, i_der]

remaining  = cfg["n_trees"] - start_tree
N, D = training_features.shape
print("\n[train] plan")
print(f"  X: N={N:,}, D={D}")
print(f"  derivatives: {derivatives}")
print(f"  base_points: {len(base_points)}")
print(f"  loss={cfg['loss']}  positive={cfg['positive']}  max_depth={cfg['max_depth']}  min_size={cfg['min_size']}")
print(f"  max_n_split={cfg['max_n_split']}  learning_rate={cfg['learning_rate']}")
print(f"  resume: {start_tree}/{cfg['n_trees']} trees already trained → training {remaining} more.\n")
Xj = jnp.asarray(training_features)

for n in range(start_tree, cfg["n_trees"]):
    t0 = time.time()

    # Per-tree progress bar (one tick per depth)
    with tqdm(total=cfg["max_depth"] + 1,
              desc=f"Tree {n+1}/{cfg['n_trees']}",
              leave=True,
              dynamic_ncols=True) as pbar:

        def _cb(depth, active_nodes):
            # depth starts at 0; update exactly once per depth
            pbar.set_postfix_str(f"depth={depth} active={active_nodes}")
            pbar.update(1)

        # Build one tree
        tree = jb.build_tree(
            Xj,
            pack_W(W, derivatives),
            base_mat_jax,
            base_mat_pos_jax,
            jb.BuildConfig(
                max_depth   = cfg["max_depth"],
                min_size    = cfg["min_size"],
                positive    = cfg["positive"],
                loss        = cfg["loss"],
                max_n_split = cfg["max_n_split"],   # keep if you added this earlier
            ),
            derivatives,
            progress_cb=_cb,  # <-- tqdm callback
        )
        trees.append(tree)

    # Update weights (on dict W)
    pred = np.asarray(jb.predict_tree(tree, Xj))  # (N, 1+M)
    N = pred.shape[0]
    delta = (W[()].reshape(N, 1) * (pred[:, 1:] / pred[:, [0]]))
    eta = 1.0 if (cfg.get("learn_global_score", False) and n == 0) else cfg["learning_rate"]
    for i_der, der in enumerate(derivatives[1:]):
        W[der] = W[der] - eta * delta[:, i_der]

    # Collect stats for logging
    n_nodes   = int(tree.split_feat.shape[0])
    is_leaf_a = np.asarray(tree.is_leaf)
    n_leaves  = int(is_leaf_a.sum())
    sf        = np.asarray(tree.split_feat)
    used_feats = np.unique(sf[(sf >= 0) & (~is_leaf_a)])
    eta = 1.0 if (cfg.get("learn_global_score", False) and n == 0) else cfg["learning_rate"]

    save_checkpoint(filename, trees, derivatives, cfg, n_trained=n+1)
    print(f"[train] tree {n+1:4d}/{cfg['n_trees']:4d} | leaves={n_leaves:4d} | used_feats={used_feats.tolist()} | eta={eta:g} | {time.time()-t0:.2f}s")

bit = BITState(trees=trees, derivatives=derivatives, cfg=cfg)
print("Training complete. Total trees:", bit.n_trees)

# --------------------------------------------------------------------
# Test data + plots (unchanged)
# --------------------------------------------------------------------

test_data_filename = os.path.join(user.data_directory, args.model, data_model.name, "test_%i"%args.nTraining)+'.pkl'
if args.overwrite in ["all", "data"] or not os.path.exists(test_data_filename):
    test_features, test_weights, test_observers = data_model.getEvents(args.nTraining, return_observers=True)
    print ("Created data set of size %i" % len(test_features) )
    if not os.path.exists(os.path.dirname(test_data_filename)):
        os.makedirs(os.path.dirname(test_data_filename), exist_ok=True)
    with open( test_data_filename, 'wb' ) as _file:
        pickle.dump( [test_features, test_weights, test_observers], _file )
        print ("Written test data to", test_data_filename)
else:
    with open( test_data_filename, 'rb') as _file:
        test_features, test_weights, test_observers = pickle.load( _file )
        print ("Loaded test data from ", test_data_filename, "with size", len(test_features))

if args.auto_clip is not None:
    len_before = len(test_features)
    selected = helpers.clip_quantile(test_features, args.auto_clip, return_selection = True)
    test_features = test_features[selected]
    test_weights = {k:test_weights[k][selected] for k in test_weights.keys()}
    if test_observers.size:
        test_observers = test_observers[selected] 
    print ("Auto clip efficiency (test) %4.3f is %4.3f"%( args.auto_clip, len(test_features)/len_before ) )

# delete coefficients we don't need
if model.pdf.variables is not None:
    for key in list(test_weights.keys()):
        if not all( [k in model.pdf.variables for k in key]):
            del test_weights[key]

if args.debug:

    tex = ROOT.TLatex()
    tex.SetNDC()
    tex.SetTextSize(0.06)

    # colors
    color = {}
    i_lin, i_diag, i_mixed = 0,0,0
    for i_der, der in enumerate(bit.derivatives):
        if len(der)==1:
            color[der] = ROOT.kAzure + i_lin
            i_lin+=1
        elif len(der)==2 and len(set(der))==1:
            color[der] = ROOT.kRed + i_diag
            i_diag+=1
        elif len(der)==2 and len(set(der))==2:
            color[der] = ROOT.kGreen + i_mixed
            i_mixed+=1

    # Which iterations to plot
    plot_iterations = list(range(1,10))+list(range(10,bit.n_trees+1,10))
    if type(args.plot_iterations)==type([]):
        if args.plot_iterations[0]<0:
            plot_iterations+=args.plot_iterations[1:]
        else:
            plot_iterations = args.plot_iterations
        plot_iterations.sort()

    for max_n_tree in plot_iterations:
        if max_n_tree==0: max_n_tree=1
        test_predictions = bit.vectorized_predict(test_features, max_n_tree = max_n_tree)

        w0 = test_weights[()]

        th1d_pred = {}
        th1d_truth= {}
        for i_der, der in enumerate( bit.derivatives ):
            truth_ratio = (test_weights[der] if der in test_weights else test_weights[tuple(reversed(der))])/w0
            quantiles = np.quantile(truth_ratio, q=(0.01,1-0.01))
            if len(der)==2: #quadratic
                binning = np.linspace( min([0, quantiles[0]]), quantiles[1], 21 )
            else:
                binning = np.linspace( quantiles[0], quantiles[1], 21 )
            th1d_truth[der]= helpers.make_TH1F( np.histogram( truth_ratio, bins = binning, weights=w0) )
            th1d_pred[der] = helpers.make_TH1F( np.histogram( test_predictions[:,i_der], bins = binning, weights=w0) )
            tex_name = "%s"%(",".join( der ))
            th1d_pred[der].GetXaxis().SetTitle( tex_name + " prediction" )
            th1d_pred[der].GetYaxis().SetTitle( "Number of Events" )
            th1d_truth[der].GetXaxis().SetTitle( tex_name + " truth" )
            th1d_truth[der].GetYaxis().SetTitle( "Number of Events" )

            th1d_truth[der].SetLineColor(color[der])
            th1d_truth[der].SetMarkerColor(color[der])
            th1d_truth[der].SetMarkerStyle(0)
            th1d_truth[der].SetLineWidth(2)
            th1d_truth[der].SetLineStyle(ROOT.kDashed)
            th1d_pred[der].SetLineColor(color[der])
            th1d_pred[der].SetMarkerColor(color[der])
            th1d_pred[der].SetMarkerStyle(0)
            th1d_pred[der].SetLineWidth(2)

        for observables, features, postfix in [
            ( data_model.feature_names, test_features, ""),
            ]:
            h_w0, h_ratio_prediction, h_ratio_truth, lin_binning = {}, {}, {}, {}
            for i_feature, feature in enumerate(observables):
                binning     = plot_options[feature]['binning']
                lin_binning[feature] = np.linspace(binning[1], binning[2], binning[0]+1)
                binned      = np.digitize(features[:,i_feature], lin_binning[feature] )
                mask        = np.transpose( binned.reshape(-1,1)==range(1,len(lin_binning[feature])) )

                h_w0[feature]           = np.array([  w0[m].sum() for m in mask])
                h_derivative_prediction = np.array([ (w0.reshape(-1,1)*test_predictions)[m].sum(axis=0) for m in mask])
                h_derivative_truth      = np.array([ (np.transpose(np.array([(test_weights[der] if der in test_weights else test_weights[tuple(reversed(der))]) for der in bit.derivatives])))[m].sum(axis=0) for m in mask])

                h_ratio_prediction[feature] = h_derivative_prediction/h_w0[feature].reshape(-1,1) 
                h_ratio_truth[feature]      = h_derivative_truth/h_w0[feature].reshape(-1,1)

            n_pads = len(observables)+1
            n_col  = int(sqrt(n_pads))
            n_rows = n_pads//n_col
            if n_rows*n_col<n_pads: n_rows+=1

            for logY in [False, True]:
                c1 = ROOT.TCanvas("c1","multipads",500*n_col,500*n_rows);
                c1.Divide(n_col,n_rows)

                l = ROOT.TLegend(0.2,0.1,0.9,0.85)
                stuff.append(l)
                l.SetNColumns(2)
                l.SetFillStyle(0)
                l.SetShadowColor(ROOT.kWhite)
                l.SetBorderSize(0)

                for i_feature, feature in enumerate(observables):

                    th1d_yield       = helpers.make_TH1F( (h_w0[feature], lin_binning[feature]) )
                    c1.cd(i_feature+1)
                    ROOT.gStyle.SetOptStat(0)
                    th1d_ratio_pred  = { der: helpers.make_TH1F( (h_ratio_prediction[feature][:,i_der], lin_binning[feature])) for i_der, der in enumerate( bit.derivatives ) }
                    th1d_ratio_truth = { der: helpers.make_TH1F( (h_ratio_truth[feature][:,i_der], lin_binning[feature])) for i_der, der in enumerate( bit.derivatives ) }

                    stuff.append(th1d_yield)
                    stuff.append(th1d_ratio_truth)
                    stuff.append(th1d_ratio_pred)

                    th1d_yield.SetLineColor(ROOT.kGray+2)
                    th1d_yield.SetMarkerColor(ROOT.kGray+2)
                    th1d_yield.SetMarkerStyle(0)
                    th1d_yield.GetXaxis().SetTitle(plot_options[feature]['tex'])
                    th1d_yield.SetTitle("")

                    th1d_yield.Draw("hist")

                    for i_der, der in enumerate(bit.derivatives):
                        th1d_ratio_truth[der].SetTitle("")
                        th1d_ratio_truth[der].SetLineColor(color[der])
                        th1d_ratio_truth[der].SetMarkerColor(color[der])
                        th1d_ratio_truth[der].SetMarkerStyle(0)
                        th1d_ratio_truth[der].SetLineWidth(2)
                        th1d_ratio_truth[der].SetLineStyle(ROOT.kDashed)
                        th1d_ratio_truth[der].GetXaxis().SetTitle(plot_options[feature]['tex'])

                        th1d_ratio_pred[der].SetTitle("")
                        th1d_ratio_pred[der].SetLineColor(color[der])
                        th1d_ratio_pred[der].SetMarkerColor(color[der])
                        th1d_ratio_pred[der].SetMarkerStyle(0)
                        th1d_ratio_pred[der].SetLineWidth(2)
                        th1d_ratio_pred[der].GetXaxis().SetTitle(plot_options[feature]['tex'])

                        tex_name = "%s"%(",".join( der ))
     
                        if i_feature==0:
                            l.AddEntry( th1d_ratio_truth[der], "R("+tex_name+")")
                            l.AddEntry( th1d_ratio_pred[der],  "#hat{R}("+tex_name+")")

                    if i_feature==0:
                        l.AddEntry( th1d_yield, "yield (SM)")

                    max_ = max( map( lambda h:h.GetMaximum(), th1d_ratio_truth.values() ))
                    max_ = 10**(1.5)*max_ if logY else 1.5*max_
                    min_ = min( map( lambda h:h.GetMinimum(), th1d_ratio_truth.values() ))
                    min_ = 0.1 if logY else (1.5*min_ if min_<0 else 0.75*min_)

                    if min_<-0.1:
                        min_= -0.1

                    th1d_yield_min = th1d_yield.GetMinimum()
                    th1d_yield_max = th1d_yield.GetMaximum()
                    for bin_ in range(1, th1d_yield.GetNbinsX()+1 ):
                        th1d_yield.SetBinContent( bin_, (th1d_yield.GetBinContent( bin_ ) - th1d_yield_min)/th1d_yield_max*(max_-min_)*0.95 + min_  )

                    th1d_yield   .Draw("hist")
                    ROOT.gPad.SetLogy(logY)
                    th1d_yield   .GetYaxis().SetRangeUser(min_, max_)
                    th1d_yield   .Draw("hist")
                    for h in list(th1d_ratio_truth.values()) + list(th1d_ratio_pred.values()):
                        h .Draw("hsame")

                c1.cd(len(observables)+1)
                l.Draw()

                lines = [ (0.29, 0.9, 'N_{B} =%5i'%( max_n_tree )) ]
                drawObjects = [ tex.DrawLatex(*line) for line in lines ]
                for o in drawObjects:
                    o.Draw()

                plot_directory_ = os.path.join( plot_directory, "training_plots", bit_name, "log" if logY else "lin" )
                os.makedirs( plot_directory_, exist_ok=True)
                helpers.copyIndexPHP( plot_directory_ )
                c1.Print( os.path.join( plot_directory_, "epoch%s_%05i.png"%(postfix, max_n_tree) ) )
            syncer.sync()

