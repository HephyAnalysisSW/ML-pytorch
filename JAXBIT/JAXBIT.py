# JAXBIT: Level-wise Multi-Boosted Information Trees in JAX
# ---------------------------------------------------------
# - Functional, array-backed trees (no Python recursion)
# - Level-wise split building with jnp operations
# - Works on CPU/GPU; set JAX_PLATFORMS=cpu to force CPU
# - Keeps your base-point projection & derivative-weight logic

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional

import jax
import jax.numpy as jnp

# Enable float64 for stable sums (important with negative/positive weights)
jax.config.update("jax_enable_x64", True)

# -------------------------------------------------------------
# Utilities: base points & derivative ordering
# -------------------------------------------------------------

def build_derivatives(coefficients: List[str]) -> List[Tuple[str, ...]]:
    """Return [(), all first, all second with replacement] in deterministic order."""
    import itertools
    first = sorted(list(itertools.combinations_with_replacement(coefficients, 1)))
    second = sorted(list(itertools.combinations_with_replacement(coefficients, 2)))
    return [tuple()] + first + second


def base_point_matrix(base_points: List[Dict[str, int]], derivatives: List[Tuple[str, ...]] ) -> jnp.ndarray:
    """Construct the base_point_const matrix with the 1/2 factor on diagonal second derivatives.
       Shape: (P, M) where M = 1 + n_first + n_second  (same as your numpy version).
    """
    import numpy as _np
    P = len(base_points)
    M = len(derivatives)
    mat = _np.zeros((P, M), dtype=_np.float64)
    for i_point, point in enumerate(base_points):
        for j_der, der in enumerate(derivatives):
            # product over coefficients (power if present else 0)
            prod = 1.0
            for coeff in der:
                prod *= point.get(coeff, 0)
            mat[i_point, j_der] = prod
    # 1/2 on diagonal second derivatives
    for j_der, der in enumerate(derivatives):
        if len(der) == 2 and der[0] == der[1]:
            mat[:, j_der] *= 0.5
    return jnp.asarray(mat)

# Also construct the positivity-augmented matrix [e0; base_point_const]

def base_point_matrix_for_pos(base_mat: jnp.ndarray) -> jnp.ndarray:
    M = base_mat.shape[1]
    e0 = jnp.zeros((1, M), dtype=base_mat.dtype).at[0, 0].set(1.0)
    return jnp.concatenate([e0, base_mat], axis=0)

# -------------------------------------------------------------
# Data structures for a single tree (Struct-of-arrays)
# -------------------------------------------------------------

@jax.tree_util.register_pytree_node_class
@dataclass
class Tree:
    # Arrays sized by number of nodes in a full binary tree up to max_depth
    # We allocate per depth level; for clarity we keep a flat indexing scheme.
    split_feat: jnp.ndarray   # (num_nodes, ) int64; -1 marks leaf
    threshold: jnp.ndarray    # (num_nodes, ) float64
    left: jnp.ndarray         # (num_nodes, ) int32 indices
    right: jnp.ndarray        # (num_nodes, ) int32 indices
    is_leaf: jnp.ndarray      # (num_nodes, ) bool
    leaf_value: jnp.ndarray   # (num_nodes, M) coefficient sums stored at leaves; only valid if is_leaf
    derivatives: Tuple[Tuple[str, ...], ...]  # kept for reference (static via aux in pytree)

    # JAX pytree protocol
    def tree_flatten(self):
        children = (self.split_feat, self.threshold, self.left, self.right, self.is_leaf, self.leaf_value)
        aux = {"derivatives": self.derivatives}
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        split_feat, threshold, left, right, is_leaf, leaf_value = children
        return cls(split_feat, threshold, left, right, is_leaf, leaf_value, aux["derivatives"])

# -------------------------------------------------------------
# Split search (per node): sort-based version (simple & exact)
# -------------------------------------------------------------
def _best_split_sort_impl(X: jnp.ndarray, W: jnp.ndarray, base_mat: jnp.ndarray,
                          min_size: int, positive: bool, base_mat_pos: jnp.ndarray,
                          loss_flag: int, max_n_split: int) -> Tuple[int, float, float]:
    """Compute best (feature, threshold, gain) for a node.
       If max_n_split >= 2, evaluate at ~max_n_split candidate cuts per feature;
       if max_n_split == -1, evaluate all unique thresholds (full search).
    """
    N, D = X.shape
    M = W.shape[1]
    P = base_mat.shape[0]

    # Handle degenerate cases: return a leaf immediately by signaling feat=-1
    def _degenerate_case():
        return (-1, jnp.inf, -jnp.inf)

    cond = (N < 2 * min_size)
    if cond:
        return _degenerate_case()

    # argsort all features at once: idxs (N, D)
    idxs = jnp.argsort(X, axis=0)
    # Gather sorted X and W per feature
    X_sorted = jnp.take_along_axis(X, idxs, axis=0)                 # (N, D)
    W_rep = jnp.repeat(W[:, None, :], D, axis=1)                    # (N, D, M)
    sorted_W = jnp.take_along_axis(W_rep, idxs[:, :, None], axis=0) # (N, D, M)

    # Prefix sums excluding last for candidate positions 0..N-2
    cums = jnp.cumsum(sorted_W, axis=0)                             # (N, D, M)
    total = cums[-1:, :, :]                                         # (1, D, M)
    left = cums[:-1, :, :]                                          # (N-1, D, M)
    right = total - left                                            # (N-1, D, M)

    # Unique-threshold mask & min_size mask
    diffs = X_sorted[1:, :] - X_sorted[:-1, :]                      # (N-1, D)
    uniq = diffs != 0
    if min_size > 1:
        mask_ms = jnp.zeros_like(uniq)
        mask_ms = mask_ms.at[min_size-1: (N-1)-(min_size-1), :].set(True)
        uniq = jnp.logical_and(uniq, mask_ms)

    def _apply_subsample(uniq_mask):
        # Work on the left-end thresholds X_sorted[:-1, :]
        vals = X_sorted[:-1, :]                        # (N-1, D)
        vmin = X_sorted[0:1, :]                        # (1, D)
        vmax = X_sorted[-1:, :]                        # (1, D)
        span = jnp.maximum(vmax - vmin, 1e-12)

        # Map each candidate value to a bin in [0, max_n_split-1]
        bins = jnp.floor(((vals - vmin) / span) * max_n_split).astype(jnp.int32)
        bins = jnp.clip(bins, 0, jnp.maximum(max_n_split - 1, 0))

        # Keep the first index where bin changes: True when current bin != previous bin
        prev = jnp.concatenate([(-jnp.ones((1, bins.shape[1]), dtype=jnp.int32)), bins[:-1, :]], axis=0)
        grid_mask = bins != prev    # (N-1, D)

        return jnp.logical_and(uniq_mask, grid_mask)

    uniq = jax.lax.cond(max_n_split >= 2, _apply_subsample, lambda m: m, uniq)

    # Never allow negative yields
    denL = jnp.clip(left[:, :, 0], a_min=1e-12)     # base weight sum left
    denR = jnp.clip(right[:, :, 0], a_min=1e-12)
    pos_mask = jnp.logical_and(denL > 0, denR > 0)

    # Positivity constraint (optional): all projections non-negative
    def _apply_positive():
        Lpos = jnp.matmul(left, base_mat_pos.T) >= 0                 # (N-1, D, P+1)
        Rpos = jnp.matmul(right, base_mat_pos.T) >= 0
        ok = jnp.logical_and(jnp.all(Lpos, axis=2), jnp.all(Rpos, axis=2))
        return jnp.logical_and(ok, jnp.logical_and(uniq, pos_mask))

    def _no_positive():
        return jnp.logical_and(uniq, pos_mask)

    valid = jax.lax.cond(positive, _apply_positive, _no_positive)

    # Gains
    if loss_flag == 0:  # MSE
        Lproj = jnp.matmul(left, base_mat.T)                          # (N-1, D, P)
        Rproj = jnp.matmul(right, base_mat.T)
        gains = (jnp.sum(Lproj * Lproj, axis=2) / denL) + (jnp.sum(Rproj * Rproj, axis=2) / denR)
    else:
        # CrossEntropy variant (mirrors your numpy code, reformulated to avoid nan/inf)
        rL = jnp.matmul(left, base_mat.T) / denL[:, :, None]
        rR = jnp.matmul(right, base_mat.T) / denR[:, :, None]
        gl = denL * ( 0.5 * jnp.log(jnp.power(1.0 / (1.0 + rL), 2))
                    + 0.5 * rL * jnp.log(jnp.power(rL / (1.0 + rL), 2)) ).sum(axis=2)
        gr = denR * ( 0.5 * jnp.log(jnp.power(1.0 / (1.0 + rR), 2))
                    + 0.5 * rR * jnp.log(jnp.power(rR / (1.0 + rR), 2)) ).sum(axis=2)
        gains = gl + gr
        # stabilize by subtracting min over valid entries
        min_valid = jnp.where(valid, gains, jnp.inf).min()
        gains = gains - min_valid

    gains = jnp.where(valid, gains, -jnp.inf)

    # Argmax over (N-1, D)
    flat_idx = jnp.argmax(gains.reshape(-1))
    i_split = flat_idx // D
    j_feat = flat_idx % D
    thr = X_sorted[i_split, j_feat]
    best_gain = gains[i_split, j_feat]
    return (j_feat.astype(jnp.int32), thr, best_gain)

_best_split_sort = jax.jit(_best_split_sort_impl, static_argnums=(3, 4, 6, 7))

# -------------------------------------------------------------
# Level-wise tree building
# -------------------------------------------------------------

@dataclass
class BuildConfig:
    max_depth: int = 4
    min_size: int = 50
    positive: bool = False
    loss: str = 'MSE'   # 'MSE' or 'CrossEntropy'
    max_n_split: int = 64  # -1 = full search, >=2 = subsample ~max_n_split cuts/feature

def build_tree(X: jnp.ndarray, W: jnp.ndarray, base_mat: jnp.ndarray,
               base_mat_pos: jnp.ndarray, cfg: BuildConfig,
               derivatives: Tuple[Tuple[str, ...], ...],
               progress_cb: Optional[callable] = None) -> Tree:
    """Build a tree level-wise using JAX ops.
       X: (N, D), W: (N, M)
    """
    N, D = X.shape
    M = W.shape[1]

    # Reserve a full binary tree array up to max_depth: num_nodes = 2^(d+1)-1
    num_nodes = (1 << (cfg.max_depth + 1)) - 1
    split_feat = -jnp.ones((num_nodes,), dtype=jnp.int32)
    threshold = jnp.full((num_nodes,), jnp.inf, dtype=X.dtype)
    left = jnp.zeros((num_nodes,), dtype=jnp.int32)
    right = jnp.zeros((num_nodes,), dtype=jnp.int32)
    is_leaf = jnp.zeros((num_nodes,), dtype=bool)
    leaf_value = jnp.zeros((num_nodes, M), dtype=W.dtype)

    # Work buffers: node-to-sample mapping via boolean masks
    # We keep an index list per node ID in Python for readability here.
    # For big-N production, prefer index arrays kept in device memory.

    # Because JAX-jitted loops dislike Python lists of variable shapes, we
    # perform level-wise in Python, while inner kernels stay jitted.

    # Current frontier: list of (node_id, index_array)
    frontier = [(0, jnp.arange(N, dtype=jnp.int32))]

    for depth in range(cfg.max_depth + 1):
        new_frontier = []
        for (nid, idx) in frontier:
            n_here = idx.size
            if (n_here < 2 * cfg.min_size) or (depth == cfg.max_depth):
                # Make leaf: store coefficient sum
                csum = W[idx].sum(axis=0)
                is_leaf = is_leaf.at[nid].set(True)
                leaf_value = leaf_value.at[nid].set(csum)
                continue

            # Candidate split on the subset
            Xn = X[idx]
            Wn = W[idx]

            loss_flag = 0 if cfg.loss == "MSE" else 1
            f, thr, gain = _best_split_sort(Xn, Wn, base_mat, cfg.min_size,
                                            cfg.positive, base_mat_pos, loss_flag, cfg.max_n_split)

            # If no valid split, become leaf
            if int(f) < 0 or not jnp.isfinite(gain):
                csum = Wn.sum(axis=0)
                is_leaf = is_leaf.at[nid].set(True)
                leaf_value = leaf_value.at[nid].set(csum)
                continue

            # Partition indices
            fm = Xn[:, f]
            left_mask = fm <= thr
            right_mask = jnp.logical_not(left_mask)
            n_left = int(left_mask.sum())
            n_right = int(right_mask.sum())
            if (n_left == 0) or (n_right == 0):
                csum = Wn.sum(axis=0)
                is_leaf = is_leaf.at[nid].set(True)
                leaf_value = leaf_value.at[nid].set(csum)
                continue

            # Record split and children indices (array positions)
            split_feat = split_feat.at[nid].set(f)
            threshold = threshold.at[nid].set(thr)
            lch = 2 * nid + 1
            rch = 2 * nid + 2
            left = left.at[nid].set(lch)
            right = right.at[nid].set(rch)

            # Push children to next frontier
            new_frontier.append((lch, idx[left_mask]))
            new_frontier.append((rch, idx[right_mask]))

        # progress callback (depth-level)
        if progress_cb is not None:
            try:
                progress_cb(depth, len(new_frontier))
            except Exception:
                pass

        frontier = new_frontier
        if not frontier:
            break

    return Tree(split_feat, threshold, left, right, is_leaf, leaf_value, tuple(derivatives))

# -------------------------------------------------------------
# Prediction through a tree (batched)
# -------------------------------------------------------------

@jax.jit
def predict_tree(tree: Tree, X: jnp.ndarray) -> jnp.ndarray:
    """Route all samples through the tree in a fixed-depth loop.
       Returns (N, M) coefficient sums (same shape as stored per leaf).
    """
    N = X.shape[0]
    M = tree.leaf_value.shape[1]

    # node indices for each sample start at root=0
    node_idx = jnp.zeros((N,), dtype=jnp.int32)

    def body_fun(d, node_idx):
        f = tree.split_feat
        thr = tree.threshold
        lch = tree.left
        rch = tree.right
        is_leaf = tree.is_leaf

        # For current node_idx, gather split feature and threshold
        sf = f[node_idx]
        t = thr[node_idx]
        # Decide go_left; for leaves, define go_left True but we will mask later
        # Take feature per-sample (sf may differ per sample)
        feat_vals = X[jnp.arange(N), jnp.clip(sf, 0, X.shape[1]-1)]
        go_left = feat_vals <= t
        next_idx = jnp.where(go_left, lch[node_idx], rch[node_idx])
        # Keep leaves sticky (remain there)
        next_idx = jnp.where(is_leaf[node_idx], node_idx, next_idx)
        return next_idx

    # Iterate up to the depth budget
    import math
    max_steps = math.ceil(math.log2(int(tree.split_feat.shape[0] + 1)))
    node_idx = jax.lax.fori_loop(0, max_steps, body_fun, node_idx)

    # Read out leaf values
    out = tree.leaf_value[node_idx]
    return out

# -------------------------------------------------------------
# Boosting: build many trees and update W
# -------------------------------------------------------------


@dataclass
class BoostConfig:
    n_trees: int = 100
    learning_rate: float = 0.2
    learn_global_score: bool = False
    # tree cfg
    max_depth: int = 4
    min_size: int = 50
    positive: bool = False
    loss: str = 'MSE'
    max_n_split: int = 64  # -1 = full search, >=2 = subsample ~max_n_split cuts/feature


def boost(X: jnp.ndarray, W0_dict: Dict[Tuple[str, ...], jnp.ndarray],
          base_points: List[Dict[str, int]], coefficients: List[str],
          cfg: BoostConfig) -> Tuple[List[Tree], Tuple[Tuple[str, ...], ...]]:
    """Train a boosted forest.
       W0_dict: dict mapping derivative tuples to arrays (N,) including ()
    """
    # Derivatives ordering & constants
    derivatives = tuple(build_derivatives(coefficients))
    # Pack W matrix [w0 | w1..]
    w0 = W0_dict[tuple()].reshape(-1, 1)
    W_cols = [w0]
    for der in derivatives[1:]:
        # allow symmetric key like tuple(reversed(der)) if missing
        if der in W0_dict:
            W_cols.append(W0_dict[der].reshape(-1, 1))
        else:
            W_cols.append(W0_dict[tuple(reversed(der))].reshape(-1, 1))
    W = jnp.concatenate(W_cols, axis=1)

    base_mat = base_point_matrix(base_points, list(derivatives))
    base_mat_pos = base_point_matrix_for_pos(base_mat)

    trees: List[Tree] = []

    # First tree may store only the score vector if requested (same semantics)
    for t in range(cfg.n_trees):
        get_only_score = (t == 0 and cfg.learn_global_score)
        # Build a tree
        tree = build_tree(
            X, W, base_mat, base_mat_pos,
            BuildConfig(max_depth=cfg.max_depth, min_size=cfg.min_size,
                        positive=cfg.positive, loss=cfg.loss, max_n_split=cfg.max_n_split),
            derivatives,
        )
        trees.append(tree)

        # Predict on training data
        pred = predict_tree(tree, X)         # (N, M)
        p0 = pred[:, :1]
        pder = pred[:, 1:]
        lr = 1.0 if get_only_score else cfg.learning_rate
        # Δ only on derivative columns
        dW = W[:, :1] * (pder / jnp.clip(p0, 1e-12))
        W = W.at[:, 1:].add(-lr * dW)

    return trees, derivatives

# -------------------------------------------------------------
# Inference helpers for the forest (like your vectorized_predict)
# -------------------------------------------------------------

def predict_forest(trees: List[Tree], X: jnp.ndarray, learning_rate: float,
                   learn_global_score: bool, summed: bool = True,
                   last_tree_counts_full: bool = False) -> jnp.ndarray:
    """Return either per-tree contributions or their sum (like original code).
       Output is (N, K) with K = number of derivative columns (excluding base ())."""
    T = len(trees)
    preds = []
    for k, tree in enumerate(trees):
        pred = predict_tree(tree, X)  # (N, M)
        ratios = pred[:, 1:] / jnp.clip(pred[:, :1], 1e-12)
        preds.append(ratios)
    P = jnp.stack(preds, axis=0)      # (T, N, K)

    lr = jnp.full((T,), learning_rate)
    if last_tree_counts_full and T > 0:
        lr = lr.at[-1].set(1.0)
    if learn_global_score and T > 0:
        lr = lr.at[0].set(1.0)

    if summed:
        return jnp.tensordot(lr, P, axes=1)   # (N, K)
    else:
        return (lr[:, None, None] * P)        # (T, N, K)

# -------------------------------------------------------------
# Minimal example (replace with your toy_models when integrating)
# -------------------------------------------------------------
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    N, D = 20000, 8
    Kcoeff = ['theta1']
    X = jax.random.normal(key, (N, D))

    # Fake weights dictionary with derivatives for demo
    ders = build_derivatives(Kcoeff)
    W0_dict = {(): jnp.abs(jax.random.normal(key, (N,))) + 1.0}
    for der in ders[1:]:
        W0_dict[der] = jax.random.normal(key, (N,))

    # Base points (example: first + second orders for single coeff)
    base_points = [{ 'theta1': 1 }, { 'theta1': 2 }]

    cfg = BoostConfig(n_trees=5, learning_rate=0.2, learn_global_score=False,
                      max_depth=3, min_size=64, positive=False, loss='MSE')

    trees, derivatives = boost(X, W0_dict, base_points, Kcoeff, cfg)
    out = predict_forest(trees, X, cfg.learning_rate, cfg.learn_global_score, summed=True)
    print("Built", len(trees), "trees; output shape:", out.shape)

