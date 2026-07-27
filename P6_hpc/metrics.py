# ============================================================================
# metrics.py -- quantitative evaluation for the HPC congestion case study.
#
#   Route B (no ground truth) : internal validity indices on real snapshots.
#   Route A (synthetic labels) : Monet's overlap score + precision/recall,
#                                plus ARI / NMI at the node level.
#   Constraint satisfaction    : how well a clustering obeys the logic axioms,
#                                so a constraint-aware method is not judged only
#                                by a constraint-blind (K-means-favouring) yard-stick.
# ============================================================================
from __future__ import annotations
import numpy as np
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score,
                             adjusted_rand_score, normalized_mutual_info_score)
from sklearn.preprocessing import StandardScaler

from torus import coord_distance, neighbor_pairs


def feature_matrix(tab, pts_col="PTs", coord_weight=1.0, pts_weight=1.0):
    """Standardised (x, y, z, PTs) feature matrix (moved here from the removed
    baselines.py). Used by the internal-validity indices."""
    X = np.stack([tab["x"].values, tab["y"].values, tab["z"].values,
                  tab[pts_col].values], axis=1).astype(float)
    Xs = StandardScaler().fit_transform(X)
    Xs[:, :3] *= coord_weight
    Xs[:, 3] *= pts_weight
    return Xs


# --------------------------------------------------------------------------
# Route B : internal validity indices
# --------------------------------------------------------------------------
def internal_indices(tab, labels, pts_col="PTs", drop_background=True, **feat_kw):
    """Silhouette / Davies-Bouldin / Calinski-Harabasz on standardised
    (coords + PTs) space. Background (label 0) points are excluded by default,
    since they are 'unclustered' rather than a real cluster."""
    X = feature_matrix(tab, pts_col=pts_col, **feat_kw)
    y = np.asarray(labels)
    if drop_background:
        keep = y != 0
        X, y = X[keep], y[keep]
    n_clusters = len(np.unique(y))
    if n_clusters < 2 or len(y) < 3:
        return {"silhouette": np.nan, "davies_bouldin": np.nan,
                "calinski_harabasz": np.nan, "n_clusters": n_clusters}
    return {"silhouette": float(silhouette_score(X, y)),
            "davies_bouldin": float(davies_bouldin_score(X, y)),
            "calinski_harabasz": float(calinski_harabasz_score(X, y)),
            "n_clusters": int(n_clusters)}


# --------------------------------------------------------------------------
# Route A : synthetic-ground-truth region matching (Monet Appendix D scoring)
# --------------------------------------------------------------------------
def monet_overlap_score(true_labels, pred_labels):
    """Monet's region-overlap score, precision and recall.

    For actual regions A_i and predicted regions B_j, greedily match each A_i
    (smallest first) to the B_j with the largest overlap, then
        score = (1/n * sum_i |A_i & B_ji| / |A_i union B_ji|) * (n / max(n, m))
    which is 1 for a perfect match and degrades with mismatch or over-
    segmentation. Precision/recall use best-overlap region pairs.
    """
    true_labels = np.asarray(true_labels); pred_labels = np.asarray(pred_labels)
    A = [np.where(true_labels == r)[0] for r in np.unique(true_labels) if r != 0]
    B = [np.where(pred_labels == r)[0] for r in np.unique(pred_labels) if r != 0]
    n, m = len(A), len(B)
    if n == 0:
        return {"overlap_score": np.nan, "precision": np.nan, "recall": np.nan,
                "n_true": n, "n_pred": m}
    Bsets = [set(b.tolist()) for b in B]
    used = set()
    iou_sum = prec_sum = rec_sum = 0.0
    for a in sorted(A, key=len):              # smallest actual region first
        aset = set(a.tolist())
        best, best_iou, best_inter = None, -1.0, 0
        for k, bset in enumerate(Bsets):
            if k in used:
                continue
            inter = len(aset & bset)
            union = len(aset | bset)
            iou = inter / union if union else 0.0
            if iou > best_iou:
                best, best_iou, best_inter = k, iou, inter
        if best is not None:
            used.add(best)
            iou_sum += best_iou
            prec_sum += best_inter / max(len(Bsets[best]), 1)
            rec_sum += best_inter / max(len(aset), 1)
    matched = max(len(used), 1)
    score = (iou_sum / n) * (n / max(n, m))
    return {"overlap_score": float(score),
            "precision": float(prec_sum / matched),
            "recall": float(rec_sum / matched),
            "n_true": n, "n_pred": m}


def point_level_agreement(true_labels, pred_labels):
    """ARI and NMI at the node level (needs true labels; use with synthetic data
    or against Monet output treated as a reference)."""
    return {"ARI": float(adjusted_rand_score(true_labels, pred_labels)),
            "NMI": float(normalized_mutual_info_score(true_labels, pred_labels))}


# --------------------------------------------------------------------------
# Region segmentation (binary: is the flagged congestion node-set correct?)
# --------------------------------------------------------------------------
def region_segmentation(true_binary, pred_binary):
    """Node-level precision/recall/F1/IoU of the predicted congestion region
    against the (synthetic) ground-truth region. Inputs are 0/1 per node."""
    t = np.asarray(true_binary).astype(bool)
    p = np.asarray(pred_binary).astype(bool)
    tp = int((t & p).sum()); fp = int((~t & p).sum()); fn = int((t & ~p).sum())
    prec = tp / (tp + fp) if (tp + fp) else (1.0 if tp == 0 and fp == 0 else 0.0)
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
    return {"precision": prec, "recall": rec, "f1": f1, "iou": iou,
            "tp": tp, "fp": fp, "fn": fn}


# --------------------------------------------------------------------------
# Constraint satisfaction (how well the logic axioms are obeyed)
# --------------------------------------------------------------------------
def constraint_satisfaction(tab, labels, dims=(24, 24, 24),
                            th_close=2, th_similarity=4.0, pts_col="PTs", wrap=True):
    """Fraction of axiom-relevant node pairs that the clustering respects.

    * axiom-3 (closeness): among spatially close pairs with *similar* PTs
      (<= th_similarity), fraction assigned to the same cluster.
    * axiom-4 (similarity): among spatially close pairs with *dissimilar* PTs
      (> th_similarity), fraction assigned to different clusters.
    Both restricted to spatial neighbours, matching the local nature of the
    constraints and keeping the count tractable on the full torus.
    """
    coords = np.stack([tab["x"].values, tab["y"].values, tab["z"].values], axis=1).astype(int)
    pts = tab[pts_col].values.astype(float)
    labels = np.asarray(labels)
    pairs = neighbor_pairs(coords, dims, th_close, wrap=wrap)
    if len(pairs) == 0:
        return {"axiom3_closeness": np.nan, "axiom4_similarity": np.nan}
    i, j = pairs[:, 0], pairs[:, 1]
    dpts = np.abs(pts[i] - pts[j])
    same = labels[i] == labels[j]

    sim = dpts <= th_similarity
    dis = dpts > th_similarity
    a3 = same[sim].mean() if sim.any() else np.nan            # want same cluster
    a4 = (~same[dis]).mean() if dis.any() else np.nan         # want different cluster
    return {"axiom3_closeness": float(a3), "axiom4_similarity": float(a4)}


# --------------------------------------------------------------------------
# Convenience: run the full metric suite for one clustering
# --------------------------------------------------------------------------
def evaluate(tab, labels, true_labels=None, dims=(24, 24, 24),
             th_close=2, th_similarity=4.0, pts_col="PTs"):
    res = {}
    res.update({f"internal.{k}": v for k, v in
                internal_indices(tab, labels, pts_col=pts_col).items()})
    res.update({f"constraint.{k}": v for k, v in
                constraint_satisfaction(tab, labels, dims, th_close,
                                        th_similarity, pts_col).items()})
    if true_labels is not None:
        res.update({f"synthetic.{k}": v for k, v in
                    monet_overlap_score(true_labels, labels).items()})
        res.update({f"synthetic.{k}": v for k, v in
                    point_level_agreement(true_labels, labels).items()})
    return res
