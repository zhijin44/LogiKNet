# ============================================================================
# cluster_label.py -- "K-means then LTN labeler" congestion detection.
#
# Pipeline (matches the paper's forward-discovery + backward-validation story):
#   1. K-means partitions nodes on (torus coords + weighted PTs).
#   2. An LTN-style fuzzy predicate labels each cluster as congestion or not,
#      using Monet-derived knowledge:
#         High(c)        : cluster mean PTs is in the High band (>= 25%)
#         Homogeneous(c) : intra-cluster PTs spread <= th_similarity (4%)
#         Congested(c)   = High(c) AND Homogeneous(c)      (product t-norm)
#   3. Predicted congestion region = union of clusters with Congested(c) >= tau.
#      Snapshot is "congested" iff at least one cluster fires.
#
# The fuzzy predicate is evaluated in numpy here (portable, no torch needed);
# it is the exact grounding an LTNtorch Predicate would compute, so the LTN
# version is a drop-in (same product t-norm as ltn.fuzzy_ops.AndProd).
# ============================================================================
from __future__ import annotations
from collections import deque
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from torus import build_adjacency


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def high_membership(mean_pts, high_band=25.0, sharp=3.0):
    """Fuzzy truth that a cluster's mean PTs sits in the High band."""
    return _sigmoid((mean_pts - high_band) / sharp)


def homogeneous_membership(std_pts, th_similarity=4.0, sharp=2.0):
    """Fuzzy truth that a cluster's PTs are homogeneous (spread <= th_similarity)."""
    return _sigmoid((th_similarity - std_pts) / sharp)


def region_grow(tab, seed_pred, dims=(24, 24, 24), growth_floor=15.0,
                th_similarity=4.0, pts_col="PTs", wrap=True):
    """Grow the seed region outward along torus adjacency (Monet-style).

    Starting from the confidently-labelled seed nodes, absorb a neighbour j of
    an in-region node i iff:
        PTs[j] >= growth_floor            (still congested-ish; not background)
        |PTs[j] - PTs[i]| <= th_similarity (local gradient step, same rule
                                            Monet uses to group similar links)
    This recovers the tapering boundary nodes K-means orphaned, lifting recall
    while the growth_floor + similarity brake keep precision from collapsing.
    Returns a (N,) binary array (>= the seeds).
    """
    coords = np.stack([tab["x"].values, tab["y"].values, tab["z"].values], axis=1).astype(int)
    pts = tab[pts_col].values.astype(float)
    adj = build_adjacency(coords, dims, wrap=wrap)
    region = np.asarray(seed_pred).astype(bool).copy()
    frontier = deque(np.where(region)[0].tolist())
    while frontier:
        i = frontier.popleft()
        for j in adj[i]:
            if (not region[j]) and pts[j] >= growth_floor and \
               abs(pts[j] - pts[i]) <= th_similarity:
                region[j] = True
                frontier.append(j)
    return region.astype(int)


def kmeans_then_label(tab, k=24, pts_col="PTs", pts_weight=3.0,
                      high_band=25.0, th_similarity=4.0, tau=0.5, seed=0,
                      grow=False, dims=(24, 24, 24), growth_floor=15.0):
    """Cluster then label congestion (optionally followed by region growth).

    Returns a dict with:
        node_pred    : (N,) binary, 1 = node detected as congested, else 0
                       (seeds only, or grown if grow=True)
        node_seed    : (N,) binary seeds before growth (== node_pred if grow=False)
        kmeans_labels: (N,) raw K-means cluster id (1..k)
        congested    : bool, snapshot-level congestion flag
        clusters     : per-cluster diagnostics (id, size, mean/std PTs,
                       High, Homogeneous, Congested truth, labeled bool)
    """
    coords = np.stack([tab["x"].values, tab["y"].values, tab["z"].values], axis=1).astype(float)
    pts = tab[pts_col].values.astype(float)

    # feature matrix: standardised coords + PTs, PTs up-weighted so the
    # (small, high-PTs) congestion nodes separate from the large background.
    X = np.concatenate([coords, pts[:, None]], axis=1)
    Xs = StandardScaler().fit_transform(X)
    Xs[:, 3] *= pts_weight

    k_eff = min(k, len(tab))
    labels = KMeans(n_clusters=k_eff, random_state=seed, n_init=10).fit_predict(Xs) + 1

    seed_pred = np.zeros(len(tab), dtype=int)
    clusters = []
    for c in np.unique(labels):
        members = labels == c
        mp = float(pts[members].mean())
        sp = float(pts[members].std())
        h = float(high_membership(mp, high_band))
        hom = float(homogeneous_membership(sp, th_similarity))
        cong = h * hom                     # AndProd (product t-norm)
        labeled = cong >= tau
        if labeled:
            seed_pred[members] = 1
        clusters.append({"id": int(c), "size": int(members.sum()),
                         "mean_pts": mp, "std_pts": sp,
                         "High": h, "Homogeneous": hom, "Congested": cong,
                         "labeled": bool(labeled)})

    node_pred = seed_pred
    if grow and seed_pred.any():
        node_pred = region_grow(tab, seed_pred, dims=dims, growth_floor=growth_floor,
                                th_similarity=th_similarity, pts_col=pts_col)
    return {"node_pred": node_pred, "node_seed": seed_pred, "kmeans_labels": labels,
            "congested": bool(node_pred.any()), "clusters": clusters}
