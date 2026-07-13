# ============================================================================
# baselines.py -- clustering baselines for the HPC congestion case study.
#   * kmeans_cluster        : geometric K-means on (coords + PTs).
#   * monet_region_growing  : faithful numpy port of Monet's 4-stage region-
#                             growth segmentation (NSDI'20 Section 4.2).
# Both return an integer label per node (label 0 reserved for "unclustered"/
# background); comparable to the LogiK-Net output.
# ============================================================================
from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from torus import build_adjacency


def feature_matrix(tab, pts_col="PTs", coord_weight=1.0, pts_weight=1.0):
    """Standardised (x, y, z, PTs) feature matrix used by geometric methods."""
    X = np.stack([tab["x"].values, tab["y"].values, tab["z"].values,
                  tab[pts_col].values], axis=1).astype(float)
    Xs = StandardScaler().fit_transform(X)
    Xs[:, :3] *= coord_weight
    Xs[:, 3] *= pts_weight
    return Xs


def kmeans_cluster(tab, k=16, pts_col="PTs", seed=0, **kw):
    X = feature_matrix(tab, pts_col=pts_col, **kw)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(X) + 1          # shift so labels start at 1
    return labels


def monet_region_growing(tab, dims=(24, 24, 24), pts_col="PTs",
                         theta_p=4.0, theta_r=4.0, sigma=20,
                         significance_pts=5.0, wrap=True):
    """Monet region-growth segmentation on node-level PTs.

    Stages (NSDI'20 Section 4.2):
      1. group neighbouring nodes whose PTs differ by <= theta_p;
      2. merge neighbouring regions whose *average* PTs differ by <= theta_r;
      3. merge regions smaller than sigma into the nearest adjacent region;
      4. discard any remaining region with fewer than sigma nodes (-> label 0).
    delta (neighbourhood radius) is fixed to the torus 1-hop adjacency, matching
    Monet's locality assumption; theta_p, theta_r, sigma default to Monet's values.
    """
    coords = np.stack([tab["x"].values, tab["y"].values, tab["z"].values], axis=1).astype(int)
    pts = tab[pts_col].values.astype(float)
    N = len(pts)
    adj = build_adjacency(coords, dims, wrap=wrap)

    # ---- stage 1: union-find over similar neighbouring nodes ---------------
    parent = list(range(N))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(N):
        if pts[i] < significance_pts:
            continue
        for j in adj[i]:
            if pts[j] >= significance_pts and abs(pts[i] - pts[j]) <= theta_p:
                union(i, j)

    # region id per node (0 for below-significance / background)
    reg = np.zeros(N, dtype=int)
    roots = {}
    for i in range(N):
        if pts[i] < significance_pts:
            continue
        r = find(i)
        if r not in roots:
            roots[r] = len(roots) + 1
        reg[i] = roots[r]

    # ---- stage 2: merge adjacent regions with similar mean PTs -------------
    reg = _merge_similar_regions(reg, adj, pts, theta_r)

    # ---- stage 3+4: absorb / discard small regions -------------------------
    reg = _handle_small_regions(reg, adj, pts, sigma)
    return _relabel(reg)


def _region_means(reg, pts):
    means = {}
    for r in np.unique(reg):
        if r == 0:
            continue
        means[r] = pts[reg == r].mean()
    return means


def _merge_similar_regions(reg, adj, pts, theta_r):
    changed = True
    while changed:
        changed = False
        means = _region_means(reg, pts)
        # find adjacent region pairs
        pair = None
        for i in range(len(reg)):
            ri = reg[i]
            if ri == 0:
                continue
            for j in adj[i]:
                rj = reg[j]
                if rj != 0 and rj != ri and abs(means[ri] - means[rj]) <= theta_r:
                    pair = (ri, rj)
                    break
            if pair:
                break
        if pair:
            keep, drop = min(pair), max(pair)
            reg[reg == drop] = keep
            changed = True
    return reg


def _handle_small_regions(reg, adj, pts, sigma):
    sizes = {r: int((reg == r).sum()) for r in np.unique(reg) if r != 0}
    small = [r for r, s in sizes.items() if s < sigma]
    for r in small:
        members = np.where(reg == r)[0]
        # nearest adjacent region by |mean PTs difference|
        neigh_regs = {}
        for i in members:
            for j in adj[i]:
                rj = reg[j]
                if rj != 0 and rj != r:
                    neigh_regs.setdefault(rj, []).append(pts[j])
        if neigh_regs:
            rmean = pts[members].mean()
            best = min(neigh_regs, key=lambda rr: abs(np.mean(neigh_regs[rr]) - rmean))
            reg[reg == r] = best
        else:
            reg[reg == r] = 0        # isolated small region -> discard
    return reg


def _relabel(reg):
    out = np.zeros_like(reg)
    for new, r in enumerate([x for x in np.unique(reg) if x != 0], start=1):
        out[reg == r] = new
    return out
