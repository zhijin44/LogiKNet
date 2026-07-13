# ============================================================================
# torus.py -- 3D-torus geometry helpers shared by every module.
# ============================================================================
from __future__ import annotations
import numpy as np


def coord_distance(a: np.ndarray, b: np.ndarray, dims, wrap: bool = True) -> np.ndarray:
    """Manhattan (hop) distance between torus coordinates.

    a : (..., 3) integer coordinates
    b : (..., 3) integer coordinates
    dims : iterable of 3 ints (X, Y, Z extent)
    wrap : if True use minimum-image (torus wrap-around) distance per axis.
    Returns hop distance with the same leading shape as the broadcast of a, b.
    """
    a = np.asarray(a); b = np.asarray(b)
    d = np.abs(a - b)
    if wrap:
        dims = np.asarray(dims)
        d = np.minimum(d, dims - d)
    return d.sum(axis=-1)


def neighbor_pairs(coords: np.ndarray, dims, th_close: int, wrap: bool = True):
    """Return array of index pairs (i, j), i < j, whose coordinate distance <= th_close.

    Uses a coordinate hash so it is O(n * neighbourhood) rather than O(n^2),
    which keeps it tractable on the full 27,648-node torus.
    """
    dims = tuple(int(x) for x in dims)
    coords = np.asarray(coords, dtype=int)
    index = {tuple(c): i for i, c in enumerate(coords)}

    # Enumerate the offset stencil within th_close hops (Manhattan).
    offs = []
    r = int(th_close)
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dz in range(-r, r + 1):
                if 0 < abs(dx) + abs(dy) + abs(dz) <= r:
                    offs.append((dx, dy, dz))

    pairs = []
    for c, i in index.items():
        for (dx, dy, dz) in offs:
            nb = ((c[0] + dx) % dims[0], (c[1] + dy) % dims[1], (c[2] + dz) % dims[2]) \
                if wrap else (c[0] + dx, c[1] + dy, c[2] + dz)
            j = index.get(nb)
            if j is not None and i < j:
                pairs.append((i, j))
    if not pairs:
        return np.empty((0, 2), dtype=int)
    return np.unique(np.array(pairs, dtype=int), axis=0)


def build_adjacency(coords: np.ndarray, dims, wrap: bool = True):
    """Adjacency list (dict: node index -> list of 1-hop neighbour indices) on the torus."""
    dims = tuple(int(x) for x in dims)
    coords = np.asarray(coords, dtype=int)
    index = {tuple(c): i for i, c in enumerate(coords)}
    adj = {i: [] for i in range(len(coords))}
    stencil = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    for c, i in index.items():
        for (dx, dy, dz) in stencil:
            nb = ((c[0] + dx) % dims[0], (c[1] + dy) % dims[1], (c[2] + dz) % dims[2]) \
                if wrap else (c[0] + dx, c[1] + dy, c[2] + dz)
            j = index.get(nb)
            if j is not None:
                adj[i].append(j)
    return adj
