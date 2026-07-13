# ============================================================================
# synthetic.py -- Monet-style synthetic congestion generator (Route A).
#
# Reproduces the ground-truth benchmark of the Monet NSDI'20 paper (Appendix D):
#   * a 24 x 24 x 24 cube representing the Blue Waters Gemini 3D torus;
#   * 1..8 random cuboid congestion regions, each side in [3, 9] links;
#   * each region raises credit- or inq-stall by a random value in [20, 50] %;
#   * additive Gaussian noise N(mu=0, sigma=2.5) on every link.
# It returns a node feature table (same shape as data_loader.load_snapshot)
# together with per-node ground-truth region labels, enabling IoU / precision /
# recall / ARI / NMI evaluation that needs no real labels.
# ============================================================================
from __future__ import annotations
import numpy as np
import pandas as pd


def make_grid(dims=(24, 24, 24)) -> np.ndarray:
    xs, ys, zs = (np.arange(d) for d in dims)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(int)


def generate_sample(dims=(24, 24, 24),
                    regions_min=1, regions_max=8,
                    cuboid_min=3, cuboid_max=9,
                    stall_min=20.0, stall_max=50.0,
                    noise_mu=0.0, noise_sigma=2.5,
                    significance_pts=5.0,
                    rng: np.random.Generator | None = None):
    """Generate one synthetic snapshot.

    Returns
    -------
    tab : DataFrame with x,y,z and PTs (and per-axis PTs_X/Y/Z set equal to PTs);
          the same columns downstream clustering/metrics expect.
    labels : (N,) int array of ground-truth region id per node (0 = background).
    """
    rng = rng or np.random.default_rng()
    coords = make_grid(dims)
    N = coords.shape[0]
    cindex = {tuple(c): i for i, c in enumerate(coords)}

    pts = np.zeros(N, dtype=float)
    labels = np.zeros(N, dtype=int)

    n_regions = rng.integers(regions_min, regions_max + 1)
    for r in range(1, n_regions + 1):
        side = rng.integers(cuboid_min, cuboid_max + 1, size=3)
        origin = np.array([rng.integers(0, dims[k]) for k in range(3)])
        stall = rng.uniform(stall_min, stall_max)
        for dx in range(side[0]):
            for dy in range(side[1]):
                for dz in range(side[2]):
                    c = ((origin[0] + dx) % dims[0],
                         (origin[1] + dy) % dims[1],
                         (origin[2] + dz) % dims[2])
                    i = cindex[c]
                    pts[i] += stall          # accumulate (regions may overlap)
                    labels[i] = r            # last writer wins on overlap
    pts = pts + rng.normal(noise_mu, noise_sigma, size=N)
    pts = np.clip(pts, 0.0, 100.0)

    # nodes whose PTs never crossed the significance floor are background
    labels[pts < significance_pts] = 0

    tab = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "z": coords[:, 2],
                        "PTs": pts, "PTs_X": pts, "PTs_Y": pts, "PTs_Z": pts,
                        "time": 0})
    return tab, labels


def generate_dataset(n_samples=100, seed=0, **kwargs):
    """Yield (tab, labels) for n_samples independent synthetic snapshots."""
    rng = np.random.default_rng(seed)
    for _ in range(n_samples):
        yield generate_sample(rng=rng, **kwargs)
