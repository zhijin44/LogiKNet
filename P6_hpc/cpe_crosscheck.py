# ============================================================================
# cpe_crosscheck.py -- external validation against Cray congestion-protection
# events (CPEs). A CPE is an independent, model-external signal that congestion
# really occurred at a given time (xtnlrd throttling) and, where available, in a
# given region of the torus (xtnetwatch link/router faults). Matching detected
# congestion clusters to logged CPEs turns the qualitative case study into a
# semi-supervised validation.
#
# Expected CPE log format (CSV), one row per event:
#   time            : Unix epoch of the event (America/Chicago), OR ISO string
#   x, y, z         : (optional) torus coordinates implicated by the event
#   severity        : (optional) free-text / numeric severity
# If per-event coordinates are unavailable, only temporal coincidence is scored.
# ============================================================================
from __future__ import annotations
import numpy as np
import pandas as pd

from torus import coord_distance


def load_cpe_log(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True)
    if "time" not in df.columns:
        raise ValueError("CPE log must have a 'time' column (epoch or ISO).")
    if not np.issubdtype(df["time"].dtype, np.number):
        df["time"] = pd.to_datetime(df["time"]).astype("int64") // 10**9
    return df


def temporal_coincidence(snapshot_time: int, cpe_df: pd.DataFrame,
                         window_seconds: int = 300) -> dict:
    """Was a CPE logged within +/- window of this snapshot?"""
    dt = (cpe_df["time"].values.astype(np.int64) - int(snapshot_time))
    hit = np.abs(dt) <= window_seconds
    return {"cpe_within_window": bool(hit.any()),
            "n_cpe_in_window": int(hit.sum()),
            "nearest_cpe_dt_s": int(dt[np.argmin(np.abs(dt))]) if len(dt) else None}


def spatial_coincidence(tab, labels, cpe_df: pd.DataFrame, snapshot_time: int,
                        dims=(24, 24, 24), window_seconds: int = 300,
                        th_close: int = 2, pts_col="PTs", high_pts: float = 25.0,
                        wrap=True) -> dict:
    """Do the detected high-congestion clusters sit where CPEs were logged?

    For each CPE within the time window that carries coordinates, check whether a
    node of a *High* cluster (mean PTs >= high_pts) lies within th_close hops.
    Reports the fraction of coordinate-bearing CPEs matched (recall-like) and the
    fraction of High clusters that coincide with some CPE (precision-like).
    """
    coords = np.stack([tab["x"].values, tab["y"].values, tab["z"].values], axis=1).astype(int)
    pts = tab[pts_col].values.astype(float)
    labels = np.asarray(labels)

    in_win = cpe_df[np.abs(cpe_df["time"].values.astype(np.int64) - int(snapshot_time))
                    <= window_seconds]
    has_xyz = all(c in in_win.columns for c in ("x", "y", "z")) and len(in_win)
    if not has_xyz:
        return {"note": "no coordinate-bearing CPEs in window; temporal check only",
                **temporal_coincidence(snapshot_time, cpe_df, window_seconds)}

    # High clusters = clusters whose mean PTs exceeds high_pts
    high_clusters = [c for c in np.unique(labels) if c != 0 and
                     pts[labels == c].mean() >= high_pts]
    high_nodes = np.isin(labels, high_clusters)
    hc_coords = coords[high_nodes]

    cpe_coords = in_win[["x", "y", "z"]].values.astype(int)
    matched_cpe = 0
    for cc in cpe_coords:
        d = coord_distance(hc_coords, cc[None, :], dims, wrap=wrap)
        if (d <= th_close).any():
            matched_cpe += 1
    # cluster-side precision
    matched_clusters = 0
    for c in high_clusters:
        cnodes = coords[labels == c]
        ok = False
        for cc in cpe_coords:
            if (coord_distance(cnodes, cc[None, :], dims, wrap=wrap) <= th_close).any():
                ok = True; break
        matched_clusters += int(ok)

    return {"n_cpe_with_xyz": int(len(cpe_coords)),
            "n_high_clusters": int(len(high_clusters)),
            "cpe_recall": matched_cpe / max(len(cpe_coords), 1),
            "high_cluster_precision": matched_clusters / max(len(high_clusters), 1),
            **temporal_coincidence(snapshot_time, cpe_df, window_seconds)}
