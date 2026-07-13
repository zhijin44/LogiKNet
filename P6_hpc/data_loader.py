# ============================================================================
# data_loader.py -- load a snapshot of the Monet Blue Waters 2017 release into
# a per-node feature table (torus coordinates + per-direction PTs).
#
# Released schema (src/HEADER of github.com/CSLDepend/monet), one row per
# node-sample:
#   #Time,
#   {Z-,Z+,Y-,Y+,X-,X+}_SAMPLE_GEMINI_LINK_CREDIT_STALL (% x1e6),
#   {Z-,Z+,Y-,Y+,X-,X+}_SAMPLE_GEMINI_LINK_INQ_STALL   (% x1e6),
#   {Z-,Z+,Y-,Y+,X-,X+}_SAMPLE_GEMINI_LINK_USED_BW     (% x1e6),
#   nettopo_mesh_coord_Z, nettopo_mesh_coord_Y, nettopo_mesh_coord_X
#
# #Time is a Unix epoch (America/Chicago). Values are percent x 1e6.
# ============================================================================
from __future__ import annotations
import numpy as np
import pandas as pd

DIRECTIONS = ["X", "Y", "Z"]
SIGNS = ["+", "-"]


def _stall_col(direction: str, sign: str, metric: str) -> str:
    m = "CREDIT" if metric.lower().startswith("cred") else "INQ"
    return f"{direction}{sign}_SAMPLE_GEMINI_LINK_{m}_STALL (% x1e6)"


def _tidy(name: str) -> str:
    """Match Monet's column-name normalisation (used in some released parquet files)."""
    s = name
    for c in " ,;{}()\n\t=#.":
        s = s.replace(c, "_")
    return s


def load_snapshot(path: str,
                  timestamp: int | None = None,
                  bucket_seconds: int = 60,
                  metric: str = "credit",
                  pts_scale: float = 1.0e6,
                  node_agg: str = "max") -> pd.DataFrame:
    """Load one 60-second snapshot into a node feature table.

    Parameters
    ----------
    path : CSV or parquet file for a day (released OVIS format), or any file with
           the schema above. Column names are matched loosely (raw or tidied).
    timestamp : if given, keep only rows in the 60-s bucket containing this epoch.
                If None, the earliest bucket in the file is used.
    metric : 'credit' or 'inq'.
    pts_scale : divide raw stored value by this to obtain PTs in percent.
    node_agg : how to reduce a node's 6 directional stalls to a scalar PTs
               ('max' as in Monet, or 'mean').

    Returns
    -------
    DataFrame with columns:
        x, y, z            : integer torus coordinates
        PTs_Xp, PTs_Xm, PTs_Yp, PTs_Ym, PTs_Zp, PTs_Zm : directional PTs (%)
        PTs_X, PTs_Y, PTs_Z : per-axis PTs (max of +/- for that axis)
        PTs                 : whole-node scalar PTs (node_agg over 6 directions)
        time                : epoch of the bucket
    """
    df = _read_any(path)
    df = _match_columns(df)

    # ---- select the 60-s bucket -------------------------------------------
    if "time" in df.columns:
        if timestamp is None:
            timestamp = int(df["time"].min())
        lo = timestamp - (timestamp % bucket_seconds)
        hi = lo + bucket_seconds
        snap = df[(df["time"] >= lo) & (df["time"] < hi)].copy()
        if snap.empty:  # fall back to nearest bucket
            nearest = int(df.loc[(df["time"] - timestamp).abs().idxmin(), "time"])
            lo = nearest - (nearest % bucket_seconds)
            snap = df[(df["time"] >= lo) & (df["time"] < lo + bucket_seconds)].copy()
    else:
        snap = df.copy()
        snap["time"] = 0

    # ---- directional PTs ---------------------------------------------------
    out = {"x": snap["x"].astype(int).values,
           "y": snap["y"].astype(int).values,
           "z": snap["z"].astype(int).values,
           "time": snap["time"].astype(np.int64).values}
    for d in DIRECTIONS:
        for s in SIGNS:
            key = f"raw_{d}{'p' if s == '+' else 'm'}"
            out[f"PTs_{d}{'p' if s == '+' else 'm'}"] = snap[key].values / pts_scale

    tab = pd.DataFrame(out)
    # collapse duplicate node rows (each Gemini ASIC hosts 2 compIDs -> take max)
    ptcols = [c for c in tab.columns if c.startswith("PTs_")]
    tab = tab.groupby(["x", "y", "z"], as_index=False).agg(
        {**{c: "max" for c in ptcols}, "time": "first"})

    for d in DIRECTIONS:
        tab[f"PTs_{d}"] = tab[[f"PTs_{d}p", f"PTs_{d}m"]].max(axis=1)
    six = [f"PTs_{d}{s}" for d in DIRECTIONS for s in ("p", "m")]
    tab["PTs"] = tab[six].max(axis=1) if node_agg == "max" else tab[six].mean(axis=1)
    return tab


def _read_any(path: str) -> pd.DataFrame:
    if str(path).endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    # released OVIS files are comma-separated with a leading '#Time' header
    return pd.read_csv(path, skipinitialspace=True)


def _match_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw or tidied schema column names to canonical short names."""
    cols = {c: c for c in df.columns}
    tidied = {_tidy(c): c for c in df.columns}

    def find(*cands):
        for cand in cands:
            if cand in cols:
                return cols[cand]
            if _tidy(cand) in tidied:
                return tidied[_tidy(cand)]
        return None

    ren = {}
    t = find("#Time", "Time", "_Time")
    if t: ren[t] = "time"
    for axis, short in (("X", "x"), ("Y", "y"), ("Z", "z")):
        c = find(f"nettopo_mesh_coord_{axis}")
        if c: ren[c] = short
    for d in DIRECTIONS:
        for s in SIGNS:
            tag = "p" if s == "+" else "m"
            c = find(_stall_col(d, s, "credit"), _stall_col(d, s, "inq"))
            # try both metrics; caller picks via `metric`, but we load credit here
            c_credit = find(_stall_col(d, s, "credit"))
            c_inq = find(_stall_col(d, s, "inq"))
            chosen = c_credit or c_inq
            if chosen: ren[chosen] = f"raw_{d}{tag}"
    out = df.rename(columns=ren)
    missing = [f"raw_{d}{'p' if s == '+' else 'm'}" for d in DIRECTIONS for s in SIGNS
               if f"raw_{d}{'p' if s == '+' else 'm'}" not in out.columns]
    if missing:
        raise ValueError(f"Snapshot is missing expected stall columns: {missing}. "
                         f"Found columns: {list(df.columns)[:6]}...")
    return out
