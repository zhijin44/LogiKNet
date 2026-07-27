# ============================================================================
# data_loader.py -- load a snapshot of the Monet Blue Waters 2017 release into
# a per-node feature table (torus coordinates + per-direction PTs).
#
# Released schema (the dataset HEADER file), one row per node-sample, in this
# exact column order:
#   #Time,
#   {Z-,Z+,Y-,Y+,X-,X+}_SAMPLE_GEMINI_LINK_CREDIT_STALL (% x1e6),
#   {Z-,Z+,Y-,Y+,X-,X+}_SAMPLE_GEMINI_LINK_INQ_STALL   (% x1e6),
#   {Z-,Z+,Y-,Y+,X-,X+}_SAMPLE_GEMINI_LINK_USED_BW     (% x1e6),
#   nettopo_mesh_coord_Z, nettopo_mesh_coord_Y, nettopo_mesh_coord_X
#
# IMPORTANT: the released day files are HEADERLESS CSV (data starts on line 1).
# This loader auto-detects that and supplies the canonical column names.
#
# #Time is a Unix epoch (US Central time). PTs(%) = credit-stall column / 1e6.
# Credit stall = inter-switch flit stall (the congestion signal / PTs);
# inq stall    = intra-switch flit stall (tracked separately).
# ============================================================================
from __future__ import annotations
import numpy as np
import pandas as pd

DIRECTIONS = ["X", "Y", "Z"]
SIGNS = ["+", "-"]

# HEADER lists directions in this order; used to build the canonical names.
_HDR_DIRS = ["Z-", "Z+", "Y-", "Y+", "X-", "X+"]


def _canonical_columns() -> list[str]:
    cols = ["#Time"]
    for kind in ("CREDIT_STALL", "INQ_STALL", "USED_BW"):
        for d in _HDR_DIRS:
            cols.append(f"{d}_SAMPLE_GEMINI_LINK_{kind} (% x1e6)")
    cols += ["nettopo_mesh_coord_Z", "nettopo_mesh_coord_Y", "nettopo_mesh_coord_X"]
    return cols


CANONICAL_COLUMNS = _canonical_columns()   # 22 names, matching the release exactly


def _stall_col(direction: str, sign: str, metric: str) -> str:
    m = "CREDIT" if metric.lower().startswith("cred") else "INQ"
    return f"{direction}{sign}_SAMPLE_GEMINI_LINK_{m}_STALL (% x1e6)"


def _tidy(name: str) -> str:
    s = name
    for c in " ,;{}()\n\t=#.":
        s = s.replace(c, "_")
    return s


# --------------------------------------------------------------------------
# Low-level reading (handles both headered and headerless releases)
# --------------------------------------------------------------------------
def _has_header(path: str) -> bool:
    with open(path, "r") as f:
        first = f.readline()
    return ("SAMPLE_GEMINI" in first) or first.lstrip().startswith("#Time")


def _read_any(path, chunksize=None, nrows=None):
    p = str(path)
    if p.endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    if _has_header(path):
        return pd.read_csv(p, skipinitialspace=True, chunksize=chunksize, nrows=nrows)
    # headerless released file -> supply canonical names
    return pd.read_csv(p, header=None, names=CANONICAL_COLUMNS,
                       skipinitialspace=True, chunksize=chunksize, nrows=nrows)


# --------------------------------------------------------------------------
# Efficient snapshot extraction from a large day file (early-stop on time)
# --------------------------------------------------------------------------
def extract_snapshot(src: str, timestamp: int, dst: str | None = None,
                     bucket_seconds: int = 60, chunksize: int = 2_000_000) -> pd.DataFrame:
    """Pull just the rows of one 60-s snapshot out of a (possibly multi-GB) day
    file, streaming in chunks. Optionally writes the small snapshot to `dst`.

    NOTE: the released day files are NOT globally time-sorted (they are written
    in node/partition blocks, so each timestamp recurs throughout the file).
    We therefore scan the whole file -- no early stop. One full pass over a
    ~2.3 GB day is a few seconds with the C csv engine; on the command line
    `awk -F, -v ts=<t> '$1==ts' <day> > snap.csv` is the fast equivalent.

    Returns the raw snapshot rows (canonical columns); pass the same `timestamp`
    to `load_snapshot` afterwards, or feed `dst` back to it.
    """
    lo = timestamp - (timestamp % bucket_seconds)
    hi = lo + bucket_seconds
    parts = []
    for chunk in _read_any(src, chunksize=chunksize):
        tcol = chunk.columns[0]                      # #Time is always column 0
        t = chunk[tcol].astype(np.int64)
        sel = chunk[(t >= lo) & (t < hi)]
        if len(sel):
            parts.append(sel)
    if not parts:
        raise ValueError(f"No rows found for timestamp {timestamp} (bucket {lo}-{hi}) in {src}")
    snap = pd.concat(parts, ignore_index=True)
    if dst:
        snap.to_csv(dst, index=False)                # writes WITH canonical header
    return snap


# --------------------------------------------------------------------------
# Snapshot -> node feature table
# --------------------------------------------------------------------------
def load_snapshot(path: str,
                  timestamp: int | None = None,
                  bucket_seconds: int = 60,
                  metric: str = "credit",
                  pts_scale: float = 1.0e6,
                  node_agg: str = "max") -> pd.DataFrame:
    """Load one 60-second snapshot into a node feature table.

    `path` may be a small extracted snapshot or a full day file (headered or
    headerless). Returns a DataFrame with:
        x, y, z                                   integer torus coordinates
        PTs_Xp..PTs_Zm                            directional PTs (%)
        PTs_X, PTs_Y, PTs_Z                       per-axis PTs (max of +/-)
        PTs                                       whole-node scalar PTs
        time                                      epoch of the snapshot
    """
    df = _read_any(path)
    df = _match_columns(df, metric=metric)
    return snapshot_from_frame(df, timestamp, bucket_seconds, pts_scale, node_agg)


def snapshot_from_frame(df: pd.DataFrame, timestamp=None, bucket_seconds=60,
                        pts_scale=1.0e6, node_agg="max") -> pd.DataFrame:
    """Build the node table from an already-matched frame (columns: time, x, y, z,
    raw_Xp..raw_Zm). Kept separate so extract_snapshot output can be reused."""
    if "time" in df.columns:
        if timestamp is None:
            timestamp = int(df["time"].min())
        lo = timestamp - (timestamp % bucket_seconds)
        snap = df[(df["time"] >= lo) & (df["time"] < lo + bucket_seconds)].copy()
        if snap.empty:
            nearest = int(df.loc[(df["time"] - timestamp).abs().idxmin(), "time"])
            lo = nearest - (nearest % bucket_seconds)
            snap = df[(df["time"] >= lo) & (df["time"] < lo + bucket_seconds)].copy()
    else:
        snap = df.copy()
        snap["time"] = 0

    out = {"x": snap["x"].astype(int).values,
           "y": snap["y"].astype(int).values,
           "z": snap["z"].astype(int).values,
           "time": snap["time"].astype(np.int64).values}
    for d in DIRECTIONS:
        for s in SIGNS:
            tag = "p" if s == "+" else "m"
            out[f"PTs_{d}{tag}"] = snap[f"raw_{d}{tag}"].values / pts_scale

    tab = pd.DataFrame(out)
    ptcols = [c for c in tab.columns if c.startswith("PTs_")]
    tab = tab.groupby(["x", "y", "z"], as_index=False).agg(
        {**{c: "max" for c in ptcols}, "time": "first"})   # 2 compIDs/coord -> max

    for d in DIRECTIONS:
        tab[f"PTs_{d}"] = tab[[f"PTs_{d}p", f"PTs_{d}m"]].max(axis=1)
    six = [f"PTs_{d}{s}" for d in DIRECTIONS for s in ("p", "m")]
    tab["PTs"] = tab[six].max(axis=1) if node_agg == "max" else tab[six].mean(axis=1)
    return tab


def _match_columns(df: pd.DataFrame, metric: str = "credit") -> pd.DataFrame:
    """Map schema column names (raw, tidied, or canonical) to short internal
    names. `metric` selects which stall family becomes the PTs source."""
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
            chosen = find(_stall_col(d, s, metric)) or \
                     find(_stall_col(d, s, "inq" if metric == "credit" else "credit"))
            if chosen: ren[chosen] = f"raw_{d}{tag}"
    out = df.rename(columns=ren)
    missing = [f"raw_{d}{'p' if s == '+' else 'm'}" for d in DIRECTIONS for s in SIGNS
               if f"raw_{d}{'p' if s == '+' else 'm'}" not in out.columns]
    if missing:
        raise ValueError(f"Snapshot is missing expected stall columns: {missing}. "
                         f"Found columns: {list(df.columns)[:6]}...")
    return out
