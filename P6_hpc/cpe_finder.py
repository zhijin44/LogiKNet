# ============================================================================
# cpe_finder.py -- locate congestion-protection-event (CPE) candidate days /
# timestamps directly from the LDMS counter telemetry.
#
# WHY THIS EXISTS
# ---------------
# A real CPE is logged by Cray's xtnlrd daemon into the netwatch/console logs
# when the stall-to-flit ratio crosses a HIGH-watermark; throttling stays on
# until values fall below the LOW-watermark (NSDI'20, Section 2). Those logs are
# NOT part of the public 140 GB counter release, so if you only have the counter
# data you cannot read exact CPE times. However, a CPE leaves a clear signature
# in the counters: a large fraction of links stay at High PTs (Percent Time
# Stalled) for a SUSTAINED window. This scanner flags those windows so you can
# pick meaningful snapshots for cross-verification.
#
# NOTE ON SEMANTICS: Monet reports that only ~8% of high-congestion events
# actually triggered a CPE, so "sustained high PTs" is a *candidate* signal
# (a Regions-Congestion-Event, RCE), not a guaranteed logged CPE. For a strict
# external check, obtain the netwatch logs from the dataset authors (UIUC DEPEND)
# and feed them to cpe_crosscheck.py instead. For picking snapshots that contain
# genuine severe congestion, this scanner is exactly what you want.
# ============================================================================
from __future__ import annotations
import argparse
import datetime
import numpy as np
import pandas as pd

from data_loader import _read_any, _match_columns, DIRECTIONS, SIGNS

CHICAGO = "America/Chicago"


def _epoch_to_local(ts: int) -> str:
    try:
        from zoneinfo import ZoneInfo
        return datetime.datetime.fromtimestamp(int(ts), ZoneInfo(CHICAGO)).isoformat()
    except Exception:
        return datetime.datetime.utcfromtimestamp(int(ts)).isoformat() + "Z(approx)"


def scan_file(path: str, metric: str = "credit", pts_scale: float = 1.0e6,
              bucket_seconds: int = 60, high_pts: float = 25.0,
              area_frac: float = 0.02, min_duration: int = 3) -> pd.DataFrame:
    """Scan one day file and return per-bucket congestion statistics.

    Parameters
    ----------
    high_pts   : PTs (%) at/above which a node counts as 'High' (Monet's High band).
    area_frac  : fraction of nodes that must be High for a bucket to be 'hot'
                 (proxy for the network-wide reach that triggers a CPE).
    min_duration : consecutive hot buckets required to call it a sustained event.

    Returns a DataFrame with one row per 60-s bucket:
        time, n_nodes, n_high, frac_high, mean_pts, p99_pts, hot(bool)
    and an attribute .events listing sustained (start, end, peak) windows.
    """
    df = _match_columns(_read_any(path))
    # per-row PTs = max over the six directional stalls of the chosen metric
    stall_cols = [f"raw_{d}{'p' if s == '+' else 'm'}" for d in DIRECTIONS for s in SIGNS]
    df["PTs"] = df[stall_cols].max(axis=1) / pts_scale

    if "time" not in df.columns:
        raise ValueError("File has no #Time column; cannot scan buckets.")
    t0 = int(df["time"].min())
    df["bucket"] = ((df["time"].astype(np.int64) - t0) // bucket_seconds)

    g = df.groupby("bucket")
    stats = pd.DataFrame({
        "time": (g["time"].min()).astype(np.int64),
        "n_nodes": g.size(),
        "n_high": g["PTs"].apply(lambda s: int((s >= high_pts).sum())),
        "mean_pts": g["PTs"].mean(),
        "p99_pts": g["PTs"].quantile(0.99),
    }).reset_index(drop=True)
    stats["frac_high"] = stats["n_high"] / stats["n_nodes"].clip(lower=1)
    stats["hot"] = stats["frac_high"] >= area_frac

    stats.attrs["events"] = _sustained_events(stats, min_duration)
    return stats


def _sustained_events(stats: pd.DataFrame, min_duration: int):
    events = []
    hot = stats["hot"].values
    i = 0
    n = len(hot)
    while i < n:
        if hot[i]:
            j = i
            while j + 1 < n and hot[j + 1]:
                j += 1
            if (j - i + 1) >= min_duration:
                seg = stats.iloc[i:j + 1]
                peak = seg.loc[seg["frac_high"].idxmax()]
                events.append({
                    "start_time": int(seg["time"].iloc[0]),
                    "end_time": int(seg["time"].iloc[-1]),
                    "duration_buckets": int(j - i + 1),
                    "peak_time": int(peak["time"]),
                    "peak_frac_high": float(peak["frac_high"]),
                    "peak_mean_pts": float(peak["mean_pts"]),
                })
            i = j + 1
        else:
            i += 1
    return events


def scan_files(paths, **kw):
    """Scan several day files; return a combined event table ranked by severity."""
    all_events = []
    for p in paths:
        stats = scan_file(p, **kw)
        for e in stats.attrs["events"]:
            e["file"] = p
            e["peak_time_local"] = _epoch_to_local(e["peak_time"])
            all_events.append(e)
    ev = pd.DataFrame(all_events)
    if len(ev):
        ev = ev.sort_values("peak_frac_high", ascending=False).reset_index(drop=True)
    return ev


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Find CPE-candidate windows in counter telemetry.")
    ap.add_argument("files", nargs="+", help="released day CSV/parquet files")
    ap.add_argument("--metric", default="credit", choices=["credit", "inq"])
    ap.add_argument("--pts-scale", type=float, default=1.0e6)
    ap.add_argument("--high-pts", type=float, default=25.0, help="High PTs band (%)")
    ap.add_argument("--area-frac", type=float, default=0.02,
                    help="fraction of nodes High for a bucket to count as hot")
    ap.add_argument("--min-duration", type=int, default=3, help="consecutive hot buckets")
    ap.add_argument("--out", default=None, help="optional CSV of candidate events")
    args = ap.parse_args()

    ev = scan_files(args.files, metric=args.metric, pts_scale=args.pts_scale,
                    high_pts=args.high_pts, area_frac=args.area_frac,
                    min_duration=args.min_duration)
    if not len(ev):
        print("No sustained high-congestion windows found with the current thresholds. "
              "Try lowering --area-frac or --high-pts.")
    else:
        cols = ["file", "peak_time", "peak_time_local", "duration_buckets",
                "peak_frac_high", "peak_mean_pts"]
        print(ev[cols].to_string(index=False))
        print(f"\n{len(ev)} candidate window(s). Use peak_time as the snapshot "
              f"timestamp for run_experiment.py real.")
        if args.out:
            ev.to_csv(args.out, index=False)
            print(f"written -> {args.out}")
