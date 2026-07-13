# ============================================================================
# run_experiment.py -- orchestrate the re-anchored HPC congestion case study.
#
# Two modes:
#   synthetic : Route-A ground-truth benchmark (no download needed). Runs
#               K-means, Monet region-growing, and (if torch/ltn present)
#               LogiK-Net over many synthetic snapshots; reports overlap score,
#               precision, recall, ARI, NMI as mean +/- std.
#   real      : load real 2017 snapshots, cluster with each method, report
#               internal validity indices + constraint satisfaction, plus
#               optional CPE cross-verification.
#
# Usage:
#   python run_experiment.py synthetic --config config.yml --out results/
#   python run_experiment.py real --config config.yml \
#          --snapshots /path/day1.csv:1483276800 /path/day2.csv:1485000000 \
#          --cpe /path/cpe_log.csv --out results/
# ============================================================================
from __future__ import annotations
import argparse, os, json
import numpy as np
import pandas as pd
import yaml

import baselines
import metrics as M
import synthetic as syn

try:
    import torch  # noqa
    import ltn    # noqa
    import logiknet_cluster
    HAS_LTN = True
except Exception:
    HAS_LTN = False


def _agg(rows):
    """mean +/- std over a list of metric dicts."""
    df = pd.DataFrame(rows)
    out = {}
    for c in df.columns:
        vals = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(vals):
            out[c] = (float(vals.mean()), float(vals.std(ddof=1) if len(vals) > 1 else 0.0))
    return out


def run_synthetic(cfg, out_dir):
    s = cfg["synthetic"]; th = cfg["thresholds"]; dims = tuple(cfg["torus"]["dims"])
    k = cfg["clustering"]["k"]
    methods = {"KMeans": lambda tab, seed: baselines.kmeans_cluster(tab, k=k, seed=seed),
               "Monet": lambda tab, seed: baselines.monet_region_growing(
                   tab, dims=dims, theta_p=th["th_similarity"], theta_r=th["th_similarity"],
                   sigma=th["min_cluster_size"], significance_pts=s["significance_pts"])}
    if HAS_LTN:
        methods["LogiK-Net"] = lambda tab, seed: logiknet_cluster.cluster(
            tab, k=k, dims=dims, th_close=th["th_close"], th_similarity=th["th_similarity"],
            epochs=cfg["logiknet"]["epochs"], seed=seed)

    per_method = {m: [] for m in methods}
    n = s["n_samples"]
    for si, (tab, labels) in enumerate(syn.generate_dataset(
            n_samples=n, seed=0,
            dims=dims, regions_min=s["regions_min"], regions_max=s["regions_max"],
            cuboid_min=s["cuboid_min"], cuboid_max=s["cuboid_max"],
            stall_min=s["stall_min"], stall_max=s["stall_max"],
            noise_mu=s["noise_mu"], noise_sigma=s["noise_sigma"],
            significance_pts=s["significance_pts"])):
        for mname, fn in methods.items():
            pred = fn(tab, si)
            res = M.evaluate(tab, pred, true_labels=labels, dims=dims,
                             th_close=th["th_close"], th_similarity=th["th_similarity"])
            per_method[mname].append(res)

    summary = {m: _agg(rows) for m, rows in per_method.items()}
    _save(summary, out_dir, "synthetic_benchmark")
    _print_table(summary, ["synthetic.overlap_score", "synthetic.precision",
                           "synthetic.recall", "synthetic.ARI", "synthetic.NMI"],
                 title="Route A -- synthetic ground-truth benchmark (mean +/- std)")
    return summary


def run_real(cfg, snapshots, cpe_path, out_dir):
    import data_loader
    th = cfg["thresholds"]; dims = tuple(cfg["torus"]["dims"])
    k = cfg["clustering"]["k"]; d = cfg["data"]
    cpe_df = None
    if cpe_path:
        import cpe_crosscheck
        cpe_df = cpe_crosscheck.load_cpe_log(cpe_path)

    methods = {"KMeans": lambda tab, seed: baselines.kmeans_cluster(tab, k=k, seed=seed),
               "Monet": lambda tab, seed: baselines.monet_region_growing(
                   tab, dims=dims, theta_p=th["th_similarity"], theta_r=th["th_similarity"],
                   sigma=th["min_cluster_size"])}
    if HAS_LTN:
        methods["LogiK-Net"] = lambda tab, seed: logiknet_cluster.cluster(
            tab, k=k, dims=dims, th_close=th["th_close"], th_similarity=th["th_similarity"],
            epochs=cfg["logiknet"]["epochs"], seed=seed)

    per_method = {m: [] for m in methods}
    cpe_rows = []
    for spec in snapshots:
        path, _, ts = spec.partition(":")
        ts = int(ts) if ts else None
        tab = data_loader.load_snapshot(path, timestamp=ts, metric=d["metric"],
                                        pts_scale=float(d["pts_scale"]),
                                        node_agg=d["node_agg"])
        snap_time = int(tab["time"].iloc[0]) if "time" in tab else 0
        for mname, fn in methods.items():
            pred = fn(tab, 0)
            res = M.evaluate(tab, pred, true_labels=None, dims=dims,
                             th_close=th["th_close"], th_similarity=th["th_similarity"])
            per_method[mname].append(res)
            if cpe_df is not None:
                import cpe_crosscheck
                cc = cpe_crosscheck.spatial_coincidence(tab, pred, cpe_df, snap_time,
                                                        dims=dims, th_close=th["th_close"])
                cpe_rows.append({"snapshot": os.path.basename(path), "method": mname, **cc})

    summary = {m: _agg(rows) for m, rows in per_method.items()}
    _save(summary, out_dir, "real_internal_indices")
    _print_table(summary, ["internal.silhouette", "internal.davies_bouldin",
                           "internal.calinski_harabasz", "constraint.axiom3_closeness",
                           "constraint.axiom4_similarity"],
                 title="Route B -- internal indices + constraint satisfaction (mean +/- std)")
    if cpe_rows:
        pd.DataFrame(cpe_rows).to_csv(os.path.join(out_dir, "cpe_crosscheck.csv"), index=False)
        print("\n[CPE cross-verification written to cpe_crosscheck.csv]")
    return summary


def _save(summary, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(summary, f, indent=2)


def _print_table(summary, keys, title=""):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)
    methods = list(summary.keys())
    header = f"{'metric':32s}" + "".join(f"{m:>15s}" for m in methods)
    print(header); print("-" * len(header))
    for k in keys:
        row = f"{k:32s}"
        for m in methods:
            if k in summary[m]:
                mean, std = summary[m][k]
                row += f"{mean:>9.3f}+/-{std:<4.2f}"
            else:
                row += f"{'--':>15s}"
        print(row)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["synthetic", "real"])
    ap.add_argument("--config", default="config.yml")
    ap.add_argument("--snapshots", nargs="*", default=[],
                    help="real mode: list of PATH[:TIMESTAMP] snapshot specs")
    ap.add_argument("--cpe", default=None, help="real mode: CPE log CSV")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if not HAS_LTN:
        print("[note] torch/LTNtorch not importable -- running K-means + Monet only. "
              "Install torch and LTNtorch to include LogiK-Net.")
    if args.mode == "synthetic":
        run_synthetic(cfg, args.out)
    else:
        run_real(cfg, args.snapshots, args.cpe, args.out)
