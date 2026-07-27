# ============================================================================
# run_step2.py -- congestion-region detection experiment (simplified scope).
#
# Goal (per the revision plan): show that the "K-means then LTN labeler"
#   (1) DIFFERENTIATES congested vs non-congested snapshots, and
#   (2) recovers the correct congestion REGION when one is present,
# scored against SYNTHETIC ground truth (the only ground truth we use).
# Real snapshots are run only for a qualitative sanity check (no scoring).
#
# Usage:
#   python run_step2.py --config config.yml                 # synthetic scoring
#   python run_step2.py --config config.yml \
#          --real snapshots/snap_1489510680.csv:1489510680   # + real qualitative
# ============================================================================
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import yaml

import synthetic as syn
import cluster_label as CL
import metrics as M


def run_synthetic(cfg, n_each=25, seed=0, grow=False, growth_floor=15.0):
    dims = tuple(cfg["torus"]["dims"])
    s = cfg["synthetic"]; th = cfg["thresholds"]; k = cfg["clustering"]["k"]
    rng = np.random.default_rng(seed)

    conf = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    seg = []
    for i in range(2 * n_each):
        congested = i < n_each
        tab, lab = syn.generate_sample(
            dims=dims,
            regions_min=(s["regions_min"] if congested else 0),
            regions_max=(s["regions_max"] if congested else 0),
            cuboid_min=s["cuboid_min"], cuboid_max=s["cuboid_max"],
            stall_min=s["stall_min"], stall_max=s["stall_max"],
            noise_mu=s["noise_mu"], noise_sigma=s["noise_sigma"],
            significance_pts=s["significance_pts"], rng=rng)
        true_bin = (lab > 0).astype(int)
        res = CL.kmeans_then_label(tab, k=k, high_band=th_high(cfg),
                                   th_similarity=th["th_similarity"], seed=0,
                                   grow=grow, dims=dims, growth_floor=growth_floor)
        truth = bool(true_bin.any())
        pred = res["congested"]
        conf["TP" if truth and pred else "FN" if truth else
             "FP" if pred else "TN"] += 1
        if truth:
            seg.append(M.region_segmentation(true_bin, res["node_pred"]))

    acc = (conf["TP"] + conf["TN"]) / sum(conf.values())
    print("\n=== Differentiation (congested vs non-congested) ===")
    print(f"  {conf}   accuracy={acc:.3f}")
    fpr = conf["FP"] / max(conf["FP"] + conf["TN"], 1)
    tpr = conf["TP"] / max(conf["TP"] + conf["FN"], 1)
    print(f"  detection rate (TPR)={tpr:.3f}  false-alarm rate (FPR)={fpr:.3f}")

    df = pd.DataFrame(seg)
    print("\n=== Region correctness on congested snapshots (mean +/- std) ===")
    for m in ["precision", "recall", "f1", "iou"]:
        print(f"  {m:10s} {df[m].mean():.3f} +/- {df[m].std():.3f}")
    return {"confusion": conf, "accuracy": acc,
            "region": {m: (float(df[m].mean()), float(df[m].std())) for m in
                       ["precision", "recall", "f1", "iou"]}}


def run_real(cfg, spec, grow=False, growth_floor=15.0):
    import data_loader as DL
    path, _, ts = spec.partition(":")
    ts = int(ts) if ts else None
    tab = DL.load_snapshot(path, timestamp=ts, metric=cfg["data"]["metric"],
                           pts_scale=float(cfg["data"]["pts_scale"]))
    res = CL.kmeans_then_label(tab, k=cfg["clustering"]["k"],
                               high_band=th_high(cfg),
                               th_similarity=cfg["thresholds"]["th_similarity"], seed=0,
                               grow=grow, dims=tuple(cfg["torus"]["dims"]),
                               growth_floor=growth_floor)
    print(f"\n=== Real snapshot {path} (qualitative, no scoring) ===")
    print(f"  congested flag: {res['congested']} | nodes flagged: {int(res['node_pred'].sum())} "
          f"| nodes PTs>=High: {int((tab['PTs'] >= th_high(cfg)).sum())}")
    for c in sorted([c for c in res["clusters"] if c["labeled"]],
                    key=lambda d: -d["mean_pts"]):
        print(f"  cluster {c['id']:2d}: size={c['size']:3d} mean_pts={c['mean_pts']:.1f}% "
              f"Cong={c['Congested']:.2f}")
    return res


def th_high(cfg):
    return float(cfg["congestion_bands"]["High"][0])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yml")
    ap.add_argument("--n-each", type=int, default=25,
                    help="synthetic congested/quiet snapshots per class")
    ap.add_argument("--real", default=None, help="PATH:TIMESTAMP for a qualitative real run")
    ap.add_argument("--grow", action="store_true", help="apply Monet-style region growth")
    ap.add_argument("--growth-floor", type=float, default=15.0,
                    help="min PTs (%) for a node to be absorbed during growth")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run_synthetic(cfg, n_each=args.n_each, grow=args.grow, growth_floor=args.growth_floor)
    if args.real:
        run_real(cfg, args.real, grow=args.grow, growth_floor=args.growth_floor)
