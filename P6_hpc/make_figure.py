# ============================================================================
# make_figure.py -- qualitative figure of a real snapshot: the 3D-torus PTs
# field and the congestion region detected by the K-means -> LTN labeler.
#
#   python make_figure.py snapshots/snap_1489510680.csv:1489510680 \
#          --title "2017-03-14 11:58 CDT" --out figures/congestion_0314.png
# ============================================================================
from __future__ import annotations
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data_loader as DL
import cluster_label as CL


def make_figure(spec, title, out, k=24, high_band=25.0, th_similarity=4.0,
                pts_scale=1.0e6, floor=5.0, grow=False, growth_floor=15.0,
                dims=(24, 24, 24)):
    path, _, ts = spec.partition(":")
    ts = int(ts) if ts else None
    tab = DL.load_snapshot(path, timestamp=ts, metric="credit", pts_scale=pts_scale)
    res = CL.kmeans_then_label(tab, k=k, high_band=high_band,
                               th_similarity=th_similarity, seed=0,
                               grow=grow, dims=dims, growth_floor=growth_floor)
    x, y, z, pts = tab.x.values, tab.y.values, tab.z.values, tab.PTs.values
    flagged = res["node_pred"] == 1

    fig = plt.figure(figsize=(14, 6))

    # -- Panel A: PTs field (show only above a small floor; rest as faint bg) --
    axA = fig.add_subplot(1, 2, 1, projection="3d")
    bg = pts < floor
    axA.scatter(x[bg], y[bg], z[bg], s=2, c="0.85", alpha=0.06, linewidths=0)
    hi = ~bg
    sc = axA.scatter(x[hi], y[hi], z[hi], c=pts[hi], cmap="inferno",
                     s=8 + pts[hi], vmin=floor, vmax=max(pts.max(), high_band),
                     alpha=0.9, linewidths=0)
    cb = fig.colorbar(sc, ax=axA, shrink=0.6, pad=0.02)
    cb.set_label("Percent Time Stalled (credit) %")
    axA.set_title(f"(a) PTs field — {title}")
    _style(axA)

    # -- Panel B: detected congestion region ---------------------------------
    axB = fig.add_subplot(1, 2, 2, projection="3d")
    axB.scatter(x[~flagged], y[~flagged], z[~flagged], s=2, c="0.85",
                alpha=0.05, linewidths=0)
    # if region-grown, show the High-band seed core vs the grown skirt
    seed = res.get("node_seed", res["node_pred"]) == 1
    grown_only = flagged & ~seed
    if grow and grown_only.any():
        axB.scatter(x[grown_only], y[grown_only], z[grown_only], s=12, c="orange",
                    alpha=0.9, linewidths=0, label=f"grown skirt ({int(grown_only.sum())})")
        axB.scatter(x[seed], y[seed], z[seed], s=18, c="crimson",
                    alpha=0.95, linewidths=0, label=f"High-band seed ({int(seed.sum())})")
    else:
        axB.scatter(x[flagged], y[flagged], z[flagged], s=18, c="crimson",
                    alpha=0.95, linewidths=0, label=f"congested ({int(flagged.sum())} nodes)")
    subtitle = "K-means → LTN labeler" + (" → region growth" if grow else "")
    axB.set_title(f"(b) Detected congestion region\n({subtitle})")
    axB.legend(loc="upper right", fontsize=9)
    _style(axB)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"congested flag: {res['congested']} | flagged nodes: {int(flagged.sum())} "
          f"| nodes PTs>=High: {int((pts >= high_band).sum())}")
    print(f"figure written -> {out}")
    return out


def _style(ax):
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_xlim(0, 23); ax.set_ylim(0, 23); ax.set_zlim(0, 23)
    ax.view_init(elev=18, azim=-60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", help="PATH:TIMESTAMP")
    ap.add_argument("--title", default="")
    ap.add_argument("--out", default="figures/congestion.png")
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--grow", action="store_true", help="apply Monet-style region growth")
    ap.add_argument("--growth-floor", type=float, default=15.0)
    args = ap.parse_args()
    make_figure(args.spec, args.title, args.out, k=args.k,
                grow=args.grow, growth_floor=args.growth_floor)
