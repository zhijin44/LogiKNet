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


def _load(spec, pts_scale):
    path, _, ts = spec.partition(":")
    ts = int(ts) if ts else None
    return DL.load_snapshot(path, timestamp=ts, metric="credit", pts_scale=pts_scale)


def _draw_field(fig, ax, x, y, z, pts, floor, high_band, title, fs=1.0):
    """Panel (a): the measured PTs field. `fs` scales all type, so a wider
    figure can be shrunk harder in LaTeX and still read at the same size."""
    bg = pts < floor
    ax.scatter(x[bg], y[bg], z[bg], s=2, c="0.85", alpha=0.06, linewidths=0)
    hi = ~bg
    sc = ax.scatter(x[hi], y[hi], z[hi], c=pts[hi], cmap="inferno",
                    s=8 + pts[hi], vmin=floor, vmax=max(pts.max(), high_band),
                    alpha=0.9, linewidths=0)
    cb = fig.colorbar(sc, ax=ax, shrink=0.62, aspect=20, pad=0.03)
    cb.set_label("Percent Time Stalled (credit) %", fontsize=LBL_FS * fs)
    cb.ax.tick_params(labelsize=TICK_FS * fs)
    ax.set_title(title, fontsize=TITLE_FS * fs, y=TITLE_Y)
    _style(ax, fs)


def _draw_region(ax, x, y, z, res, title, grow, fs=1.0, legend=True):
    """Region panel: labelled core, plus the nodes recovered by growth."""
    flagged = res["node_pred"] == 1
    core = res.get("node_seed", res["node_pred"]) == 1
    grown_only = flagged & ~core
    ax.scatter(x[~flagged], y[~flagged], z[~flagged], s=2, c="0.85",
               alpha=0.05, linewidths=0)
    if grow and grown_only.any():
        ax.scatter(x[grown_only], y[grown_only], z[grown_only], s=12, c="orange",
                   alpha=0.9, linewidths=0,
                   label=f"recovered by growth ({int(grown_only.sum())})")
        ax.scatter(x[core], y[core], z[core], s=18, c="crimson",
                   alpha=0.95, linewidths=0,
                   label=f"High-band core ({int(core.sum())})")
    else:
        ax.scatter(x[flagged], y[flagged], z[flagged], s=18, c="crimson",
                   alpha=0.95, linewidths=0,
                   label=f"labelled congested ({int(flagged.sum())})")
    ax.set_title(title, fontsize=TITLE_FS * fs, y=TITLE_Y)
    if legend:
        ax.legend(loc="upper left", fontsize=LBL_FS * fs, framealpha=0.9,
                  borderpad=0.3, handletextpad=0.2, labelspacing=0.25,
                  bbox_to_anchor=(-0.02, 0.86))
    _style(ax, fs)


def _save(fig, out):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.02)
    if not out.lower().endswith(".pdf"):
        fig.savefig(os.path.splitext(out)[0] + ".pdf",
                    bbox_inches="tight", pad_inches=0.02)
    print(f"figure written -> {out}")


def make_figure4(spec, title, out, k=24, high_band=25.0, th_similarity=4.0,
                 pts_scale=1.0e6, floor=5.0, growth_floors=(15.0, 20.0),
                 dims=(24, 24, 24)):
    """Four-panel figure: PTs field, region without growth, region at two
    growth floors -- the three rows of the region-correctness table."""
    tab = _load(spec, pts_scale)
    x, y, z, pts = tab.x.values, tab.y.values, tab.z.values, tab.PTs.values

    def run(grow, gf):
        return CL.kmeans_then_label(tab, k=k, high_band=high_band,
                                    th_similarity=th_similarity, seed=0,
                                    grow=grow, dims=dims, growth_floor=gf)

    fig = plt.figure(figsize=(11.0, 8.4))
    axes = [fig.add_subplot(2, 2, i, projection="3d") for i in range(1, 5)]

    _draw_field(fig, axes[0], x, y, z, pts, floor, high_band,
                f"(a) PTs field, {title}")

    r_none = run(False, growth_floors[0])
    _draw_region(axes[1], x, y, z, r_none, "(b) Without growth", grow=False)

    for ax, gf, tag in zip(axes[2:], growth_floors, ("(c)", "(d)")):
        r = run(True, gf)
        _draw_region(ax, x, y, z, r, f"{tag} Region growth, floor {int(gf)}",
                     grow=True)
        print(f"  floor {int(gf)}: flagged {int((r['node_pred'] == 1).sum())}")

    fig.subplots_adjust(left=0.01, right=0.93, bottom=0.01, top=0.96,
                        wspace=-0.04, hspace=0.04)
    _save(fig, out)
    print(f"  no growth: flagged {int((r_none['node_pred'] == 1).sum())} "
          f"| nodes PTs>=High: {int((pts >= high_band).sum())}")
    return out


def make_figure_row(spec, title, out, k=24, high_band=25.0, th_similarity=4.0,
                    pts_scale=1.0e6, floor=5.0, growth_floors=(15.0, 20.0),
                    dims=(24, 24, 24)):
    """Same four panels as make_figure4, laid out 1x4. The panels are narrow,
    so node counts move into the subtitles and the per-panel legends are
    replaced by one shared legend under the row."""
    tab = _load(spec, pts_scale)
    x, y, z, pts = tab.x.values, tab.y.values, tab.z.values, tab.PTs.values
    fs = 2.0   # 1x4 is shrunk ~2x harder than 2x2 in LaTeX; scale type to match

    def run(grow, gf):
        return CL.kmeans_then_label(tab, k=k, high_band=high_band,
                                    th_similarity=th_similarity, seed=0,
                                    grow=grow, dims=dims, growth_floor=gf)

    fig = plt.figure(figsize=(22.0, 5.6))
    axes = [fig.add_subplot(1, 4, i, projection="3d") for i in range(1, 5)]

    _draw_field(fig, axes[0], x, y, z, pts, floor, high_band,
                "(a) PTs field", fs=fs)

    r_none = run(False, growth_floors[0])
    n0 = int((r_none["node_pred"] == 1).sum())
    _draw_region(axes[1], x, y, z, r_none, f"(b) Without growth ({n0})",
                 grow=False, fs=fs, legend=False)

    handles = None
    for ax, gf, tag in zip(axes[2:], growth_floors, ("(c)", "(d)")):
        r = run(True, gf)
        n = int((r["node_pred"] == 1).sum())
        _draw_region(ax, x, y, z, r,
                     f"{tag} Growth, floor {int(gf)} ({n})",
                     grow=True, fs=fs, legend=False)
        handles = ax.get_legend_handles_labels()[0]
        print(f"  floor {int(gf)}: flagged {n}")

    fig.legend(handles, ["recovered by growth", f"High-band core ({n0})"],
               loc="lower center", ncol=2, fontsize=LBL_FS * fs,
               frameon=False, bbox_to_anchor=(0.5, 0.0),
               handletextpad=0.3, columnspacing=2.5)

    fig.subplots_adjust(left=0.005, right=0.965, bottom=0.17, top=0.92,
                        wspace=-0.02)
    _save(fig, out)
    print(f"  no growth: flagged {n0} "
          f"| nodes PTs>=High: {int((pts >= high_band).sum())}")
    return out


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

    fig = plt.figure(figsize=(11.0, 4.4))

    # -- Panel A: PTs field (show only above a small floor; rest as faint bg) --
    axA = fig.add_subplot(1, 2, 1, projection="3d")
    bg = pts < floor
    axA.scatter(x[bg], y[bg], z[bg], s=2, c="0.85", alpha=0.06, linewidths=0)
    hi = ~bg
    sc = axA.scatter(x[hi], y[hi], z[hi], c=pts[hi], cmap="inferno",
                     s=8 + pts[hi], vmin=floor, vmax=max(pts.max(), high_band),
                     alpha=0.9, linewidths=0)
    cb = fig.colorbar(sc, ax=axA, shrink=0.62, aspect=20, pad=0.03)
    cb.set_label("Percent Time Stalled (credit) %", fontsize=LBL_FS)
    cb.ax.tick_params(labelsize=TICK_FS)
    axA.set_title(f"(a) PTs field, {title}", fontsize=TITLE_FS, y=TITLE_Y)
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
                    alpha=0.9, linewidths=0,
                    label=f"recovered by growth ({int(grown_only.sum())})")
        axB.scatter(x[seed], y[seed], z[seed], s=18, c="crimson",
                    alpha=0.95, linewidths=0,
                    label=f"High-band core ({int(seed.sum())})")
    else:
        axB.scatter(x[flagged], y[flagged], z[flagged], s=18, c="crimson",
                    alpha=0.95, linewidths=0, label=f"congested ({int(flagged.sum())} nodes)")
    axB.set_title("(b) Detected congestion region", fontsize=TITLE_FS, y=TITLE_Y)
    axB.legend(loc="upper left", fontsize=LBL_FS, framealpha=0.9,
               borderpad=0.3, handletextpad=0.2, labelspacing=0.25,
               bbox_to_anchor=(-0.02, 0.84))
    _style(axB)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    # 3-D axes ignore tight_layout; pack the panels manually instead.
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.03, top=0.97, wspace=-0.06)
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.02)
    if not out.lower().endswith(".pdf"):
        fig.savefig(os.path.splitext(out)[0] + ".pdf",
                    bbox_inches="tight", pad_inches=0.02)
    print(f"congested flag: {res['congested']} | flagged nodes: {int(flagged.sum())} "
          f"| nodes PTs>=High: {int((pts >= high_band).sum())}")
    print(f"figure written -> {out}")
    return out


# --- shared typography / tick density -------------------------------------
TITLE_FS = 13   # panel subtitles
LBL_FS = 10     # axis labels, colorbar label, legend
TICK_FS = 9     # tick labels
TICKS = [0, 8, 16, 23]   # sparse ticks on a 24-node torus dimension
TITLE_Y = 1.02  # lift subtitles clear of the 3-D box and its tick labels


def _style(ax, fs=1.0):
    ax.set_xlabel("X", fontsize=LBL_FS * fs, labelpad=-4)
    ax.set_ylabel("Y", fontsize=LBL_FS * fs, labelpad=-4)
    # matplotlib's 3-D z-label falls outside the tight bbox and is clipped in
    # some panels; place it manually so every panel is labelled identically.
    ax.set_zlabel("")
    ax.text2D(0.98, 0.56, "Z", transform=ax.transAxes, fontsize=LBL_FS * fs,
              ha="left", va="center")
    ax.set_xlim(0, 23); ax.set_ylim(0, 23); ax.set_zlim(0, 23)
    ax.set_xticks(TICKS); ax.set_yticks(TICKS); ax.set_zticks(TICKS)
    ax.tick_params(labelsize=TICK_FS * fs, pad=-2)
    ax.view_init(elev=18, azim=-60)
    try:                       # matplotlib >= 3.6: fill the panel box
        ax.set_box_aspect((1, 1, 1), zoom=1.12)
    except TypeError:
        ax.set_box_aspect((1, 1, 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", help="PATH:TIMESTAMP")
    ap.add_argument("--title", default="")
    ap.add_argument("--out", default="figures/congestion.png")
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--grow", action="store_true", help="apply Monet-style region growth")
    ap.add_argument("--growth-floor", type=float, default=15.0)
    ap.add_argument("--four", action="store_true",
                    help="2x2 figure: field, no growth, growth at two floors")
    ap.add_argument("--row", action="store_true",
                    help="same four panels laid out 1x4")
    args = ap.parse_args()
    if args.row:
        make_figure_row(args.spec, args.title, args.out, k=args.k)
    elif args.four:
        make_figure4(args.spec, args.title, args.out, k=args.k)
    else:
        make_figure(args.spec, args.title, args.out, k=args.k,
                    grow=args.grow, growth_floor=args.growth_floor)
