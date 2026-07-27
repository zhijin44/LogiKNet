# P6_hpc — Re-anchored HPC Congestion-Monitoring Case Study

Rebuilds the Section V-B HPC congestion case study on the **public** Monet Blue
Waters 2017 dataset and adds quantitative evaluation, addressing the reviewers'
request that the case study be reproducible and quantitative rather than
illustrative.

## Scope (simplified)

- **Goal:** show the detector can (1) **differentiate** congested vs
  non-congested snapshots, and (2) recover the **correct congestion region**
  when one is present.
- **Method:** `K-means → LTN labeler` (+ optional Monet-style region growth).
- **Ground truth:** **synthetic only** — the accuracy numbers come from a
  synthetic generator with known planted regions. Real snapshots are used for
  qualitative visualization.
- **Monet [28] is a *reference*, not a baseline.** We do not re-run or compare
  against Monet region-growing; we reuse its data-mining conclusions (the
  congestion bands, the thresholds `th_close = 2`, `th_similarity = 4%`, and the
  "small-but-severe regions" / directional-bandwidth findings) as fixed anchors.

## Why re-anchor

The figures in the original submission used snapshot timestamps `@1398220860…`
= 23 Apr 2014, which predate the only public release of this data (Jan–May
2017). Rebuilding on the public release makes the telemetry reproducible from a
citable DOI.

## Dataset

**Monet – Blue Waters Network Dataset**, Illinois Data Bank,
DOI [10.13012/B2IDB-2921318_V1](https://doi.org/10.13012/B2IDB-2921318_V1).
27,648 nodes on a 24×24×24 Cray Gemini 3D torus; LDMS/gpcdr counters at 60-s
intervals. Two practical facts about the day files:

- **Headerless CSV.** Column order is given by the `HEADER` file (`#Time`, six
  directional CREDIT/INQ stall + USED_BW counters in **percent × 1e6**, then
  `nettopo_mesh_coord_Z/Y/X`). `data_loader.py` auto-detects this and applies the
  canonical names. **PTs(%) = credit-stall ÷ 1e6.**
- **Not time-sorted.** Files are written in node/partition blocks, so each
  timestamp recurs throughout the file — snapshot extraction must full-scan.
  Fast path: `awk -F, -v ts=<t> '$1==ts' <day> > snap.csv`.

Three peak snapshots are already extracted in `snapshots/`:
`snap_1489386120` (03-13 01:22 CDT), `snap_1489510680` (03-14 11:58 CDT,
strongest), `snap_1489559700` (03-15 01:35 CDT).

## Method

1. **K-means (discovery).** Each node → `(x, y, z, PTs)`, standardised with PTs
   up-weighted so small high-PTs regions separate from the background; K-means
   into K clusters.
2. **LTN labeler (validation).** For each cluster, a fuzzy predicate (product
   t-norm, matching `ltn.fuzzy_ops.AndProd`):
   `Congested(c) = High(c) AND Homogeneous(c)`, where `High` = mean PTs in
   Monet's High band (≥25%) and `Homogeneous` = PTs spread ≤ `th_similarity`.
   Clusters with truth ≥ τ (0.5) are congestion; their union is the region, and
   the snapshot is congested iff any cluster fires. Evaluated in numpy
   (identical to an LTNtorch grounding; torch only needed to *learn* it).
3. **Region growth (optional, `--grow`).** From the labelled seeds, grow along
   torus adjacency, absorbing a neighbour with PTs ≥ `growth_floor` and within
   `th_similarity` of the local level — the same seed-and-grow rule as Monet.
   Recovers the tapering boundary K-means orphaned.

## Results (synthetic ground truth; k=24, seed=0)

Config: N = 25 congested + 25 quiet synthetic snapshots (each a 24³ = 13,824-node
torus instance).

Differentiation (congested vs non-congested), over all 50 snapshots:
**accuracy 1.00** (TPR 1.00, FPR 0.00).

Region correctness, mean ± std over the 25 congested snapshots:

| variant | precision | recall | F1 | IoU |
|---|---|---|---|---|
| seeds (K-means → LTN labeler) | 1.000 | 0.75 ± 0.21 | 0.84 | 0.75 ± 0.21 |
| **+ region growth** | **1.000** | **0.85 ± 0.19** | **0.90** | **0.85 ± 0.19** |

Precision is 1.0 throughout (the strict `High ∧ Homogeneous` gate never flags
background); region growth lifts recall/IoU by ~10 points at no precision cost.
For reference, Monet's own synthetic benchmark reported ~0.81 overlap.

> Note: recall/IoU carry ~0.2 std across random draws, so for the final paper
> run a larger N (e.g. `--n-each 100`) and average over a few seeds for a stable
> figure; the mean settles in the ranges above.

## Files

| file | role |
|------|------|
| `config.yml` | torus dims, thresholds, congestion bands, synthetic params |
| `torus.py` | torus distance / neighbour pairs / adjacency |
| `data_loader.py` | headerless-aware snapshot loader + `extract_snapshot` |
| `synthetic.py` | Monet-style synthetic congestion generator (ground truth) |
| `cluster_label.py` | K-means → LTN labeler + `region_grow` |
| `metrics.py` | `region_segmentation` (+ other indices) |
| `run_step2.py` | experiment: differentiation + region correctness, ± growth |
| `make_figure.py` | qualitative torus figure of a real snapshot |

## Running

```bash
cd P6_hpc
python run_step2.py --config config.yml --grow              # synthetic scores
python run_step2.py --config config.yml --grow \
    --real snapshots/snap_1489510680.csv:1489510680          # + real qualitative
python make_figure.py snapshots/snap_1489510680.csv:1489510680 \
    --title "2017-03-14 11:58 CDT" --grow --out figures/congestion_0314_grow.png
```

## Dependencies

`numpy scipy scikit-learn pandas pyyaml matplotlib`. No torch required — the LTN
labeler is evaluated in numpy. (torch + LTNtorch only if you later learn the
predicate parameters.)
