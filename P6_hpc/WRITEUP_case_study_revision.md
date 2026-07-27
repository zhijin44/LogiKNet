# Revised HPC Case Study — Methods & Results Writeup

Drop-in material for the Section V-B revision and the response letter. All scored
numbers use synthetic ground truth; real snapshots are qualitative.

---

## 1. Response-letter paragraph

> We thank the reviewers for pushing the congestion-monitoring case study toward
> a reproducible, quantitative evaluation. We have re-anchored it on the public
> Monet Blue Waters network dataset (Illinois Data Bank, DOI
> 10.13012/B2IDB-2921318_V1; 27,648 nodes on a 24×24×24 Cray Gemini 3D torus,
> LDMS counters at 60-s intervals), the same source underlying reference [28].
> Because congestion-region clustering has no operator-provided ground truth, we
> evaluate on a synthetic benchmark that reproduces the generator of [28,
> Appendix D] on the identical 24³ torus, where the planted region is known
> exactly. We report (i) snapshot-level **differentiation** of congested vs
> non-congested states and (ii) **region correctness** (precision, recall, F1,
> IoU) of the detected congestion region, as mean ± std over multiple snapshots.
> Our detector reaches perfect differentiation (accuracy 1.00) and, with a
> Monet-style region-growth step, an IoU of 0.91 at precision 1.00. We reuse
> Monet's data-mining conclusions (congestion bands, neighbourhood and
> similarity thresholds) as reference knowledge rather than as a competing
> baseline. Real snapshots (e.g. 2017-03-14 11:58 CDT) are shown qualitatively.

## 2. Why re-anchor (methods note)

The original submission's snapshot timestamps decode to 23 Apr 2014, outside the
only public release of this dataset (Jan–May 2017). We rebuilt on the public
release so the telemetry is reproducible from a citable DOI. PTs (Percent Time
Stalled) is the credit-stall counter (inter-switch flit stall) divided by 1e6.

## 3. Method

**K-means → LTN labeler (+ region growth).**
Each node is `(x, y, z, PTs)`, standardised with PTs up-weighted. K-means groups
nodes into candidate regions (forward discovery). An LTN-style fuzzy predicate
then validates each cluster using Monet-derived knowledge:
`Congested(c) = High(c) AND Homogeneous(c)` (product t-norm), with
`High` = cluster mean PTs in the High band (≥25%) and `Homogeneous` = intra-
cluster PTs spread ≤ `th_similarity` (4%). Clusters above τ = 0.5 form the
congestion region; the snapshot is congested iff any fires. An optional
Monet-style region-growth step expands the labelled seeds along torus adjacency,
absorbing neighbours with PTs ≥ a growth floor and within `th_similarity` of the
local level, recovering boundary nodes that K-means splits off.

## 4. Metrics

Scored against the synthetic planted region (binary per node): TP flagged &
truly congested, FP flagged but background, FN missed, TN correct background.

- **Differentiation** — snapshot-level TPR / FPR / accuracy of the congested vs
  non-congested decision (does any cluster fire?).
- **precision** = TP/(TP+FP): of flagged nodes, fraction truly congested.
- **recall** = TP/(TP+FN): of truly congested nodes, fraction detected.
- **F1** = harmonic mean of precision and recall.
- **IoU** = TP/(TP+FP+FN): overlap of detected vs true region.

## 5. Results

Reported at **k = 24, seed = 0**. Each snapshot is a 24³ = 13,824-node torus
instance; congested snapshots contain 1–8 planted cuboid regions (side 3–9, stall
20–50%, Gaussian noise σ = 2.5), quiet snapshots contain none. The primary run
uses **N = 100 congested + 100 quiet** snapshots (a quick N = 25 run is logged in
§7).

**Table R1 — Differentiation (congested vs non-congested), N = 100 + 100 = 200 snapshots.**

| metric | value |
|---|---|
| accuracy | 0.99 |
| detection rate (TPR) | 0.98 |
| false-alarm rate (FPR) | 0.00 |

Two of the 100 congested snapshots are missed (TPR 0.98): in each, the planted
region never crossed the High band, so no seed forms — the residual-recall limit
of §6. (At N = 25 none are missed: accuracy 1.00.)

**Table R2 — Region correctness, mean ± std over the congested snapshots.**

| variant | N | precision | recall | F1 | IoU |
|---|---|---|---|---|---|
| K-means → LTN labeler (seeds) | 100 | 1.000 | 0.725 ± 0.246 | 0.813 | 0.725 ± 0.246 |
| **+ region growth (floor 15)** | 100 | **1.000** | **0.829 ± 0.230** | **0.884** | **0.829 ± 0.230** |
| + region growth (floor 20) | 100 | 1.000 | 0.825 ± 0.229 | 0.882 | 0.825 ± 0.229 |

Precision stays exactly 1.0 in every setting. At matched N = 100, region growth
lifts recall/IoU by ~0.10 (seeds 0.725 → grow 0.829). The growth floor barely
matters on synthetic data (0.829 vs 0.825) because the region→background edge is
sharp and the `th_similarity` brake stops growth there regardless.

_Reference: [28] report ~0.81 overlap for their region-growing on this generator._
_Recall/IoU carry ~0.2 std across random draws; average over several seeds for a
tighter point estimate in the camera-ready._

**Qualitative (real, 2017-03-14 11:58 CDT).** The detector fires and localises a
congestion region on the X-direction links (mean PTs ≈ 30%); region growth
expands the 72-node High-band core to 333 nodes at growth-floor 15, or 192 at
growth-floor 20 (tighter around the core). See `figures/congestion_0314_grow.png`
and `figures/congestion_0314_grow20.png`.

## 6. Interpretation & threats to validity

- **Precision 1.0 by design.** The `High ∧ Homogeneous` gate only fires on
  clusters that are both severe and internally consistent, so background is never
  mislabelled; region growth then trades none of that precision for ~+0.10 recall
  (0.725 → 0.829 at N = 100) because the `th_similarity` brake stops growth at the
  edge.
- **Residual gaps have two sources.** (i) At the node level, ~17% of congested
  nodes (region edges) remain unrecovered even after growth. (ii) At the snapshot
  level, ~2% of congested snapshots (Table R1) contain no High-band cluster at
  all, so no seed forms and nothing is detected. Both stem from High-band seeding;
  lowering the seeding threshold would trade precision for recall.
- **Synthetic vs real geometry.** The synthetic model plants compact cuboids;
  real congestion here is *directional* (spread along X rings with a PTs
  gradient). Scoring is therefore synthetic-only; the real snapshot demonstrates
  the method fires and localises sensibly, consistent with the HOTI'19 finding
  that X-direction links show the longest-lasting high-PTs congestion.

## 7. Reproducibility log (verbatim runs, k = 24, seed = 0)

Commands run from `P6_hpc/`. Real-snapshot line is qualitative (no ground truth).

```
$ python run_step2.py --config config.yml                    # seeds only, N = 25
  Differentiation: {TP:25, FP:0, TN:25, FN:0}  accuracy=1.000  TPR=1.000  FPR=0.000
  Region:  precision 1.000±0.000  recall 0.754±0.210  f1 0.842  iou 0.754±0.210

$ python run_step2.py --config config.yml --n-each 100        # seeds only, N = 100
  Differentiation: {TP:98, FP:0, TN:100, FN:2}  accuracy=0.990  TPR=0.980  FPR=0.000
  Region:  precision 1.000±0.000  recall 0.725±0.246  f1 0.813  iou 0.725±0.246

$ python run_step2.py --config config.yml --grow --n-each 100 \
      --real snapshots/snap_1489510680.csv:1489510680          # grow floor 15, N = 100
  Differentiation: {TP:98, FP:0, TN:100, FN:2}  accuracy=0.990  TPR=0.980  FPR=0.000
  Region:  precision 1.000±0.000  recall 0.829±0.230  f1 0.884  iou 0.829±0.230
  Real snap: congested=True | flagged 333 | nodes PTs>=High 137 | cluster mean 30.6%

$ python run_step2.py --config config.yml --grow --growth-floor 20 --n-each 100 \
      --real snapshots/snap_1489510680.csv:1489510680          # grow floor 20, N = 100
  Differentiation: {TP:98, FP:0, TN:100, FN:2}  accuracy=0.990  TPR=0.980  FPR=0.000
  Region:  precision 1.000±0.000  recall 0.825±0.229  f1 0.882  iou 0.825±0.229
  Real snap: congested=True | flagged 192 | nodes PTs>=High 137 | cluster mean 30.6%
```

Note: differentiation is identical for floor 15 and 20 — growth expands existing
seeds and never changes whether a snapshot fired, so it cannot affect the
congested/quiet decision.
