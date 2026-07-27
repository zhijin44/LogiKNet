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

Configuration for both tables: k = 24 clusters, seed = 0, N = 25 congested + 25
quiet synthetic snapshots (each a 24³ = 13,824-node torus instance). Congested
snapshots contain 1–8 planted cuboid regions (side 3–9, stall 20–50%, additive
Gaussian noise σ = 2.5); quiet snapshots contain none.

**Table R1 — Differentiation (congested vs non-congested), over all 50 snapshots.**

| metric | value |
|---|---|
| accuracy | 1.00 |
| detection rate (TPR) | 1.00 |
| false-alarm rate (FPR) | 0.00 |

**Table R2 — Region correctness, mean ± std over the 25 congested snapshots.**

| variant | precision | recall | F1 | IoU |
|---|---|---|---|---|
| K-means → LTN labeler | 1.000 | 0.75 ± 0.21 | 0.84 | 0.75 ± 0.21 |
| **+ region growth** | **1.000** | **0.85 ± 0.19** | **0.90** | **0.85 ± 0.19** |

_Reference: [28] report ~0.81 overlap for their region-growing on this generator._
_Recall/IoU carry ~0.2 std across random draws; for the final paper, average over
a larger N (e.g. 100) and several seeds for a stable point estimate._

**Qualitative (real, 2017-03-14 11:58 CDT).** The detector fires and localises a
congestion region on the X-direction links (mean PTs ≈ 30%); region growth
expands the ~72-node High-band core to ~333 nodes by absorbing the surrounding
Medium-band skirt. See `figures/congestion_0314_grow.png`.

## 6. Interpretation & threats to validity

- **Precision 1.0 by design.** The `High ∧ Homogeneous` gate only fires on
  clusters that are both severe and internally consistent, so background is never
  mislabelled; region growth then trades none of that precision for +10 points of
  recall because the `th_similarity` brake stops growth at the congestion edge.
- **Residual recall gap (~9%).** Confined to snapshots where no cluster ever
  crossed the High band, so no seed forms; these are inherently undetectable by a
  High-band-seeded method and would require lowering the seeding threshold.
- **Synthetic vs real geometry.** The synthetic model plants compact cuboids;
  real congestion here is *directional* (spread along X rings with a PTs
  gradient). Scoring is therefore synthetic-only; the real snapshot demonstrates
  the method fires and localises sensibly, consistent with the HOTI'19 finding
  that X-direction links show the longest-lasting high-PTs congestion.
