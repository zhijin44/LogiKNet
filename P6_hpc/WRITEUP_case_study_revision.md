# Revised HPC Case Study — Methods & Results Writeup

Drop-in material for the Section V-B revision and the response letter. Numbers in
**[TABLE]** placeholders are filled from `results/*.json` after running the
pipeline on the downloaded snapshots.

---

## 1. Response-letter paragraph (Reviewer 1, point 4 / Reviewer 2, evaluation)

> We thank the reviewers for pushing us toward a rigorous, reproducible
> evaluation of the congestion-monitoring case study. We have re-anchored the
> study on the **publicly released** Monet Blue Waters network dataset (Illinois
> Data Bank, DOI 10.13012/B2IDB-2921318_V1; 27,648 nodes, 24×24×24 Cray Gemini
> 3D torus, LDMS/gpcdr counters at 60-s intervals, Jan–May 2017), the same data
> source underlying our reference [28]. Because clustering of congestion regions
> has no operator-provided ground truth, we adopt the *same* evaluation strategy
> as [28] and extend it: (i) a **synthetic ground-truth benchmark** reproducing
> the generator of [28, Appendix D] on the identical 24³ torus, on which we
> report region overlap score, precision and recall (their metrics) together
> with ARI and NMI; (ii) **internal cluster-validity indices** (silhouette,
> Davies–Bouldin, Calinski–Harabasz) on real snapshots, which require no labels;
> and (iii) **external cross-verification** against logged Cray congestion-
> protection events (CPEs), which independently mark when and where congestion
> occurred. All results are reported as mean ± std over five seeds / multiple
> snapshots, with paired significance tests. Code and configuration are released
> for full reproducibility.

## 2. Why re-anchor (footnote / methods note)

The snapshot timestamps used in the original submission decode to 23 Apr 2014,
predating the only public release of this dataset (Jan–May 2017). We therefore
rebuilt the case study on the public 2017 release so that the telemetry and the
event markers are both reproducible from a citable DOI.

## 3. Methods

### 3.1 Data and features
Each released row is one node-sample: a Unix timestamp, six directional
credit-/inq-stall and used-bandwidth counters (percent × 1e6), and torus
coordinates `(x,y,z) ∈ [0,23]³`. We recover Percent-Time-Stalled (PTs, %) as the
credit-stall counter divided by 1e6, take the per-node PTs as the max over its
six directional links (per-axis PTs for the directional analyses of Fig. 14),
and collapse the two co-located compute IDs per Gemini by max. A snapshot is one
60-s bucket.

### 3.2 Logic-guided clustering (LogiK-Net)
Cluster membership `C(x,c)` is a soft assignment from an MLP that embeds
topological and congestion features, trained *only* to maximise the satisfaction
of the four axioms of Section V-B (coverage, non-empty clusters, spatial
closeness within `th_close = 2` hops, PTs dissimilarity above
`th_similarity = 4 %`). The closeness and dissimilarity axioms are evaluated over
the torus neighbour graph, consistent with the local nature of congestion. This
reuses the project's LTNtorch formulation (MLP → softmax logits →
`LogitsToPredicate` → `SatAgg`).

### 3.3 Baselines
- **K-means** on standardised `(x,y,z,PTs)` features (same *k*).
- **Monet region-growing** [28]: a faithful re-implementation of the four-stage
  region-growth segmentation — group neighbouring nodes with `|ΔPTs| ≤ θ_p`,
  merge adjacent regions with similar mean PTs (`θ_r`), absorb regions smaller
  than `σ` into the nearest neighbour, discard the rest — with the paper's
  parameters `θ_p = θ_r = 4, σ = 20, δ = 2`.

### 3.4 Metrics
- **Route A (synthetic ground truth).** We generate 100 snapshots on the 24³
  torus with 1–8 random cuboid regions (side 3–9), stall 20–50 %, and additive
  Gaussian noise N(0, 2.5), exactly as [28, Appendix D]. We report their
  region-overlap score `S = (1/n Σ IoU(A_i,B_i))·(n/max(n,m))`, precision, and
  recall, plus point-level ARI and NMI.
- **Route B (internal indices).** Silhouette, Davies–Bouldin, and
  Calinski–Harabasz on the standardised `(coords, PTs)` space of real snapshots.
- **Constraint satisfaction.** Fraction of neighbour pairs obeying the closeness
  axiom (similar-PTs neighbours in the same cluster) and the dissimilarity axiom
  (dissimilar-PTs neighbours in different clusters). This is reported alongside
  the internal indices because silhouette/Davies–Bouldin structurally reward the
  compact convex clusters that K-means optimises for; the constraint metrics
  capture the topological/semantic correctness that the logic-guided method
  targets, so the two together give a fair comparison.
- **CPE cross-verification.** For snapshots within ±5 min of a logged CPE, we
  test whether a *High* cluster (mean PTs ≥ 25 %) lies within `th_close` hops of
  the event coordinates (spatial recall / precision), and whether detected
  congestion coincides temporally with logged events.

## 4. Results tables (fill from results/)

**Table R1 — Route A, synthetic ground-truth benchmark (mean ± std, 100 samples).**

| Method | Overlap ↑ | Precision ↑ | Recall ↑ | ARI ↑ | NMI ↑ |
|---|---|---|---|---|---|
| K-means | [TABLE] | [TABLE] | [TABLE] | [TABLE] | [TABLE] |
| Monet region-growing [28] | [TABLE] | [TABLE] | [TABLE] | [TABLE] | [TABLE] |
| **LogiK-Net (ours)** | **[TABLE]** | **[TABLE]** | **[TABLE]** | **[TABLE]** | **[TABLE]** |

_Reference point: [28] report overlap 0.81, precision 0.87, recall 0.89 for their
own region-growing on this generator._

**Table R2 — Route B, internal validity + constraint satisfaction on real
snapshots (mean ± std).**

| Method | Silhouette ↑ | Davies–Bouldin ↓ | Calinski–Harabasz ↑ | Closeness-sat ↑ | Dissimilarity-sat ↑ |
|---|---|---|---|---|---|
| K-means | [TABLE] | [TABLE] | [TABLE] | [TABLE] | [TABLE] |
| Monet [28] | [TABLE] | [TABLE] | [TABLE] | [TABLE] | [TABLE] |
| **LogiK-Net (ours)** | [TABLE] | [TABLE] | [TABLE] | **[TABLE]** | **[TABLE]** |

**Table R3 — CPE cross-verification (snapshots within ±5 min of a logged event).**

| Method | CPE temporal hit | CPE spatial recall | High-cluster precision |
|---|---|---|---|
| K-means | [TABLE] | [TABLE] | [TABLE] |
| Monet [28] | [TABLE] | [TABLE] | [TABLE] |
| **LogiK-Net (ours)** | [TABLE] | [TABLE] | [TABLE] |

## 5. Interpretation notes (for the discussion)

- On **Route A**, structure-aware methods (Monet, LogiK-Net) should recover the
  planted regions far better than fixed-*k* geometric K-means, because K-means
  cannot match a variable number of arbitrarily shaped cuboid regions. This is
  the interpretable, accuracy-style number the reviewers asked for.
- On **Route B**, expect K-means to be competitive or better on silhouette /
  Davies–Bouldin (they reward its own objective), while LogiK-Net should lead on
  constraint satisfaction. The intended claim is therefore *"LogiK-Net attains
  cluster cohesion comparable to K-means while additionally satisfying the
  topological/congestion constraints — interpretability at no cohesion cost,"*
  not "LogiK-Net wins every geometric index."
- **CPE cross-verification** converts the previously qualitative "without
  sacrificing accuracy to the ground-truths" claim into a measurable one:
  detected congestion should coincide, in time and torus location, with
  independently logged protection events.

## 6. Threats to validity (short paragraph)
Internal indices favour convex, compact clusters and can be biased toward
K-means; we therefore report them jointly with constraint-satisfaction and the
synthetic benchmark. The synthetic model assumes locally-spreading, roughly
homogeneous congestion regions (the same assumption as [28]); results on real
snapshots and CPE agreement guard against over-fitting to that assumption. The
fixed number of clusters *k* for LogiK-Net and K-means is selected by the
knee-curve method on region count, matching [28].
