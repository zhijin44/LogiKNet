# LogiK-Net — Major Revision Results Ledger

Single aggregation point for all multi-seed / statistical-rigor results feeding
`logik-net-paper/Major-Revision-Response.tex` and `Results.tex`.

**Naming is normative.** Model names in this file follow the macros defined in
`Major-Revision-Response.tex` / `main.tex`. Experiment-side tags (`AttnKAN+LTN`,
`ltn_seed0.pt`, …) are recorded only in the *Source* column, never in a table
that will be transcribed into the paper.

| Paper macro | Rendered name | Experiment tag |
|---|---|---|
| — | MLP | `mlp` |
| — | Logic-MLP | `logic_mlp` |
| — | KAN (no logic) | `kan` |
| `\logickan` | Logic-KAN | `logiKNet` |
| `\hlogickan` | H-Logic-KAN | `hierarchical_logiKNet` |
| `\hlogickanp` | H-Logic-KAN* | — |
| `\attnlogiknet` | Attn-LogiK-Net (no logic) | `AttnKAN(noLTN)` / `noltn_seed*.pt` |
| `\attnlogiknet` | Attn-LogiK-Net (full) | `AttnKAN+LTN` / `ltn_seed*.pt` |
| `\ourmethod` | LogiK-Net | framework name |

---

## 0. Status board

| Block | Experiment | Protocol | Seeds | Status |
|---|---|---|---|---|
| P5 | Attn-LogiK-Net full vs. no-logic | 6-class / 18-feat | 5 | **done** |
| P1 | Model comparison (Fig. `P1-all-model`) | 13-class / 9-feat | 5 | *pending* |
| P2 | Feature-count comparison (14/16/18) | per-set | 5 | *pending* |
| P1 | Reliability-score figure | 13-class / 9-feat | 1 | not scheduled |
| P2 | Data-size comparison (S/M/L) | per-size | 1 | not scheduled |

**Agreed metric scope for the pending re-runs:** all three groups —
Predictive + Calibration + Hierarchical (see §1).
**Agreed protocol:** P1 and P2 are retrained at their *original* settings
(13-class/9-feature for P1; native feature sets for P2). P5 remains a separate
6-class ablation. `CHILD_TO_PARENT_4_9` must be re-added to `eval_metrics.py`
(referenced in `P5_attention/INTEGRATION.md` §2 but absent from the current file,
which only defines `CHILD_TO_PARENT_6`).

---

## 1. Metric definitions in force

All computed by `P5_attention/eval_metrics.py::evaluate_run`.

### 1.1 Predictive (flat — the class tree is invisible to these)

| Metric | Definition | Reviewer demand |
|---|---|---|
| `accuracy` | fraction of exact matches | continuity with v1 |
| `macro_f1` | per-class F1, unweighted mean | R1 #3 |
| `weighted_f1` | per-class F1, support-weighted | supporting |
| `macro_recall` | mean per-class recall | R1 #3 |
| `per_class_recall` | the per-class vector behind it | R1 #3 |
| `macro_fpr` | mean one-vs-rest FP/(FP+TN) | R1 #3 |
| `macro_auroc` | one-vs-rest AUROC, macro-averaged | R1 #3 |

### 1.2 Calibration (uses the full probability vector, not just argmax)

| Metric | Definition | Reviewer demand |
|---|---|---|
| `ece` | 15-bin top-label expected calibration error | R1 #3 ("a calibration metric") |
| `brier` | multiclass Brier score (proper scoring rule) | supporting |
| `reliability_diagram()` | per-bin (conf, acc, n) for plotting | optional figure |

### 1.3 Hierarchical (uses `CHILD_TO_PARENT`)

| Metric | Definition | Reviewer demand |
|---|---|---|
| `reliability` | 1.0 exact match; `partial` (=0.5) if parent correct but fine class wrong | the paper's bespoke score |
| `hierarchical_f1` | Kiritchenko: label → {self, parent}, set-overlap P/R/F1 | R1 #3 ("at minimum accompanied by") |

> ### ⚠ Identity warning — must not be misreported
>
> For a single-parent two-level tree with `partial = 0.5`, these two are the
> **same statistic**, not two independent measurements:
>
> extending every label to `{self, parent}` gives `|s_t| = |s_p| = 2` for all
> samples, so precision = recall = `Σinter / 2N`, hence
> `hierarchical_f1 = Σinter / 2N`. Simultaneously
> `reliability = (n_exact + 0.5·n_same_parent)/N = (2·n_exact + n_same_parent)/2N
> = Σinter / 2N`. **Identical by construction.**
>
> This is confirmed empirically in P5 to 15 decimal places (§2.1) and will
> reproduce in P1. Therefore:
> - Do **not** write "hierarchical-F1 corroborates the reliability score."
> - **Do** state the equivalence explicitly — it is the strongest available
>   answer to "the 0.5 partial-credit is ad hoc": the choice `partial = 0.5` is
>   precisely the value at which the bespoke score coincides with the standard
>   metric, i.e. it is not arbitrary but canonical.
> - The `partial ∈ {0.25, 0.5, 0.75}` sweep is what actually separates them and
>   is where the sensitivity argument lives.

### 1.4 Statistical rigor (R1 #2 — the comment this ledger exists to close)

| Tool | Use |
|---|---|
| `set_seed(seed)` | seeds python / numpy / torch, `cudnn.deterministic=True`, `benchmark=False` |
| `MetricTracker` | per-seed accumulation → mean ± std (`ddof=1`) |
| `wilcoxon_compare` | paired Wilcoxon signed-rank **across seeds**, one metric |
| `mcnemar_test` / `mcnemar_across_seeds` | paired McNemar **per sample**, per seed |

Minimum bar set by the reviewer: **≥ 5 seeds, mean ± std, one paired test.**
Test loaders must be built with `shuffle=False` so McNemar pairing is valid.

---

## 2. P5 — Attn-LogiK-Net (DONE)

**Protocol.** CICIoMT2024 MQTT setting. 10,000 train / 3,994 test flows,
18 features, 6 classes. `KAN_WIDTH = [18, 6, 6, 6]`, grid 5, k 3.
Attention encoder `d_model=32, n_heads=4, n_layers=2, dropout=0.1, residual=True`.
Adam, lr `1e-3`, full-batch. Seeds `[0,1,2,3,4]`.

**Epoch budget.** Fixed per variant at (measured convergence epoch) × 1.05:
Attn-LogiK-Net (full) = **656** epochs; (no logic) = **885** epochs.
*Note: `MARGIN = 0.05` in `run_multiseed.ipynb`, but the docstring and
`INTEGRATION.md` say ×1.10. The saved JSON confirms 656/885, i.e. ×1.05 was what
actually ran. Report ×1.05.*

**Source.** `P5_attention/saved/multiseed_results.json`, `.../multiseed_results.txt`,
checkpoints `P5_attention/saved/models/{ltn,noltn}_seed{0..4}.pt`.

### 2.1 Mean ± std over 5 seeds

| Metric | Attn-LogiK-Net (full) | Attn-LogiK-Net (no logic) |
|---|---|---|
| accuracy | 0.7855 ± 0.0333 | **0.8030 ± 0.0112** |
| macro-F1 | 0.7802 ± 0.0374 | **0.8020 ± 0.0109** |
| weighted-F1 | 0.7800 ± 0.0377 | **0.8022 ± 0.0109** |
| macro recall | 0.7883 ± 0.0326 | **0.8044 ± 0.0112** |
| macro FPR ↓ | 0.0427 ± 0.0066 | **0.0393 ± 0.0022** |
| macro AUROC | 0.9656 ± 0.0060 | **0.9693 ± 0.0039** |
| ECE ↓ | 0.0387 ± 0.0140 | **0.0228 ± 0.0122** |
| Brier ↓ | 0.2982 ± 0.0369 | **0.2836 ± 0.0169** |
| reliability | 0.8726 ± 0.0203 | **0.8816 ± 0.0076** |
| hierarchical-F1 | 0.8726 ± 0.0203 | **0.8816 ± 0.0076** |

Full precision (for LaTeX transcription):

```
                     accuracy   macro_f1  weighted_f1  macro_rec   macro_fpr
full     mean    0.785528292  0.780205377  0.780019247  0.788318974  0.042725666
         std     0.033269156  0.037426085  0.037713638  0.032621831  0.006615972
no-logic mean    0.802954432  0.802017027  0.802230342  0.804400913  0.039304872
         std     0.011190135  0.010854967  0.010852270  0.011175102  0.002235422

                   macro_auroc        ece        brier   reliability   hier_f1
full     mean    0.965597717  0.038679965  0.298217082  0.872633951  0.872633951
         std     0.005953343  0.014016855  0.036867002  0.020339952  0.020339952
no-logic mean    0.969294832  0.022816166  0.283616355  0.881622434  0.881622434
         std     0.003859436  0.012218548  0.016871343  0.007582185  0.007582185
```

### 2.2 Paired Wilcoxon signed-rank across the 5 seeds

| Metric | mean (full) | mean (no logic) | W | p |
|---|---|---|---|---|
| accuracy | 0.7855 | 0.8030 | 6.0 | **0.8125** |
| macro-F1 | 0.7802 | 0.8020 | 6.0 | **0.8125** |
| reliability | 0.8726 | 0.8816 | 6.0 | **0.8125** |

> **Caveat that must be stated in the letter.** With n = 5 the smallest
> attainable two-sided Wilcoxon p is 0.0625; the test cannot reach p < 0.05 at
> any effect size. p = 0.8125 is therefore evidence of *no detectable
> difference*, **not** evidence of equivalence, and the test is underpowered by
> construction. The current draft text ("statistically indistinguishable … under
> a paired Wilcoxon signed-rank test") is defensible only if this limitation is
> acknowledged. Options: (a) state the floor explicitly, (b) raise to 10 seeds
> so p < 0.05 becomes attainable, (c) report a TOST / confidence interval on the
> difference, which is the correct instrument for an equivalence claim.

### 2.3 Per-seed McNemar (paired at the sample level, 3,994 test flows)

`b01` = full wrong / no-logic right; `b10` = full right / no-logic wrong.

| Seed | b01 | b10 | χ² | p |
|---|---|---|---|---|
| 0 | 93 | 138 | 8.381 | 3.79e-03 |
| 1 | 459 | 136 | 174.259 | 8.69e-40 |
| 2 | 330 | 171 | 49.828 | 1.68e-12 |
| 3 | 91 | 152 | 14.815 | 1.19e-04 |
| 4 | 151 | 179 | 2.209 | 1.37e-01 |

Significant in **4 / 5** seeds. Direction is *not* consistent (seeds 0, 3, 4
favour the full model; seeds 1, 2 favour no-logic), which is exactly the
"same aggregate score, different errors" reading in the letter. That
inconsistency is worth stating — it strengthens the argument rather than
weakening it, because a consistent direction would contradict the
accuracy-neutrality claim.

### 2.4 Other reported figures

- Test satisfaction (full model): **0.644** — single run, from
  `AttentionEncoder.ipynb`. *Not multi-seed.* Either mark as single-run in the
  letter or recompute across the 5 saved `ltn_seed*.pt` checkpoints.
- Trainable parameters: **21.5 K** (letter §(a)). Verify against
  `sum(p.numel() for p in model.parameters() if p.requires_grad)`.

### 2.5 Known gaps in P5

1. **`per_class_recall` is not persisted.** `evaluate_run` computes it, but
   `MetricTracker.SCALAR_KEYS` excludes it, so it never reaches
   `multiseed_results.json`. The reviewer explicitly asked for per-class recall.
   Recoverable by re-running inference on the 10 saved checkpoints — see
   `P5_attention/attention_figure.ipynb`, which already reloads them.
2. **No `partial` sweep.** Only `PARTIAL = 0.5` was run. Needed for the
   "0.5 is ad hoc" rebuttal (§1.3).
3. **No standalone Transformer baseline.** `TransformerClassifier` exists in
   `attention_modules.py` and INTEGRATION.md's ablation matrix lists it, but it
   was never run. The letter claims to have "adopted the architectural idea" of
   [1]/[4]; a reader may still ask for the pure-attention row.
4. **The claim "both Attn-LogiK-Net variants improve over the corresponding
   non-attention LogiK-Net configurations" is currently unsupported** — the
   non-attention rows are `[TODO]` in the response table. This claim is
   load-bearing for R1 #1 and cannot ship until §3 is filled in.
5. `MARGIN` docstring/README inconsistency (×1.10 vs actual ×1.05) — see §2.

---

## 3. P1 — Model comparison (PENDING)

Target: `Results.tex` Fig. `P1-all-model`, and the `[TODO]` cells in
`Major-Revision-Response.tex` §Reviewer 1 / Comment 1(b).

**Protocol (to preserve).** `P1_structurelevel/KAN2LTN+hierarchy.py`:
9 features, 13 outputs (4 coarse L1 classes 0–3 + 9 fine L2 classes 4–12),
`MLP(layer_sizes=(9, 64, 32, 13))`, Adam lr `1.5e-3`.

**Hierarchy mapping required.** The 13 outputs mix both levels, so
`CHILD_TO_PARENT_4_9` is *not* a plain 6-class analogue:

```
label_L1_mapping = {"MQTT": 0, "Benign": 1, "Recon": 2, "ARP_Spoofing": 3}
label_L2_mapping = {"MQTT-DDoS-Connect_Flood": 4, "MQTT-DDoS-Publish_Flood": 5,
                    "MQTT-DoS-Connect_Flood": 6,  "MQTT-DoS-Publish_Flood": 7,
                    "MQTT-Malformed_Data": 8,     "Benign(fine)": 9,
                    "Recon-OS_Scan": 10, "Recon-Port_Scan": 11,
                    "ARP_Spoofing(fine)": 12}
CHILD_TO_PARENT_4_9 = {4:0, 5:0, 6:0, 7:0, 8:0, 9:1, 10:2, 11:2, 12:3}
```

Decide and record: is the 13-way argmax evaluated against `label_L2` only
(classes 4–12, as `compute_accuracy` does), or over all 13? This changes
`macro_f1` and `macro_fpr` materially and must match what the paper's existing
numbers meant.

### 3.1 Results table — TO FILL

| Model | Attn | KAN | Logic | Accuracy | Macro-F1 | Macro recall | Macro FPR | AUROC | ECE | Brier | Reliability | Hier-F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MLP | – | – | – | | | | | | | | | |
| Logic-MLP | – | – | ✓ | | | | | | | | | |
| KAN (no logic) | – | ✓ | – | | | | | | | | | |
| Logic-KAN | – | ✓ | ✓ | | | | | | | | | |
| H-Logic-KAN | – | ✓ | ✓ | | | | | | | | | |
| H-Logic-KAN* | – | ✓ | ✓ | | | | | | | | | |
| Attn-LogiK-Net (no logic) | ✓ | ✓ | – | | | | | | | | | |
| Attn-LogiK-Net (full) | ✓ | ✓ | ✓ | | | | | | | | | |

*(The two Attn rows here would be the 13-class re-run, not the §2 6-class
numbers. Do not mix protocols in one table.)*

### 3.2 Per-class recall — TO FILL

| Class | MLP | Logic-MLP | Logic-KAN | H-Logic-KAN | H-Logic-KAN* |
|---|---|---|---|---|---|
| MQTT-DDoS-Connect_Flood | | | | | |
| MQTT-DDoS-Publish_Flood | | | | | |
| MQTT-DoS-Connect_Flood | | | | | |
| MQTT-DoS-Publish_Flood | | | | | |
| MQTT-Malformed_Data | | | | | |
| Benign | | | | | |
| Recon-OS_Scan | | | | | |
| Recon-Port_Scan | | | | | |
| ARP_Spoofing | | | | | |

### 3.3 Significance — TO FILL

| Comparison | Metric | W | p | McNemar (sig. seeds) |
|---|---|---|---|---|
| H-Logic-KAN vs. MLP | macro-F1 | | | |
| H-Logic-KAN vs. Logic-MLP | macro-F1 | | | |
| Logic-KAN vs. Logic-MLP | macro-F1 | | | |
| H-Logic-KAN vs. Logic-KAN | macro-F1 | | | |

### 3.4 Reliability partial-credit sweep — TO FILL

| Model | partial=0.25 | partial=0.50 | partial=0.75 |
|---|---|---|---|
| MLP | | | |
| H-Logic-KAN | | | |
| Attn-LogiK-Net (full) | | | |

Single-run values in the current paper, for regression-checking the re-run:
`R_MLP = 0.7615`, `R_H-Logic-KAN = 0.8671`
(`Results.tex` Fig. `P1-reliability-score` caption).

---

## 4. P2 — Feature-count comparison (PENDING)

Target: `Results.tex` Fig. `P2-kan-mlp-compare`, `P2-feature-score`.
Sources: `P2_featurelevel/{14,16,18}features.txt`, `KAN_2_LTN.ipynb`.

| Features | Model | Accuracy | Macro-F1 | Macro recall | Macro FPR | AUROC | ECE | Brier | Reliability | Hier-F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 14 | Logic-MLP | | | | | | | | | |
| 14 | Logic-KAN | | | | | | | | | |
| 16 | Logic-MLP | | | | | | | | | |
| 16 | Logic-KAN | | | | | | | | | |
| 18 | Logic-MLP | | | | | | | | | |
| 18 | Logic-KAN | | | | | | | | | |

Significance — TO FILL

| Feature set | Comparison | Metric | W | p |
|---|---|---|---|---|
| 14 | Logic-KAN vs. Logic-MLP | macro-F1 | | |
| 16 | Logic-KAN vs. Logic-MLP | macro-F1 | | |
| 18 | Logic-KAN vs. Logic-MLP | macro-F1 | | |

---

## 5. Reproducibility record

| Item | Value |
|---|---|
| Seeds | `[0, 1, 2, 3, 4]` |
| Seeding | `eval_metrics.set_seed` — python / numpy / torch, `cudnn.deterministic=True`, `benchmark=False` |
| Test loader | `shuffle=False`, full-batch — required for valid McNemar pairing |
| Scaler | `StandardScaler` fit on train only; `mean_`/`scale_` stored in every checkpoint |
| Device | CUDA if available, else CPU. **MPS deliberately excluded** — LTN Constants stay on CPU while the model moves to GPU, producing a device mismatch |
| std convention | sample std, `ddof=1` |
| P5 checkpoints | `P5_attention/saved/models/{ltn,noltn}_seed{0..4}.pt` |
| P1 checkpoints | `P1_structurelevel/efficiency/model_weights/*.pt` (single-run, pre-revision) |
| P2 checkpoints | `P2_featurelevel/ Compare_datasize/model_weights/*.pt` (single-run, pre-revision) |

---

## 6. Open decisions

1. **Seed count.** 5 satisfies the reviewer's stated minimum but caps Wilcoxon at
   p ≥ 0.0625. If any comparison in §3.3 needs to *claim* significance, 10 seeds
   are required. Decide before launching P1, not after.
2. **Multiple comparisons.** §3.3 runs ≥ 4 paired tests on the same seed set.
   Either apply Holm–Bonferroni or state explicitly that p-values are
   uncorrected and exploratory.
3. **Equivalence vs. non-significance.** See §2.2 caveat. Applies to any
   "logic is accuracy-neutral" wording.
4. **Standalone Transformer row.** Run `TransformerClassifier` or defend its
   absence explicitly in the letter.
5. **13-way vs. 9-way evaluation target for P1.** See §3.
