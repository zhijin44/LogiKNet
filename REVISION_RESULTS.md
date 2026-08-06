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
| P7 | C2 backward ablation (KAN\* block) | 6-class / 18-feat, train 10k | 5 | **done** |
| P1 | Model comparison (Fig. `P1-all-model`) | 6-class / 18-feat | 5 | *harness ready* |
| P2 | Feature-count comparison (14/16/18) | per-set | 5 | *pending* |
| P1 | Reliability-score figure | 13-class / 9-feat | 1 | not scheduled |
| P2 | Data-size comparison (S/M/L) | per-size | 1 | not scheduled |

**Agreed metric scope for the pending re-runs:** all three groups —
Predictive + Calibration + Hierarchical (see §1).
**Agreed protocol:** P1 and P2 are retrained at their *original* settings.
P1 turns out to share P5's 6-class / 18-feature setup, so `CHILD_TO_PARENT_6`
covers both and no new mapping is needed (see §3). P1 trains on
`logiKNet_train_35945.csv` — the original 90/10 split — whereas P5 used the
10,000-row subsample, so the two are **not** directly comparable to each other.

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

## 2B. P7 — C2 backward-module ablation (DONE)

Fills the KAN block of `tab: ablation-backward` (Results.tex §V-C).
Harness: `P7_ablation/run_multiseed_c2.py`. Pilot: `P7_ablation/C2_backward_ablation.ipynb`.

**Protocol.** 10,000 train / 3,994 test, 18 features, 6 classes.
Forward module held fixed at KAN `[18,6,6,6]`, grid 5, k 3 — deliberately the
same backbone as inside P5's `AttentionKANModel`, so this block and P5's
attention block differ by exactly the attention encoder. Adam lr `1e-3`,
full batch, seeds `[0,1,2,3,4]`.

**Epoch budget.** Per-variant convergence × 1.05 (same treatment as P5;
verified the formula reproduces P5's 625→656 and 843→885 exactly):
KAN\* 600→**630**, Logic-KAN\* 800→**840**, H-Logic-KAN\* 800→**840**.
Convergence read off the test-accuracy plateau in a single seed-42 pilot, then
held constant across all seeds.

**Source.** `P7_ablation/saved/multiseed_results.{txt,json}`,
`hierarchical.json`, checkpoints `saved/models/{key}_seed{0..4}.pt`.

### 2B.1 Mean ± std over 5 seeds

| Metric | KAN\* (no logic) | Logic-KAN\* | H-Logic-KAN\* |
|---|---|---|---|
| accuracy | 0.8074 ± 0.0041 | **0.8162 ± 0.0068** | 0.8143 ± 0.0030 |
| macro-F1 | 0.8071 ± 0.0040 | **0.8159 ± 0.0067** | 0.8140 ± 0.0030 |
| macro recall | 0.8085 ± 0.0039 | **0.8174 ± 0.0066** | 0.8156 ± 0.0029 |
| macro FPR ↓ | 0.0384 ± 0.0008 | **0.0367 ± 0.0014** | 0.0370 ± 0.0006 |
| macro AUROC | 0.9711 ± 0.0011 | **0.9715 ± 0.0018** | 0.9709 ± 0.0010 |
| ECE ↓ | 0.0191 ± 0.0042 | **0.0174 ± 0.0034** | 0.0212 ± 0.0061 |
| Brier ↓ | 0.2773 ± 0.0055 | **0.2655 ± 0.0073** | 0.2700 ± 0.0046 |
| reliability (w=0.5) | 0.8842 ± 0.0037 | **0.8900 ± 0.0053** | 0.8891 ± 0.0032 |
| hierarchical-F1 | 0.8842 ± 0.0037 | **0.8900 ± 0.0053** | 0.8891 ± 0.0032 |

`reliability@0.5 == hierarchical_f1` to every stored digit, for all three
models — the identity in §1.3 confirmed a second time.

### 2B.2 Partial-credit sweep

| Model | R(0.25) | R(0.5) | R(0.75) |
|---|---|---|---|
| KAN\* | 0.8458 ± 0.0039 | 0.8842 ± 0.0037 | 0.9225 ± 0.0035 |
| Logic-KAN\* | 0.8531 ± 0.0060 | 0.8900 ± 0.0053 | 0.9269 ± 0.0047 |
| H-Logic-KAN\* | 0.8517 ± 0.0031 | 0.8891 ± 0.0032 | 0.9265 ± 0.0034 |

Model ranking is invariant across w — the "0.5 is ad hoc" objection is answered
empirically, not just algebraically.

### 2B.3 Significance

| Comparison | Metric | W | p | McNemar |
|---|---|---|---|---|
| Logic-KAN\* vs KAN\* | accuracy | 1.0 | 0.125 | 3/5 seeds |
| Logic-KAN\* vs KAN\* | macro-F1 | 1.0 | 0.125 | — |
| H-Logic-KAN\* vs Logic-KAN\* | accuracy | 5.0 | 0.625 | 1/5 seeds |
| H-Logic-KAN\* vs KAN\* | accuracy | **0.0** | 0.0625 | 3/5 seeds |

`W = 0` for H-Logic-KAN\* vs KAN\* means **all five seeds** ordered them the
same way — the strongest result attainable at n=5. Logic-KAN\* has `W = 1`,
i.e. one seed (seed 4) reversed.

### 2B.4 Per-class recall (mean over seeds)

| Class | KAN\* | Logic-KAN\* | H-Logic-KAN\* |
|---|---|---|---|
| MQTT-DDoS-Connect_Flood | 0.6348 | 0.6406 | 0.6412 |
| MQTT-DDoS-Publish_Flood | 0.7575 | 0.7747 | 0.7709 |
| MQTT-DoS-Connect_Flood | 0.8056 | 0.8094 | 0.8094 |
| MQTT-DoS-Publish_Flood | 0.8831 | 0.8914 | 0.8831 |
| MQTT-Malformed_Data | 0.9064 | 0.9203 | **0.9278** |
| Benign | 0.8637 | 0.8680 | 0.8612 |

### 2B.5 What the numbers say

1. **The logic tensor does buy accuracy** — +0.9 pp over cross-entropy, plus the
   best calibration in the table. This is *stronger* than the P5-only result
   suggested, where logic looked accuracy-neutral.
2. **The hierarchy rule buys stability, not accuracy.** H-Logic-KAN\* is
   0.2 pp *below* Logic-KAN\* (p = 0.625, inside one std), but its seed-to-seed
   std is less than half (0.0030 vs 0.0068) and its advantage over no-logic is
   unanimous across seeds. Consistent with its logical content: it constrains
   only the MQTT/Benign boundary and says nothing about the five sub-classes.
3. **⚠ The attention encoder does NOT help.** See §2C.

---

## 2C. Attention encoder: negative result, and how it is now framed

**Status: resolved.** The response letter's claim that "both `\attnlogiknet{}`
variants improve over the corresponding non-attention configurations" was
falsified by the C2 ablation and has been **rewritten** (2026-08-05).

### The measurement

Both C2 blocks share an identical KAN configuration and rule set, so the
comparison isolates the attention encoder exactly:

Both blocks of Table C2 share the identical KAN backbone `[18,6,6,6]` and the
identical rule set, so the comparison isolates the attention encoder exactly:

| Backward module | Without attention | With attention | Δ |
|---|---|---|---|
| None (cross-entropy) | KAN\* **0.8074 ± 0.0041** | Attn-LogiK-Net (no logic) 0.8030 ± 0.0112 | **−0.4 pp** |
| Logic + hierarchy | H-Logic-KAN\* **0.8143 ± 0.0030** | Attn-LogiK-Net (full) 0.7855 ± 0.0333 | **−2.9 pp** |

The attention encoder *reduces* accuracy in both configurations, and inflates
the variance by 3–10×.

`Major-Revision-Response.tex`, Reviewer #1 Comment #1(b), currently states:

> "the attention encoder is where the additional predictive accuracy comes
> from: both \attnlogiknet{} variants improve over the corresponding
> non-attention \ourmethod{} configurations, confirming that cross-feature
> interaction … is genuinely useful on this data"

### The resolution adopted

The accuracy claim was **dropped**, not softened. The attention encoder is now
justified by what it demonstrably provides — an **extended input pipeline** —
rather than by accuracy. Rationale: self-attention can only exploit structure
present in the input, and CICIoMT2024 as used here presents each flow as a
single flat vector, with no cross-record context to attend over. The null result
is therefore *consistent with* the mechanism rather than evidence against it.

Four deployment settings now argued in the letter, §(a):

1. **Temporal windows** — tokenise W consecutive flows from one device. Needed
   for classes a single flow cannot disambiguate: a low-rate DoS flow is
   individually indistinguishable from benign, and DoS-vs-DDoS is a statement
   about the *population* of flows.
2. **Multi-device aggregation** — tokens are devices; the encoder learns which
   device pairs co-vary. Recon and ARP-spoofing campaigns show as correlated
   cross-device behaviour while each device looks unremarkable alone.
3. **Multi-protocol fusion** — Wi-Fi / MQTT / Bluetooth feature blocks of
   different arity as token groups, without padding to a common flat schema.
4. **Variable feature sets** — the 18→16→14 pruning study needs no input-layer
   re-architecting.

The letter also now turns the negative result to the reviewer's advantage: a
stronger attention-based forward module, placed in the identical protocol, does
*not* overtake the KAN — which answers "are the gains an artifact of a weak
reference point?" with a direct **no**.

**Not done, optional:** `d_model=32, n_heads=4, n_layers=2` was never swept. A
brief sweep would let the letter say the configuration was tuned rather than
assumed. Worth doing if a reviewer presses on the negative result.

### Naming in §IV-C (decided 2026-08-05)

Sections IV-A/IV-B are micro-level studies where the `*` suffix distinguishes
architectural variants. §IV-C is a high-level module comparison, so the rows
drop both the asterisk and the architecture: **KAN**, **Logic-KAN**,
**H-Logic-KAN**. The backbone is stated once in the protocol paragraph (all
KAN-based rows use the two-hidden-layer configuration introduced as
H-Logic-KAN\* in §IV-B) so the mapping stays traceable.

### ⚠ Section renumbering side-effect

Inserting §IV-C pushed **Implementation Efficiency from IV-C to IV-D**. Two
places in the response letter cite `Sec.~IV-C-3` (the "Detection Failue" typo
fix, and the threat-rate minor comment). The reviewer's *quoted* comments must
keep the old numbering; **our own replies** must use IV-D-3. One has been fixed;
check the threat-rate reply when it is written.

---

## 3. P1 — Model comparison (PENDING)

Target: `Results.tex` Fig. `P1-all-model`, and the `[TODO]` cells in
`Major-Revision-Response.tex` §Reviewer 1 / Comment 1(b).

**Harness:** `P1_structurelevel/run_multiseed_p1.py`
**Figure:** `P1_structurelevel/plot_p1_performance.ipynb`
`KAN_2_LTN_hierarchy.ipynb` is left untouched as the record of the original logic.

> **Correction to an earlier version of this file.** P1 was previously recorded
> here as a 13-class / 9-feature setup requiring a new `CHILD_TO_PARENT_4_9`
> mapping. That was read off `KAN2LTN+hierarchy.py`, which is a *different*
> experiment. The notebook behind Fig. `P1-all-model` is
> **6-class / 18-feature**, identical to P5: `label_L2` 0–5, `label_L1` =
> MQTT(0)/Benign(1). `CHILD_TO_PARENT_6` applies unchanged and no new mapping
> is needed.

### 3.1 Protocol (from `KAN_2_LTN_hierarchy.ipynb`)

| Item | Value |
|---|---|
| Data | `logiKNet_train_35945.csv` / `logiKNet_test_3994.csv` (90/10 split of `filtered_train_l_2_6.csv`) |
| Features | 18 (`X_COLUMNS`) |
| Classes | 6 (`label_L2` 0–5); 0–4 → MQTT, 5 → Benign |
| Optimiser | Adam, lr `1e-3`, **full batch** |
| Epochs | 401 (0–400), same for every variant |
| KAN | `grid=5, k=3` |
| Scaling | `StandardScaler` fit on train only |
| Seeds | `[0,1,2,3,4]` (original single run used seed 42) |

### 3.2 Model roster

| Key | Paper name | Macro | Backbone | Width | Loss |
|---|---|---|---|---|---|
| `mlp` | MLP | — | MLP | `[18,10,6]` | cross-entropy |
| `logic_mlp` | Logic-MLP | — | MLP | `[18,10,6]` | LTN, flat rules |
| `kan` | KAN (no logic) | — | KAN | `[18,10,6]` | cross-entropy |
| `logic_kan` | Logic-KAN | `\logickan` | KAN | `[18,10,6]` | LTN, flat rules |
| `h_logic_kan` | H-Logic-KAN | `\hlogickan` | KAN | `[18,10,6]` | LTN + hierarchy |
| `h_logic_kan_star` | H-Logic-KAN* | `\hlogickanp` | KAN | `[18,6,6,6]` | LTN + hierarchy |

"Flat rules" = the six per-class `Forall` (notebook cell 10).
"+ hierarchy" = those six plus `Forall(x_MQTT, Not(P(x_MQTT, l_Benign)))` (cell 13).

**Width discrepancy, resolved.** Notebook cell 13 builds `KAN([18,6,6,6])` for
H-LogiKNet, but `efficiency/logiKNet.py:302` reloads `hierarchical_logiKNet.pt`
as `KAN([18,10,6])` — and `load_state_dict` must match, so `[18,10,6]` is what
actually trained; the notebook cell was edited after the run. Confirmed by the
author: **H-Logic-KAN = `[18,10,6]`** (one hidden layer), **H-Logic-KAN* =
`[18,6,6,6]`** (the deeper architectural variant). Without this the two models
would be identical and the figure's two curves unjustifiable.

`kan` (KAN trained with plain cross-entropy) did not previously exist. It is the
cell that separates the KAN contribution from the logic contribution — the exact
question Reviewer #1 raised.

### 3.3 Results table — TO FILL

Generated as LaTeX by the last cell of `plot_p1_performance.ipynb`.

| Model | Hier. | KAN | Logic | Accuracy | Macro-F1 | Macro recall | Macro FPR | AUROC | ECE | Brier |
|---|---|---|---|---|---|---|---|---|---|---|
| MLP | – | – | – | | | | | | | |
| Logic-MLP | – | – | ✓ | | | | | | | |
| KAN (no logic) | – | ✓ | – | | | | | | | |
| Logic-KAN | – | ✓ | ✓ | | | | | | | |
| H-Logic-KAN | ✓ | ✓ | ✓ | | | | | | | |
| H-Logic-KAN* | ✓ | ✓ | ✓ | | | | | | | |

### 3.4 Hierarchical evaluation (separate) — TO FILL

Written to `p1_multiseed/hierarchical.json`.

| Model | R(w=0.25) | R(w=0.5) | R(w=0.75) | Hier-F1 |
|---|---|---|---|---|
| MLP | | | | |
| Logic-MLP | | | | |
| KAN (no logic) | | | | |
| Logic-KAN | | | | |
| H-Logic-KAN | | | | |
| H-Logic-KAN* | | | | |

Reminder from §1.3: the `R(w=0.5)` and `Hier-F1` columns will be identical.
Report the equivalence, not two "independent" numbers.

Single-run values in the current paper, for regression-checking:
`R_MLP = 0.7615`, `R_H-Logic-KAN = 0.8671` (`Results.tex`, Fig.
`P1-reliability-score` caption). Final test accuracies in the existing logs:
MLP `0.6134`, Logic-KAN `0.782`, H-Logic-KAN* `0.766`.

### 3.5 Per-class recall — TO FILL

Printed by the harness; stored per run in `run__{key}__seed{n}.json`.

### 3.6 Significance — TO FILL

Written to `p1_multiseed/significance.json`. Pairs tested: H-Logic-KAN vs
{MLP, Logic-MLP, Logic-KAN, KAN}, Logic-KAN vs {Logic-MLP, KAN}, Logic-MLP vs
MLP, H-Logic-KAN* vs H-Logic-KAN. Metrics: accuracy, macro-F1, macro recall,
reliability. Wilcoxon across seeds + McNemar per seed.

See §6.1: with 5 seeds every Wilcoxon p-value is ≥ 0.0625, so none of these can
reach p < 0.05 no matter how large the gap.

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
