# Attention encoder + statistical rigor — integration guide

This wires the new files into `KAN2LTN+hierarchy.py` to answer reviewer
concerns **1.1 / 2.3** (missing Transformer/attention baseline + ablation) and
**1.2 / 1.3** (multi-seed mean±std, significance tests, proper IDS metrics).

New files (same folder):

- `attention_modules.py` — `FeatureAttentionEncoder`, `AttentionKANModel`, `TransformerClassifier`
- `eval_metrics.py` — metrics bundle, `MetricTracker`, Wilcoxon + McNemar

All three models expose the identical `forward(x, training=False) -> logits[B,13]`
interface as `utils.MLP`, so they drop into `LogitsToPredicate` with no change
to the LTN backward path.

> Note: torch could not be executed in the build sandbox (disk limit), so run
> the built-in smoke tests on the cluster first:
> `python attention_modules.py` and `python eval_metrics.py`.
> `eval_metrics.py` was verified end-to-end here on synthetic data.

---

## 1. The one-line model swap

In `KAN2LTN+hierarchy.py`, the model is created at line ~354:

```python
mlp = MLP(layer_sizes=(9, 64, 32, 13)).to(device)
P = ltn.Predicate(LogitsToPredicate(mlp))
```

Replace `mlp` with whichever variant the current run needs. `IN, OUT = 9, 13`.

```python
from attention_modules import AttentionKANModel, TransformerClassifier
from utils import MultiKANModel   # already used for the KAN logits model

IN, OUT = 9, 13

if MODEL == "mlp":                       # baseline
    model = MLP(layer_sizes=(IN, 64, 32, OUT))
elif MODEL == "kan":                     # KAN, no logic-encoder change
    model = MultiKANModel(kan)           # your existing built MultKAN
elif MODEL == "transformer":             # NEW standalone baseline (HiViT-style)
    model = TransformerClassifier(IN, OUT, d_model=32, n_heads=4, n_layers=2)
elif MODEL == "attn_kan":                # NEW LogiK-Net + attention encoder
    model = AttentionKANModel(IN, MultiKANModel(kan),
                              d_model=32, n_heads=4, n_layers=2)
model = model.to(device)
P = ltn.Predicate(LogitsToPredicate(model))
```

The training loop, LTN rules, `compute_sat_level`, and `compute_accuracy` are
untouched — they all call `P(...)`.

### Ablation matrix (Reviewer 2.3)

Run each row with the swap above; report all in one table:

| Tag | Model | KAN | Logic (LTN) | Attention |
|-----|-------|-----|-------------|-----------|
| MLP            | `mlp`         | – | – (train w/ CE) | – |
| Logic-MLP      | `mlp`         | – | ✓ | – |
| KAN            | `kan`         | ✓ | – (CE) | – |
| KAN+Logic      | `kan`         | ✓ | ✓ | – |
| Transformer    | `transformer` | – | – (CE) | ✓ |
| Transformer+Logic | `transformer` | – | ✓ | ✓ |
| **LogiK-Net+Attn (full)** | `attn_kan` | ✓ | ✓ | ✓ |

"– (CE)" = train with plain cross-entropy instead of the satisfaction loss
(swap `loss = 1. - sat_agg` for `F.cross_entropy(logits, labels)`), which
isolates the logic contribution.

---

## 2. Wrap the existing training in a 5-seed loop

Move the build+train+eval block into a function and loop seeds. Minimal sketch:

```python
from eval_metrics import (set_seed, evaluate_run, MetricTracker,
                          wilcoxon_compare, mcnemar_across_seeds,
                          CHILD_TO_PARENT_4_9)

tracker = MetricTracker()
SEEDS = [0, 1, 2, 3, 4]

for MODEL in ["mlp", "kan", "transformer", "attn_kan"]:
    for seed in SEEDS:
        set_seed(seed)                     # seeds python/numpy/torch + cudnn
        model = build_model(MODEL).to(device)   # the swap block above
        P = ltn.Predicate(LogitsToPredicate(model))
        optimizer = torch.optim.Adam(P.parameters(), lr=0.0015)
        train(...)                         # your existing 51-epoch loop
        res = evaluate_run(model, test_loader, n_classes=13, device=device,
                           child_to_parent=CHILD_TO_PARENT_4_9, partial=0.5)
        tracker.add(MODEL, seed, res)

print(tracker.summary())                   # mean +/- std for every metric
```

`evaluate_run` assumes `test_loader` yields `(data, label_L1, label_L2)` and
uses `label_L2` as the 13-way target (matches `compute_accuracy`). If you
evaluate on a different target, change `label_L2` in
`eval_metrics.collect_predictions`.

---

## 3. Metrics reported (Reviewer 1.3)

`evaluate_run` returns, per run: `accuracy`, `macro_f1`, `weighted_f1`,
`macro_recall`, `macro_fpr`, `macro_auroc`, `per_class_recall`, plus
`reliability` (your hierarchical score) **and** `hierarchical_f1` (the standard
hierarchical F1 the reviewer asked you to report alongside the bespoke score).

For the reliability metric, sweep `partial` ∈ {0.25, 0.5, 0.75} to show the
0.5 choice is not load-bearing — directly answers "the 0.5 partial-credit is
ad hoc."

---

## 4. Significance tests (Reviewer 1.2)

Across-seed paired test (e.g. macro-F1), full vs. each baseline:

```python
for base in ["mlp", "kan", "transformer"]:
    print(wilcoxon_compare(tracker, "attn_kan", base, metric="macro_f1"))
```

Per-seed McNemar on the test predictions (paired at the sample level):

```python
print(mcnemar_across_seeds(tracker, "attn_kan", "transformer"))
```

Report mean±std and the p-values; with 5 seeds use Wilcoxon for the metric
comparison and McNemar for prediction-level paired significance.

---

## 5. Interpretability hook (Reviewer 2.4)

Both attention models cache the last-layer feature-feature attention map:

```python
_ = model(some_batch, training=False)
A = model.last_attention()   # [B, F, F] (or [B, F+1, F+1] for transformer w/ CLS)
```

Average `A` over the test set to get a feature-interaction importance matrix.
Cross-check it against the features your LTN rules and KAN pruning rely on — if
attention concentrates on the same features, that is *independent* evidence the
explanation is faithful, which is the gap Reviewer 2 flagged for satisfaction
scores.

---

## Suggested hyperparameters (start here, then tune)

`d_model=32, n_heads=4, n_layers=2, dropout=0.1`, Adam lr `1.5e-3`, 51 epochs,
batch 64 — matching your current KAN/MLP run so the comparison is controlled.
Keep the attention block light (1–2 layers) as in HiViT-IDS [4], so the
parameter/FLOP story stays favorable versus CKAN [1].
