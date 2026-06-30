"""
Evaluation metrics + multi-seed statistical rigor for LogiK-Net
(ToN major revision, reviewer concerns R1-2 and R1-3).

This module matches the ACTUAL P5_attention setup:
  * 6 fine classes (label_L2 0..5); classes 0..4 are MQTT sub-attacks (L1=0),
    class 5 is Benign (L1=1).
  * loaders yield (data, labels) where `labels` is the integer label_L2.
  * models expose forward(x, training=False) -> logits  [B, 6]
    (AttentionKANModel / MultiKANModel / MLP / TransformerClassifier).

It provides everything the reviewers asked for:

  R1-2 (statistical rigor)
    - set_seed(seed)            reproducible runs (python/numpy/torch + cudnn)
    - MetricTracker            accumulate per-seed metrics -> mean +/- std
    - wilcoxon_compare(...)    paired Wilcoxon signed-rank across seeds
    - mcnemar_test(...)        paired McNemar on two models' test predictions
    - mcnemar_across_seeds(..) McNemar per seed for a model pair

  R1-3 (standard IDS metrics)
    - accuracy, macro-F1, weighted-F1
    - per-class recall, macro recall
    - macro false-positive rate (FPR)
    - macro AUROC (one-vs-rest)
    - calibration: expected calibration error (ECE) + multiclass Brier score
    - hierarchical_f1 (standard) reported ALONGSIDE the paper's reliability score

Dependencies: numpy, scipy, scikit-learn (already used in the repo).

--------------------------------------------------------------------------- #
QUICK START
--------------------------------------------------------------------------- #
A) Multi-seed run from inside a training notebook/script
   (you supply the model-building + training -- only the seed changes):

    from eval_metrics import (set_seed, evaluate_run, MetricTracker,
                              wilcoxon_compare, mcnemar_across_seeds,
                              CHILD_TO_PARENT_6, N_CLASSES)

    tracker = MetricTracker()
    for seed in range(5):                       # >= 5 seeds (R1-2)
        set_seed(seed)
        model = build_and_train(seed)           # your AttentionEncoder loop
        res = evaluate_run(model, test_loader, n_classes=N_CLASSES,
                           device=device, child_to_parent=CHILD_TO_PARENT_6)
        tracker.add("LogiKNet+Attn", seed, res)
    print(tracker.summary())                    # mean +/- std table

    # paired significance across seeds (e.g. macro_f1):
    print(wilcoxon_compare(tracker, "LogiKNet+Attn", "Attn-noLTN",
                           metric="macro_f1"))

B) Evaluate an already-saved checkpoint (single seed) from the shell:

    python eval_metrics.py --ckpt ./saved/attn_kan_2_6.pt \
                           --test ../P1_structurelevel/efficiency/input_files/logiKNet_test_3994.csv
"""

import argparse
import numpy as np
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, confusion_matrix, roc_auc_score,
)
from scipy.stats import wilcoxon


# --------------------------------------------------------------------------- #
#  dataset constants (P5_attention 6-class hierarchy)
# --------------------------------------------------------------------------- #
N_CLASSES = 6
BENIGN_L2 = 5
LABEL_L2_NAMES = {
    0: "MQTT-DDoS-Connect_Flood", 1: "MQTT-DDoS-Publish_Flood",
    2: "MQTT-DoS-Connect_Flood",  3: "MQTT-DoS-Publish_Flood",
    4: "MQTT-Malformed_Data",     5: "Benign",
}
# fine class -> parent (L1):  0..4 -> MQTT (0), 5 -> Benign (1)
CHILD_TO_PARENT_6 = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1}


# --------------------------------------------------------------------------- #
#  reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed):
    """Seed python/numpy/torch for a reproducible run."""
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True   # reproducible (slightly slower)
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
#  collect predictions from a logits model + a (data, labels) loader
# --------------------------------------------------------------------------- #
def collect_predictions(model, loader, device="cpu"):
    """Run a logits model over a 2-tuple loader and return arrays.

    Matches utils.DataLoader, which yields (data, labels) where `labels` is the
    integer label_L2 used for the 6-way argmax.

    Returns: y_true (np int), y_pred (np int), probs (np float [N, C]).
    """
    import torch
    model.eval()
    all_true, all_pred, all_prob = [], [], []
    with torch.no_grad():
        for data, labels in loader:
            data = data.to(device)
            logits = model(data, training=False)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            all_true.append(np.asarray(labels.cpu().numpy()).reshape(-1))
            all_pred.append(preds.cpu().numpy())
            all_prob.append(probs.cpu().numpy())
    return (np.concatenate(all_true).astype(int),
            np.concatenate(all_pred).astype(int),
            np.concatenate(all_prob))


# --------------------------------------------------------------------------- #
#  metrics
# --------------------------------------------------------------------------- #
def macro_fpr(y_true, y_pred, n_classes):
    """Macro-averaged one-vs-rest false-positive rate = mean_c FP/(FP+TN)."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    fprs, total = [], cm.sum()
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        tn = total - tp - fp - fn
        denom = fp + tn
        if denom > 0:
            fprs.append(fp / denom)
    return float(np.mean(fprs)) if fprs else 0.0


def macro_auroc(y_true, probs, n_classes):
    """Macro one-vs-rest AUROC over the classes present in y_true."""
    present = np.unique(y_true)
    if len(present) < 2:
        return float("nan")
    try:
        y_onehot = np.eye(n_classes)[y_true]
        cols = present                       # restrict to present classes
        return float(roc_auc_score(
            y_onehot[:, cols], probs[:, cols], average="macro", multi_class="ovr"))
    except ValueError:
        return float("nan")


def expected_calibration_error(y_true, probs, n_bins=15):
    """Top-label Expected Calibration Error (ECE).

    Bins test points by predicted confidence (max softmax prob), then averages
    |accuracy - confidence| over bins, weighted by bin population. 0 = perfectly
    calibrated. This is the standard reliability-diagram summary statistic.
    """
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(y_true)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        acc_bin = correct[mask].mean()
        conf_bin = conf[mask].mean()
        ece += (mask.sum() / n) * abs(acc_bin - conf_bin)
    return float(ece)


def brier_score(y_true, probs, n_classes):
    """Multiclass Brier score = mean squared error between the predicted
    probability vector and the one-hot target. Lower is better (proper scoring
    rule; complements ECE as a calibration/sharpness measure)."""
    y_onehot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))


def reliability_diagram(y_true, probs, n_bins=15):
    """Return per-bin (confidence, accuracy, count) for plotting a reliability
    diagram. Not aggregated -- handy for Figs in the rebuttal."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            out.append((0.5 * (lo + hi), float("nan"), 0))
        else:
            out.append((float(conf[mask].mean()),
                        float(correct[mask].mean()), int(mask.sum())))
    return out


def hierarchical_reliability(y_true, y_pred, child_to_parent, partial=0.5):
    """The paper's reliability score.

    Full credit (1.0) for an exact fine-level match; `partial` credit
    (default 0.5) when the fine prediction is wrong but the PARENT (L1) is
    correct. Reviewer R1-3 calls 0.5 ad hoc -- report this alongside
    hierarchical_f1 and treat `partial` as a sensitivity knob (sweep .25/.5/.75).
    """
    score = 0.0
    for t, p in zip(y_true, y_pred):
        if t == p:
            score += 1.0
        elif child_to_parent.get(int(t)) == child_to_parent.get(int(p)):
            score += partial
    return score / len(y_true)


def hierarchical_f1(y_true, y_pred, child_to_parent):
    """Standard hierarchical F1 (Kiritchenko-style): extend each label to its
    {self, parent} set and compute set-overlap precision/recall, then F1."""
    def ext(idx):
        s = {("C", int(idx))}                 # tag fine class
        par = child_to_parent.get(int(idx))
        if par is not None:
            s.add(("P", par))                 # tag parent (distinct namespace)
        return s
    inter = tp = pp = 0.0
    for t, p in zip(y_true, y_pred):
        st, sp = ext(t), ext(p)
        inter += len(st & sp)
        tp += len(st)
        pp += len(sp)
    prec = inter / pp if pp else 0.0
    rec = inter / tp if tp else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def evaluate_run(model, loader, n_classes=N_CLASSES, device="cpu",
                 child_to_parent=CHILD_TO_PARENT_6, partial=0.5, n_bins=15):
    """Compute the full metric bundle for one trained model on one loader."""
    y_true, y_pred, probs = collect_predictions(model, loader, device)
    res = {
        "accuracy":     accuracy_score(y_true, y_pred),
        "macro_f1":     f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1":  f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_fpr":    macro_fpr(y_true, y_pred, n_classes),
        "macro_auroc":  macro_auroc(y_true, probs, n_classes),
        "ece":          expected_calibration_error(y_true, probs, n_bins),
        "brier":        brier_score(y_true, probs, n_classes),
    }
    per_class = recall_score(y_true, y_pred, average=None,
                             labels=list(range(n_classes)), zero_division=0)
    res["per_class_recall"] = {c: float(r) for c, r in enumerate(per_class)}
    if child_to_parent is not None:
        res["reliability"] = hierarchical_reliability(
            y_true, y_pred, child_to_parent, partial)
        res["hierarchical_f1"] = hierarchical_f1(y_true, y_pred, child_to_parent)
    # keep predictions so McNemar can be run later between models on the same seed
    res["_y_true"] = y_true
    res["_y_pred"] = y_pred
    res["_probs"] = probs
    return res


# --------------------------------------------------------------------------- #
#  multi-seed aggregation
# --------------------------------------------------------------------------- #
class MetricTracker:
    """Collect per-seed metric dicts per model, then summarize mean +/- std."""

    SCALAR_KEYS = ["accuracy", "macro_f1", "weighted_f1", "macro_recall",
                   "macro_fpr", "macro_auroc", "ece", "brier",
                   "reliability", "hierarchical_f1"]

    def __init__(self):
        self.store = defaultdict(lambda: defaultdict(list))   # model -> metric -> [seeds]
        self.preds = defaultdict(dict)                        # model -> seed -> (yt, yp)

    def add(self, model_name, seed, res):
        for k in self.SCALAR_KEYS:
            if k in res and res[k] is not None:
                self.store[model_name][k].append(res[k])
        if "_y_true" in res:
            self.preds[model_name][seed] = (res["_y_true"], res["_y_pred"])

    def mean_std(self, model_name, metric):
        vals = np.array(self.store[model_name][metric], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            return float("nan"), float("nan")
        return float(vals.mean()), float(vals.std(ddof=1) if len(vals) > 1 else 0.0)

    def summary(self, metrics=None):
        metrics = metrics or [m for m in self.SCALAR_KEYS
                              if any(self.store[mdl][m] for mdl in self.store)]
        lines = []
        header = f"{'model':22s} " + " ".join(f"{m:>15s}" for m in metrics)
        lines.append(header)
        lines.append("-" * len(header))
        for mdl in self.store:
            cells = []
            for m in metrics:
                mu, sd = self.mean_std(mdl, m)
                cells.append(f"{mu:.3f}+/-{sd:.3f}")
            lines.append(f"{mdl:22s} " + " ".join(f"{c:>15s}" for c in cells))
        return "\n".join(lines)

    def series(self, model_name, metric):
        """Per-seed values (paired ordering by seed) for significance tests."""
        return list(self.store[model_name][metric])


# --------------------------------------------------------------------------- #
#  significance tests
# --------------------------------------------------------------------------- #
def wilcoxon_compare(tracker, model_a, model_b, metric="macro_f1"):
    """Paired Wilcoxon signed-rank test across seeds for one metric.
    Use with >= ~5 seeds. Returns dict with statistic and p-value."""
    a = np.array(tracker.series(model_a, metric), dtype=float)
    b = np.array(tracker.series(model_b, metric), dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if n < 1 or np.allclose(a, b):
        return {"metric": metric, "n": n, "statistic": float("nan"),
                "p_value": float("nan"), "note": "identical or empty"}
    stat, p = wilcoxon(a, b)
    return {"metric": metric, "model_a": model_a, "model_b": model_b, "n": n,
            "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "statistic": float(stat), "p_value": float(p)}


def mcnemar_test(y_true, y_pred_a, y_pred_b, exact=True):
    """McNemar paired test on two models' predictions over the SAME test set.
    Compares discordant pairs b01 (a wrong, b right) vs b10 (a right, b wrong).
    Exact binomial for small discordant counts, else continuity-corrected chi2."""
    from scipy.stats import binomtest, chi2
    correct_a = (np.asarray(y_pred_a) == np.asarray(y_true))
    correct_b = (np.asarray(y_pred_b) == np.asarray(y_true))
    b01 = int(np.sum(~correct_a & correct_b))
    b10 = int(np.sum(correct_a & ~correct_b))
    n = b01 + b10
    if n == 0:
        return {"b01": b01, "b10": b10, "statistic": 0.0, "p_value": 1.0}
    if exact and n < 25:
        p = binomtest(min(b01, b10), n, 0.5).pvalue
        stat = float(min(b01, b10))
    else:
        stat = (abs(b01 - b10) - 1) ** 2 / n
        p = float(chi2.sf(stat, df=1))
    return {"b01": b01, "b10": b10, "statistic": float(stat), "p_value": float(p)}


def mcnemar_across_seeds(tracker, model_a, model_b):
    """Run McNemar per seed (predictions aligned on the same fixed-order test
    set) and return the list of per-seed results."""
    out = []
    seeds = sorted(set(tracker.preds[model_a]) & set(tracker.preds[model_b]))
    for s in seeds:
        yt_a, yp_a = tracker.preds[model_a][s]
        _yt_b, yp_b = tracker.preds[model_b][s]
        out.append({"seed": s, **mcnemar_test(yt_a, yp_a, yp_b)})
    return out


# --------------------------------------------------------------------------- #
#  checkpoint reload + test-loader helpers (so a saved .pt can be scored)
# --------------------------------------------------------------------------- #
def load_attention_kan_checkpoint(ckpt_path, device="cpu", grid=5, k=3):
    """Rebuild an AttentionKANModel from a checkpoint saved by the notebooks
    (model_state + config) and return (model, ckpt). grid/k default to the
    values used in the notebooks (KAN(width=..., grid=5, k=3))."""
    import torch
    from kan import KAN
    from utils import MultiKANModel
    from attention_modules import AttentionKANModel

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    kan = KAN(width=cfg["KAN_WIDTH"], grid=grid, k=k,
              seed=cfg.get("SEED", 42), device=device)
    model = AttentionKANModel(cfg["IN_FEATURES"], MultiKANModel(kan),
                              **cfg["ATTN"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def make_test_loader_from_csv(test_csv, ckpt, device="cpu", batch_size=None):
    """Build a fixed-order (data, labels) test loader from a CSV using the
    scaler stored in the checkpoint. label target = label_L2."""
    import torch
    import pandas as pd
    from utils import DataLoader
    cfg = ckpt["config"]
    df = pd.read_csv(test_csv)
    X = df[cfg["X_columns"]].values.astype(np.float32)
    X = (X - np.asarray(ckpt["scaler_mean"])) / np.asarray(ckpt["scaler_scale"])
    y = df["label_L2"].values.astype(int)
    bs = batch_size or X.shape[0]            # full-batch, deterministic order
    return DataLoader(
        data=torch.tensor(X, dtype=torch.float32, device=device),
        labels=torch.tensor(y, dtype=torch.long, device=device),
        batch_size=bs, shuffle=False)        # shuffle=False -> paired McNemar safe


def evaluate_checkpoint(ckpt_path, test_csv, device="cpu"):
    """Convenience: reload a checkpoint, build its test loader, and return the
    full metric bundle for that single seed."""
    model, ckpt = load_attention_kan_checkpoint(ckpt_path, device=device)
    loader = make_test_loader_from_csv(test_csv, ckpt, device=device)
    return evaluate_run(model, loader, n_classes=N_CLASSES, device=device,
                        child_to_parent=CHILD_TO_PARENT_6)


def save_seed_results(path, model_name, results_by_seed):
    """Persist a model's per-seed metrics + predictions to a single .npz.

    `results_by_seed`: dict {seed: res} where each `res` is an evaluate_run()
    output (must include the scalar metrics and `_y_true` / `_y_pred`).
    The saved bundle is everything load_seed_results() needs to rebuild a
    MetricTracker for mean+/-std and the paired McNemar/Wilcoxon tests --
    so the LTN and no-LTN notebooks can be run separately and compared later.
    """
    seeds = sorted(results_by_seed)
    arrs = {"model_name": np.array(model_name), "seeds": np.array(seeds)}
    for k in MetricTracker.SCALAR_KEYS:
        arrs[f"metric__{k}"] = np.array(
            [float(results_by_seed[s].get(k, np.nan)) for s in seeds], dtype=float)
    s0 = seeds[0]
    arrs["y_true"] = np.asarray(results_by_seed[s0]["_y_true"])
    arrs["y_pred"] = np.stack(
        [np.asarray(results_by_seed[s]["_y_pred"]) for s in seeds])
    np.savez_compressed(path, **arrs)
    return path


def load_seed_results(path, tracker=None):
    """Rebuild (or extend) a MetricTracker from a saved .npz bundle."""
    tracker = tracker or MetricTracker()
    d = np.load(path, allow_pickle=True)
    name = str(d["model_name"])
    seeds = d["seeds"].tolist()
    y_true = d["y_true"]
    for i, s in enumerate(seeds):
        res = {}
        for k in MetricTracker.SCALAR_KEYS:
            key = f"metric__{k}"
            if key in d.files:
                v = float(d[key][i])
                if not np.isnan(v):
                    res[k] = v
        res["_y_true"] = y_true
        res["_y_pred"] = d["y_pred"][i]
        tracker.add(name, int(s), res)
    return tracker


def _print_run(res, title=""):
    if title:
        print(f"\n=== {title} ===")
    for k in MetricTracker.SCALAR_KEYS:
        if k in res:
            print(f"  {k:14s}: {res[k]:.4f}")
    if "per_class_recall" in res:
        print("  per_class_recall:")
        for c, r in res["per_class_recall"].items():
            print(f"     {c} {LABEL_L2_NAMES.get(c, ''):26s}: {r:.4f}")


# --------------------------------------------------------------------------- #
#  CLI / self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LogiK-Net evaluation metrics")
    ap.add_argument("--ckpt", help="path to a saved .pt checkpoint")
    ap.add_argument("--test", help="path to the test CSV (with label_L2)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if args.ckpt and args.test:
        res = evaluate_checkpoint(args.ckpt, args.test, device=args.device)
        _print_run(res, title=f"checkpoint: {args.ckpt}")
    else:
        # synthetic self-test (no torch needed) on the 6-class hierarchy
        rng = np.random.default_rng(0)
        n = 600
        y_true = rng.integers(0, N_CLASSES, size=n)
        y_pred = y_true.copy()
        flip = rng.random(n) < 0.15
        y_pred[flip] = rng.integers(0, N_CLASSES, size=flip.sum())
        probs = np.eye(N_CLASSES)[y_pred] * 0.7 + rng.random((n, N_CLASSES)) * 0.3
        probs /= probs.sum(1, keepdims=True)

        print("self-test (synthetic, 6-class):")
        print("  accuracy  :", round(accuracy_score(y_true, y_pred), 4))
        print("  macro_f1  :", round(f1_score(y_true, y_pred, average='macro'), 4))
        print("  macro_fpr :", round(macro_fpr(y_true, y_pred, N_CLASSES), 4))
        print("  macro_auroc:", round(macro_auroc(y_true, probs, N_CLASSES), 4))
        print("  ece       :", round(expected_calibration_error(y_true, probs), 4))
        print("  brier     :", round(brier_score(y_true, probs, N_CLASSES), 4))
        print("  reliability:", round(hierarchical_reliability(
            y_true, y_pred, CHILD_TO_PARENT_6), 4))
        print("  hier_f1   :", round(hierarchical_f1(
            y_true, y_pred, CHILD_TO_PARENT_6), 4))
        y_pred_b = y_true.copy()
        flip2 = rng.random(n) < 0.25
        y_pred_b[flip2] = rng.integers(0, N_CLASSES, size=flip2.sum())
        print("  mcnemar   :", mcnemar_test(y_true, y_pred, y_pred_b))
        print("\n(no --ckpt/--test given; ran synthetic self-test only)")
