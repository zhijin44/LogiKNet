#!/usr/bin/env python
"""
P1 multi-seed harness  (ToN major revision, Reviewer #1 Comments #2 and #3).

Re-runs the six P1 model variants over several seeds, recording per-epoch
training curves AND a full end-of-training metric bundle for each run, so that
Fig. `P1-all-model` can be redrawn with error bands and the response letter's
ablation table can be filled with mean +/- std plus paired significance tests.

This file does NOT modify `KAN_2_LTN_hierarchy.ipynb`; it reimplements the same
protocol so the original notebook stays as the record of the initial logic.

--------------------------------------------------------------------------- #
PROTOCOL (taken verbatim from KAN_2_LTN_hierarchy.ipynb)
--------------------------------------------------------------------------- #
  data      : logiKNet_train_35945.csv / logiKNet_test_3994.csv
              (the original 90/10 split of filtered_train_l_2_6.csv)
  features  : 18                     labels: label_L2 in 0..5
  hierarchy : classes 0..4 -> MQTT (L1=0), class 5 -> Benign (L1=1)
  optimiser : Adam, lr 1e-3, FULL batch
  epochs    : 401 (0..400) for every variant
  KAN       : grid=5, k=3
  scaling   : StandardScaler fit on train only

  Only the seed changes between runs. Seed 42 was the original single run;
  the default sweep is seeds 0-4.

--------------------------------------------------------------------------- #
MODEL ROSTER
--------------------------------------------------------------------------- #
  key                paper name        backbone         width        loss
  ------------------ ----------------- ---------------- ------------ ----------
  mlp                MLP               MLP              [18,10,6]    CE
  logic_mlp          Logic-MLP         MLP              [18,10,6]    LTN flat
  kan                KAN (no logic)    KAN              [18,10,6]    CE
  logic_kan          Logic-KAN         KAN              [18,10,6]    LTN flat
  h_logic_kan        H-Logic-KAN       KAN              [18,10,6]    LTN + hier
  h_logic_kan_star   H-Logic-KAN*      KAN              [18,6,6,6]   LTN + hier

  "LTN flat" = the six per-class Forall rules (notebook cell 10).
  "LTN + hier" = those six PLUS  Forall(x_MQTT, Not(P(x_MQTT, l_Benign)))
                 (notebook cell 13).
  H-Logic-KAN uses [18,10,6] per efficiency/logiKNet.py:302, which must match
  hierarchical_logiKNet.pt to load; H-Logic-KAN* is the deeper variant.

--------------------------------------------------------------------------- #
USAGE
--------------------------------------------------------------------------- #
  # full sweep (long -- run under nohup/tmux on the cluster)
  python run_multiseed_p1.py

  # wiring check: 2 models, 1 seed, 5 epochs, small train file
  python run_multiseed_p1.py --fast

  # subset / resume (already-finished runs are skipped unless --overwrite)
  python run_multiseed_p1.py --models mlp logic_kan --seeds 0 1 2

  # re-aggregate + significance without retraining
  python run_multiseed_p1.py --aggregate-only

Every (model, seed) run is written to disk the moment it finishes, so an
interrupted sweep can be resumed with the same command.

Outputs land in --out (default ./p1_multiseed/):
  curve__{key}__seed{n}.npz   per-epoch train_loss / test_acc / train_sat / test_sat
  preds__{key}__seed{n}.npz   y_true, y_pred, probs   (fixed test order)
  run__{key}__seed{n}.json    end-of-training scalar metrics
  models/{key}_seed{n}.pt     state_dict + config + scaler
  summary.json                mean +/- std per model per metric
  hierarchical.json           reliability partial-sweep + hierarchical_f1
  significance.json           paired Wilcoxon (across seeds) + McNemar (per seed)
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                    # P1 utils.py
sys.path.insert(0, os.path.join(HERE, '..', 'P5_attention'))  # eval_metrics.py

from utils import MLP, LogitsToPredicate, MultiKANModel, DataLoader  # noqa: E402
from kan import KAN                                                   # noqa: E402
import ltn                                                            # noqa: E402
import ltn.fuzzy_ops                                                  # noqa: E402
from eval_metrics import (                                            # noqa: E402
    set_seed, evaluate_run, MetricTracker, wilcoxon_compare,
    mcnemar_test, hierarchical_reliability, hierarchical_f1,
    CHILD_TO_PARENT_6, LABEL_L2_NAMES,
)


# --------------------------------------------------------------------------- #
#  protocol constants
# --------------------------------------------------------------------------- #
X_COLUMNS = [
    'Header_Length', 'Protocol Type', 'Duration', 'Rate', 'Srate',
    'IPv', 'LLC',
    'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number',
    'Magnitue', 'Radius', 'Covariance',
]
IN_FEATURES = len(X_COLUMNS)        # 18
N_CLASSES = 6                       # label_L2 in 0..5
BENIGN_L2 = 5
LR = 1e-3
EPOCHS = 401                        # matches the 401-line log files
GRID, K = 5, 3
PARTIAL_SWEEP = (0.25, 0.5, 0.75)   # reviewer: "0.5 partial credit is ad hoc"

DEFAULT_TRAIN = os.path.join(HERE, 'efficiency', 'input_files',
                             'logiKNet_train_35945.csv')
DEFAULT_TEST = os.path.join(HERE, 'efficiency', 'input_files',
                            'logiKNet_test_3994.csv')
FAST_TRAIN = os.path.join(HERE, 'efficiency', 'input_files',
                          'logiKNet_test_3994.csv')

# key -> (paper name, backbone, width, loss)
#   loss: 'ce' | 'ltn_flat' | 'ltn_hier'
MODELS = {
    'mlp':              ('MLP',            'mlp', [IN_FEATURES, 10, N_CLASSES], 'ce'),
    'logic_mlp':        ('Logic-MLP',      'mlp', [IN_FEATURES, 10, N_CLASSES], 'ltn_flat'),
    'kan':              ('KAN (no logic)', 'kan', [IN_FEATURES, 10, N_CLASSES], 'ce'),
    'logic_kan':        ('Logic-KAN',      'kan', [IN_FEATURES, 10, N_CLASSES], 'ltn_flat'),
    'h_logic_kan':      ('H-Logic-KAN',    'kan', [IN_FEATURES, 10, N_CLASSES], 'ltn_hier'),
    'h_logic_kan_star': ('H-Logic-KAN*',   'kan', [IN_FEATURES, 6, 6, N_CLASSES], 'ltn_hier'),
}

# comparisons reported in significance.json
SIG_PAIRS = [
    ('h_logic_kan', 'mlp'),
    ('h_logic_kan', 'logic_mlp'),
    ('h_logic_kan', 'logic_kan'),
    ('h_logic_kan', 'kan'),
    ('logic_kan', 'logic_mlp'),
    ('logic_kan', 'kan'),
    ('logic_mlp', 'mlp'),
    ('h_logic_kan_star', 'h_logic_kan'),
]
SIG_METRICS = ['accuracy', 'macro_f1', 'macro_recall', 'reliability']


# --------------------------------------------------------------------------- #
#  data
# --------------------------------------------------------------------------- #
def check_not_lfs_pointer(path):
    """The CSVs are Git LFS-tracked (.gitattributes: *.csv filter=lfs).

    Without `git lfs pull` they are ~130-byte pointer stubs that pandas happily
    parses into a 1-column frame, producing a baffling KeyError later.
    """
    with open(path, 'rb') as f:
        if f.read(40).startswith(b'version https://git-lfs'):
            raise RuntimeError(
                f'{path} is a Git LFS pointer, not the actual CSV.\n'
                'Run from the repo root:  git lfs install && git lfs pull')


def load_data(train_path, test_path, device):
    for p in (train_path, test_path):
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        check_not_lfs_pointer(p)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    missing = [c for c in X_COLUMNS if c not in train_df.columns]
    if missing:
        raise KeyError(f'missing feature columns in {train_path}: {missing}')

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(train_df[X_COLUMNS])
    Xte = scaler.transform(test_df[X_COLUMNS])
    ytr = train_df['label_L2'].values.astype(int)
    yte = test_df['label_L2'].values.astype(int)

    train_loader = DataLoader(
        data=torch.tensor(Xtr, dtype=torch.float32, device=device),
        labels=torch.tensor(ytr, dtype=torch.long, device=device),
        batch_size=len(train_df), shuffle=False)
    # shuffle=False keeps the test order fixed -> McNemar pairing stays valid
    test_loader = DataLoader(
        data=torch.tensor(Xte, dtype=torch.float32, device=device),
        labels=torch.tensor(yte, dtype=torch.long, device=device),
        batch_size=len(test_df), shuffle=False)

    print(f'train {Xtr.shape} | test {Xte.shape}')
    print('train label_L2 counts:', np.bincount(ytr, minlength=N_CLASSES))
    print('test  label_L2 counts:', np.bincount(yte, minlength=N_CLASSES))
    return train_loader, test_loader, scaler


# --------------------------------------------------------------------------- #
#  LTN scaffolding  (notebook cells 8, 10, 13)
# --------------------------------------------------------------------------- #
class LTNRules:
    """The six per-class rules, optionally plus the hierarchical constraint.

    Constants are built on `device`; the original notebook left them on CPU,
    which silently breaks on CUDA (predicate gets a CPU tensor for a GPU model).
    """

    def __init__(self, device, hierarchical):
        self.hierarchical = hierarchical
        self.Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
        self.Forall = ltn.Quantifier(
            ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier='f')
        self.SatAgg = ltn.fuzzy_ops.SatAgg()
        eye = torch.eye(N_CLASSES, device=device)
        self.consts = [ltn.Constant(eye[c]) for c in range(N_CLASSES)]

    def sat(self, loader, P):
        sat_level = 0
        for data, labels in loader:
            terms = []
            for c in range(N_CLASSES):
                sub = data[labels == c]
                if sub.size(0) == 0:        # skip empty class in this batch
                    continue
                v = ltn.Variable(f'x_c{c}', sub)
                terms.append(self.Forall(v, P(v, self.consts[c])))
            if self.hierarchical:
                sub = data[labels < BENIGN_L2]
                if sub.size(0) > 0:
                    v = ltn.Variable('x_MQTT', sub)
                    # MQTT traffic is not Benign
                    terms.append(self.Forall(
                        v, self.Not(P(v, self.consts[BENIGN_L2]))))
            sat_level = self.SatAgg(*terms)
        return sat_level


# --------------------------------------------------------------------------- #
#  build + train
# --------------------------------------------------------------------------- #
def build_model(key, seed, device):
    _, backbone, width, _ = MODELS[key]
    if backbone == 'mlp':
        return MLP(layer_sizes=tuple(width)).to(device)
    kan = KAN(width=width, grid=GRID, k=K, seed=seed, device=device,
              auto_save=False, save_act=False)
    return MultiKANModel(kan).to(device)


@torch.no_grad()
def test_accuracy(loader, model):
    correct = total = 0
    for data, labels in loader:
        preds = model(data, training=False).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    return correct / total


def train_one(key, seed, train_loader, test_loader, device, epochs,
              eval_every=1, verbose_every=50):
    """Train one (model, seed) and return (model, curves dict)."""
    _, _, _, loss_kind = MODELS[key]
    set_seed(seed)
    model = build_model(key, seed, device)

    if loss_kind == 'ce':
        criterion = torch.nn.CrossEntropyLoss()
        params, P, rules = model.parameters(), None, None
    else:
        rules = LTNRules(device, hierarchical=(loss_kind == 'ltn_hier'))
        P = ltn.Predicate(LogitsToPredicate(model))
        params = P.parameters()
    optimizer = torch.optim.Adam(params, lr=LR)

    nan = float('nan')
    curves = {k: [] for k in
              ('epoch', 'train_loss', 'test_acc', 'train_sat', 'test_sat')}
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        if loss_kind == 'ce':
            logits = model(train_loader.data, training=True)
            loss = criterion(logits, train_loader.labels)
            train_sat = nan
        else:
            sat = rules.sat(train_loader, P)
            loss = 1. - sat
            train_sat = sat.item()
        loss.backward()
        optimizer.step()

        if epoch % eval_every == 0 or epoch == epochs - 1:
            model.eval()
            acc = test_accuracy(test_loader, model)
            if loss_kind == 'ce':
                test_sat = nan
            else:
                with torch.no_grad():
                    test_sat = rules.sat(test_loader, P).item()
            curves['epoch'].append(epoch)
            curves['train_loss'].append(loss.item())
            curves['test_acc'].append(acc)
            curves['train_sat'].append(train_sat)
            curves['test_sat'].append(test_sat)
            if verbose_every and epoch % verbose_every == 0:
                s = '' if np.isnan(train_sat) else f' | sat {train_sat:.3f}'
                print(f'    epoch {epoch:4d} | loss {loss.item():.4f} '
                      f'| acc {acc:.4f}{s} | {time.time()-t0:.0f}s', flush=True)

    curves = {k: np.asarray(v, dtype=float) for k, v in curves.items()}
    return model, curves


# --------------------------------------------------------------------------- #
#  persistence
# --------------------------------------------------------------------------- #
def paths_for(out_dir, key, seed):
    return dict(
        curve=os.path.join(out_dir, f'curve__{key}__seed{seed}.npz'),
        preds=os.path.join(out_dir, f'preds__{key}__seed{seed}.npz'),
        run=os.path.join(out_dir, f'run__{key}__seed{seed}.json'),
        model=os.path.join(out_dir, 'models', f'{key}_seed{seed}.pt'),
    )


def save_run(out_dir, key, seed, model, curves, res, scaler, elapsed):
    p = paths_for(out_dir, key, seed)
    os.makedirs(os.path.join(out_dir, 'models'), exist_ok=True)

    np.savez_compressed(p['curve'], **curves)
    np.savez_compressed(p['preds'], y_true=res['_y_true'],
                        y_pred=res['_y_pred'], probs=res['_probs'])

    scalars = {k: v for k, v in res.items() if not k.startswith('_')}
    scalars['per_class_recall'] = {str(c): float(r) for c, r
                                   in res['per_class_recall'].items()}
    with open(p['run'], 'w') as f:
        json.dump({'model': key, 'name': MODELS[key][0], 'seed': seed,
                   'width': MODELS[key][2], 'loss': MODELS[key][3],
                   'epochs': int(len(curves['epoch'])),
                   'seconds': round(elapsed, 1),
                   'metrics': scalars}, f, indent=2, default=float)

    torch.save({'model_state': model.state_dict(), 'model_key': key,
                'seed': seed,
                'config': {'IN_FEATURES': IN_FEATURES, 'N_CLASSES': N_CLASSES,
                           'WIDTH': MODELS[key][2], 'BACKBONE': MODELS[key][1],
                           'LOSS': MODELS[key][3], 'X_columns': X_COLUMNS,
                           'GRID': GRID, 'K': K},
                'scaler_mean': scaler.mean_, 'scaler_scale': scaler.scale_},
               p['model'])


def load_tracker(out_dir, keys, seeds):
    """Rebuild a MetricTracker from whatever runs are already on disk."""
    tracker, found = MetricTracker(), {}
    for key in keys:
        for seed in seeds:
            p = paths_for(out_dir, key, seed)
            if not (os.path.exists(p['run']) and os.path.exists(p['preds'])):
                continue
            with open(p['run']) as f:
                m = json.load(f)['metrics']
            d = np.load(p['preds'])
            res = {k: v for k, v in m.items() if k != 'per_class_recall'}
            res['_y_true'], res['_y_pred'] = d['y_true'], d['y_pred']
            tracker.add(key, seed, res)
            found.setdefault(key, []).append(seed)
    return tracker, found


# --------------------------------------------------------------------------- #
#  aggregation: summary / hierarchical / significance
# --------------------------------------------------------------------------- #
def write_summary(out_dir, tracker, found):
    summary = {}
    for key, seeds in found.items():
        row = {'name': MODELS[key][0], 'width': MODELS[key][2],
               'loss': MODELS[key][3], 'seeds': sorted(seeds), 'n': len(seeds)}
        for m in MetricTracker.SCALAR_KEYS:
            if tracker.store[key][m]:
                mu, sd = tracker.mean_std(key, m)
                row[m] = {'mean': mu, 'std': sd,
                          'values': tracker.series(key, m)}
        summary[key] = row
    path = os.path.join(out_dir, 'summary.json')
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2, default=float)

    order = [k for k in MODELS if k in summary]
    cols = ['accuracy', 'macro_f1', 'macro_recall', 'macro_fpr',
            'macro_auroc', 'ece', 'brier']
    print('\n' + '=' * 78)
    print('PREDICTIVE + CALIBRATION  (mean +/- std over seeds)')
    print('=' * 78)
    print(f"{'model':<17}" + ''.join(f'{c:>17}' for c in cols))
    for k in order:
        cells = ''.join(
            f"{summary[k][c]['mean']:.4f}+/-{summary[k][c]['std']:.4f}".rjust(17)
            if c in summary[k] else 'n/a'.rjust(17) for c in cols)
        print(f'{MODELS[k][0]:<17}' + cells)
    return summary, path


def write_hierarchical(out_dir, keys, seeds):
    """Separate hierarchical evaluation: reliability over the partial sweep,
    plus standard hierarchical-F1.

    NOTE: at partial = 0.5 these two are the SAME statistic for a single-parent
    two-level tree -- extending each label to {self, parent} gives
    |s_t| = |s_p| = 2, so precision = recall = sum(inter)/2N = hierarchical_f1,
    while reliability = (2*n_exact + n_same_parent)/2N is the same quantity.
    The sweep over 0.25 / 0.75 is what actually separates them, and is the
    answer to "the 0.5 partial-credit is ad hoc".
    """
    out = {'partials': list(PARTIAL_SWEEP), 'models': {}}
    for key in keys:
        per_seed = {f'reliability@{p}': [] for p in PARTIAL_SWEEP}
        per_seed['hierarchical_f1'] = []
        got = []
        for seed in seeds:
            p = paths_for(out_dir, key, seed)
            if not os.path.exists(p['preds']):
                continue
            d = np.load(p['preds'])
            yt, yp = d['y_true'], d['y_pred']
            for partial in PARTIAL_SWEEP:
                per_seed[f'reliability@{partial}'].append(
                    hierarchical_reliability(yt, yp, CHILD_TO_PARENT_6, partial))
            per_seed['hierarchical_f1'].append(
                hierarchical_f1(yt, yp, CHILD_TO_PARENT_6))
            got.append(seed)
        if not got:
            continue
        row = {'name': MODELS[key][0], 'seeds': got}
        for m, vals in per_seed.items():
            a = np.asarray(vals, dtype=float)
            row[m] = {'mean': float(a.mean()),
                      'std': float(a.std(ddof=1) if len(a) > 1 else 0.0),
                      'values': [float(v) for v in a]}
        out['models'][key] = row

    path = os.path.join(out_dir, 'hierarchical.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)

    cols = [f'reliability@{p}' for p in PARTIAL_SWEEP] + ['hierarchical_f1']
    print('\n' + '=' * 78)
    print('HIERARCHICAL  (mean +/- std over seeds)')
    print('=' * 78)
    print(f"{'model':<17}" + ''.join(f'{c:>19}' for c in cols))
    for k in [m for m in MODELS if m in out['models']]:
        r = out['models'][k]
        print(f'{MODELS[k][0]:<17}' + ''.join(
            f"{r[c]['mean']:.4f}+/-{r[c]['std']:.4f}".rjust(19) for c in cols))
    print('\nnote: reliability@0.5 == hierarchical_f1 by construction '
          '(single-parent 2-level tree).')
    return out, path


def write_significance(out_dir, tracker, found, seeds):
    sig = {'wilcoxon': [], 'mcnemar': [],
           'note': ('Wilcoxon is paired across seeds; with n=5 the smallest '
                    'attainable two-sided p is 0.0625, so p<0.05 is '
                    'unreachable regardless of effect size. p-values are '
                    'uncorrected across the pairs listed here.')}
    for a, b in SIG_PAIRS:
        if a not in found or b not in found:
            continue
        for metric in SIG_METRICS:
            if not (tracker.store[a][metric] and tracker.store[b][metric]):
                continue
            r = wilcoxon_compare(tracker, a, b, metric=metric)
            r['name_a'], r['name_b'] = MODELS[a][0], MODELS[b][0]
            sig['wilcoxon'].append(r)

        shared = sorted(set(found[a]) & set(found[b]))
        for seed in shared:
            da = np.load(paths_for(out_dir, a, seed)['preds'])
            db = np.load(paths_for(out_dir, b, seed)['preds'])
            r = mcnemar_test(da['y_true'], da['y_pred'], db['y_pred'])
            sig['mcnemar'].append({'model_a': a, 'model_b': b,
                                   'name_a': MODELS[a][0], 'name_b': MODELS[b][0],
                                   'seed': int(seed), **r})

    path = os.path.join(out_dir, 'significance.json')
    with open(path, 'w') as f:
        json.dump(sig, f, indent=2, default=float)

    print('\n' + '=' * 78)
    print('PAIRED WILCOXON ACROSS SEEDS')
    print('=' * 78)
    for r in sig['wilcoxon']:
        if r['metric'] not in ('accuracy', 'macro_f1'):
            continue
        print(f"{r['name_a']:>16} vs {r['name_b']:<16} {r['metric']:<13} "
              f"{r.get('mean_a', float('nan')):.4f} vs "
              f"{r.get('mean_b', float('nan')):.4f} | p = {r['p_value']:.4g}")

    print('\nMcNemar, significant seeds per pair (alpha = 0.05):')
    for a, b in SIG_PAIRS:
        rows = [r for r in sig['mcnemar']
                if r['model_a'] == a and r['model_b'] == b]
        if rows:
            n_sig = sum(r['p_value'] < 0.05 for r in rows)
            print(f'  {MODELS[a][0]:>16} vs {MODELS[b][0]:<16} '
                  f'{n_sig}/{len(rows)} seeds')
    return sig, path


# --------------------------------------------------------------------------- #
#  main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description='P1 multi-seed re-run (Reviewer #1 comments #2 / #3)')
    ap.add_argument('--models', nargs='+', default=list(MODELS),
                    choices=list(MODELS))
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--train', default=DEFAULT_TRAIN)
    ap.add_argument('--test', default=DEFAULT_TEST)
    ap.add_argument('--out', default=os.path.join(HERE, 'p1_multiseed'))
    ap.add_argument('--device', default=None,
                    help='cuda:0 | cpu (default: cuda if available)')
    ap.add_argument('--eval-every', type=int, default=1,
                    help='epochs between test evaluations (1 = original)')
    ap.add_argument('--overwrite', action='store_true',
                    help='redo runs that already have output files')
    ap.add_argument('--aggregate-only', action='store_true',
                    help='skip training, just rebuild summary/hier/significance')
    ap.add_argument('--fast', action='store_true',
                    help='wiring check: 2 models, 1 seed, 5 epochs, small file')
    args = ap.parse_args()

    if args.fast:
        args.models = ['mlp', 'h_logic_kan']
        args.seeds, args.epochs, args.train = [0], 5, FAST_TRAIN
        args.out = os.path.join(HERE, 'p1_multiseed_fast')
        print('[FAST] wiring check only -- results are not meaningful')

    # MPS is deliberately excluded: LTN Constants stay on CPU while the model
    # moves to the GPU -> "expected all tensors on the same device".
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out, exist_ok=True)
    print(f'device: {device} | out: {args.out}')
    print(f'models: {args.models}')
    print(f'seeds : {args.seeds} | epochs: {args.epochs}\n')

    if not args.aggregate_only:
        train_loader, test_loader, scaler = load_data(
            args.train, args.test, device)
        total = len(args.models) * len(args.seeds)
        done = 0
        for key in args.models:
            for seed in args.seeds:
                done += 1
                p = paths_for(args.out, key, seed)
                if os.path.exists(p['run']) and not args.overwrite:
                    print(f'[{done}/{total}] {MODELS[key][0]} seed {seed} '
                          f'-- already done, skipping')
                    continue
                print(f'[{done}/{total}] {MODELS[key][0]} '
                      f'(width {MODELS[key][2]}, loss {MODELS[key][3]}) '
                      f'seed {seed}', flush=True)
                t0 = time.time()
                model, curves = train_one(
                    key, seed, train_loader, test_loader, device,
                    args.epochs, eval_every=args.eval_every)
                res = evaluate_run(model, test_loader, n_classes=N_CLASSES,
                                   device=device,
                                   child_to_parent=CHILD_TO_PARENT_6,
                                   partial=0.5)
                elapsed = time.time() - t0
                save_run(args.out, key, seed, model, curves, res, scaler,
                         elapsed)
                print(f'    -> acc {res["accuracy"]:.4f} | '
                      f'macroF1 {res["macro_f1"]:.4f} | '
                      f'reliab {res["reliability"]:.4f} | '
                      f'{elapsed:.0f}s\n', flush=True)
                del model
    else:
        print('[aggregate-only] skipping training\n')

    tracker, found = load_tracker(args.out, args.models, args.seeds)
    if not found:
        print('no completed runs found in', args.out)
        return
    _, sp = write_summary(args.out, tracker, found)
    _, hp = write_hierarchical(args.out, args.models, args.seeds)
    _, gp = write_significance(args.out, tracker, found, args.seeds)

    print('\n' + '=' * 78)
    print('PER-CLASS RECALL (mean over seeds)')
    print('=' * 78)
    hdr = f"{'class':<28}" + ''.join(
        f'{MODELS[k][0]:>17}' for k in args.models if k in found)
    print(hdr)
    for c in range(N_CLASSES):
        line = f'{c} {LABEL_L2_NAMES[c]:<26}'
        for k in args.models:
            if k not in found:
                continue
            vals = []
            for seed in found[k]:
                with open(paths_for(args.out, k, seed)['run']) as f:
                    vals.append(json.load(f)['metrics']['per_class_recall'][str(c)])
            line += f'{np.mean(vals):>17.4f}'
        print(line)

    print(f'\nwrote:\n  {sp}\n  {hp}\n  {gp}')
    print(f'  curves + predictions + checkpoints under {args.out}')
    print('\nNext: open plot_p1_performance.ipynb to render the figure.')


if __name__ == '__main__':
    main()
