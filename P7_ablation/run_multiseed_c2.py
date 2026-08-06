#!/usr/bin/env python
"""
C2 backward-module ablation -- 5-seed run.

Fills the KAN block of Table C2 in `Results.tex` (`tab: ablation-backward`) with
mean +/- std over seeds, plus the paired significance tests.

Mirrors `P5_attention/run_multiseed.ipynb`: fixed per-variant epoch budget =
(measured convergence epoch) x (1 + MARGIN), looped over seeds, then a
mean+/-std table with paired Wilcoxon (across seeds) and McNemar (per seed).

--------------------------------------------------------------------------- #
PROTOCOL  (identical to the C2 single-run notebook)
--------------------------------------------------------------------------- #
  data      : logiKNet_train_10000.csv / logiKNet_test_3994.csv
  features  : 18                      labels: label_L2 in 0..5
  hierarchy : classes 0..4 -> MQTT (L1=0), class 5 -> Benign (L1=1)
  forward   : KAN [18,6,6,6], grid 5, k 3   -- HELD FIXED across all rows
  optimiser : Adam, lr 1e-3, FULL batch
  seeds     : 0..4

Only the BACKWARD module changes between rows:

  key                 name              backward module              loss
  ------------------- ----------------- ---------------------------- ---------
  kan_star_ce         KAN* (no logic)   none                         CE
  logic_kan_star      Logic-KAN*        LTN, six per-class rules     1 - sat
  h_logic_kan_star    H-Logic-KAN*      those six + MQTT-not-Benign  1 - sat

The forward module width [18,6,6,6] is the KAN backbone inside P5's
AttentionKANModel, so `h_logic_kan_star` IS H-Logic-KAN* and differs from
Attn-LogiK-Net (full) by exactly the attention encoder. The attention block of
Table C2 therefore comes from P5 and is NOT retrained here.

--------------------------------------------------------------------------- #
USAGE
--------------------------------------------------------------------------- #
  python run_multiseed_c2.py                       # full 3 x 5 run
  python run_multiseed_c2.py --fast                # wiring check (20 epochs)
  python run_multiseed_c2.py --variants logic_kan_star --seeds 0 1
  python run_multiseed_c2.py --aggregate-only      # re-tabulate, no training

Each (variant, seed) is written to disk the moment it finishes, so an
interrupted run resumes with the same command. Requires `pip install LTNtorch`.

Outputs (default ./saved/):
  models/{key}_seed{n}.pt      state_dict + config + scaler + per-run metrics
  preds__{key}__seed{n}.npz    y_true, y_pred, probs   (fixed test order)
  multiseed_results.txt        mean +/- std table
  multiseed_results.json       summary + significance + per-seed values
  hierarchical.json            reliability partial sweep + hierarchical F1
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
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'P1_structurelevel'))   # utils.py
sys.path.insert(0, os.path.join(HERE, '..', 'P5_attention'))        # eval_metrics.py

from utils import LogitsToPredicate, MultiKANModel, MLP, DataLoader   # noqa: E402
from kan import KAN                                                   # noqa: E402
import ltn                                                            # noqa: E402
import ltn.fuzzy_ops                                                  # noqa: E402
from eval_metrics import (                                            # noqa: E402
    set_seed, evaluate_run, MetricTracker, wilcoxon_compare,
    mcnemar_test, hierarchical_reliability, hierarchical_f1,
    CHILD_TO_PARENT_6, LABEL_L2_NAMES,
)


# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #
TRAIN_PATH = os.path.join(HERE, '..', 'P1_structurelevel', 'efficiency',
                          'input_files', 'logiKNet_train_10000.csv')
TEST_PATH = os.path.join(HERE, '..', 'P1_structurelevel', 'efficiency',
                         'input_files', 'logiKNet_test_3994.csv')

X_COLUMNS = [
    'Header_Length', 'Protocol Type', 'Duration', 'Rate', 'Srate',
    'IPv', 'LLC',
    'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number',
    'Magnitue', 'Radius', 'Covariance',
]
IN_FEATURES = len(X_COLUMNS)        # 18
N_CLASSES = 6
BENIGN_L2 = 5
KAN_WIDTH = [IN_FEATURES, 6, 6, N_CLASSES]   # forward module, HELD FIXED
GRID, K = 5, 3
LR = 1e-3
SEEDS = [0, 1, 2, 3, 4]

MLP_WIDTH = [IN_FEATURES, 6, 6, N_CLASSES]   # depth-matched to KAN_WIDTH

# key -> (display name, backward module)
VARIANTS = {
    'kan_star_ce':      ('KAN* (no logic)', 'none'),
    'logic_kan_star':   ('Logic-KAN*',      'flat'),
    'h_logic_kan_star': ('H-Logic-KAN*',    'hier'),
    # C1 forward-axis row: the SAME backward module as h_logic_kan_star, but an
    # MLP forward path. NOT the paper's Section IV-A Logic-MLP, which is
    # MLP(18,10,6) with the FLAT rule set -- different model, same label in the
    # C1 table (the caption states the control).
    'logic_mlp_hier':   ('Logic-MLP',       'hier'),
}
BACKBONE = {'kan_star_ce': 'kan', 'logic_kan_star': 'kan',
            'h_logic_kan_star': 'kan', 'logic_mlp_hier': 'mlp'}

# Convergence epochs measured in C2_backward_ablation.ipynb (single run, seed 42),
# the fixed budget is CONVERGE x (1 + MARGIN).
CONVERGE = {
    'kan_star_ce':      600,
    'logic_kan_star':   800,
    'h_logic_kan_star': 800,
    'logic_mlp_hier':   800,
}
MARGIN = 0.05                     # set to 0.0 if CONVERGE already IS the budget
EPOCHS = {k: (None if v is None else int(round(v * (1 + MARGIN))))
          for k, v in CONVERGE.items()}

PARTIAL = 0.5
PARTIAL_SWEEP = (0.25, 0.5, 0.75)   # reviewer: "the 0.5 partial credit is ad hoc"

# paired comparisons reported in the significance section
SIG_PAIRS = [
    ('logic_kan_star',   'kan_star_ce'),      # does the logic tensor help?
    ('h_logic_kan_star', 'logic_kan_star'),   # does the hierarchy rule help?
    ('h_logic_kan_star', 'kan_star_ce'),      # full backward vs. none
    ('h_logic_kan_star', 'logic_mlp_hier'),   # C1 forward axis: KAN vs MLP
]
SIG_METRICS = ['accuracy', 'macro_f1', 'macro_recall', 'reliability']


# --------------------------------------------------------------------------- #
#  Data
# --------------------------------------------------------------------------- #
def load_data(train_path, test_path, device):
    for p in (train_path, test_path):
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        # CSVs are Git LFS-tracked; without `git lfs pull` they are pointer stubs
        with open(p, 'rb') as f:
            if f.read(40).startswith(b'version https://git-lfs'):
                raise RuntimeError(
                    f'{p} is a Git LFS pointer, not the actual CSV.\n'
                    'Run from the repo root:  git lfs install && git lfs pull')

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    missing = [c for c in X_COLUMNS + ['label_L2'] if c not in train_df.columns]
    if missing:
        raise KeyError(f'missing columns in {train_path}: {missing}')

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(train_df[X_COLUMNS])
    Xte = scaler.transform(test_df[X_COLUMNS])
    ytr = train_df['label_L2'].values.astype(int)
    yte = test_df['label_L2'].values.astype(int)

    train_loader = DataLoader(
        data=torch.tensor(Xtr, dtype=torch.float32, device=device),
        labels=torch.tensor(ytr, dtype=torch.long, device=device),
        batch_size=len(train_df), shuffle=False)
    # shuffle=False -> fixed test order -> McNemar pairing stays valid
    test_loader = DataLoader(
        data=torch.tensor(Xte, dtype=torch.float32, device=device),
        labels=torch.tensor(yte, dtype=torch.long, device=device),
        batch_size=len(test_df), shuffle=False)

    print('train shape:', Xtr.shape, '| test shape:', Xte.shape)
    print('train L2 dist:', np.bincount(ytr, minlength=N_CLASSES))
    print('test  L2 dist:', np.bincount(yte, minlength=N_CLASSES))
    return train_loader, test_loader, scaler


# --------------------------------------------------------------------------- #
#  LTN setup  (constants built on `device`, unlike the original notebooks)
# --------------------------------------------------------------------------- #
class LTNRules:
    """Six per-class rules, plus MQTT-is-not-Benign when hierarchical=True.

    That one extra Forall is the ONLY difference between the `flat` and `hier`
    rows -- it is the sole place the class tree enters training.
    """

    def __init__(self, device):
        self.Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
        self.Forall = ltn.Quantifier(
            ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier='f')
        self.SatAgg = ltn.fuzzy_ops.SatAgg()
        eye = torch.eye(N_CLASSES, device=device)
        self.consts = [ltn.Constant(eye[c]) for c in range(N_CLASSES)]

    def sat(self, loader, P, hierarchical):
        sat_level = 0
        for data, labels in loader:
            terms = []
            for c in range(N_CLASSES):
                sub = data[labels == c]
                if sub.size(0) == 0:
                    continue
                v = ltn.Variable(f'x_c{c}', sub)
                terms.append(self.Forall(v, P(v, self.consts[c])))
            if hierarchical:
                sub = data[labels < BENIGN_L2]
                if sub.size(0) > 0:
                    v = ltn.Variable('x_MQTT', sub)
                    terms.append(self.Forall(
                        v, self.Not(P(v, self.consts[BENIGN_L2]))))
            sat_level = self.SatAgg(*terms)
        return sat_level


# --------------------------------------------------------------------------- #
#  Build + train
# --------------------------------------------------------------------------- #
def build_model(key, seed, device):
    if BACKBONE[key] == 'mlp':
        return MLP(layer_sizes=tuple(MLP_WIDTH)).to(device)
    kan = KAN(width=KAN_WIDTH, grid=GRID, k=K, seed=seed, device=device,
              auto_save=False, save_act=False)
    return MultiKANModel(kan).to(device)


def train_ce(model, train_loader, epochs, **_):
    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = criterion(model(train_loader.data, training=True),
                         train_loader.labels)
        loss.backward()
        opt.step()
    return model, None


def train_ltn(model, train_loader, epochs, rules, hierarchical):
    P = ltn.Predicate(LogitsToPredicate(model))
    opt = torch.optim.Adam(P.parameters(), lr=LR)
    for _epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = 1. - rules.sat(train_loader, P, hierarchical)
        loss.backward()
        opt.step()
    return model, P


def train_variant(key, seed, train_loader, rules, device, epochs):
    _, backward = VARIANTS[key]
    set_seed(seed)
    model = build_model(key, seed, device)
    if backward == 'none':
        return train_ce(model, train_loader, epochs)
    return train_ltn(model, train_loader, epochs, rules,
                     hierarchical=(backward == 'hier'))


# --------------------------------------------------------------------------- #
#  Persistence
# --------------------------------------------------------------------------- #
def paths_for(save_dir, key, seed):
    return dict(
        model=os.path.join(save_dir, 'models', f'{key}_seed{seed}.pt'),
        preds=os.path.join(save_dir, f'preds__{key}__seed{seed}.npz'),
    )


def save_run(save_dir, key, seed, model, res, scaler, epochs, elapsed):
    p = paths_for(save_dir, key, seed)
    os.makedirs(os.path.join(save_dir, 'models'), exist_ok=True)

    np.savez_compressed(p['preds'], y_true=res['_y_true'],
                        y_pred=res['_y_pred'], probs=res['_probs'])
    torch.save({
        'model_state': model.state_dict(),
        'variant': key, 'backward': VARIANTS[key][1], 'seed': seed,
        'epochs': epochs, 'seconds': round(elapsed, 1),
        'config': {'IN_FEATURES': IN_FEATURES, 'N_CLASSES': N_CLASSES,
                   'BACKBONE': BACKBONE[key],
                   'WIDTH': MLP_WIDTH if BACKBONE[key] == 'mlp' else KAN_WIDTH,
                   'KAN_WIDTH': KAN_WIDTH, 'GRID': GRID, 'K': K,
                   'X_columns': X_COLUMNS, 'BENIGN_L2': BENIGN_L2,
                   'label_L2_names': LABEL_L2_NAMES},
        'scaler_mean': scaler.mean_, 'scaler_scale': scaler.scale_,
        'metrics': {k: v for k, v in res.items() if not k.startswith('_')},
    }, p['model'])
    return p['model']


def load_tracker(save_dir, keys, seeds):
    """Rebuild a MetricTracker from whatever is already on disk."""
    tracker, found = MetricTracker(), {}
    for key in keys:
        for seed in seeds:
            p = paths_for(save_dir, key, seed)
            if not (os.path.exists(p['model']) and os.path.exists(p['preds'])):
                continue
            ck = torch.load(p['model'], map_location='cpu', weights_only=False)
            d = np.load(p['preds'])
            res = {k: v for k, v in ck['metrics'].items()
                   if k != 'per_class_recall'}
            res['_y_true'], res['_y_pred'] = d['y_true'], d['y_pred']
            tracker.add(key, seed, res)
            found.setdefault(key, []).append(seed)
    return tracker, found


# --------------------------------------------------------------------------- #
#  Reporting
# --------------------------------------------------------------------------- #
def report(save_dir, tracker, found, seeds):
    order = [k for k in VARIANTS if k in found]

    # ---- mean +/- std ----
    txt = tracker.summary()
    print('\n' + '=' * 78)
    print('MEAN +/- STD OVER SEEDS')
    print('=' * 78)
    print(txt)
    with open(os.path.join(save_dir, 'multiseed_results.txt'), 'w') as f:
        f.write(txt + '\n')

    summary = {}
    for key in order:
        row = {'name': VARIANTS[key][0], 'backward': VARIANTS[key][1],
               'epochs': EPOCHS[key], 'seeds': sorted(found[key])}
        for m in MetricTracker.SCALAR_KEYS:
            if tracker.store[key][m]:
                mu, sd = tracker.mean_std(key, m)
                row[m] = {'mean': mu, 'std': sd,
                          'values': tracker.series(key, m)}
        summary[key] = row

    # ---- Table C2 layout ----
    BACKWARD_LABEL = {'none': 'None (cross-entropy)',
                      'flat': 'Logic tensor',
                      'hier': 'Logic + hierarchy'}
    cols = [('accuracy', 'Acc.'), ('macro_f1', 'Macro-F1'),
            ('macro_recall', 'Macro Rec.'), ('macro_fpr', 'Macro FPR'),
            ('macro_auroc', 'AUROC')]
    print('\n' + '=' * 100)
    print('TABLE C2 -- KAN block  (paste into tab: ablation-backward)')
    print('=' * 100)
    print(f"{'Backward module':<22}{'Model':<17}" +
          ''.join(f'{h:>16}' for _, h in cols))
    for key in order:
        r = summary[key]
        line = f"{BACKWARD_LABEL[r['backward']]:<22}{r['name']:<17}"
        for m, _ in cols:
            line += f"{r[m]['mean']:.4f}+/-{r[m]['std']:.4f}".rjust(16)
        print(line)

    print('\nLaTeX rows:')
    for key in order:
        r = summary[key]
        cells = ' & '.join(f"${r[m]['mean']:.3f}\\pm{r[m]['std']:.3f}$"
                           for m, _ in cols)
        print(f"    {BACKWARD_LABEL[r['backward']]} & {r['name']} & {cells} \\\\")

    # ---- hierarchical (Table C3) with the partial sweep ----
    hier = {'partials': list(PARTIAL_SWEEP), 'models': {}}
    for key in order:
        acc = {f'reliability@{p}': [] for p in PARTIAL_SWEEP}
        acc['hierarchical_f1'] = []
        for seed in sorted(found[key]):
            d = np.load(paths_for(save_dir, key, seed)['preds'])
            yt, yp = d['y_true'], d['y_pred']
            for p in PARTIAL_SWEEP:
                acc[f'reliability@{p}'].append(
                    hierarchical_reliability(yt, yp, CHILD_TO_PARENT_6, p))
            acc['hierarchical_f1'].append(
                hierarchical_f1(yt, yp, CHILD_TO_PARENT_6))
        row = {'name': VARIANTS[key][0], 'seeds': sorted(found[key])}
        for m, vals in acc.items():
            a = np.asarray(vals, dtype=float)
            row[m] = {'mean': float(a.mean()),
                      'std': float(a.std(ddof=1) if len(a) > 1 else 0.0),
                      'values': [float(v) for v in a]}
        hier['models'][key] = row

    hcols = [f'reliability@{p}' for p in PARTIAL_SWEEP] + ['hierarchical_f1']
    print('\n' + '=' * 100)
    print('HIERARCHICAL + CALIBRATION  (feeds tab: ablation-hierarchical)')
    print('=' * 100)
    print(f"{'Model':<17}" + ''.join(f'{c:>19}' for c in hcols) +
          f"{'ECE':>16}{'Brier':>16}")
    for key in order:
        r, s = hier['models'][key], summary[key]
        print(f"{r['name']:<17}" +
              ''.join(f"{r[c]['mean']:.4f}+/-{r[c]['std']:.4f}".rjust(19)
                      for c in hcols) +
              f"{s['ece']['mean']:.4f}+/-{s['ece']['std']:.4f}".rjust(16) +
              f"{s['brier']['mean']:.4f}+/-{s['brier']['std']:.4f}".rjust(16))
    print('\nnote: reliability@0.5 == hierarchical_f1 by construction for this')
    print('      single-parent two-level tree. Report the equivalence, not two')
    print('      "independent" numbers.')
    with open(os.path.join(save_dir, 'hierarchical.json'), 'w') as f:
        json.dump(hier, f, indent=2, default=float)

    # ---- per-class recall ----
    print('\n' + '=' * 100)
    print('PER-CLASS RECALL (mean over seeds)')
    print('=' * 100)
    print(f"{'class':<28}" + ''.join(f'{VARIANTS[k][0]:>18}' for k in order))
    pcr = {}
    for key in order:
        mats = []
        for seed in sorted(found[key]):
            ck = torch.load(paths_for(save_dir, key, seed)['model'],
                            map_location='cpu', weights_only=False)
            v = ck['metrics']['per_class_recall']
            mats.append([float(v[c]) for c in range(N_CLASSES)])
        pcr[key] = np.asarray(mats).mean(axis=0)
    for c in range(N_CLASSES):
        print(f'{c} {LABEL_L2_NAMES[c]:<26}' +
              ''.join(f'{pcr[k][c]:>18.4f}' for k in order))

    # ---- significance ----
    sig = {'wilcoxon': [], 'mcnemar': [],
           'note': ('Wilcoxon is paired across seeds; with n=5 the smallest '
                    'attainable two-sided p is 0.0625, so p<0.05 is unreachable '
                    'regardless of effect size. p-values are uncorrected.')}
    print('\n' + '=' * 100)
    print('PAIRED SIGNIFICANCE')
    print('=' * 100)
    for a, b in SIG_PAIRS:
        if a not in found or b not in found:
            continue
        for metric in SIG_METRICS:
            if not (tracker.store[a][metric] and tracker.store[b][metric]):
                continue
            r = wilcoxon_compare(tracker, a, b, metric=metric)
            r['name_a'], r['name_b'] = VARIANTS[a][0], VARIANTS[b][0]
            sig['wilcoxon'].append(r)
            if metric in ('accuracy', 'macro_f1'):
                print(f"Wilcoxon {VARIANTS[a][0]:>16} vs {VARIANTS[b][0]:<16}"
                      f"{metric:<14}"
                      f"{r.get('mean_a', float('nan')):.4f} vs "
                      f"{r.get('mean_b', float('nan')):.4f} | "
                      f"p = {r['p_value']:.4g}")
        shared = sorted(set(found[a]) & set(found[b]))
        n_sig = 0
        for seed in shared:
            da = np.load(paths_for(save_dir, a, seed)['preds'])
            db = np.load(paths_for(save_dir, b, seed)['preds'])
            r = mcnemar_test(da['y_true'], da['y_pred'], db['y_pred'])
            r.update({'model_a': a, 'model_b': b, 'seed': int(seed),
                      'name_a': VARIANTS[a][0], 'name_b': VARIANTS[b][0]})
            sig['mcnemar'].append(r)
            n_sig += int(r['p_value'] < 0.05)
        if shared:
            print(f"McNemar  {VARIANTS[a][0]:>16} vs {VARIANTS[b][0]:<16}"
                  f"significant in {n_sig}/{len(shared)} seeds")
    print('\nNOTE: n=5 -> the smallest attainable Wilcoxon p is 0.0625.')
    print('      A non-significant result here is "not detectable", NOT')
    print('      "equivalent". State this, or raise the seed count.')

    out = {'seeds': seeds, 'converge': CONVERGE, 'margin': MARGIN,
           'epochs': EPOCHS, 'kan_width': KAN_WIDTH, 'lr': LR,
           'train_path': TRAIN_PATH, 'test_path': TEST_PATH,
           'summary': summary, 'significance': sig,
           'per_class_recall': {k: pcr[k].tolist() for k in order}}
    with open(os.path.join(save_dir, 'multiseed_results.json'), 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nwrote:\n  {os.path.join(save_dir, 'multiseed_results.txt')}"
          f"\n  {os.path.join(save_dir, 'multiseed_results.json')}"
          f"\n  {os.path.join(save_dir, 'hierarchical.json')}"
          f"\n  checkpoints under {os.path.join(save_dir, 'models')}")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description='C2 backward-module ablation, 5 seeds')
    ap.add_argument('--variants', nargs='+', default=list(VARIANTS),
                    choices=list(VARIANTS))
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--save-dir', default=os.path.join(HERE, 'saved'))
    ap.add_argument('--device', default=None, help='cuda:0 | cpu')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--aggregate-only', action='store_true')
    ap.add_argument('--fast', action='store_true',
                    help='wiring check: 20 epochs, 1 seed, separate save dir')
    args = ap.parse_args()

    epochs_by_key = dict(EPOCHS)
    missing_budget = [k for k in args.variants
                      if epochs_by_key.get(k) is None and not args.aggregate_only]
    if missing_budget and not args.fast:
        raise SystemExit(
            f'no measured epoch budget for {missing_budget}.\n'
            'Run the pilot cell in C2_backward_ablation.ipynb, read plateau@ off\n'
            'the accuracy trace, and set CONVERGE[...] in this file.')
    if args.fast:
        args.seeds = [0]
        epochs_by_key = {k: 20 for k in VARIANTS}
        args.save_dir = os.path.join(HERE, 'saved_fast')
        print('[FAST] wiring check only -- results are not meaningful')

    # MPS deliberately excluded: LTN Constants stay on CPU while the model moves
    # to the GPU -> "expected all tensors on the same device".
    device = torch.device(args.device) if args.device else torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    print(f'device: {device} | save dir: {args.save_dir}')
    print(f'variants: {args.variants}')
    print(f'seeds   : {args.seeds}')
    print(f'converge: {CONVERGE}')
    print(f'margin  : x{1 + MARGIN:.2f}')
    print(f'epochs  : {epochs_by_key}\n')

    if not args.aggregate_only:
        train_loader, test_loader, scaler = load_data(
            TRAIN_PATH, TEST_PATH, device)
        rules = LTNRules(device)
        total = len(args.variants) * len(args.seeds)
        done = 0
        for key in args.variants:
            n_epochs = epochs_by_key[key]
            for seed in args.seeds:
                done += 1
                p = paths_for(args.save_dir, key, seed)
                if os.path.exists(p['model']) and not args.overwrite:
                    print(f'[{done}/{total}] {VARIANTS[key][0]} seed {seed} '
                          f'-- already done, skipping')
                    continue
                t0 = time.time()
                model, _P = train_variant(key, seed, train_loader, rules,
                                          device, n_epochs)
                res = evaluate_run(model, test_loader, n_classes=N_CLASSES,
                                   device=device,
                                   child_to_parent=CHILD_TO_PARENT_6,
                                   partial=PARTIAL)
                elapsed = time.time() - t0
                ckpt = save_run(args.save_dir, key, seed, model, res, scaler,
                                n_epochs, elapsed)
                print(f'[{done}/{total}] {VARIANTS[key][0]} seed {seed} '
                      f'| epochs {n_epochs} | acc {res["accuracy"]:.4f} '
                      f'| macroF1 {res["macro_f1"]:.4f} '
                      f'| reliab {res["reliability"]:.4f} '
                      f'| {elapsed:.0f}s | saved {os.path.basename(ckpt)}',
                      flush=True)
                del model
        print('\ndone. models saved under',
              os.path.join(args.save_dir, 'models'))
    else:
        print('[aggregate-only] skipping training\n')

    tracker, found = load_tracker(args.save_dir, args.variants, args.seeds)
    if not found:
        print('no completed runs found in', args.save_dir)
        return
    report(args.save_dir, tracker, found, args.seeds)


if __name__ == '__main__':
    main()
