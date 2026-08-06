#!/usr/bin/env python
"""
Export per-sample test predictions from P5's saved checkpoints.

INFERENCE ONLY -- nothing is trained and no existing file is overwritten.

Why this exists
---------------
`run_multiseed.ipynb` computed per-sample predictions, used them for the
in-memory McNemar test, and then discarded them: `MetricTracker.SCALAR_KEYS`
persists only scalar metrics, so `multiseed_results.json` has no `y_pred`.

That makes two things impossible after the fact:

  * the reliability partial-credit sweep at w = 0.25 / 0.75, which needs
    y_true / y_pred (the saved JSON has only w = 0.5);
  * any re-derivation of McNemar, which is paired at the sample level.

Both are recoverable by replaying the 10 saved checkpoints over the test set.
The models are deterministic in eval mode, so the exported predictions are
exactly the ones the original run produced -- which this script verifies by
comparing recomputed accuracy against the accuracy stored inside each
checkpoint, and aborting if they disagree.

What it writes (all NEW files, in ./saved/)
-------------------------------------------
  preds__{ltn,noltn}__seed{n}.npz   y_true, y_pred, probs  (P7's schema)
  hierarchical.json                 reliability sweep + hierarchical F1
                                    + per-seed McNemar between the two variants

`multiseed_results.json` and `multiseed_results.txt` are NOT touched.

Usage
-----
  python export_preds.py                    # all 10 checkpoints
  python export_preds.py --variants ltn     # one variant
  python export_preds.py --overwrite        # redo existing .npz
  python export_preds.py --aggregate-only   # rebuild hierarchical.json only

Requires `pip install LTNtorch` only transitively (ltn is not imported here --
this script needs no LTN machinery, since satisfaction is not recomputed).
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'P1_structurelevel'))   # utils.py

from utils import MultiKANModel                              # noqa: E402
from attention_modules import AttentionKANModel              # noqa: E402
from kan import KAN                                          # noqa: E402
from eval_metrics import (                                   # noqa: E402
    hierarchical_reliability, hierarchical_f1, mcnemar_test,
    CHILD_TO_PARENT_6, N_CLASSES,
)


# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #
SAVE_DIR = os.path.join(HERE, 'saved')
MODEL_DIR = os.path.join(SAVE_DIR, 'models')
TEST_PATH = os.path.join(HERE, '..', 'P1_structurelevel', 'efficiency',
                         'input_files', 'logiKNet_test_3994.csv')

# checkpoint tag -> display name (the names used in Results.tex Table C3)
VARIANTS = {
    'ltn':   'Attn-LogiK-Net (full)',
    'noltn': 'Attn-LogiK-Net (no logic)',
}
SEEDS = [0, 1, 2, 3, 4]
PARTIAL_SWEEP = (0.25, 0.5, 0.75)
GRID, K = 5, 3
ACC_TOL = 1e-6          # recomputed vs. stored accuracy


def npz_path(tag, seed):
    return os.path.join(SAVE_DIR, f'preds__{tag}__seed{seed}.npz')


def ckpt_path(tag, seed):
    return os.path.join(MODEL_DIR, f'{tag}_seed{seed}.pt')


# --------------------------------------------------------------------------- #
#  Rebuild + score one checkpoint
# --------------------------------------------------------------------------- #
def rebuild(ck, device):
    """Reconstruct AttentionKANModel from a checkpoint saved by run_multiseed.

    pykan's Symbolic_KANLayer holds un-picklable lambdas, so only the
    state_dict was stored. The seed passed to KAN() affects nothing but the
    random init, which load_state_dict immediately overwrites.
    """
    cfg = ck['config']
    kan = KAN(width=cfg['KAN_WIDTH'], grid=GRID, k=K,
              seed=ck.get('seed', cfg.get('SEED', 42)), device=device,
              auto_save=False, save_act=False)
    model = AttentionKANModel(cfg['IN_FEATURES'], MultiKANModel(kan),
                              **cfg['ATTN']).to(device)
    model.load_state_dict(ck['model_state'])
    model.eval()
    return model


def load_test_features(ck, test_df):
    """Normalise the test set with THIS checkpoint's own stored scaler."""
    cfg = ck['config']
    X = test_df[cfg['X_columns']].values.astype(np.float32)
    X = (X - np.asarray(ck['scaler_mean'])) / np.asarray(ck['scaler_scale'])
    return X.astype(np.float32)


@torch.no_grad()
def predict(model, X, device, chunk=1024):
    """Forward pass in the ORIGINAL row order (no shuffling) -> pairing holds."""
    preds, probs = [], []
    for s in range(0, X.shape[0], chunk):
        xb = torch.tensor(X[s:s + chunk], dtype=torch.float32, device=device)
        p = torch.softmax(model(xb, training=False), dim=1)
        probs.append(p.cpu().numpy())
        preds.append(p.argmax(dim=1).cpu().numpy())
    return (np.concatenate(preds).astype(np.int64),
            np.concatenate(probs).astype(np.float32))


def export_one(tag, seed, test_df, y_true, device, overwrite, tol):
    out = npz_path(tag, seed)
    if os.path.exists(out) and not overwrite:
        print(f'  {tag} seed {seed}: already exported, skipping')
        return None

    src = ckpt_path(tag, seed)
    if not os.path.exists(src):
        print(f'  {tag} seed {seed}: MISSING checkpoint {src}')
        return None

    ck = torch.load(src, map_location=device, weights_only=False)
    model = rebuild(ck, device)
    X = load_test_features(ck, test_df)
    y_pred, probs = predict(model, X, device)

    acc_new = float((y_pred == y_true).mean())
    acc_old = float(ck.get('metrics', {}).get('accuracy', float('nan')))

    # Integrity gate: if the rebuilt model does not reproduce the accuracy
    # recorded at training time, the reconstruction is wrong and the exported
    # predictions would silently corrupt every downstream number.
    if not np.isnan(acc_old) and abs(acc_new - acc_old) > tol:
        raise SystemExit(
            f'MISMATCH for {tag} seed {seed}: recomputed accuracy {acc_new:.10f} '
            f'!= stored {acc_old:.10f} (tol {tol:g}).\n'
            'The checkpoint was not reconstructed faithfully -- refusing to '
            'write predictions. Check KAN grid/k, the ATTN config, or whether '
            'the model was saved in a different pykan version.')

    np.savez_compressed(out, y_true=y_true, y_pred=y_pred, probs=probs)
    flag = 'ok' if np.isnan(acc_old) else f'matches stored ({acc_old:.4f})'
    print(f'  {tag} seed {seed}: acc {acc_new:.4f}  {flag}  -> '
          f'{os.path.basename(out)}')
    del model
    return acc_new


# --------------------------------------------------------------------------- #
#  Aggregate
# --------------------------------------------------------------------------- #
def aggregate(variants, seeds):
    out = {'partials': list(PARTIAL_SWEEP),
           'source': 'export_preds.py (inference over saved checkpoints)',
           'models': {}, 'mcnemar': []}

    for tag in variants:
        acc = {f'reliability@{p}': [] for p in PARTIAL_SWEEP}
        acc['hierarchical_f1'] = []
        got = []
        for seed in seeds:
            p = npz_path(tag, seed)
            if not os.path.exists(p):
                continue
            d = np.load(p)
            yt, yp = d['y_true'], d['y_pred']
            for w in PARTIAL_SWEEP:
                acc[f'reliability@{w}'].append(
                    hierarchical_reliability(yt, yp, CHILD_TO_PARENT_6, w))
            acc['hierarchical_f1'].append(
                hierarchical_f1(yt, yp, CHILD_TO_PARENT_6))
            got.append(seed)
        if not got:
            continue
        row = {'name': VARIANTS[tag], 'seeds': got}
        for m, vals in acc.items():
            a = np.asarray(vals, dtype=float)
            row[m] = {'mean': float(a.mean()),
                      'std': float(a.std(ddof=1) if len(a) > 1 else 0.0),
                      'values': [float(v) for v in a]}
        out['models'][tag] = row

    # per-seed McNemar between the two attention variants
    if all(t in out['models'] for t in ('ltn', 'noltn')):
        shared = sorted(set(out['models']['ltn']['seeds']) &
                        set(out['models']['noltn']['seeds']))
        for seed in shared:
            a = np.load(npz_path('ltn', seed))
            b = np.load(npz_path('noltn', seed))
            r = mcnemar_test(a['y_true'], a['y_pred'], b['y_pred'])
            r.update({'seed': int(seed), 'model_a': 'ltn', 'model_b': 'noltn',
                      'name_a': VARIANTS['ltn'], 'name_b': VARIANTS['noltn']})
            out['mcnemar'].append(r)

    path = os.path.join(SAVE_DIR, 'hierarchical.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2, default=float)
    return out, path


def report(agg):
    order = [t for t in VARIANTS if t in agg['models']]
    if not order:
        print('nothing to report')
        return

    cols = [f'reliability@{p}' for p in PARTIAL_SWEEP] + ['hierarchical_f1']
    print('\n' + '=' * 92)
    print('HIERARCHICAL MEASURES  (mean +/- std over seeds)')
    print('=' * 92)
    print(f"{'Model':<28}" + ''.join(f'{c:>19}' for c in cols))
    for t in order:
        r = agg['models'][t]
        print(f"{r['name']:<28}" + ''.join(
            f"{r[c]['mean']:.4f}+/-{r[c]['std']:.4f}".rjust(19) for c in cols))

    print('\nSanity: reliability@0.5 == hierarchical_f1 for a single-parent')
    print('two-level tree. Difference per model:')
    for t in order:
        r = agg['models'][t]
        d = abs(r['reliability@0.5']['mean'] - r['hierarchical_f1']['mean'])
        print(f"  {r['name']:<28} {d:.2e}  {'OK' if d < 1e-12 else 'UNEXPECTED'}")

    if agg['mcnemar']:
        n_sig = sum(m['p_value'] < 0.05 for m in agg['mcnemar'])
        print(f"\nMcNemar, {VARIANTS['ltn']} vs {VARIANTS['noltn']}: "
              f"significant in {n_sig}/{len(agg['mcnemar'])} seeds")
        for m in agg['mcnemar']:
            print(f"    seed {m['seed']}: b01 {m['b01']:>4} b10 {m['b10']:>4} "
                  f"| p {m['p_value']:.4g}")

    # ready-to-paste cells for the C3 table
    print('\n' + '=' * 92)
    print('CELLS FOR TABLE C3  (R_0.25 and R_0.75 were the [TODO] ones)')
    print('=' * 92)
    for t in order:
        r = agg['models'][t]
        cells = ' & '.join(
            f"${r[f'reliability@{p}']['mean']:.3f}\\pm"
            f"{r[f'reliability@{p}']['std']:.3f}$" for p in PARTIAL_SWEEP)
        hf = (f"${r['hierarchical_f1']['mean']:.3f}\\pm"
              f"{r['hierarchical_f1']['std']:.3f}$")
        print(f"    {r['name']:<28} & {cells} & {hf} \\\\")
    print('\n(ECE and Brier for these rows are already in multiseed_results.json')
    print(' and unchanged -- this script does not recompute them.)')


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description='Export per-sample predictions from P5 checkpoints')
    ap.add_argument('--variants', nargs='+', default=list(VARIANTS),
                    choices=list(VARIANTS))
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--test', default=TEST_PATH)
    ap.add_argument('--device', default=None, help='cuda:0 | cpu')
    ap.add_argument('--overwrite', action='store_true',
                    help='re-export .npz files that already exist')
    ap.add_argument('--aggregate-only', action='store_true',
                    help='skip inference, rebuild hierarchical.json from .npz')
    ap.add_argument('--tol', type=float, default=ACC_TOL,
                    help='accuracy tolerance for the integrity check')
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    print(f'checkpoints: {MODEL_DIR}')
    print(f'output     : {SAVE_DIR}\n')

    if not args.aggregate_only:
        if not os.path.exists(args.test):
            raise FileNotFoundError(args.test)
        with open(args.test, 'rb') as f:
            if f.read(40).startswith(b'version https://git-lfs'):
                raise RuntimeError(
                    f'{args.test} is a Git LFS pointer, not the actual CSV.\n'
                    'Run from the repo root:  git lfs install && git lfs pull')

        test_df = pd.read_csv(args.test)
        y_true = test_df['label_L2'].values.astype(np.int64)
        print(f'test set: {len(test_df)} flows | '
              f'class counts {np.bincount(y_true, minlength=N_CLASSES)}\n')

        print('exporting predictions:')
        for tag in args.variants:
            for seed in args.seeds:
                export_one(tag, seed, test_df, y_true, device,
                           args.overwrite, args.tol)
        print()
    else:
        print('[aggregate-only] skipping inference\n')

    agg, path = aggregate(args.variants, args.seeds)
    report(agg)
    print(f'\nwrote {path}')
    print('multiseed_results.json / .txt were NOT modified.')


if __name__ == '__main__':
    main()
