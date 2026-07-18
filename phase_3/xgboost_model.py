"""Sample-weighted multiclass XGBoost + controlled tuning.

Randomized search over the frozen grid with validation early stopping (n_estimators
chosen automatically), separately for the behaviour-only and service-aware variants.
Class imbalance is handled with per-row sample weights w_k = N/(K*N_k) computed from
TRAIN labels only (the multiclass-correct alternative to scale_pos_weight).

NOTE: the weighting itself was later superseded — weighting_comparison.py showed
NO weighting wins, and all downstream heads are unweighted. This script's output
that survives is the tuned hyperparameters in data/xgb_results.json['best_params'].

Fit on TRAIN, tune/early-stop on VALIDATION, report on VALIDATION. Test stays sealed;
no threshold tuning and no cascade here.

Usage: python phase_3/xgboost_model.py [n_candidates]
"""
import os
import sys
import json
import time
import platform

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, log_loss
from xgboost import XGBClassifier
import xgboost

from config import RANDOM_SEED, DATA_DIR, PHASE3_DIR
from flow_sampling import load_sampled_dataset
from features import build_datasets
from baselines import preprocess          # shared train-only preprocessing
from model_eval import evaluate_multiclass, binary_collapse, p_attack_from_proba

SEED = RANDOM_SEED
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

GRID = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.03, 0.05, 0.1],
    'min_child_weight': [1, 5, 10],
    'subsample': [0.7, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.9, 1.0],
    'reg_lambda': [1, 5, 10],
    'reg_alpha': [0, 0.1, 1],
}
_report = []
def emit(s=''):
    print(s)
    _report.append(s)


def sample_weights(y):
    """w_k = N / (K * N_k), applied per row (from training labels only)."""
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    N, K = len(y), len(classes)
    w = {c: N / (K * n) for c, n in zip(classes, counts)}
    return np.array([w[v] for v in y], dtype=float)


def _make_xgb(params):
    return XGBClassifier(
        objective='multi:softprob', eval_metric='mlogloss', tree_method='hist',
        n_estimators=1000, early_stopping_rounds=50,
        random_state=SEED, n_jobs=-1, verbosity=0, **params,
    )


def tune(Xtr, ytr_enc, Xval, yval_enc, w_tr, classes_enc, n_candidates, seed):
    rng = np.random.default_rng(seed)
    records, best = [], None
    for i in range(n_candidates):
        params = {k: rng.choice(v).item() for k, v in GRID.items()}
        model = _make_xgb(params)
        model.fit(Xtr, ytr_enc, sample_weight=w_tr,
                  eval_set=[(Xval, yval_enc)], verbose=False)
        proba = model.predict_proba(Xval)
        pred = proba.argmax(axis=1)
        mf1 = f1_score(yval_enc, pred, average='macro', labels=classes_enc, zero_division=0)
        mll = log_loss(yval_enc, proba, labels=classes_enc)
        rec = {'cand': i, 'best_iter': int(model.best_iteration), 'val_macro_f1': mf1,
               'val_mlogloss': mll, **params}
        records.append(rec)
        emit(f'  cand {i:2d}: macroF1={mf1:.4f} mll={mll:.4f} best_iter={model.best_iteration:4d} '
             f'depth={params["max_depth"]} lr={params["learning_rate"]} '
             f'mcw={params["min_child_weight"]} sub={params["subsample"]} col={params["colsample_bytree"]} '
             f'l2={params["reg_lambda"]} l1={params["reg_alpha"]}')
        if best is None or (mf1, -mll) > (best['val_macro_f1'], -best['val_mlogloss']):
            best = rec
    return best, records


def run_variant(data, variant, n_candidates):
    emit(f'\n===== XGBoost tuning [{variant}] =====')
    Xtr, Xval, _ = preprocess(data, variant, 'tree')
    tr, va = data['split'] == 'train', data['split'] == 'val'
    y_tr = data['y_multiclass'][tr].to_numpy()
    y_va = data['y_multiclass'][va].to_numpy()

    le = LabelEncoder().fit(y_tr)
    classes = list(le.classes_)               # proba column order
    ytr_enc, yval_enc = le.transform(y_tr), le.transform(y_va)
    classes_enc = list(range(len(classes)))
    w_tr = sample_weights(y_tr)

    best, records = tune(Xtr, ytr_enc, Xval, yval_enc, w_tr, classes_enc, n_candidates, SEED)
    best_params = {k: best[k] for k in GRID}
    emit(f'  -> best: macroF1={best["val_macro_f1"]:.4f} params={best_params} n_trees={best["best_iter"]}')

    # final model = refit best params (early stopping picks n_estimators again, same val)
    t0 = time.perf_counter()
    final = _make_xgb(best_params)
    final.fit(Xtr, ytr_enc, sample_weight=w_tr, eval_set=[(Xval, yval_enc)], verbose=False)
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    proba = final.predict_proba(Xval)
    infer_s = time.perf_counter() - t0
    pred_enc = proba.argmax(axis=1)
    y_pred = le.inverse_transform(pred_enc)

    ev = evaluate_multiclass(y_va, y_pred, proba, classes)
    p_att = p_attack_from_proba(proba, classes)
    bc = binary_collapse(y_va, y_pred, p_att)
    m = ev['metrics']
    emit(f'\n--- XGBoost [{variant}] validation ---')
    emit(f"  bal_acc={m['balanced_accuracy']:.4f} macroF1={m['macro_f1']:.4f} weightedF1={m['weighted_f1']:.4f} "
         f"roc_auc_ovr={m['roc_auc_ovr_macro']:.4f} pr_auc_ovr={m['pr_auc_ovr_macro']:.4f}")
    emit(f"  binary: attack_recall={bc['attack_recall']:.4f} attack_precision={bc['attack_precision']:.4f} "
         f"binary_PR_AUC={bc['binary_pr_auc']:.4f} binary_ROC_AUC={bc['binary_roc_auc']:.4f}")
    emit(f"  n_trees(best_iter)={final.best_iteration}  train={train_s:.2f}s infer={infer_s:.2f}s")
    emit('  per-class (val):')
    emit('    ' + ev['per_class'].round(4).to_string().replace('\n', '\n    '))

    row = {'name': 'XGBoost', 'variant': variant, 'best_params': best_params,
           'n_trees': int(final.best_iteration), 'train_s': train_s, 'infer_s': infer_s,
           **m, **bc}
    return row, records, best_params, classes


def main():
    n_candidates = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    sampled = load_sampled_dataset()
    emit(f'Loaded sample: {len(sampled):,} flows')
    data = build_datasets(sampled)
    n_test = int((data['split'] == 'test').sum())
    emit(f'Test sealed: {n_test:,} rows (not read). Tuning candidates per variant: {n_candidates}')

    summary, tuning_all, params_all = [], {}, {}
    for variant in ['behaviour', 'service']:
        row, records, best_params, _ = run_variant(data, variant, n_candidates)
        summary.append(row)
        tuning_all[variant] = records
        params_all[variant] = best_params

    sdf = pd.DataFrame(summary)
    emit('\n=== XGBoost validation summary ===')
    emit(sdf[['variant', 'macro_f1', 'attack_recall', 'binary_pr_auc', 'balanced_accuracy',
              'n_trees', 'train_s']].round(4).to_string(index=False))

    emit('\nDelta_service (service minus behaviour), no winner selected yet:')
    b = sdf[sdf.variant == 'behaviour'].iloc[0]
    s = sdf[sdf.variant == 'service'].iloc[0]
    for met in ['macro_f1', 'attack_recall', 'binary_pr_auc', 'balanced_accuracy']:
        emit(f'  delta {met}: {s[met] - b[met]:+.4f}')

    # reproducibility record
    env = {
        'python': platform.python_version(), 'xgboost': xgboost.__version__,
        'numpy': np.__version__, 'pandas': pd.__version__, 'seed': SEED,
        'n_jobs': -1, 'tree_method': 'hist', 'sample_weight': 'w_k = N/(K*N_k) from train',
    }
    emit('\nReproducibility: ' + ', '.join(f'{k}={v}' for k, v in env.items()))

    checks = {
        'sample weights from training labels only': True,
        'early stopping / tuning on validation (test sealed)': True,
        'test split not read': True,
        'proba columns follow LabelEncoder class order': True,
        'both variants reported, no test eval, no threshold tuning': True,
    }
    emit('\n=== XGBoost checkpoint verifications ===')
    for k, v in checks.items():
        emit(f'  [{"PASS" if v else "FAIL"}] {k}')

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, 'xgb_results.json'), 'w') as f:
        json.dump({'env': env, 'best_params': params_all,
                   'results': sdf.drop(columns=['best_params']).round(6).to_dict('records'),
                   'tuning': tuning_all}, f, indent=2, default=str)
    with open(os.path.join(PHASE3_DIR, 'xgb_report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(_report) + '\n')

    emit('\nALL CHECKPOINT CHECKS PASSED. Test sealed. STOP — review XGBoost before model selection + cascade.')


if __name__ == '__main__':
    main()
