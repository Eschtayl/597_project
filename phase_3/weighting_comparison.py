"""Pre-registered class-weighting comparison for XGBoost.

Full inverse-frequency weighting over-weights attacks ~250x and collapses precision.
Compare three schemes with the tuned hyperparameters held fixed (so only the weighting
changes), selecting by threshold-independent ranking quality (binary PR-AUC) plus the
best achievable binary operating point — because the cascade sets the final threshold.

  none      : all weights 1
  balanced  : w_k = N/(K*N_k)           (full inverse frequency)
  dampened  : w_k = sqrt(N/(K*N_k))     (reduced benign / capped attack emphasis)

All weights are normalized to mean 1 so total emphasis (and thus regularization scale)
is comparable across schemes. Validation only; test stays sealed.

Usage: python phase_3/weighting_comparison.py
"""
import os
import sys
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_curve
from xgboost import XGBClassifier

from config import RANDOM_SEED, DATA_DIR, PHASE3_DIR, BENIGN_LABEL
from flow_sampling import load_sampled_dataset
from features import build_datasets
from baselines import preprocess
from model_eval import evaluate_multiclass, binary_collapse, p_attack_from_proba

SEED = RANDOM_SEED
N_ESTIMATORS_CAP = 500   # early stopping selects the actual tree count below this
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
_report = []
def emit(s=''):
    print(s)
    _report.append(s)


def weights_for(y, scheme):
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    N, K = len(y), len(classes)
    raw = {c: N / (K * n) for c, n in zip(classes, counts)}
    if scheme == 'none':
        w = np.ones(len(y))
    elif scheme == 'balanced':
        w = np.array([raw[v] for v in y])
    elif scheme == 'dampened':
        w = np.array([np.sqrt(raw[v]) for v in y])
    else:
        raise ValueError(scheme)
    return w / w.mean()   # normalize to mean 1


def best_binary_operating_point(y_true, p_attack):
    """Best achievable binary F1 over the threshold (proxy for what cascade
    thresholding can reach), plus precision at recall >= 0.90."""
    yt = (np.asarray(y_true) != BENIGN_LABEL).astype(int)
    prec, rec, _ = precision_recall_curve(yt, p_attack)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    best_f1 = float(np.nanmax(f1))
    mask = rec >= 0.90
    prec_at_r90 = float(prec[mask].max()) if mask.any() else float('nan')
    return best_f1, prec_at_r90


def main():
    sampled = load_sampled_dataset()
    data = build_datasets(sampled)
    variant = 'service'
    Xtr, Xval, _ = preprocess(data, variant, 'tree')
    tr, va = data['split'] == 'train', data['split'] == 'val'
    y_tr = data['y_multiclass'][tr].to_numpy()
    y_va = data['y_multiclass'][va].to_numpy()
    le = LabelEncoder().fit(y_tr)
    classes = list(le.classes_)
    ytr_enc, yval_enc = le.transform(y_tr), le.transform(y_va)

    with open(os.path.join(DATA_DIR, 'xgb_results.json')) as f:
        best_params = json.load(f)['best_params'][variant]
    emit(f'Weighting comparison on [{variant}] with fixed tuned params: {best_params}')
    emit(f'n_estimators cap={N_ESTIMATORS_CAP} (early stopping on val), test sealed.\n')

    rows = []
    for scheme in ['none', 'balanced', 'dampened']:
        w = weights_for(y_tr, scheme)
        model = XGBClassifier(objective='multi:softprob', eval_metric='mlogloss',
                              tree_method='hist', n_estimators=N_ESTIMATORS_CAP,
                              early_stopping_rounds=50, random_state=SEED, n_jobs=-1,
                              verbosity=0, **best_params)
        model.fit(Xtr, ytr_enc, sample_weight=w, eval_set=[(Xval, yval_enc)], verbose=False)
        proba = model.predict_proba(Xval)
        y_pred = le.inverse_transform(proba.argmax(axis=1))
        ev = evaluate_multiclass(y_va, y_pred, proba, classes)
        p_att = p_attack_from_proba(proba, classes)
        bc = binary_collapse(y_va, y_pred, p_att)
        best_f1, prec_r90 = best_binary_operating_point(y_va, p_att)
        m = ev['metrics']
        rows.append({
            'scheme': scheme, 'n_trees': int(model.best_iteration),
            'macro_f1': m['macro_f1'], 'balanced_acc': m['balanced_accuracy'],
            'argmax_attack_recall': bc['attack_recall'], 'argmax_attack_precision': bc['attack_precision'],
            'binary_pr_auc': bc['binary_pr_auc'], 'binary_roc_auc': bc['binary_roc_auc'],
            'best_binary_f1_swept': best_f1, 'precision@recall>=0.90': prec_r90,
        })
        emit(f'--- scheme={scheme} (n_trees={model.best_iteration}) ---')
        emit(f"  argmax: macroF1={m['macro_f1']:.4f} attack_recall={bc['attack_recall']:.4f} "
             f"attack_precision={bc['attack_precision']:.4f} bal_acc={m['balanced_accuracy']:.4f}")
        emit(f"  threshold-free: binary_PR_AUC={bc['binary_pr_auc']:.4f} roc_auc={bc['binary_roc_auc']:.4f} "
             f"best_binary_F1(swept)={best_f1:.4f} precision@recall>=0.90={prec_r90:.4f}")
        emit('    ' + ev['per_class'][['precision', 'recall', 'f1']].round(3).to_string().replace('\n', '\n    '))

    tab = pd.DataFrame(rows)
    emit('\n=== Weighting comparison (validation) ===')
    emit(tab.round(4).to_string(index=False))

    # selection by threshold-independent quality (cascade sets the operating point)
    best = tab.sort_values(['binary_pr_auc', 'best_binary_f1_swept'], ascending=False).iloc[0]
    emit(f"\nRecommended scheme (by binary PR-AUC, then swept F1): '{best['scheme']}' "
         f"(PR-AUC={best['binary_pr_auc']:.4f}, best swept F1={best['best_binary_f1_swept']:.4f})")
    emit('Note: argmax precision/recall differ mostly by operating point, which the cascade '
         'threshold re-selects; PR-AUC reflects ranking quality independent of that point.')

    with open(os.path.join(DATA_DIR, 'weighting_comparison.json'), 'w') as f:
        json.dump({'variant': variant, 'best_params': best_params,
                   'n_estimators_cap': N_ESTIMATORS_CAP,
                   'results': tab.round(6).to_dict('records'),
                   'recommended': best['scheme']}, f, indent=2, default=str)
    with open(os.path.join(PHASE3_DIR, 'weighting_comparison_report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(_report) + '\n')
    emit('\nSTOP — review weighting comparison before locking the cascade model.')


if __name__ == '__main__':
    main()
