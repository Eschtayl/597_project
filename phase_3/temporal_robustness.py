"""Blocked temporal-fold robustness evaluation.

Expanding-window protocol on the winning configuration (XGB multiclass,
service-aware, locked unweighted params): folds are contiguous time blocks
within each class (splits.blocked_temporal_folds); model k trains on folds < k
and predicts fold k. Out-of-time predictions are pooled across folds for the
headline numbers; per-fold class support is reported because thin classes (XSS)
are unstable. Blocked-CV uncertainty is conservative (folds share training data).

Early stopping uses the latest 10% of each training block (by ts_first) — still
strictly earlier than the test fold, so no temporal leakage.

Usage: python phase_3/temporal_robustness.py
"""
import os
import sys
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from config import RANDOM_SEED, DATA_DIR, PHASE3_DIR, LABEL_COL, BENIGN_LABEL
from flow_sampling import load_sampled_dataset
from features import (add_engineered_behaviour, add_service_fields, clean_frame,
                      build_preprocessor, numeric_categorical_cols)
from splits import blocked_temporal_folds, temporal_fold_plan, FOLD_COL
from model_eval import evaluate_multiclass, binary_collapse, p_attack_from_proba

SEED = RANDOM_SEED
RESULTS_PATH = os.path.join(DATA_DIR, 'temporal_robustness.json')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
_report = []
def emit(s=''):
    print(s)
    _report.append(s)


def main():
    sampled = load_sampled_dataset()
    df = add_service_fields(add_engineered_behaviour(sampled.reset_index(drop=True)))
    df = blocked_temporal_folds(df)

    with open(os.path.join(DATA_DIR, 'xgb_results.json'), encoding='utf-8') as f:
        params = json.load(f)['best_params']['service']
    num, cat = numeric_categorical_cols('service')
    X_all = clean_frame(df[num + cat])
    y_all = df[LABEL_COL]

    emit('=== Blocked temporal folds — XGB MC service-aware, locked params ===')
    emit('\nClass support per temporal fold:')
    support = pd.crosstab(df[FOLD_COL], df[LABEL_COL])
    emit(support.to_string())

    pooled_true, pooled_pred, pooled_s = [], [], []
    fold_rows = []
    for train_folds, test_fold in temporal_fold_plan():
        tr_mask = df[FOLD_COL].isin(train_folds)
        te_mask = df[FOLD_COL] == test_fold

        # time-ordered early-stop slice: latest 10% of each CLASS's training
        # block (per-class, matching the per-class fold construction — a global
        # cut would drop late-starting captures from the fit slice entirely)
        fit_parts, es_parts = [], []
        for _, grp in df[tr_mask].groupby(LABEL_COL):
            order = grp['ts_first'].sort_values().index
            cut = max(int(len(order) * 0.9), 1)
            fit_parts.append(order[:cut]); es_parts.append(order[cut:])
        fit_idx = fit_parts[0].append(fit_parts[1:])
        es_idx = es_parts[0].append(es_parts[1:])

        pre = build_preprocessor(num, cat, 'tree')
        Xfit = pre.fit_transform(X_all.loc[fit_idx])
        Xes = pre.transform(X_all.loc[es_idx])
        Xte = pre.transform(X_all.loc[te_mask])

        le = LabelEncoder().fit(y_all)   # fixed class set across folds
        model = XGBClassifier(objective='multi:softprob', eval_metric='mlogloss',
                              tree_method='hist', n_estimators=500,
                              early_stopping_rounds=50, random_state=SEED,
                              n_jobs=-1, verbosity=0, **params)
        model.fit(Xfit, le.transform(y_all.loc[fit_idx]),
                  eval_set=[(Xes, le.transform(y_all.loc[es_idx]))], verbose=False)

        proba = model.predict_proba(Xte)
        y_pred = le.inverse_transform(proba.argmax(axis=1))
        y_true = y_all.loc[te_mask].to_numpy()
        s = p_attack_from_proba(proba, list(le.classes_))
        bc = binary_collapse(y_true, y_pred, s)
        ev = evaluate_multiclass(y_true, y_pred, proba, list(le.classes_))
        fold_rows.append({'test_fold': test_fold, 'n_train': int(tr_mask.sum()),
                          'n_test': int(te_mask.sum()), 'n_trees': int(model.best_iteration),
                          'macro_f1': ev['metrics']['macro_f1'],
                          'binary_pr_auc': bc['binary_pr_auc'],
                          'attack_recall': bc['attack_recall'],
                          'attack_precision': bc['attack_precision']})
        emit(f"\nfold {test_fold} (train folds {train_folds}): n_train={tr_mask.sum():,} "
             f"n_test={te_mask.sum():,} trees={model.best_iteration}")
        emit(f"  macroF1={ev['metrics']['macro_f1']:.4f}  PR-AUC={bc['binary_pr_auc']:.4f}  "
             f"attackP={bc['attack_precision']:.4f}  attackR={bc['attack_recall']:.4f}")

        pooled_true.append(y_true); pooled_pred.append(y_pred); pooled_s.append(s)

    y_true = np.concatenate(pooled_true)
    y_pred = np.concatenate(pooled_pred)
    s = np.concatenate(pooled_s)
    classes = sorted(y_all.unique())
    ev = evaluate_multiclass(y_true, y_pred, None, classes)
    bc = binary_collapse(y_true, y_pred, s)

    emit('\n=== Pooled out-of-time performance (folds 1-3) ===')
    emit(f"macroF1={ev['metrics']['macro_f1']:.4f}  binary_PR_AUC={bc['binary_pr_auc']:.4f}  "
         f"attackP={bc['attack_precision']:.4f}  attackR={bc['attack_recall']:.4f}")
    emit('\nPooled per-class (out-of-time):')
    emit(ev['per_class'].round(4).to_string())

    xss_n = int(ev['per_class'].loc['XSS', 'support']) if 'XSS' in ev['per_class'].index else 0
    emit(f'\nCaveats: XSS out-of-time support is thin (n={xss_n}) — its temporal estimates '
         'are UNSTABLE and must not be read as conclusive (report with the stratified-split '
         'numbers as primary). Blocked-CV folds share training data, so uncertainty is '
         'understated; treat these as conservative robustness checks, not new headline metrics.')

    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump({'per_fold': fold_rows,
                   'pooled': {'macro_f1': ev['metrics']['macro_f1'],
                              'binary_pr_auc': bc['binary_pr_auc'],
                              'attack_precision': bc['attack_precision'],
                              'attack_recall': bc['attack_recall']},
                   'pooled_per_class': ev['per_class'].to_dict(orient='index'),
                   'support_per_fold': support.to_dict(orient='index')}, f, indent=1)
    emit(f'\nSaved -> {RESULTS_PATH}')

    with open(os.path.join(PHASE3_DIR, 'temporal_robustness_report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(_report) + '\n')


if __name__ == '__main__':
    main()
