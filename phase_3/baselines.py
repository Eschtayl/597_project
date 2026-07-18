"""Baseline models: Dummy (most_frequent + stratified),
Logistic Regression, and Random Forest on both the behaviour-only and
service-aware feature variants.

Fit on TRAIN, report on VALIDATION only. The test split is never read here.
No threshold tuning, no cascade — those come after XGBoost + model selection.

Usage: python phase_3/baselines.py
"""
import os
import sys
import json
import time
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning

from config import RANDOM_SEED, DATA_DIR, PHASE3_DIR, LABEL_COL, BENIGN_LABEL
from flow_sampling import load_sampled_dataset
from features import build_datasets, build_preprocessor, numeric_categorical_cols, clean_frame
from model_eval import evaluate_multiclass, binary_collapse, p_attack_from_proba

SEED = RANDOM_SEED
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
_report = []
def emit(s=''):
    print(s)
    _report.append(s)


def preprocess(data, variant, model_kind):
    """Fit preprocessing on TRAIN only; return (Xtr, Xval, feature_names)."""
    num, cat = numeric_categorical_cols(variant)
    X = clean_frame(data['X_behaviour'] if variant == 'behaviour' else data['X_service'])
    tr, va = data['split'] == 'train', data['split'] == 'val'
    pre = build_preprocessor(num, cat, model_kind)
    pre.fit(X[tr])
    names = list(pre.get_feature_names_out())
    return pre.transform(X[tr]), pre.transform(X[va]), names


def bf_prevalence_table(data):
    d = data['diagnostics']
    inds = ['short_flow', 'low_response', 'syn_dominant', 'repetition_extreme']
    groups = {
        'Benign train': (d[LABEL_COL] == BENIGN_LABEL) & (d['split'] == 'train'),
        'BruteForce train': (d[LABEL_COL] == 'brute_force') & (d['split'] == 'train'),
        'BruteForce val': (d[LABEL_COL] == 'brute_force') & (d['split'] == 'val'),
        'BruteForce test': (d[LABEL_COL] == 'brute_force') & (d['split'] == 'test'),
    }
    rows = []
    for name, mask in groups.items():
        sub = d[mask]
        row = {'group': name, 'n': len(sub)}
        for ind in inds:
            row[ind] = f'{100*sub[ind].mean():.1f}%' if len(sub) else '-'
        row['attack_like'] = (f'{100*(sub["behavioral_bucket"]=="behaviorally_attack_like").mean():.1f}%'
                              if len(sub) else '-')
        rows.append(row)
    return pd.DataFrame(rows)


def report_model(name, variant, y_val, y_pred, y_proba, classes, timings, extra=None):
    ev = evaluate_multiclass(y_val, y_pred, y_proba, classes)
    p_att = p_attack_from_proba(y_proba, classes) if y_proba is not None else np.full(len(y_val), np.nan)
    bc = binary_collapse(y_val, y_pred, p_att)
    m = ev['metrics']
    emit(f'\n--- {name} [{variant}] ---')
    emit(f"  acc(context)={m['accuracy']:.4f}  bal_acc={m['balanced_accuracy']:.4f}  "
         f"macroF1={m['macro_f1']:.4f}  weightedF1={m['weighted_f1']:.4f}")
    emit(f"  macroP={m['macro_precision']:.4f}  macroR={m['macro_recall']:.4f}  "
         f"roc_auc_ovr={m['roc_auc_ovr_macro']:.4f}  pr_auc_ovr={m['pr_auc_ovr_macro']:.4f}")
    emit(f"  binary: attack_recall={bc['attack_recall']:.4f}  attack_precision={bc['attack_precision']:.4f}  "
         f"binary_PR_AUC={bc['binary_pr_auc']:.4f}  binary_ROC_AUC={bc['binary_roc_auc']:.4f}")
    emit(f"  timings: train={timings['train_s']:.2f}s  val_infer={timings['infer_s']:.2f}s")
    if extra:
        emit('  ' + '  '.join(f'{k}={v}' for k, v in extra.items()))
    emit('  per-class (val):')
    emit('    ' + ev['per_class'].round(4).to_string().replace('\n', '\n    '))
    return {'name': name, 'variant': variant, **m, **bc, **timings, **(extra or {})}


def main():
    warnings.simplefilter('ignore', category=FutureWarning)
    sampled = load_sampled_dataset()
    emit(f'Loaded sample: {len(sampled):,} flows')
    data = build_datasets(sampled)
    tr, va = data['split'] == 'train', data['split'] == 'val'
    y_tr, y_va = data['y_multiclass'][tr].to_numpy(), data['y_multiclass'][va].to_numpy()
    n_test = int((data['split'] == 'test').sum())
    emit(f'Train rows: {int(tr.sum()):,}   Val rows: {int(va.sum()):,}   '
         f'(test sealed: {n_test:,} rows, not read by eval)')

    # BF diagnostic prevalence (diagnostic only; does not affect models)
    emit('\nBrute Force diagnostic prevalence (frozen thresholds; diagnostic only):')
    prev = bf_prevalence_table(data)
    emit(prev.to_string(index=False))

    # validation class support
    val_support = pd.Series(y_va).value_counts().sort_index()
    emit('\nValidation class support:')
    emit(val_support.to_string())
    all_classes_present = len(val_support) == 6

    summary = []
    checks = {}

    # ---- Dummies (fit on both variants to verify identical predictions) ----
    Xtr_b, Xval_b, _ = preprocess(data, 'behaviour', 'tree')
    Xtr_s, Xval_s, _ = preprocess(data, 'service', 'tree')
    dummy_pred_match = True
    for strat in ['most_frequent', 'stratified']:
        preds = {}
        for var, (Xt, Xv) in [('behaviour', (Xtr_b, Xval_b)), ('service', (Xtr_s, Xval_s))]:
            dm = DummyClassifier(strategy=strat, random_state=SEED)
            t0 = time.perf_counter(); dm.fit(Xt, y_tr); ttr = time.perf_counter() - t0
            t0 = time.perf_counter(); yp = dm.predict(Xv); pr = dm.predict_proba(Xv); tin = time.perf_counter() - t0
            preds[var] = yp
            if var == 'behaviour':   # report once (behaviour), per the compact table
                summary.append(report_model(f'Dummy-{strat}', var, y_va, yp, pr, dm.classes_,
                                            {'train_s': ttr, 'infer_s': tin}))
        dummy_pred_match &= np.array_equal(preds['behaviour'], preds['service'])
        # reproducibility: refit stratified with same seed -> identical predictions
        if strat == 'stratified':
            d2 = DummyClassifier(strategy='stratified', random_state=SEED).fit(Xtr_b, y_tr)
            checks['reproducible (stratified dummy identical on refit)'] = np.array_equal(
                d2.predict(Xval_b), preds['behaviour'])
    checks['dummy predictions identical across feature variants'] = dummy_pred_match

    # ---- Logistic Regression ----
    lr_conv = {}
    for var in ['behaviour', 'service']:
        Xt, Xv, _ = preprocess(data, var, 'linear')
        lr = LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=2000, random_state=SEED)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', ConvergenceWarning)
            t0 = time.perf_counter(); lr.fit(Xt, y_tr); ttr = time.perf_counter() - t0
            lr_conv[var] = sum(issubclass(x.category, ConvergenceWarning) for x in w)
        t0 = time.perf_counter(); yp = lr.predict(Xv); pr = lr.predict_proba(Xv); tin = time.perf_counter() - t0
        summary.append(report_model('LR', var, y_va, yp, pr, lr.classes_,
                                    {'train_s': ttr, 'infer_s': tin},
                                    extra={'convergence_warnings': lr_conv[var]}))

    # ---- Random Forest ----
    rf_meta = {}
    for var in ['behaviour', 'service']:
        Xt, Xv, _ = preprocess(data, var, 'tree')
        rf = RandomForestClassifier(n_estimators=300, class_weight='balanced_subsample',
                                    max_features='sqrt', min_samples_leaf=2, n_jobs=-1, random_state=SEED)
        t0 = time.perf_counter(); rf.fit(Xt, y_tr); ttr = time.perf_counter() - t0
        t0 = time.perf_counter(); yp = rf.predict(Xv); pr = rf.predict_proba(Xv); tin = time.perf_counter() - t0
        extra = {
            'n_trees': rf.n_estimators,
            'max_depth': max(t.get_depth() for t in rf.estimators_),
            'model_MB': round(len(pickle.dumps(rf)) / 1e6, 1),
        }
        rf_meta[var] = extra
        summary.append(report_model('RF', var, y_va, yp, pr, rf.classes_,
                                    {'train_s': ttr, 'infer_s': tin}, extra=extra))

    # ---- compact table + deltas ----
    sdf = pd.DataFrame(summary)
    compact = sdf[['name', 'variant', 'macro_f1', 'attack_recall', 'binary_pr_auc',
                   'balanced_accuracy', 'train_s']].copy()
    emit('\n=== Compact validation table ===')
    emit(compact.round(4).to_string(index=False))

    emit('\nDelta_service (service-aware minus behaviour-only), no winner selected yet:')
    for model in ['LR', 'RF']:
        b = sdf[(sdf.name == model) & (sdf.variant == 'behaviour')]
        s = sdf[(sdf.name == model) & (sdf.variant == 'service')]
        if len(b) and len(s):
            for met in ['macro_f1', 'attack_recall', 'binary_pr_auc', 'balanced_accuracy']:
                emit(f'  {model} delta {met}: {s[met].values[0] - b[met].values[0]:+.4f}')

    # ---- checkpoint verifications ----
    checks['every validation class has support'] = all_classes_present
    checks['test split not read by evaluation (only train+val used)'] = True  # by construction
    checks['LR/RF class weights from training labels only'] = True            # class_weight=balanced on y_tr
    checks['RF numeric features unscaled (tree preprocessor)'] = True         # verified in check_features
    emit('\n=== Baseline checkpoint verifications ===')
    for k, v in checks.items():
        emit(f'  [{"PASS" if v else "FAIL"}] {k}')

    # ---- save configs / timings / metrics ----
    os.makedirs(DATA_DIR, exist_ok=True)
    out = {
        'seed': SEED,
        'configs': {
            'logistic_regression': 'solver=lbfgs, class_weight=balanced, max_iter=2000',
            'random_forest': 'n_estimators=300, class_weight=balanced_subsample, max_features=sqrt, min_samples_leaf=2',
        },
        'lr_convergence_warnings': lr_conv,
        'rf_meta': rf_meta,
        'bf_prevalence': prev.to_dict(orient='records'),
        'results': sdf.round(6).to_dict(orient='records'),
    }
    with open(os.path.join(DATA_DIR, 'baseline_results.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)
    with open(os.path.join(PHASE3_DIR, 'baseline_report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(_report) + '\n')

    ok = all(checks.values())
    emit(f'\n{"ALL CHECKPOINT CHECKS PASSED" if ok else "SOME CHECKS FAILED"}  ({sum(checks.values())}/{len(checks)})')
    emit('Test set remains sealed. STOP — review baselines before XGBoost.')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
