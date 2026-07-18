"""Ablation ladder + computational overhead.

New runs (validation only, test sealed; original sample + split, so rows are
comparable with baselines.py / xgboost_model.py):
  * XGB MC, locked unweighted params, behaviour WITHOUT engineered features
  * XGB MC, locked unweighted params, behaviour (with engineered)
  * XGB MC, locked unweighted params, service-aware

Ladder table assembled from cached results:
  Dummy -> LR -> RF -> XGB(-engineered) -> XGB(behaviour) -> XGB(service)
  -> threshold-tuned final cascade (test numbers from cascade_results.json).

Computational overhead: throughput of the frozen cascade head (preprocessor
transform + predict) on the candidate-flow set -> ms per Phase 2 alert.

Usage: python phase_3/ablations.py
"""
import os
import sys
import json
import time

import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from config import RANDOM_SEED, DATA_DIR, PHASE3_DIR
from flow_sampling import load_sampled_dataset
from features import (build_datasets, build_preprocessor, clean_frame,
                      BEHAVIOUR_NUMERIC, ENGINEERED_NUMERIC,
                      SERVICE_CATEGORICAL, SERVICE_BINARY)
from model_eval import evaluate_multiclass, binary_collapse, p_attack_from_proba
from cascade_heads import SCORES_PATH, MODELS_DIR

SEED = RANDOM_SEED
ABLATION_PATH = os.path.join(DATA_DIR, 'ablation_results.json')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
_report = []
def emit(s=''):
    print(s)
    _report.append(s)

CONFIGS = {
    'xgb_behaviour_no_engineered': ([c for c in BEHAVIOUR_NUMERIC if c not in ENGINEERED_NUMERIC], []),
    'xgb_behaviour': (BEHAVIOUR_NUMERIC, []),
    'xgb_service': (BEHAVIOUR_NUMERIC + SERVICE_BINARY, SERVICE_CATEGORICAL),
}


def run_config(name, data, num, cat, params):
    X_src = data['X_service'] if cat else data['X_behaviour']
    # X_behaviour lacks service cols; both variants contain all behaviour numerics
    X = clean_frame(X_src[num + cat])
    tr, va = data['split'] == 'train', data['split'] == 'val'
    pre = build_preprocessor(num, cat, 'tree')
    Xtr, Xva = pre.fit_transform(X[tr]), pre.transform(X[va])

    y_tr = data['y_multiclass'][tr].to_numpy()
    y_va = data['y_multiclass'][va].to_numpy()
    le = LabelEncoder().fit(y_tr)
    model = XGBClassifier(objective='multi:softprob', eval_metric='mlogloss',
                          tree_method='hist', n_estimators=500, early_stopping_rounds=50,
                          random_state=SEED, n_jobs=-1, verbosity=0, **params)
    model.fit(Xtr, le.transform(y_tr), eval_set=[(Xva, le.transform(y_va))], verbose=False)

    proba = model.predict_proba(Xva)
    y_pred = le.inverse_transform(proba.argmax(axis=1))
    s = p_attack_from_proba(proba, list(le.classes_))
    ev = evaluate_multiclass(y_va, y_pred, proba, list(le.classes_))
    bc = binary_collapse(y_va, y_pred, s)
    row = {'config': name, 'n_features_raw': len(num) + len(cat),
           'n_trees': int(model.best_iteration),
           'macro_f1': ev['metrics']['macro_f1'],
           'balanced_accuracy': ev['metrics']['balanced_accuracy'],
           'binary_pr_auc': bc['binary_pr_auc'],
           'attack_precision': bc['attack_precision'],
           'attack_recall': bc['attack_recall']}
    emit(f"  {name:<28} macroF1={row['macro_f1']:.4f}  PR-AUC={row['binary_pr_auc']:.4f}  "
         f"attackP={row['attack_precision']:.4f}  attackR={row['attack_recall']:.4f}  "
         f"trees={row['n_trees']}")
    return row


def ladder_from_cached(new_rows):
    """Assemble the model-progression ladder (Dummy -> LR -> RF -> XGBoost ->
    final cascade) from cached result files."""
    rows = []
    with open(os.path.join(DATA_DIR, 'baseline_results.json'), encoding='utf-8') as f:
        base = json.load(f)
    for r in base['results']:
        rows.append({'stage': f"{r['name']} [{r['variant']}]",
                     'macro_f1': r.get('macro_f1'),
                     'binary_pr_auc': r.get('binary_pr_auc')})
    for r in new_rows:
        rows.append({'stage': r['config'], 'macro_f1': r['macro_f1'],
                     'binary_pr_auc': r['binary_pr_auc']})
    with open(os.path.join(DATA_DIR, 'cascade_results.json'), encoding='utf-8') as f:
        casc = json.load(f)
    t = casc['test_metrics']
    rows.append({'stage': f"final cascade (test): {casc['frozen_config']['head']} "
                          f"policy {casc['frozen_config']['policy']}",
                 'macro_f1': None, 'binary_pr_auc': None,
                 'fp_reduction': t['fp_reduction'], 'tp_retention': t['tp_retention'],
                 'precision_gain': f"{t['precision_phase2']:.3f}->{t['precision_cascade']:.3f}"})
    return rows


def overhead_timing():
    """Cascade computational overhead: frozen-head scoring throughput on the
    candidate-flow set + per-alert cost estimate."""
    head = joblib.load(os.path.join(MODELS_DIR, 'head_mc_service_cascade.joblib'))
    scores = pd.read_parquet(SCORES_PATH)
    from features import add_engineered_behaviour, add_service_fields, numeric_categorical_cols
    unified = pd.read_parquet(os.path.join(DATA_DIR, 'unified_flows_full.parquet'))
    cand = unified[unified['LogicalFlowID'].isin(set(scores['LogicalFlowID']))].reset_index(drop=True)
    cand = add_service_fields(add_engineered_behaviour(cand))
    num, cat = numeric_categorical_cols('service')
    Xc = clean_frame(cand[num + cat])

    t0 = time.perf_counter()
    Xt = head['preprocessor'].transform(Xc)
    t1 = time.perf_counter()
    head['model'].predict_proba(Xt)
    t2 = time.perf_counter()
    n = len(Xc)
    emit(f'\nComputational overhead (frozen head MC-service, {n:,} candidate flows):')
    emit(f'  preprocess: {1000*(t1-t0):.0f} ms   predict: {1000*(t2-t1):.0f} ms   '
         f'total {1e6*(t2-t0)/n:.1f} us/flow')
    emit(f'  Phase 3 adds one flow lookup + one XGBoost inference per Phase 2 alert '
         f'(O(trees x depth) per flow, trees={head["model"].best_iteration}); '
         f'training is offline. Two-phase complexity = Phase 2 IF O(t_IF x log n) '
         f'per packet + Phase 3 O(t_XGB x d) per alerted flow only.')
    return {'n_flows': n, 'preprocess_ms': 1000*(t1-t0), 'predict_ms': 1000*(t2-t1),
            'us_per_flow': 1e6*(t2-t0)/n}


def main():
    sampled = load_sampled_dataset()
    data = build_datasets(sampled)
    with open(os.path.join(DATA_DIR, 'xgb_results.json'), encoding='utf-8') as f:
        best = json.load(f)['best_params']

    emit('=== Engineered-feature / service ablation (XGB MC, locked unweighted params; '
         'validation only) ===')
    new_rows = []
    for name, (num, cat) in CONFIGS.items():
        params = best['service'] if cat else best['behaviour']
        new_rows.append(run_config(name, data, num, cat, params))

    emit('\n=== Model progression ladder (validation; cascade row = sealed test) ===')
    ladder = ladder_from_cached(new_rows)
    emit(pd.DataFrame(ladder).to_string(index=False))

    timing = overhead_timing()

    with open(ABLATION_PATH, 'w', encoding='utf-8') as f:
        json.dump({'ablation_rows': new_rows, 'ladder': ladder, 'overhead': timing}, f, indent=1)
    emit(f'\nSaved -> {ABLATION_PATH}')

    with open(os.path.join(PHASE3_DIR, 'ablation_report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(_report) + '\n')


if __name__ == '__main__':
    main()
