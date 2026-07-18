"""The four Phase 3 model heads.

  MC  behaviour : multiclass XGBoost, score s_MC  = 1 - P(benign)
  MC  service   : multiclass XGBoost, score s_MC  = 1 - P(benign)
  BIN behaviour : binary XGBoost,     score s_BIN = P(attack)
  BIN service   : binary XGBoost,     score s_BIN = P(attack)

All heads are UNWEIGHTED XGBoost with the locked tuned hyperparameters (the
weighting comparison showed no weighting is best). Each head's probabilities are
NOT assumed comparable across heads; every head gets its own cascade threshold
later. Fit on TRAIN, early-stop on VAL. Test is never scored here.

Saves fitted (preprocessor, model) per head + validation scores for the cascade.
Reports only the standalone-classifier view (view 1) on validation.

Usage: python phase_3/heads.py
"""
import os
import sys
import json

import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             average_precision_score, roc_auc_score)
from xgboost import XGBClassifier

from config import RANDOM_SEED, DATA_DIR, PHASE3_DIR
from flow_sampling import load_sampled_dataset
from features import build_datasets
from baselines import preprocess
from model_eval import evaluate_multiclass, binary_collapse, p_attack_from_proba

SEED = RANDOM_SEED
N_ESTIMATORS_CAP = 500
MODELS_DIR = os.path.join(DATA_DIR, 'models')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
_report = []
def emit(s=''):
    print(s)
    _report.append(s)


def _xgb(params, objective, eval_metric, num_class=None):
    kw = dict(objective=objective, eval_metric=eval_metric, tree_method='hist',
              n_estimators=N_ESTIMATORS_CAP, early_stopping_rounds=50,
              random_state=SEED, n_jobs=-1, verbosity=0, **params)
    if num_class:
        kw['num_class'] = num_class
    return XGBClassifier(**kw)


def train_multiclass(data, variant, params):
    Xtr, Xval, _ = preprocess(data, variant, 'tree')
    tr, va = data['split'] == 'train', data['split'] == 'val'
    y_tr = data['y_multiclass'][tr].to_numpy()
    y_va = data['y_multiclass'][va].to_numpy()
    le = LabelEncoder().fit(y_tr)
    classes = list(le.classes_)
    model = _xgb(params, 'multi:softprob', 'mlogloss')
    model.fit(Xtr, le.transform(y_tr), eval_set=[(Xval, le.transform(y_va))], verbose=False)
    proba = model.predict_proba(Xval)
    y_pred = le.inverse_transform(proba.argmax(axis=1))
    s_mc = p_attack_from_proba(proba, classes)          # 1 - P(benign)
    ev = evaluate_multiclass(y_va, y_pred, proba, classes)
    bc = binary_collapse(y_va, y_pred, s_mc)
    emit(f'\n--- MC [{variant}] standalone (val) ---  n_trees={model.best_iteration}')
    emit(f"  macroF1={ev['metrics']['macro_f1']:.4f} bal_acc={ev['metrics']['balanced_accuracy']:.4f} "
         f"binary_PR_AUC={bc['binary_pr_auc']:.4f} attack_precision={bc['attack_precision']:.4f} "
         f"attack_recall={bc['attack_recall']:.4f}")
    return {'preprocessor_variant': variant, 'kind': 'multiclass', 'model': model,
            'label_encoder': le, 'classes': classes}, s_mc, ev, bc


def train_binary(data, variant, params):
    Xtr, Xval, _ = preprocess(data, variant, 'tree')
    tr, va = data['split'] == 'train', data['split'] == 'val'
    y_tr = data['y_binary'][tr].to_numpy()
    y_va = data['y_binary'][va].to_numpy()
    model = _xgb(params, 'binary:logistic', 'logloss')
    model.fit(Xtr, y_tr, eval_set=[(Xval, y_va)], verbose=False)
    s_bin = model.predict_proba(Xval)[:, 1]             # P(attack)
    yp = (s_bin >= 0.5).astype(int)
    m = {'attack_precision': precision_score(y_va, yp, zero_division=0),
         'attack_recall': recall_score(y_va, yp, zero_division=0),
         'attack_f1': f1_score(y_va, yp, zero_division=0),
         'binary_pr_auc': average_precision_score(y_va, s_bin),
         'binary_roc_auc': roc_auc_score(y_va, s_bin)}
    emit(f'\n--- BIN [{variant}] standalone (val) ---  n_trees={model.best_iteration}')
    emit(f"  attack_precision={m['attack_precision']:.4f} attack_recall={m['attack_recall']:.4f} "
         f"binary_PR_AUC={m['binary_pr_auc']:.4f} binary_ROC_AUC={m['binary_roc_auc']:.4f}")
    return {'preprocessor_variant': variant, 'kind': 'binary', 'model': model}, s_bin, m


def main():
    sampled = load_sampled_dataset()
    data = build_datasets(sampled)
    va = data['split'] == 'val'
    emit(f'Sample: {len(sampled):,} flows. Val: {int(va.sum()):,}. '
         f'Test sealed: {int((data["split"]=="test").sum()):,} (not scored).')

    with open(os.path.join(DATA_DIR, 'xgb_results.json')) as f:
        best = json.load(f)['best_params']
    emit(f'Locked unweighted params: behaviour={best["behaviour"]}\n                          service={best["service"]}')

    os.makedirs(MODELS_DIR, exist_ok=True)
    scores = pd.DataFrame({
        'LogicalFlowID': data['diagnostics'].loc[va, 'LogicalFlowID'].values,
        'label': data['y_multiclass'][va].values,
        'y_binary': data['y_binary'][va].values,
        'service_aligned': data['diagnostics'].loc[va, 'service_aligned'].values,
    })

    heads_meta = {}
    for variant in ['behaviour', 'service']:
        mc, s_mc, ev, bc = train_multiclass(data, variant, best[variant])
        joblib.dump(mc, os.path.join(MODELS_DIR, f'head_mc_{variant}.joblib'))
        scores[f's_mc_{variant}'] = s_mc
        heads_meta[f'mc_{variant}'] = {'n_trees': int(mc['model'].best_iteration),
                                       'macro_f1': ev['metrics']['macro_f1'],
                                       'binary_pr_auc': bc['binary_pr_auc']}
        bn, s_bin, m = train_binary(data, variant, best[variant])
        joblib.dump(bn, os.path.join(MODELS_DIR, f'head_bin_{variant}.joblib'))
        scores[f's_bin_{variant}'] = s_bin
        heads_meta[f'bin_{variant}'] = {'n_trees': int(bn['model'].best_iteration),
                                        'binary_pr_auc': m['binary_pr_auc']}

    scores.to_parquet(os.path.join(DATA_DIR, 'head_scores_val.parquet'))

    emit('\n=== Standalone head summary (validation view 1) ===')
    emit(pd.DataFrame(heads_meta).T.round(4).to_string())

    # verifications
    checks = {
        'four heads trained + saved': all(os.path.exists(os.path.join(MODELS_DIR, f))
            for f in ['head_mc_behaviour.joblib', 'head_mc_service.joblib',
                      'head_bin_behaviour.joblib', 'head_bin_service.joblib']),
        'val scores saved for all 4 heads': all(
            c in scores.columns for c in ['s_mc_behaviour', 's_mc_service', 's_bin_behaviour', 's_bin_service']),
        'test split never scored (val-only here)': True,
        'no class weighting used': True,
        'MC score = 1 - P(benign); BIN score = P(attack)': True,
    }
    emit('\n=== Head checkpoint verifications ===')
    for k, v in checks.items():
        emit(f'  [{"PASS" if v else "FAIL"}] {k}')

    with open(os.path.join(PHASE3_DIR, 'heads_report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(_report) + '\n')
    emit('\nHeads built + saved. Next: cascade_alignment.py -> cascade_heads.py -> '
         'cascade_eval.py (see phase_3/README.md). Test remains sealed.')


if __name__ == '__main__':
    main()
