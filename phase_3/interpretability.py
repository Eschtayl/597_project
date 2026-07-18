"""Interpretability of the frozen cascade head.

  * Global importance: XGBoost gain + permutation importance (binary PR-AUC drop,
    permuting RAW feature columns before the preprocessor, on the leakage-clean
    validation split the head was early-stopped on).
  * Per-class importance: mean |SHAP| per class via native XGBoost pred_contribs
    (TreeSHAP; no external shap dependency).
  * Case studies: SHAP explanations for one representative cascade-test alert of
    each kind — rejected benign FP, retained attack, wrongly rejected attack,
    wrongly retained benign — using the frozen config's decision flows.

Usage: python phase_3/interpretability.py
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.metrics import average_precision_score

from config import RANDOM_SEED, DATA_DIR, PHASE3_DIR, LABEL_COL, BENIGN_LABEL
from flow_sampling import load_sampled_dataset
from features import (build_datasets, clean_frame, add_engineered_behaviour,
                      add_service_fields, numeric_categorical_cols)
from model_eval import p_attack_from_proba
from cascade_alignment import ALIGNMENT_PATH, UNIFIED_FULL_PATH
from cascade_heads import MODELS_DIR, SCORES_PATH, build_clean_sample, candidate_lfid_set
from cascade_eval import alert_scores, cascade_keep, RESULTS_PATH as CASCADE_RESULTS

SEED = RANDOM_SEED
OUT_JSON = os.path.join(DATA_DIR, 'interpretability.json')
N_PERM_REPEATS = 3
N_SHAP_SAMPLE = 5000

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
_report = []
def emit(s=''):
    print(s)
    _report.append(s)


def rebuild_clean_val():
    """Deterministically rebuild the leakage-clean sample the cascade heads were
    trained on (same seed/inputs as cascade_heads.py) and return its datasets."""
    aligned = pd.read_parquet(ALIGNMENT_PATH)
    unified = pd.read_parquet(UNIFIED_FULL_PATH)
    sampled = load_sampled_dataset()
    cand_val = candidate_lfid_set(aligned[aligned['phase2_split'] == 'val'])
    cand_test = candidate_lfid_set(aligned[aligned['phase2_split'] == 'test'])
    del aligned
    # original split reproduced directly (identical to build_datasets' internal
    # call — feature adders don't change row order — but far lighter on memory)
    from splits import grouped_stratified_split
    diag0 = grouped_stratified_split(sampled)[['LogicalFlowID', 'split']]
    new_sampled, split_by_lfid, _ = build_clean_sample(
        sampled, diag0, cand_val, cand_test, unified)
    del sampled, diag0
    return build_datasets(new_sampled, split_by_lfid=split_by_lfid), unified


def head_score(head, X_raw):
    """1 - P(benign) for a raw (already engineered) feature frame."""
    Xt = head['preprocessor'].transform(clean_frame(X_raw))
    proba = head['model'].predict_proba(Xt)
    return p_attack_from_proba(proba, head['classes'])


def permutation_importance(head, X_raw, y_bin, cols, n_repeats=N_PERM_REPEATS):
    rng = np.random.default_rng(SEED)
    base = average_precision_score(y_bin, head_score(head, X_raw))
    emit(f'  baseline binary PR-AUC on clean val: {base:.4f}')
    drops = {}
    for c in cols:
        d = []
        for _ in range(n_repeats):
            Xp = X_raw.copy()
            Xp[c] = Xp[c].sample(frac=1.0, random_state=int(rng.integers(0, 2**31))).values
            d.append(base - average_precision_score(y_bin, head_score(head, Xp)))
        drops[c] = float(np.mean(d))
    return base, drops


def shap_contribs(head, X_raw):
    """TreeSHAP contributions (n, n_class, n_feat+1) + transformed feature names."""
    Xt = head['preprocessor'].transform(clean_frame(X_raw))
    names = list(head['preprocessor'].get_feature_names_out())
    booster = head['model'].get_booster()
    dm = xgb.DMatrix(Xt, feature_names=[f'f{i}' for i in range(Xt.shape[1])])
    contribs = np.asarray(booster.predict(dm, pred_contribs=True))
    if contribs.ndim == 2:   # older xgboost: (n, n_class*(n_feat+1)) flattened
        n_class = len(head['classes'])
        contribs = contribs.reshape(len(contribs), n_class, -1)
    return contribs, names, Xt


def main():
    head = joblib.load(os.path.join(MODELS_DIR, 'head_mc_service_cascade.joblib'))
    classes = head['classes']
    benign_idx = classes.index(BENIGN_LABEL)
    num, cat = numeric_categorical_cols('service')
    cols = num + cat

    emit('=== Interpretability — frozen cascade head (MC service-aware) ===')

    emit('\nRebuilding leakage-clean sample (deterministic)...')
    data, unified = rebuild_clean_val()
    va = data['split'] == 'val'
    X_val = data['X_service'].loc[va, cols].reset_index(drop=True)
    y_val_bin = data['y_binary'][va].to_numpy()
    del data   # free the full-sample matrices; only the val slice is needed

    # ---------------- global gain importance ----------------
    gains = head['model'].feature_importances_
    names = list(head['preprocessor'].get_feature_names_out())
    gain_tbl = (pd.DataFrame({'feature': names, 'gain': gains})
                .sort_values('gain', ascending=False).head(20).reset_index(drop=True))
    emit('\nTop 20 features by gain importance:')
    emit(gain_tbl.round(4).to_string(index=False))

    # ---------------- permutation importance ----------------
    emit(f'\nPermutation importance (raw columns, {N_PERM_REPEATS} repeats, '
         'metric = binary PR-AUC drop):')
    base, drops = permutation_importance(head, X_val, y_val_bin, cols)
    perm_tbl = (pd.DataFrame({'feature': list(drops), 'pr_auc_drop': list(drops.values())})
                .sort_values('pr_auc_drop', ascending=False).head(20).reset_index(drop=True))
    emit(perm_tbl.round(4).to_string(index=False))

    # ---------------- per-class mean |SHAP| ----------------
    emit(f'\nPer-class mean |SHAP| (TreeSHAP, {N_SHAP_SAMPLE:,}-row val sample), top 8 per class:')
    sub = X_val.sample(n=min(N_SHAP_SAMPLE, len(X_val)), random_state=SEED)
    contribs, tnames, _ = shap_contribs(head, sub)
    per_class = {}
    for k, cls in enumerate(classes):
        mean_abs = np.abs(contribs[:, k, :-1]).mean(axis=0)   # drop bias term
        top = (pd.Series(mean_abs, index=tnames).sort_values(ascending=False).head(8))
        per_class[cls] = {f: float(v) for f, v in top.items()}
        emit(f'  {cls}: ' + ', '.join(f'{f} ({v:.3f})' for f, v in top.items()))

    # ---------------- four representative cascade-test cases ----------------
    emit('\n=== Representative cascade-test case studies (SHAP, benign-logit flipped '
         '= push toward attack) ===')
    with open(CASCADE_RESULTS, encoding='utf-8') as f:
        frozen = json.load(f)['frozen_config']
    aligned = pd.read_parquet(ALIGNMENT_PATH)
    test = aligned[aligned['phase2_split'] == 'test'].reset_index(drop=True)
    flow_scores = pd.read_parquet(SCORES_PATH)
    s, usable, dflow = alert_scores(test, flow_scores, frozen['head'], frozen['policy'])
    keep = cascade_keep(s, frozen['tau'])
    y = test['y_true'].to_numpy()

    def pick(mask, order_scores, largest):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return None
        vals = order_scores[idx]
        return int(idx[np.nanargmax(vals) if largest else np.nanargmin(vals)])

    cases = {
        'rejected_benign_FP (correct suppression)': pick((y == 0) & ~keep, s, False),
        'retained_attack (correct keep)': pick((y == 1) & keep & usable, s, True),
        'wrongly_rejected_attack (missed by cascade)': pick((y == 1) & ~keep, s, True),
        'wrongly_retained_benign (residual FP)': pick((y == 0) & keep & usable, s, True),
    }

    unified_idx = unified.set_index('LogicalFlowID')
    case_out = {}
    for name, i in cases.items():
        if i is None:
            emit(f'\n{name}: no example in test alerts')
            continue
        lfid = dflow[i]
        row = unified_idx.loc[[lfid]].reset_index()
        row = add_service_fields(add_engineered_behaviour(row))
        contribs_i, tnames_i, _ = shap_contribs(head, row[cols])
        toward_attack = -contribs_i[0, benign_idx, :-1]     # + pushes away from benign
        order = np.argsort(-np.abs(toward_attack))[:8]
        emit(f"\n{name}")
        emit(f"  packet label={test['label'].iloc[i]}  score={s[i]:.4f} "
             f"(tau={frozen['tau']:.4f})  flow={lfid}")
        feats = []
        for j in order:
            fname = tnames_i[j]
            raw_val = row[fname].iloc[0] if fname in row.columns else None
            feats.append({'feature': fname, 'shap_toward_attack': float(toward_attack[j]),
                          'raw_value': (None if raw_val is None or
                                        (isinstance(raw_val, float) and not np.isfinite(raw_val))
                                        else (float(raw_val) if isinstance(raw_val, (int, float, np.number))
                                              else str(raw_val)))})
            emit(f"    {fname:<32} shap={toward_attack[j]:+.4f}"
                 + (f"  value={raw_val}" if raw_val is not None else ''))
        case_out[name] = {'lfid': lfid, 'score': float(s[i]),
                          'packet_label': str(test['label'].iloc[i]), 'top_features': feats}

    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump({'gain_top20': gain_tbl.to_dict(orient='records'),
                   'permutation_baseline_pr_auc': base,
                   'permutation_top20': perm_tbl.to_dict(orient='records'),
                   'per_class_mean_abs_shap': per_class,
                   'case_studies': case_out}, f, indent=1)
    emit(f'\nSaved -> {OUT_JSON}')

    with open(os.path.join(PHASE3_DIR, 'interpretability_report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(_report) + '\n')


if __name__ == '__main__':
    main()
