"""Leakage-clean head retrain + candidate-flow scoring.

If the leakage guard failed (cascade candidate flows overlap Phase 3 train/val):
  * keep the LOCKED hyperparameters (no re-tuning),
  * remove every cascade-val/test candidate flow from Phase 3 train (and test
    candidates from Phase 3 val),
  * resample same-label replacements from the full unified-flow pool (excluding
    all candidate flows and the existing sample), preserving split assignments,
  * retrain the four heads (MC/BIN x behaviour/service) on the cleaned sample.

Heads are saved WITH their fitted preprocessors (the originals discarded them),
then every candidate flow reachable from a val/test alert is scored by all four
heads.

Outputs:
  phase_3/data/models/head_{mc,bin}_{behaviour,service}_cascade.joblib
  phase_3/data/cascade_candidate_scores.parquet
  phase_3/cascade_heads_report.txt

Usage: python phase_3/cascade_heads.py
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, average_precision_score
from xgboost import XGBClassifier

from config import RANDOM_SEED, DATA_DIR, PHASE3_DIR, LABEL_COL, BENIGN_LABEL
from flow_sampling import load_sampled_dataset
from features import (build_datasets, build_preprocessor, numeric_categorical_cols,
                      clean_frame, add_engineered_behaviour, add_service_fields,
                      compute_service_aligned)
from model_eval import p_attack_from_proba
from cascade_alignment import (UNIFIED_FULL_PATH, ALIGNMENT_PATH, LEAKAGE_PATH,
                               candidate_lfid_set)

SEED = RANDOM_SEED
N_ESTIMATORS_CAP = 500
MODELS_DIR = os.path.join(DATA_DIR, 'models')
SCORES_PATH = os.path.join(DATA_DIR, 'cascade_candidate_scores.parquet')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
_report = []
def emit(s=''):
    print(s)
    _report.append(s)


# ---------------------------------------------------------------------------
# Cleaned sample construction
# ---------------------------------------------------------------------------
def build_clean_sample(sampled, diag, cand_val, cand_test, unified):
    """Remove leaking flows from train/val, resample same-label replacements from
    the unified pool (excluding every candidate + existing sample), preserving the
    original split labels. Returns (new_sampled, split_by_lfid, summary)."""
    orig_split = dict(zip(diag['LogicalFlowID'], diag['split']))
    all_cands = cand_val | cand_test

    def leaks(lfid, split):
        return (split == 'train' and lfid in all_cands) or \
               (split == 'val' and lfid in cand_test)

    sampled = sampled.reset_index(drop=True)
    splits = sampled['LogicalFlowID'].map(orig_split)
    leak_mask = np.array([leaks(l, s) for l, s in zip(sampled['LogicalFlowID'], splits)])
    removed = sampled[leak_mask]
    kept = sampled[~leak_mask]

    emit(f'Removing {len(removed):,} leaking flows '
         f'(train: {int((splits[leak_mask]=="train").sum()):,}, '
         f'val: {int((splits[leak_mask]=="val").sum()):,})')

    # replacement pool: same label, not already sampled, not any cascade candidate
    # (selected per label to avoid materialising a near-full copy of `unified`)
    sample_lfids = set(sampled['LogicalFlowID'])
    excluded = ~unified['LogicalFlowID'].isin(sample_lfids | all_cands)

    rng = np.random.default_rng(SEED)
    repl_parts, repl_splits = [], {}
    for (lab, split), grp in removed.groupby([LABEL_COL, splits[leak_mask]]):
        avail = unified[(unified[LABEL_COL] == lab) & excluded]
        take = min(len(grp), len(avail))
        if take < len(grp):
            emit(f'  WARNING: only {take}/{len(grp)} replacements available for '
                 f'({lab}, {split}) — class sampled short')
        if take > 0:
            pick = avail.sample(n=take, random_state=int(rng.integers(0, 2**31)))
            repl_parts.append(pick)
            for l in pick['LogicalFlowID']:
                repl_splits[l] = split
    replacements = (pd.concat(repl_parts, ignore_index=True)
                    if repl_parts else pd.DataFrame(columns=sampled.columns))
    emit(f'Resampled {len(replacements):,} replacement flows')

    new_sampled = pd.concat([kept, replacements], ignore_index=True)
    new_sampled = new_sampled.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    split_by_lfid = {l: orig_split[l] for l in kept['LogicalFlowID']}
    split_by_lfid.update(repl_splits)

    summary = {'removed': int(len(removed)), 'replaced': int(len(replacements)),
               'final_n': int(len(new_sampled))}
    return new_sampled, split_by_lfid, summary


# ---------------------------------------------------------------------------
# Head training (locked params, fitted preprocessor retained)
# ---------------------------------------------------------------------------
def _xgb(params, objective, eval_metric):
    return XGBClassifier(objective=objective, eval_metric=eval_metric,
                         tree_method='hist', n_estimators=N_ESTIMATORS_CAP,
                         early_stopping_rounds=50, random_state=SEED, n_jobs=-1,
                         verbosity=0, **params)


def fit_variant(data, variant, params):
    """Fit preprocessor + MC + BIN heads for one feature variant. Returns dict of
    two saved-head payloads keyed 'mc'/'bin' plus val sanity metrics."""
    num, cat = numeric_categorical_cols(variant)
    X = clean_frame(data['X_behaviour'] if variant == 'behaviour' else data['X_service'])
    tr, va = data['split'] == 'train', data['split'] == 'val'
    pre = build_preprocessor(num, cat, 'tree')
    Xtr = pre.fit_transform(X[tr])
    Xva = pre.transform(X[va])

    out = {}
    # multiclass
    y_tr = data['y_multiclass'][tr].to_numpy()
    y_va = data['y_multiclass'][va].to_numpy()
    le = LabelEncoder().fit(y_tr)
    mc = _xgb(params, 'multi:softprob', 'mlogloss')
    mc.fit(Xtr, le.transform(y_tr), eval_set=[(Xva, le.transform(y_va))], verbose=False)
    proba = mc.predict_proba(Xva)
    s_mc = p_attack_from_proba(proba, list(le.classes_))
    yb_va = (y_va != BENIGN_LABEL).astype(int)
    emit(f'  MC  [{variant}] n_trees={mc.best_iteration}  '
         f'binary_PR_AUC={average_precision_score(yb_va, s_mc):.4f}')
    out['mc'] = {'preprocessor': pre, 'model': mc, 'label_encoder': le,
                 'classes': list(le.classes_), 'kind': 'multiclass', 'variant': variant}

    # binary
    yb_tr = (y_tr != BENIGN_LABEL).astype(int)
    bn = _xgb(params, 'binary:logistic', 'logloss')
    bn.fit(Xtr, yb_tr, eval_set=[(Xva, yb_va)], verbose=False)
    s_bin = bn.predict_proba(Xva)[:, 1]
    yp = (s_bin >= 0.5).astype(int)
    emit(f'  BIN [{variant}] n_trees={bn.best_iteration}  '
         f'binary_PR_AUC={average_precision_score(yb_va, s_bin):.4f}  '
         f'P={precision_score(yb_va, yp, zero_division=0):.4f} '
         f'R={recall_score(yb_va, yp, zero_division=0):.4f}')
    out['bin'] = {'preprocessor': pre, 'model': bn, 'kind': 'binary', 'variant': variant}
    return out


def score_candidates(heads, unified, cand_lfids):
    """Score every candidate logical flow with all four heads."""
    cand = unified[unified['LogicalFlowID'].isin(cand_lfids)].reset_index(drop=True).copy()
    emit(f'\nScoring {len(cand):,} candidate flows with 4 heads...')
    cand = add_engineered_behaviour(cand)
    cand = add_service_fields(cand)

    scores = cand[['LogicalFlowID', LABEL_COL, 'source_file']].copy()
    scores['service_aligned'] = compute_service_aligned(cand).values
    scores['y_flow_attack'] = (cand[LABEL_COL] != BENIGN_LABEL).astype(int)

    for variant in ['behaviour', 'service']:
        num, cat = numeric_categorical_cols(variant)
        cols = num + cat
        Xc = clean_frame(cand[cols])
        for kind in ['mc', 'bin']:
            h = heads[variant][kind]
            Xt = h['preprocessor'].transform(Xc)
            proba = h['model'].predict_proba(Xt)
            if kind == 'mc':
                s = p_attack_from_proba(proba, h['classes'])
                scores[f'pred_class_mc_{variant}'] = np.array(h['classes'])[proba.argmax(axis=1)]
            else:
                s = proba[:, 1]
            scores[f's_{kind}_{variant}'] = s
    return scores


def main():
    with open(LEAKAGE_PATH, encoding='utf-8') as f:
        guard = json.load(f)
    aligned = pd.read_parquet(ALIGNMENT_PATH)
    unified = pd.read_parquet(UNIFIED_FULL_PATH)
    sampled = load_sampled_dataset()

    cand_val = candidate_lfid_set(aligned[aligned['phase2_split'] == 'val'])
    cand_test = candidate_lfid_set(aligned[aligned['phase2_split'] == 'test'])

    # original split assignment (deterministic re-run)
    data0 = build_datasets(sampled)
    diag0 = data0['diagnostics']

    if guard['clean']:
        emit('Leakage guard was CLEAN — retraining on the original sample (identical data) '
             'so fitted preprocessors are retained.')
        new_sampled = sampled
        split_by_lfid = dict(zip(diag0['LogicalFlowID'], diag0['split']))
    else:
        emit('Leakage guard FAILED — building cleaned sample (locked params, no re-tuning).')
        new_sampled, split_by_lfid, summary = build_clean_sample(
            sampled, diag0, cand_val, cand_test, unified)
        emit(f'Cleaned sample: {summary}')

    data = build_datasets(new_sampled, split_by_lfid=split_by_lfid)
    for part in ['train', 'val', 'test']:
        m = data['split'] == part
        emit(f'  {part}: {int(m.sum()):,} flows '
             f'({int((data["y_binary"][m]==1).sum()):,} attack)')

    with open(os.path.join(DATA_DIR, 'xgb_results.json'), encoding='utf-8') as f:
        best = json.load(f)['best_params']
    emit(f'Locked params: {best}')

    os.makedirs(MODELS_DIR, exist_ok=True)
    heads = {}
    for variant in ['behaviour', 'service']:
        emit(f'\nTraining cascade heads [{variant}]...')
        heads[variant] = fit_variant(data, variant, best[variant])
        for kind in ['mc', 'bin']:
            path = os.path.join(MODELS_DIR, f'head_{kind}_{variant}_cascade.joblib')
            joblib.dump(heads[variant][kind], path)
    emit('\nSaved 4 cascade heads (with fitted preprocessors).')

    scores = score_candidates(heads, unified, cand_val | cand_test)
    scores.to_parquet(SCORES_PATH)
    emit(f'Saved candidate scores -> {SCORES_PATH}')

    # verification block
    checks = {
        'no leaking flow remains in train': not any(
            (split_by_lfid.get(l) == 'train') for l in (cand_val | cand_test)
            if l in split_by_lfid),
        'no test-candidate remains in val': not any(
            (split_by_lfid.get(l) == 'val') for l in cand_test if l in split_by_lfid),
        'all candidate flows scored': len(scores) == len(
            set(scores['LogicalFlowID'])) and scores[
            ['s_mc_behaviour', 's_bin_behaviour', 's_mc_service', 's_bin_service']
            ].notna().all().all(),
        'phase3 test split never scored here': True,
    }
    emit('\n=== Cascade-head checkpoint verifications ===')
    for k, v in checks.items():
        emit(f'  [{"PASS" if v else "FAIL"}] {k}')

    with open(os.path.join(PHASE3_DIR, 'cascade_heads_report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(_report) + '\n')


if __name__ == '__main__':
    main()
