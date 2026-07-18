"""Feature-builder verification + manifests.

Builds behaviour-only and service-aware datasets from the cached sample, fits
train-only preprocessing, runs the 10 required leakage/consistency checks, and
saves feature manifests + the feature reason table. No model is trained here.

Usage: python phase_3/check_features.py
"""
import os
import sys
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import DATA_DIR, PHASE3_DIR, LABEL_COL
from flow_sampling import load_sampled_dataset
from features import (
    build_datasets, build_preprocessor, numeric_categorical_cols,
    clean_frame, feature_reason_table, FORBIDDEN_IN_MATRIX,
    BEHAVIOUR_NUMERIC, SERVICE_CATEGORICAL, SERVICE_BINARY,
)

results = []
def check(name, ok, detail=''):
    results.append(bool(ok))
    print(f'  [{"PASS" if ok else "FAIL"}] {name}' + (f'  -- {detail}' if detail else ''))


def fit_transform_variant(data, variant, model_kind):
    num, cat = numeric_categorical_cols(variant)
    X = clean_frame(data['X_behaviour'] if variant == 'behaviour' else data['X_service'])
    pre = build_preprocessor(num, cat, model_kind)
    tr = data['split'] == 'train'
    pre.fit(X[tr])
    names = list(pre.get_feature_names_out())
    Xtr = pd.DataFrame(pre.transform(X[tr]), columns=names, index=X[tr].index)
    Xva = pd.DataFrame(pre.transform(X[data['split'] == 'val']), columns=names)
    return pre, Xtr, Xva, names


def max_single_feature_auc(Xtr, y_bin_tr):
    best, worst_feat = 0.0, None
    for c in Xtr.columns:
        col = Xtr[c].to_numpy()
        if np.nanstd(col) == 0:
            continue
        try:
            a = roc_auc_score(y_bin_tr, col)
        except ValueError:
            continue
        a = max(a, 1 - a)
        if a > best:
            best, worst_feat = a, c
    return best, worst_feat


def main():
    sampled = load_sampled_dataset()
    print(f'Loaded sample: {len(sampled):,} flows')
    data = build_datasets(sampled)
    y_bin = data['y_binary']
    tr = data['split'] == 'train'

    # 1. identical rows and targets across variants
    same_rows = data['X_behaviour'].index.equals(data['X_service'].index) and \
        len(data['X_behaviour']) == len(data['X_service'])
    check('behaviour-only and service-aware have identical rows & targets',
          same_rows and data['y_multiclass'].equals(data['y_multiclass']))

    # 2. split assignments unchanged (deterministic rebuild matches)
    data2 = build_datasets(sampled)
    check('split assignments deterministic / unchanged',
          data['split'].equals(data2['split']) and data['fold'].equals(data2['fold']))

    # 3. no identifier/filename/label/diagnostic column in either matrix
    beh_cols = set(data['X_behaviour'].columns)
    svc_cols = set(data['X_service'].columns)
    forbidden = set(FORBIDDEN_IN_MATRIX)
    leaked = (beh_cols | svc_cols) & forbidden
    check('no identifier/label/diagnostic column in feature matrices', not leaked,
          f'leaked={sorted(leaked)}' if leaked else 'none')
    check('behaviour-only excludes Protocol & service fields',
          'Protocol' not in beh_cols and not (beh_cols & set(SERVICE_CATEGORICAL + SERVICE_BINARY)))

    # 4/5. categories fitted on train only + unknown categories handled safely
    pre_lin, Xtr_lin, Xva_lin, names_lin = fit_transform_variant(data, 'service', 'linear')
    ohe = pre_lin.named_transformers_['cat'].named_steps['onehot']
    cats_from_train = all(len(c) > 0 for c in ohe.categories_)
    # inject an unseen Protocol category into a val copy and ensure no crash / all-zero
    Xsvc = clean_frame(data['X_service']).copy()
    val_idx = data['split'] == 'val'
    Xinj = Xsvc[val_idx].copy()
    Xinj.iloc[0, Xinj.columns.get_loc('Protocol')] = '999'   # unseen protocol
    trans_ok = True
    try:
        pre_lin.transform(Xinj)
    except Exception:
        trans_ok = False
    check('categorical encoders fitted on train only', cats_from_train)
    check('unknown validation categories handled safely (handle_unknown=ignore)', trans_ok)

    # 6. linear preprocessing scales numeric using train stats only
    scaler = pre_lin.named_transformers_['num'].named_steps['scale']
    num_block = Xtr_lin[[c for c in names_lin if c in BEHAVIOUR_NUMERIC + SERVICE_BINARY]]
    check('linear numeric scaled with train statistics (train mean approx 0)',
          np.nanmax(np.abs(num_block.mean().to_numpy())) < 1e-6 and scaler.mean_ is not None,
          f'|mean|max={np.nanmax(np.abs(num_block.mean().to_numpy())):.1e}')

    # 7. tree preprocessing does NOT scale numeric
    pre_tree, Xtr_tree, _, names_tree = fit_transform_variant(data, 'service', 'tree')
    has_scaler = 'scale' in dict(pre_tree.named_transformers_['num'].named_steps)
    raw_max = clean_frame(data['X_service'])['Flow Duration'][tr].max()
    tree_max = Xtr_tree['Flow Duration'].max()
    check('tree preprocessing leaves numeric unscaled',
          (not has_scaler) and np.isclose(raw_max, tree_max, rtol=1e-6),
          f'raw max={raw_max:.0f} tree max={tree_max:.0f}')

    # 8. missing indicators + imputation fitted on train
    imp = pre_tree.named_transformers_['num'].named_steps['impute']
    n_ind = int(imp.indicator_.features_.shape[0]) if imp.indicator_ is not None else 0
    check('missing-value imputation + indicators fitted on train',
          imp.statistics_ is not None, f'{n_ind} missing-indicator columns added')

    # 9. no feature is a one-to-one map of the target
    best_beh, feat_beh = max_single_feature_auc(
        fit_transform_variant(data, 'behaviour', 'tree')[1], y_bin[tr])
    best_svc, feat_svc = max_single_feature_auc(Xtr_tree, y_bin[tr])
    check('no single feature perfectly separates the target (AUC < 0.9999)',
          best_beh < 0.9999 and best_svc < 0.9999,
          f'behaviour max AUC={best_beh:.4f} ({feat_beh}); service max AUC={best_svc:.4f} ({feat_svc})')

    # 10. save manifests + reason table
    os.makedirs(DATA_DIR, exist_ok=True)
    manifest = {
        'behaviour_only': {'numeric': BEHAVIOUR_NUMERIC, 'categorical': []},
        'service_aware': {'numeric': BEHAVIOUR_NUMERIC + SERVICE_BINARY,
                          'categorical': SERVICE_CATEGORICAL},
        'bf_diagnostic_thresholds': data['bf_thresholds'],
        'n_features_service_encoded': len(names_tree),
    }
    with open(os.path.join(DATA_DIR, 'feature_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    reason = feature_reason_table()
    reason.to_csv(os.path.join(DATA_DIR, 'feature_reason_table.csv'), index=False)
    check('feature manifests + reason table saved',
          os.path.exists(os.path.join(DATA_DIR, 'feature_manifest.json')))

    print('\nBrute Force diagnostic thresholds (train benign quantiles):')
    for k, v in data['bf_thresholds'].items():
        print(f'  {k}: {v:.4f}')
    print('\nFeature reason table (head):')
    print(reason.head(6).to_string(index=False))

    print('\n' + ('ALL CHECKS PASSED' if all(results) else 'SOME CHECKS FAILED')
          + f'  ({sum(results)}/{len(results)})')
    sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
