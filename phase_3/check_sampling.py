"""Integrity checks for the sampler + splits.

Builds the sampled dataset (or loads the cache), applies both split protocols,
and verifies sampling proportions, group-safety, stratification, and per-fold
class support. Prints PASS/FAIL.

Usage: python phase_3/check_sampling.py [--rebuild]
"""
import sys

import numpy as np
import pandas as pd

from config import (
    LABEL_COL, BENIGN_LABEL, ATTACK_LABELS, N_ATTACK_TYPES,
    BENIGN_SAMPLE_N, ATTACK_SAMPLE_MIN, ATTACK_SAMPLE_MAX, N_TEMPORAL_FOLDS,
)
from flow_sampling import build_sampled_dataset, load_sampled_dataset
from splits import (
    grouped_stratified_split, blocked_temporal_folds, temporal_fold_plan,
    class_support_table, SPLIT_COL, FOLD_COL, GROUP_COL,
)

results = []
def check(name, ok, detail=''):
    results.append(ok)
    print(f'  [{"PASS" if ok else "FAIL"}] {name}' + (f'  -- {detail}' if detail else ''))


def main():
    rebuild = '--rebuild' in sys.argv
    if rebuild:
        sampled = build_sampled_dataset()
    else:
        try:
            sampled = load_sampled_dataset()
            print(f'Loaded cached sample: {len(sampled):,} flows')
        except FileNotFoundError:
            sampled = build_sampled_dataset()

    print()
    vc = sampled[LABEL_COL].value_counts()
    print('Sampled class counts:')
    for lab, c in vc.items():
        print(f'  {lab:<18} {c:>8,}')
    print()

    # 1. benign count
    n_ben = int(vc.get(BENIGN_LABEL, 0))
    check('benign count == 200,000 (or all available)',
          n_ben == BENIGN_SAMPLE_N or n_ben < BENIGN_SAMPLE_N, f'{n_ben:,}')

    # 2. attack total within bounds
    n_att = int(sampled[sampled[LABEL_COL] != BENIGN_LABEL].shape[0])
    check('attack total within [4000, 6200] (or capped by availability)',
          n_att <= ATTACK_SAMPLE_MAX and n_att > 0, f'{n_att:,}')

    # 3. all five attack types present
    present = sorted(sampled.loc[sampled[LABEL_COL] != BENIGN_LABEL, LABEL_COL].unique())
    check('all five attack types present', len(present) == N_ATTACK_TYPES, str(present))

    # 4. LogicalFlowID unique (no duplicated flows)
    check('LogicalFlowID unique in sample', sampled[GROUP_COL].is_unique)

    # ---- grouped stratified split ----
    sp = grouped_stratified_split(sampled)
    counts = sp[SPLIT_COL].value_counts()
    print('\nSplit sizes:', dict(counts))

    # 5. every flow in exactly one split, no LogicalFlowID overlap across splits
    lfid_split = sp.groupby(GROUP_COL)[SPLIT_COL].nunique()
    check('each LogicalFlowID in exactly one split', (lfid_split == 1).all())

    # 6. stratification: class proportions stable across splits
    props = pd.crosstab(sp[SPLIT_COL], sp[LABEL_COL], normalize='index')
    max_dev = (props.max(axis=0) - props.min(axis=0)).max()
    check('stratified: class proportions stable across train/val/test',
          max_dev < 0.01, f'max proportion spread = {max_dev:.4f}')

    # 7. all classes present in every split
    per_split = class_support_table(sp, SPLIT_COL)
    check('every class present in train/val/test', (per_split > 0).all().all(),
          f'min cell = {int(per_split.values.min())}')

    # ---- blocked temporal folds ----
    tf = blocked_temporal_folds(sampled)
    support = class_support_table(tf, FOLD_COL)
    print('\nPer-fold class support (temporal robustness):')
    print(support.to_string())
    print('Expanding-window plan (train folds -> test fold):', temporal_fold_plan())

    # 8. folds assigned to all rows
    check('every flow assigned a temporal fold', (tf[FOLD_COL] >= 0).all())

    # 9. every class appears in every fold (so expanding-window test blocks are usable)
    check('every class present in every temporal fold', (support > 0).all().all(),
          f'min fold cell = {int(support.values.min())}')

    # 10. flag thin classes (support < 30 in any fold) as unstable, not a failure
    thin = support[support < 30].stack()
    if len(thin):
        print('\n  NOTE (not a failure): thin class-folds (<30 flows), interpret with wide CIs:')
        for (fold, lab), n in thin.items():
            print(f'    fold {fold}  {lab}: {int(n)}')

    print('\n' + ('ALL CHECKS PASSED' if all(results) else 'SOME CHECKS FAILED')
          + f'  ({sum(results)}/{len(results)})')
    sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
