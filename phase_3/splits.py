"""Dataset partitioning.

Two complementary protocols over the sampled unified flows:

  * grouped stratified split -> PRIMARY per-class metrics. The group unit is the
    LogicalFlowID; because aggregation yields exactly one row per LogicalFlowID,
    a stratified row split is already group-safe (no logical flow spans two
    partitions). Reused raw Flow IDs became distinct LogicalFlowIDs (distinct
    conversations) and may legitimately fall in different partitions.

  * blocked temporal folds -> ROBUSTNESS. Within each class, order flows by
    start time and use expanding-window folds (earlier blocks train, next block
    tests). Pool out-of-time predictions across folds; class support per fold is
    reported because thin classes (XSS) can be unstable.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    RANDOM_SEED, LABEL_COL, TEST_SIZE, VAL_SIZE, N_TEMPORAL_FOLDS,
)

SPLIT_COL = 'split'
FOLD_COL = 'temporal_fold'
GROUP_COL = 'LogicalFlowID'


def grouped_stratified_split(sampled, seed=RANDOM_SEED, test_size=TEST_SIZE, val_size=VAL_SIZE):
    """Add a `split` column with values train/val/test, stratified by label.
    Group unit is LogicalFlowID (one row per flow => group-safe by construction)."""
    df = sampled.reset_index(drop=True).copy()
    y = df[LABEL_COL]

    idx = np.arange(len(df))
    train_val_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=y,
    )
    # val_size is a fraction of the whole -> rescale relative to the train+val part
    rel_val = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=rel_val, random_state=seed, stratify=y.iloc[train_val_idx],
    )

    df[SPLIT_COL] = 'train'
    df.loc[val_idx, SPLIT_COL] = 'val'
    df.loc[test_idx, SPLIT_COL] = 'test'
    return df


def blocked_temporal_folds(sampled, n_folds=N_TEMPORAL_FOLDS):
    """Add a `temporal_fold` column (0..n_folds-1) assigned within each class by
    start-time order, so expanding-window folds keep every class present. Fold k
    is used as the out-of-time test block for the model trained on folds < k."""
    df = sampled.reset_index(drop=True).copy()
    df[FOLD_COL] = -1
    for lab, grp in df.groupby(LABEL_COL):
        order = grp.sort_values('ts_first').index
        # contiguous time blocks per class
        fold_ids = np.floor(np.linspace(0, n_folds, len(order), endpoint=False)).astype(int)
        fold_ids = np.clip(fold_ids, 0, n_folds - 1)
        df.loc[order, FOLD_COL] = fold_ids
    return df


def temporal_fold_plan(n_folds=N_TEMPORAL_FOLDS):
    """Expanding-window (train folds -> test fold) plan; fold 0 is train-only."""
    return [(list(range(k)), k) for k in range(1, n_folds)]


def class_support_table(df, group_col):
    """Counts of each label within each split/fold value."""
    return pd.crosstab(df[group_col], df[LABEL_COL])
