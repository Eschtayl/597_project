import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from config import (
    FLOW_PATH,
    RANDOM_SEED,
    LABEL_COL,
    BENIGN_LABEL,
    ORIG_LABEL_COL,
    FLOW_ID_COL,
    FLOW_IDENTIFIER_COLS,
    TEST_SIZE,
    VAL_SIZE,
)
from helpers import (
    benign_load_and_label,
    attacks_load_and_label,
    benign_sampler_200k,
    attack_sampler,
)


# -------------------------
# Loading and aggregation
# -------------------------

def load_flow_files_raw(path=FLOW_PATH):
    all_csvs = glob.glob(os.path.join(path, "*.csv"))
    df_benign = benign_load_and_label(all_csvs)
    df_attack = attacks_load_and_label(all_csvs)
    df_combined = pd.concat([df_benign, df_attack], ignore_index=True)

    # Filename-derived 'label' is authoritative; drop the shipped 'Label'
    if ORIG_LABEL_COL in df_combined.columns:
        df_combined = df_combined.drop(columns=[ORIG_LABEL_COL])

    df_combined = df_combined.replace([np.inf, -np.inf], np.nan)
    return df_combined


def _agg_func_for(col):
    # Flows over 2 min are split into segments, so aggregate each feature by its meaning.
    if col in FLOW_IDENTIFIER_COLS or col == LABEL_COL:
        return 'first'
    lc = col.lower()
    if 'init win' in lc:
        return 'mean'
    if lc.endswith('max'):
        return 'max'
    if lc.endswith('min'):
        return 'min'
    for keyword in ('mean', 'std', 'var', 'avg', 'ratio', '/s'):
        if keyword in lc:
            return 'mean'
    return 'sum'


def aggregate_by_flow_id(df, flow_id_col=FLOW_ID_COL):
    if flow_id_col not in df.columns:
        raise ValueError(f'Expected "{flow_id_col}" column in flow data, got {df.columns.tolist()[:10]}...')

    agg_rules = {}
    for col in df.columns:
        if col == flow_id_col:
            continue
        agg_rules[col] = _agg_func_for(col)

    df_agg = df.groupby(flow_id_col, as_index=False).agg(agg_rules)
    print(f'Aggregated {len(df):,} flow rows into {len(df_agg):,} unique flows')
    return df_agg


# -------------------------
# Sampling
# -------------------------

def sample_flow_dataset(df_agg, seed=RANDOM_SEED):
    # Task 1.1 proportions
    df_benign = df_agg[df_agg[LABEL_COL] == BENIGN_LABEL].copy()
    df_attack = df_agg[df_agg[LABEL_COL] != BENIGN_LABEL].copy()

    df_benign_sample = benign_sampler_200k(df_benign, seed=seed)
    df_attack_sample = attack_sampler(df_attack, label_col=LABEL_COL, random_seed=seed)

    df_combined = pd.concat([df_benign_sample, df_attack_sample], ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f'Sampled flow dataset: {len(df_combined):,} rows '
          f'(benign {len(df_benign_sample):,}, attack {len(df_attack_sample):,})')
    return df_combined


# -------------------------
# Preprocessing
# -------------------------

def preprocess_flow_features(df, fit_scaler=True, scaler=None):
    # log1p compresses heavy tails; RobustScaler limits outlier influence.
    identifier_cols_present = [c for c in FLOW_IDENTIFIER_COLS if c in df.columns]
    df_identifiers = df[identifier_cols_present].copy()

    df_numeric = df.drop(columns=identifier_cols_present + [LABEL_COL], errors='ignore').copy()

    for col in df_numeric.columns:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')

    df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)
    medians = df_numeric.median(numeric_only=True)
    df_numeric = df_numeric.fillna(medians)
    df_numeric = df_numeric.fillna(0.0)

    df_log = np.log1p(df_numeric.abs())

    if fit_scaler:
        scaler = RobustScaler()
        arr = scaler.fit_transform(df_log)
    else:
        if scaler is None:
            raise ValueError('scaler=None but fit_scaler=False')
        arr = scaler.transform(df_log)

    df_scaled = pd.DataFrame(arr, columns=df_numeric.columns, index=df_numeric.index)

    df_final = pd.concat([
        df_identifiers.reset_index(drop=True),
        df_scaled.reset_index(drop=True),
    ], axis=1)
    if LABEL_COL in df.columns:
        df_final[LABEL_COL] = df[LABEL_COL].reset_index(drop=True)

    return df_final, scaler


# -------------------------
# Top-level entry points
# -------------------------

def prepare_flow_data(path=FLOW_PATH, seed=RANDOM_SEED):
    # Returns the sampled/preprocessed frame, the full aggregated flow set used for
    # matching Phase 2 alerts, and the fitted scaler.
    print('Loading raw flow CSVs')
    df_raw = load_flow_files_raw(path)
    print(f'Raw flow rows: {len(df_raw):,}')

    print('Aggregating flows by Flow ID')
    df_agg = aggregate_by_flow_id(df_raw)

    print('Sampling flow dataset (Task 1.1 proportions)')
    df_sample = sample_flow_dataset(df_agg, seed=seed)

    print('Preprocessing flow features (log1p + RobustScaler)')
    df_pre, scaler = preprocess_flow_features(df_sample, fit_scaler=True)
    print(f'Preprocessed flow shape: {df_pre.shape}')
    return df_pre, df_agg, scaler


def make_train_val_test_splits(df_pre, seed=RANDOM_SEED):
    identifier_cols_present = [c for c in FLOW_IDENTIFIER_COLS if c in df_pre.columns]
    labels = df_pre[LABEL_COL].values
    y_bin = (labels != BENIGN_LABEL).astype(int)

    x = df_pre.drop(columns=identifier_cols_present + [LABEL_COL], errors='ignore')

    x_trainval, x_test, y_trainval, y_test, labels_trainval, labels_test = train_test_split(
        x, y_bin, labels, test_size=TEST_SIZE, random_state=seed, stratify=y_bin
    )
    x_train, x_val, y_train, y_val, labels_train, labels_val = train_test_split(
        x_trainval, y_trainval, labels_trainval,
        test_size=VAL_SIZE, random_state=seed, stratify=y_trainval,
    )

    print(f'Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}')
    return (x_train, x_val, x_test,
            y_train, y_val, y_test,
            labels_train, labels_val, labels_test)
