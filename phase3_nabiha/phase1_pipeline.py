"""
taylor_pipeline.py
===================
Phase 1 data loading + preprocessing, ported from Taylor's shared repo:
https://github.com/Eschtayl/597_project/blob/main/helpers.py

WHY A DIRECT PORT RATHER THAN A REIMPLEMENTATION?
The whole point of using shared Phase 1 preprocessing is that every
teammate's model trains on IDENTICAL features. Re-implementing "the same
idea" from the flowchart risks subtle mismatches (log(x) vs log1p(x),
StandardScaler vs RobustScaler, exactly which columns count as
"identifiers"). This file ports the actual functions, so our features
are byte-for-byte consistent with Taylor's and, by extension, comparable
to Vansh's Phase 3 results built on the same foundation.

ONE DOCUMENTED DEVIATION: THE BENIGN SAMPLE SIZE
Taylor's original `benign_sampler_200k` hard-requires 200,000 benign
rows and raises a ValueError below that. Our real benign pool (pooled
across 4 files) is ~44,895 rows -- well short of 200,000. This is a
data-availability blocker, not a bug in his code or ours: there simply
isn't 200,000 rows of real benign flow data available to this team yet.

SCOPE EXCEPTION (documented, same pattern as our earlier 40K/200K
adjustment before this integration): `benign_sampler_scope_adjusted`
below replaces the hard 200,000 requirement with a sample size capped
at whatever the real pool actually supports, and prints/logs this
explicitly every run so it is never a silent substitution. State this
exactly this way in your report: "Phase 1's benign sampler specifies
200,000 rows; our team's currently available real benign data supports
at most ~44,895, so this run scope-adjusts to that ceiling."
"""

import glob
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

import config

# ============================================================
# DATA LOADING (ported near-verbatim from Taylor's helpers.py)
# ============================================================

def benign_load_and_label(all_csvs):
    """
    Finds all csvs with "Benign" at the start of the filename, pools
    them, and labels every row 'benign'.
    """
    benign_files_flow = [f for f in all_csvs if os.path.basename(f).startswith("Benign")]
    df_benign_flow = pd.concat(
        (pd.read_csv(f).assign(label='benign') for f in benign_files_flow),
        ignore_index=True
    )
    return df_benign_flow


def attacks_load_and_label(all_csvs):
    """
    Filters, loads, and labels all attack files based on filename
    substrings. Every row gets a label matching its source file's
    attack type.
    """
    attack_files = [f for f in all_csvs if not os.path.basename(f).startswith("Benign")]

    substring_map = {
        'DDoS-HTTP': 'DDOS-HTTP_flood',
        'DoS-HTTP': 'DoS-HTTP',
        'Spoofing': 'DNS_spoofing',
        'XSS': 'XSS',
        'BruteForce': 'brute_force',
    }

    processed_dfs = []
    for file_path in attack_files:
        df = pd.read_csv(file_path, low_memory=False)
        filename = os.path.basename(file_path)

        assigned_label = 'Unknown Attack'
        for substring, label in substring_map.items():
            if substring in filename:
                assigned_label = label
                break

        df['label'] = assigned_label
        processed_dfs.append(df)

    return pd.concat(processed_dfs, ignore_index=True)


def benign_sampler_scope_adjusted(df_benign, seed=0):
    """
    SCOPE-ADJUSTED VERSION of Taylor's benign_sampler_200k.

    Original spec: exactly 200,000 rows, hard-enforced via ValueError.
    This version samples min(config.BENIGN_TARGET, len(df_benign)) rows
    instead, and prints the adjustment explicitly every time it fires,
    so the deviation is never silent.
    """
    available = len(df_benign)
    target = config.BENIGN_TARGET

    if available < 200_000:
        actual_n = min(target, available)
        print(f"  [SCOPE EXCEPTION] Phase 1 spec requires 200,000 benign rows; "
              f"only {available:,} are available in the pooled real data. "
              f"Sampling {actual_n:,} instead (see taylor_pipeline.py docstring).")
    else:
        actual_n = target

    return df_benign.sample(n=actual_n, random_state=seed)


def attack_sampler(df_attacks, label_col='label', random_seed=None):
    """
    Ported verbatim from Taylor's helpers.py -- unmodified, since our
    real attack pools (2,000-4,000 rows per category) comfortably cover
    the 4,000-6,200 total / 5-way split this function draws (800-1,240
    rows per category at most).
    """
    unique_attack_count = df_attacks[label_col].nunique()
    if unique_attack_count != 5:
        raise ValueError(f"Dataset integrity failure: Expected 5 attack types, found {unique_attack_count}.")

    rng = np.random.default_rng(seed=random_seed)
    num_attack_samples = rng.integers(4000, 6200, endpoint=True)
    samples_per_attack = num_attack_samples // 5

    df_attack_sampled = (df_attacks.groupby(label_col)
                          .sample(n=samples_per_attack, random_state=random_seed))

    total_sampled = len(df_attack_sampled)
    assert 4000 <= total_sampled <= 6200, f"Sample size {total_sampled} out of bounds [4000-6200]."
    assert df_attack_sampled[label_col].nunique() == 5, "Output is missing attack classes."
    assert df_attack_sampled[label_col].isna().sum() == 0, "Null values found in label column."

    return df_attack_sampled


def sample_traffic(df_benign, df_attack, seed=0):
    """Samples benign (scope-adjusted) and attack traffic, combines them."""
    df_sampled_benign = benign_sampler_scope_adjusted(df_benign, seed=seed)
    df_sampled_attack = attack_sampler(df_attack, random_seed=seed)
    return pd.concat([df_sampled_benign, df_sampled_attack], ignore_index=True)


def load_csv(path):
    """
    Finds all csvs in `path` (expects Taylor's folder convention: a
    'flow_based' directory containing only real flow-level CSVs -- NOT
    the raw packet-level files, which have a different schema entirely).
    """
    all_csvs = glob.glob(os.path.join(path, "*.csv"))
    df_benign = benign_load_and_label(all_csvs)
    df_attack = attacks_load_and_label(all_csvs)
    return df_benign, df_attack


# ============================================================
# PREPROCESSING (ported near-verbatim from Taylor's helpers.py)
# ============================================================

def shuffle_and_segregate(df_combined, seed=0):
    """Shuffles the combined dataset and splits features from labels."""
    df_combined = df_combined.sample(frac=1, random_state=seed).reset_index(drop=True)
    labels = df_combined['label']
    df_features = df_combined.drop(columns=['label'])
    return df_features, labels


def handle_missing_data(df_features):
    """
    Missing-data feature engineering, ported verbatim:
    - placeholder -1 values mapped to NaN
    - infinities capped at the max observed finite value, with an
      "_is_infinite" indicator column preserving that it happened
    - numeric NaNs filled with median, with an "_is_missing" indicator
    - categorical NaNs filled with the literal string 'unknown'
    """
    df_clean = df_features.copy()
    df_clean = df_clean.replace({"-1": np.nan, -1: np.nan})

    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    categorical_cols = df_clean.select_dtypes(exclude=[np.number]).columns

    inf_mask = df_clean[numeric_cols].isin([np.inf, -np.inf])
    inf_cols = inf_mask.any()[inf_mask.any()].index
    for col in inf_cols:
        df_clean[f"{col}_is_infinite"] = df_clean[col].isin([np.inf, -np.inf]).astype(int)
        max_finite_val = df_clean.loc[df_clean[col] != np.inf, col].max()
        df_clean[col] = df_clean[col].replace(np.inf, max_finite_val)

    missing_numeric_pct = df_clean[numeric_cols].isna().mean()
    numeric_missing_cols = missing_numeric_pct[missing_numeric_pct > 0].index
    for col in numeric_missing_cols:
        df_clean[f"{col}_is_missing"] = df_clean[col].isna().astype(int)

    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
    df_clean[categorical_cols] = df_clean[categorical_cols].fillna('unknown')

    return df_clean


def one_hot_encoder(df):
    """
    One-hot encodes packet-level categorical columns. On flow-level
    data (our case), none of these columns exist -- pd.get_dummies with
    an empty column list is a safe no-op, so this stays a faithful port
    even though it does nothing for us specifically.
    """
    df = df.drop(columns=['http_host'], errors='ignore')
    cols_to_encode = ['http_request_method', 'handshake_version', 'http_content_type']
    df_encoded = pd.get_dummies(
        df, columns=[c for c in cols_to_encode if c in df.columns], dtype=int
    )
    return df_encoded


def split_identifiers(df_clean):
    """
    Separates columns that identify a machine/user (not a measure of
    network behavior) from the numeric feature set, via keyword
    matching on column names. Notably includes 'port' and 'protocol'
    as identifier keywords -- meaning Src/Dst Port and Protocol are
    EXCLUDED from the model here. (This is a real fix relative to our
    earlier pipeline, which had left ports in as numeric features --
    see the README for why that was flagged as a leakage-adjacent risk.)
    """
    identifier_keywords = [
        'ip', 'port', 'mac', 'timestamp', 'flow_id', 'protocol',
        'server', 'host', 'user_agent', 'oui', 'uri', 'content_type',
    ]

    identifier_cols = []
    for col in df_clean.columns:
        words = col.lower().replace(' ', '_').replace('-', '_').split('_')
        if any(keyword in words for keyword in identifier_keywords):
            identifier_cols.append(col)

    df_identifiers = df_clean[identifier_cols].copy()
    df_numeric = df_clean.drop(columns=identifier_cols).copy()
    return df_numeric, df_identifiers


def feature_cleaner(df_features):
    """Runs missing-data handling, one-hot encoding, then identifier splitting."""
    df_clean_features = handle_missing_data(df_features)
    df_clean_hot = one_hot_encoder(df_clean_features)
    df_numeric, df_identifiers = split_identifiers(df_clean_hot)
    return df_numeric, df_identifiers


def log_and_scale(df_numeric, df_identifiers, labels):
    """
    Final transform: coerce any remaining hex/text columns to numeric,
    log1p-transform (compresses heavy-tailed flow features like byte
    counts and inter-arrival times), then RobustScale (median/IQR-based
    -- more resistant to the outliers common in network traffic than
    a mean/std-based StandardScaler).
    """
    object_cols = df_numeric.select_dtypes(include=['object', 'string']).columns
    for col in object_cols:
        df_numeric[col] = df_numeric[col].apply(
            lambda x: int(x, 16) if isinstance(x, str) and str(x).startswith('0x') else x
        )
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')

    if not object_cols.empty:
        df_numeric[object_cols] = df_numeric[object_cols].fillna(df_numeric[object_cols].median())

    df_numeric = df_numeric.dropna(axis=1, how='all')
    df_numeric = df_numeric.select_dtypes(include=[np.number])

    df_log_transformed = np.log1p(df_numeric.abs())

    scaler = RobustScaler()
    scaled_array = scaler.fit_transform(df_log_transformed)
    df_scaled = pd.DataFrame(scaled_array, columns=df_numeric.columns)

    df_final = pd.concat([
        df_identifiers.reset_index(drop=True),
        df_scaled.reset_index(drop=True),
        labels.reset_index(drop=True),
    ], axis=1)

    return df_final, scaler


# ============================================================
# ORCHESTRATION (new -- wires the above into one call for our pipeline)
# ============================================================

def build_preprocessed_dataset(seed=None):
    """
    Runs the full Phase 1 pipeline end to end: load -> sample -> shuffle
    -> clean -> encode -> split identifiers -> log+scale.

    Returns
    -------
    df_final : the fully preprocessed dataset (Taylor's df_preprocessed),
        including identifier columns (kept for traceability / grouping,
        e.g. Src IP for our group-aware split), scaled numeric features,
        and the 'label' column.
    scaler : fitted RobustScaler (kept for reference/reuse if needed).
    """
    active_seed = seed if seed is not None else config.SEED

    df_benign, df_attack = load_csv(config.TAYLOR_FLOW_DIR)
    df_combined_sampled = sample_traffic(df_benign, df_attack, seed=active_seed)

    df_features, labels = shuffle_and_segregate(df_combined_sampled, seed=active_seed)
    df_numeric, df_identifiers = feature_cleaner(df_features)
    df_final, scaler = log_and_scale(df_numeric, df_identifiers, labels)

    print(f"\nPhase 1 (Taylor's pipeline) output shape: {df_final.shape}")
    print(df_final['label'].value_counts())

    return df_final, scaler
