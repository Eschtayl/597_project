"""
preprocessing.py
================
Turns the raw combined dataset into a clean numeric feature matrix (X)
and a binary target vector (y), ready for model training.

THREE DISTINCT PROBLEMS THIS FILE SOLVES

1. IDENTIFIER LEAKAGE
   Columns like Flow ID, Src IP, Dst IP, and Timestamp describe WHO was
   involved in a connection, not WHAT the connection behaved like. If a
   model is allowed to see these, it can "solve" the classification task
   by memorizing specific IP addresses seen during training rather than
   learning generalizable behavioral patterns -- this looks great on your
   training/test metrics (since the same IPs may appear in both, if you
   split carelessly) but will fail completely on any new deployment
   where different IPs are involved. We remove these columns entirely.

2. UNCLEAN NUMERIC DATA
   Flow-level features often include RATE columns (e.g. Flow Bytes/s,
   Flow Packets/s) computed as (something / flow_duration). When a flow
   has near-zero duration, this division can produce `inf` or `NaN`
   values. Most ML models cannot handle inf/NaN inputs at all -- they
   will error out or silently produce garbage. We detect and repair these.

3. FEATURE SCALE MISMATCH (for Logistic Regression specifically)
   Logistic Regression computes a weighted sum of all features. If one
   feature ranges 0-5 and another ranges 0-50,000, the model's learned
   weights will be dominated by whichever feature happens to have larger
   raw numbers -- not whichever is actually most predictive. We fix this
   with StandardScaler, which rescales every feature to have mean 0 and
   standard deviation 1, putting them all on equal footing.
   (Decision Trees do NOT need this -- a tree just asks "is this value
   above or below a threshold?", which works the same regardless of
   scale. So we scale for Logistic Regression only, in models.py.)
"""

import numpy as np
import pandas as pd

import config


def split_features_and_target(df):
    """
    Separate the dataset into a target (y), a raw feature frame that
    still needs cleaning (df_features), the original string labels
    (labels_raw), and a group array (groups) used for leakage-safe
    splitting in evaluate.py.

    TARGET ENCODING DEPENDS ON config.TASK_MODE
    - "binary": y is 0/1 -- 0=Benign, 1=Attack (any of the 5 types).
      Matches the Phase 2->3 cascade design (a single Attack/Benign
      alert feeds forward into Phase 3's threshold logic).
    - "multiclass": y is the actual category string (Benign, XSS,
      DDoS-HTTP_Flood, etc.), left as-is for sklearn to handle directly
      (sklearn's classifiers accept string labels natively). This mode
      produces metrics (notably macro-F1) directly comparable to
      teammate Vansh's Phase 3 results, which report macro-F1 across
      attack categories.

    WHY RETURN groups HERE?
    config.GROUP_COL (Src IP) identifies which flows came from the same
    source host. We need this alongside X and y so evaluate.py can do a
    GROUP-AWARE split -- keeping every flow from a given IP entirely in
    train OR entirely in test, never split across both. Without this,
    a model could partly learn "this specific host's behavior" instead
    of a generalizable attack pattern, inflating test performance in a
    way that would not hold up on genuinely new traffic.
    """
    labels_raw = df[config.LABEL_COL].copy()

    if config.TASK_MODE == "binary":
        y = (labels_raw != "Benign").astype(int)
    elif config.TASK_MODE == "multiclass":
        y = labels_raw.copy()
    else:
        raise ValueError(f"Unknown config.TASK_MODE: {config.TASK_MODE!r}")

    groups = df[config.GROUP_COL].copy() if config.GROUP_COL in df.columns else None

    identifier_cols_present = [
        c for c in config.IDENTIFIER_COLS + [config.LABEL_COL]
        if c in df.columns
    ]
    df_features = df.drop(columns=identifier_cols_present)

    return df_features, y, labels_raw, groups


def clean_numeric_features(df_features):
    """
    Keep only numeric columns, then repair inf/-inf and NaN values.

    WHY MEDIAN, NOT MEAN, FOR FILLING MISSING VALUES?
    Network traffic features are frequently skewed by outliers (e.g. one
    unusually long-lived flow can drag a "mean flow duration" column way
    up). The median is robust to outliers in a way the mean is not, so
    filling gaps with the median avoids injecting extreme values into
    rows that were otherwise perfectly fine.

    WHY REPLACE inf FIRST, THEN FILL NaN?
    pandas' .fillna() does not touch inf/-inf values -- they are valid
    floats as far as pandas is concerned, just not usable ones for our
    purposes. We must explicitly convert them to NaN first so the
    fillna step can catch them too.
    """
    df_numeric = df_features.select_dtypes(include=[np.number]).copy()

    n_dropped_cols = df_features.shape[1] - df_numeric.shape[1]
    if n_dropped_cols > 0:
        dropped = set(df_features.columns) - set(df_numeric.columns)
        print(f"  [INFO] Dropped {n_dropped_cols} non-numeric column(s) "
              f"not already excluded as identifiers: {dropped}")

    inf_count = np.isinf(df_numeric.to_numpy(dtype=float, na_value=0)).sum()
    if inf_count > 0:
        print(f"  [INFO] Found {inf_count} inf/-inf values -- converting to NaN")
    df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)

    nan_count = df_numeric.isna().sum().sum()
    if nan_count > 0:
        print(f"  [INFO] Found {nan_count} NaN values -- filling with column medians")
    df_numeric = df_numeric.fillna(df_numeric.median(numeric_only=True))

    # Safety net: if an entire column was NaN (median itself would be NaN),
    # fall back to 0 so no NaNs survive into model training.
    remaining_nans = df_numeric.isna().sum().sum()
    if remaining_nans > 0:
        print(f"  [WARNING] {remaining_nans} NaN values remained after median "
              f"fill (likely an all-NaN column) -- filling remaining with 0")
        df_numeric = df_numeric.fillna(0)

    return df_numeric


def preprocess(df):
    """
    Full preprocessing pipeline: raw combined DataFrame ->
    (X, y, labels_raw, groups).

    This is the single function main.py/evaluate.py calls; it exists so
    the full sequence of steps (split -> clean) is documented and tested
    as one unit, while the individual steps above remain independently
    testable and reusable.
    """
    df_features, y, labels_raw, groups = split_features_and_target(df)
    X = clean_numeric_features(df_features)
    return X, y, labels_raw, groups


def split_features_and_target_taylor(df_final):
    """
    Adapter for Taylor's Phase 1 output (df_final from
    taylor_pipeline.build_preprocessed_dataset). Unlike our original
    split_features_and_target, the numeric features here are ALREADY
    log-transformed and RobustScaled -- Taylor's log_and_scale did that
    work already, so this function only needs to separate target,
    identifiers (kept for the group-aware split), and the already-clean
    numeric feature matrix.

    Returns
    -------
    X : already-preprocessed numeric feature matrix, ready for both
        Logistic Regression and the Decision Tree directly (no further
        scaling needed -- see models.py, which skips its own scaling
        step when Taylor-preprocessed data is used).
    y : target (binary 0/1 or multiclass string, per config.TASK_MODE)
    labels_raw : original label strings, kept for reference
    groups : Src IP column, for the group-aware split
    """
    labels_raw = df_final[config.TAYLOR_LABEL_COL].copy()

    if config.TASK_MODE == "binary":
        y = (labels_raw != config.TAYLOR_BENIGN_VALUE).astype(int)
    elif config.TASK_MODE == "multiclass":
        y = labels_raw.copy()
    else:
        raise ValueError(f"Unknown config.TASK_MODE: {config.TASK_MODE!r}")

    groups = df_final[config.GROUP_COL].copy() if config.GROUP_COL in df_final.columns else None

    # Everything that isn't the label and isn't an identifier column is
    # already-scaled numeric feature data, courtesy of Taylor's pipeline.
    # We detect identifier columns the same way his split_identifiers did
    # (by column name), rather than re-deriving our own IDENTIFIER_COLS
    # list, so this stays consistent with his actual output structure.
    identifier_keywords = [
        'ip', 'port', 'mac', 'timestamp', 'flow_id', 'protocol',
        'server', 'host', 'user_agent', 'oui', 'uri', 'content_type',
    ]
    identifier_cols = []
    for col in df_final.columns:
        if col == config.TAYLOR_LABEL_COL:
            continue
        words = col.lower().replace(' ', '_').replace('-', '_').split('_')
        if any(keyword in words for keyword in identifier_keywords):
            identifier_cols.append(col)

    X = df_final.drop(columns=identifier_cols + [config.TAYLOR_LABEL_COL])

    return X, y, labels_raw, groups


def preprocess_taylor(df_final):
    """
    Entry point used by evaluate.py when running the Taylor-integrated
    pipeline. Named separately from preprocess() (the legacy path) so
    both remain available and it's always clear which one produced a
    given result.
    """
    return split_features_and_target_taylor(df_final)
