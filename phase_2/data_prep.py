import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from config import (
    PACKET_PATH,
    LABEL_COL,
    BENIGN_LABEL,
    IDENTIFIER_KEYWORDS,
    RANDOM_SEED,
    TEST_SIZE,
)
from helpers import (
    load_csv,
    shuffle_and_segregate,
    feature_cleaner,
    log_and_scale,
)


def prepare_packet_data(path=PACKET_PATH, seed=RANDOM_SEED):
    # Load, sample, preprocess packet-level data (Phase 1 pipeline)
    df_combined_sampled = load_csv(path)
    df_features, labels = shuffle_and_segregate(df_combined_sampled, seed=seed)
    df_numeric, df_identifiers = feature_cleaner(df_features)
    df_preprocessed, fitted_scaler = log_and_scale(df_numeric, df_identifiers, labels)
    return df_preprocessed, fitted_scaler


def get_identifier_columns(df):
    # Find columns that identify a machine or user
    identifier_cols = []
    for col in df.columns:
        words = col.lower().replace(' ', '_').replace('-', '_').split('_')
        if any(keyword in words for keyword in IDENTIFIER_KEYWORDS):
            identifier_cols.append(col)
    return identifier_cols


def split_features_and_labels(df_preprocessed, label_col=LABEL_COL):
    # Keep identifiers aside; train only on scaled numeric behaviour features
    labels = df_preprocessed[label_col].copy()
    identifier_cols = get_identifier_columns(df_preprocessed)
    drop_cols = [label_col] + [c for c in identifier_cols if c in df_preprocessed.columns]
    x_features = df_preprocessed.drop(columns=drop_cols)
    # Binary ground truth for evaluation only (not used in unsupervised training)
    y_true = (labels != BENIGN_LABEL).astype(int)
    return x_features, labels, y_true, identifier_cols


def split_train_test(x_features, labels, y_true, test_size=TEST_SIZE, seed=RANDOM_SEED):
    # Stratified split so attack rate is similar in train and test
    x_train, x_test, labels_train, labels_test, y_train, y_test = train_test_split(
        x_features,
        labels,
        y_true,
        test_size=test_size,
        random_state=seed,
        stratify=y_true,
    )
    return x_train, x_test, labels_train, labels_test, y_train, y_test


def choose_threshold(anomaly_scores, y_true, n_candidates=200):
    # Pick threshold on train scores by max F1 (labels used only for threshold selection)
    y_true = np.asarray(y_true)
    scores = np.asarray(anomaly_scores)
    lo, hi = np.percentile(scores, 1), np.percentile(scores, 99)
    candidates = np.linspace(lo, hi, n_candidates)

    best_threshold = candidates[0]
    best_f1 = -1.0
    for threshold in candidates:
        y_pred = (scores >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f'Threshold chosen by max F1 on train: {best_threshold:.6f} (F1={best_f1:.4f})')
    return float(best_threshold)


def generate_alerts(anomaly_scores, threshold):
    # Flag rows at or above threshold as anomalous (1)
    return (np.asarray(anomaly_scores) >= threshold).astype(int)
