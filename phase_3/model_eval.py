"""Shared metric computation for Phase 3 models (multiclass + binary collapse).

All functions take predictions/probabilities that are already aligned to an
explicit `classes` order (the classifier's `classes_`), so probability columns
are never assumed by position.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    f1_score, confusion_matrix, roc_auc_score, average_precision_score,
    precision_score, recall_score,
)
from sklearn.preprocessing import label_binarize

from config import BENIGN_LABEL


def _safe(fn, *a, **k):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return float(fn(*a, **k))
    except Exception:
        return float('nan')


def evaluate_multiclass(y_true, y_pred, y_proba, classes):
    """Return a dict of multiclass metrics + per-class table + confusion matrix.
    `y_proba` columns must be in `classes` order."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    per_class = pd.DataFrame({'precision': p, 'recall': r, 'f1': f, 'support': s}, index=classes)

    metrics = {
        'accuracy': _safe(accuracy_score, y_true, y_pred),
        'balanced_accuracy': _safe(balanced_accuracy_score, y_true, y_pred),
        'macro_precision': _safe(precision_score, y_true, y_pred, average='macro', zero_division=0),
        'macro_recall': _safe(recall_score, y_true, y_pred, average='macro', zero_division=0),
        'macro_f1': _safe(f1_score, y_true, y_pred, average='macro', zero_division=0),
        'weighted_f1': _safe(f1_score, y_true, y_pred, average='weighted', zero_division=0),
    }

    present_all = set(classes) <= set(np.unique(y_true))
    if y_proba is not None and present_all:
        metrics['roc_auc_ovr_macro'] = _safe(
            roc_auc_score, y_true, y_proba, multi_class='ovr', average='macro', labels=classes)
        yb = label_binarize(y_true, classes=classes)
        metrics['pr_auc_ovr_macro'] = _safe(average_precision_score, yb, y_proba, average='macro')
    else:
        metrics['roc_auc_ovr_macro'] = float('nan')
        metrics['pr_auc_ovr_macro'] = float('nan')

    return {'metrics': metrics, 'per_class': per_class,
            'confusion': confusion_matrix(y_true, y_pred, labels=classes)}


def binary_collapse(y_true, y_pred, p_attack):
    """Benign-vs-any-attack view derived from multiclass predictions.
    `p_attack = 1 - P(benign)`; threshold 0.5 is descriptive only here."""
    yt = (np.asarray(y_true) != BENIGN_LABEL).astype(int)
    yp = (np.asarray(y_pred) != BENIGN_LABEL).astype(int)
    return {
        'attack_precision': _safe(precision_score, yt, yp, zero_division=0),
        'attack_recall': _safe(recall_score, yt, yp, zero_division=0),
        'attack_f1': _safe(f1_score, yt, yp, zero_division=0),
        'binary_pr_auc': _safe(average_precision_score, yt, p_attack),
        'binary_roc_auc': _safe(roc_auc_score, yt, p_attack),
    }


def p_attack_from_proba(y_proba, classes):
    """1 - P(benign), using the class order to locate the benign column."""
    classes = list(classes)
    if BENIGN_LABEL not in classes:
        return np.full(len(y_proba), np.nan)
    return 1.0 - np.asarray(y_proba)[:, classes.index(BENIGN_LABEL)]
