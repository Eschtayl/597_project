import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)

from config import RESULTS_FILE, SAVED_FIGS_DIR, BENIGN_LABEL


def ensure_figs_dir(path=SAVED_FIGS_DIR):
    os.makedirs(path, exist_ok=True)
    return path


def compute_detection_metrics(y_true, y_pred, anomaly_scores=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'fpr': fp / (fp + tn) if (fp + tn) else 0.0,
        'fnr': fn / (fn + tp) if (fn + tp) else 0.0,
        'auc_roc': None,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
    }
    if anomaly_scores is not None:
        metrics['auc_roc'] = roc_auc_score(y_true, anomaly_scores)
    return metrics


def per_attack_detection_rate(labels, y_true, y_pred):
    # Detection rate = recall within each attack type
    labels = np.asarray(labels)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rates = {}
    attack_types = sorted({lab for lab in labels if lab != BENIGN_LABEL})
    for attack in attack_types:
        mask = labels == attack
        n = int(mask.sum())
        detected = int(((y_pred == 1) & mask).sum())
        rates[attack] = detected / n if n else 0.0
    return rates


def write_metrics_to_file(metrics, attack_rates, threshold, path=RESULTS_FILE, model_name='Isolation Forest'):
    with open(path, 'a') as f:
        f.write(f'\n### PHASE 2 — {model_name} ###\n\n')
        f.write(f'Threshold: {threshold}\n')
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1: {metrics['f1']:.4f}\n")
        f.write(f"FPR: {metrics['fpr']:.4f}\n")
        f.write(f"FNR: {metrics['fnr']:.4f}\n")
        if metrics['auc_roc'] is not None:
            f.write(f"AUC-ROC: {metrics['auc_roc']:.4f}\n")
        f.write(
            f"Confusion matrix [tn, fp, fn, tp]: "
            f"[{metrics['tn']}, {metrics['fp']}, {metrics['fn']}, {metrics['tp']}]\n"
        )
        f.write('\nPer-attack detection rate:\n')
        for attack, rate in attack_rates.items():
            f.write(f'  {attack}: {rate:.4f}\n')
        f.write('\n')
    print(f'Appended results to {path}')


def print_metrics(metrics, attack_rates):
    print('--- Detection metrics (test) ---')
    print(f"precision: {metrics['precision']:.4f}")
    print(f"recall:    {metrics['recall']:.4f}")
    print(f"f1:        {metrics['f1']:.4f}")
    print(f"fpr:       {metrics['fpr']:.4f}")
    print(f"fnr:       {metrics['fnr']:.4f}")
    if metrics['auc_roc'] is not None:
        print(f"auc_roc:   {metrics['auc_roc']:.4f}")
    print(
        f"cm tn/fp/fn/tp: "
        f"{metrics['tn']}/{metrics['fp']}/{metrics['fn']}/{metrics['tp']}"
    )
    print('--- Per-attack detection rate ---')
    for attack, rate in attack_rates.items():
        print(f'  {attack}: {rate:.4f}')
    print()


def visualise_confusion_matrix(y_true, y_pred, model_name='Isolation Forest',
                               save_name='if_confusion_matrix.png'):
    ensure_figs_dir()
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=['benign', 'attack'], ax=ax, colorbar=False
    )
    ax.set_title(f'{model_name} — confusion matrix (test)')
    fig.savefig(os.path.join(SAVED_FIGS_DIR, save_name), bbox_inches='tight')
    plt.close(fig)


def visualise_roc_curve(y_true, anomaly_scores, model_name='Isolation Forest',
                        save_name='if_roc_curve.png'):
    ensure_figs_dir()
    fpr, tpr, _ = roc_curve(y_true, anomaly_scores)
    auc = roc_auc_score(y_true, anomaly_scores)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    ax.plot([0, 1], [0, 1], linestyle='--', color='grey')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title(f'{model_name} — ROC (test)')
    ax.legend(loc='lower right')
    fig.savefig(os.path.join(SAVED_FIGS_DIR, save_name), bbox_inches='tight')
    plt.close(fig)


def visualise_score_distribution(anomaly_scores, y_true, model_name='Isolation Forest',
                                 save_name='if_score_distribution.png'):
    ensure_figs_dir()
    y_true = np.asarray(y_true)
    scores = np.asarray(anomaly_scores)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(scores[y_true == 0], bins=80, alpha=0.5, density=True, label='benign')
    ax.hist(scores[y_true == 1], bins=80, alpha=0.5, density=True, label='attack')
    ax.set_xlabel('Anomaly score')
    ax.set_ylabel('Density')
    ax.set_title(f'{model_name} — score distribution (test)')
    ax.legend()
    fig.savefig(os.path.join(SAVED_FIGS_DIR, save_name), bbox_inches='tight')
    plt.close(fig)
