import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, roc_auc_score, roc_curve,
)

from config import RESULTS_FILE, SAVED_FIGS_DIR, BENIGN_LABEL


def _ensure_figs_dir():
    if not os.path.exists(SAVED_FIGS_DIR):
        os.makedirs(SAVED_FIGS_DIR)


# -------------------------
# Metrics
# -------------------------

def compute_supervised_metrics(y_true, y_pred, scores=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {}
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['accuracy'] = accuracy_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics['tn'] = int(tn)
    metrics['fp'] = int(fp)
    metrics['fn'] = int(fn)
    metrics['tp'] = int(tp)
    metrics['fpr'] = fp / max(fp + tn, 1)
    metrics['fnr'] = fn / max(fn + tp, 1)

    if scores is not None and len(np.unique(y_true)) > 1:
        metrics['auc_roc'] = roc_auc_score(y_true, scores)
    else:
        metrics['auc_roc'] = float('nan')
    return metrics


def per_attack_detection_rate(labels, y_pred):
    labels = np.asarray(labels)
    y_pred = np.asarray(y_pred)
    result = {}
    unique_attacks = [lb for lb in np.unique(labels) if lb != BENIGN_LABEL]
    for attack in unique_attacks:
        mask = labels == attack
        if mask.sum() == 0:
            continue
        result[str(attack)] = float(y_pred[mask].mean())
    return result


# -------------------------
# Reporting
# -------------------------

def print_metrics(name, metrics, attack_rates=None):
    print(f'--- {name} ---')
    print(f'  accuracy:  {metrics["accuracy"]:.4f}')
    print(f'  precision: {metrics["precision"]:.4f}')
    print(f'  recall:    {metrics["recall"]:.4f}')
    print(f'  f1:        {metrics["f1"]:.4f}')
    print(f'  fpr:       {metrics["fpr"]:.4f}')
    print(f'  fnr:       {metrics["fnr"]:.4f}')
    print(f'  auc_roc:   {metrics["auc_roc"]:.4f}')
    print(f'  cm tn/fp/fn/tp: {metrics["tn"]}/{metrics["fp"]}/{metrics["fn"]}/{metrics["tp"]}')
    if attack_rates:
        print(f'  Per-attack detection rate:')
        for k, v in attack_rates.items():
            print(f'    {k}: {v:.4f}')


def write_metrics_to_file(name, metrics, attack_rates=None, extras=None):
    lines = []
    lines.append(f'=== {name} ===')
    lines.append(f'accuracy:  {metrics["accuracy"]:.4f}')
    lines.append(f'precision: {metrics["precision"]:.4f}')
    lines.append(f'recall:    {metrics["recall"]:.4f}')
    lines.append(f'f1:        {metrics["f1"]:.4f}')
    lines.append(f'fpr:       {metrics["fpr"]:.4f}')
    lines.append(f'fnr:       {metrics["fnr"]:.4f}')
    lines.append(f'auc_roc:   {metrics["auc_roc"]:.4f}')
    lines.append(f'cm tn/fp/fn/tp: {metrics["tn"]}/{metrics["fp"]}/{metrics["fn"]}/{metrics["tp"]}')
    if attack_rates:
        lines.append('Per-attack detection rate:')
        for k, v in attack_rates.items():
            lines.append(f'  {k}: {v:.4f}')
    if extras:
        for k, v in extras.items():
            lines.append(f'{k}: {v}')
    lines.append('')
    with open(RESULTS_FILE, 'a') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'Appended results to {RESULTS_FILE}')


# -------------------------
# Two-stage refinement
# -------------------------

def compute_two_stage_predictions(y_phase2_pred_all, y_phase3_pred_on_matched, matched_mask):
    # A matched alert survives only if Phase 3 agrees; unmatched alerts keep Phase 2's verdict.
    y_phase2_pred_all = np.asarray(y_phase2_pred_all)
    matched_mask = np.asarray(matched_mask).astype(bool)
    y_phase3_pred_on_matched = np.asarray(y_phase3_pred_on_matched)

    y_two_stage = np.zeros_like(y_phase2_pred_all)
    alert_indices = np.where(y_phase2_pred_all == 1)[0]

    matched_j = np.cumsum(matched_mask.astype(int)) - 1

    kept = 0
    dropped = 0
    unmatched_kept = 0
    for i in range(len(alert_indices)):
        packet_idx = alert_indices[i]
        if matched_mask[i]:
            j = int(matched_j[i])
            if y_phase3_pred_on_matched[j] == 1:
                y_two_stage[packet_idx] = 1
                kept += 1
            else:
                dropped += 1
        else:
            y_two_stage[packet_idx] = 1
            unmatched_kept += 1

    print(f'Two-stage refinement: {kept} kept, {dropped} dropped, '
          f'{unmatched_kept} kept-unmatched')
    return y_two_stage, {'kept': kept, 'dropped': dropped, 'unmatched_kept': unmatched_kept}


def compute_fp_reduction(phase2_metrics, two_stage_metrics):
    fp2 = phase2_metrics['fp']
    fp3 = two_stage_metrics['fp']
    if fp2 == 0:
        pct = 0.0
    else:
        pct = 100.0 * (fp2 - fp3) / fp2
    return {'phase2_fp': fp2, 'two_stage_fp': fp3, 'fp_reduction_pct': pct}


# -------------------------
# Plots
# -------------------------

def plot_confusion_matrix(y_true, y_pred, name, save_name):
    _ensure_figs_dir()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap='Blues')
    for (i, j), v in np.ndenumerate(cm):
        colour = 'white' if v > cm.max() / 2 else 'black'
        ax.text(j, i, str(v), ha='center', va='center', color=colour)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['benign', 'attack'])
    ax.set_yticklabels(['benign', 'attack'])
    ax.set_xlabel('predicted')
    ax.set_ylabel('actual')
    ax.set_title(f'Confusion matrix: {name}')
    fig.tight_layout()
    fig.savefig(os.path.join(SAVED_FIGS_DIR, save_name))
    plt.close(fig)


def plot_roc_curve(y_true, scores, name, save_name):
    _ensure_figs_dir()
    if len(np.unique(y_true)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC curve: {name}')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(SAVED_FIGS_DIR, save_name))
    plt.close(fig)


def plot_feature_importance(model, feature_names, name, save_name, top_k=20):
    _ensure_figs_dir()
    importances = getattr(model, 'feature_importances_', None)
    if importances is None:
        return
    importances = np.asarray(importances)
    top_k = min(top_k, len(importances))
    order = np.argsort(importances)[::-1][:top_k]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.3 * top_k)))
    ax.barh(range(len(order)), importances[order][::-1])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feature_names[i] for i in order][::-1])
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_k} features: {name}')
    fig.tight_layout()
    fig.savefig(os.path.join(SAVED_FIGS_DIR, save_name))
    plt.close(fig)
