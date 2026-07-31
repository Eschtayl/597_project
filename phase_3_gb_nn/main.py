import argparse
import sys
import time
from contextlib import contextmanager

import numpy as np

import config
from config import RANDOM_SEED, PHASE2_DIR

from flow_prep import (
    prepare_flow_data,
    make_train_val_test_splits,
)
from flow_matching import match_packets_to_flows

from models.gradient_boosting import train_gradient_boosting, predict_gradient_boosting
from models.neural_net import train_neural_net, predict_neural_net

from evaluation import (
    compute_supervised_metrics,
    per_attack_detection_rate,
    print_metrics,
    write_metrics_to_file,
    compute_two_stage_predictions,
    compute_fp_reduction,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
)

# phase_2 and phase_3 both define config/models/data_prep/evaluation, so the phase_2
# imports are isolated to avoid one shadowing the other.
_PHASE2_COLLIDING_NAMES = [
    'config', 'helpers', 'data_prep', 'eda', 'evaluation',
    'models', 'models.autoencoder', 'models.isolation_forest', 'models.ensemble',
]


@contextmanager
def _phase2_import_context():
    saved = {}
    for name in _PHASE2_COLLIDING_NAMES:
        if name in sys.modules:
            saved[name] = sys.modules.pop(name)
    sys.path.insert(0, PHASE2_DIR)
    try:
        yield
    finally:
        sys.path.remove(PHASE2_DIR)
        for name in list(sys.modules.keys()):
            if name in _PHASE2_COLLIDING_NAMES:
                del sys.modules[name]
        sys.modules.update(saved)


def run_phase2_get_alerts():
    # Rerun the Phase 2 autoencoder to obtain the alerts Phase 3 refines.
    print('\n=== Rerunning Phase 2 (tuned AE) for two-stage evaluation ===')
    with _phase2_import_context():
        from data_prep import (
            prepare_packet_data,
            split_features_and_labels,
            split_train_test,
            choose_threshold,
            generate_alerts,
        )
        from models.autoencoder import tune_autoencoder, get_anomaly_scores as get_ae_scores

        print('Preparing packet data')
        df_preprocessed, _ = prepare_packet_data()
        x_features, labels, y_true, identifier_cols = split_features_and_labels(df_preprocessed)
        print(f'Feature matrix: {x_features.shape}, identifiers held out: {len(identifier_cols)}')

        x_train, x_test, labels_train, labels_test, y_train, y_test = split_train_test(
            x_features, labels, y_true
        )
        print(f'Train: {x_train.shape}, Test: {x_test.shape}')

        print('Tuning Phase 2 autoencoder (best architecture)')
        t_train_start = time.time()
        ae_model, best_dims, _results = tune_autoencoder(x_train, y_train=y_train)
        ae_train_time = time.time() - t_train_start

        t_infer_start = time.time()
        train_scores = get_ae_scores(ae_model, x_train)
        test_scores = get_ae_scores(ae_model, x_test)
        ae_infer_time = time.time() - t_infer_start

        threshold = choose_threshold(train_scores, y_train)
        y_pred_test = generate_alerts(test_scores, threshold)

    y_test_arr = y_test.values if hasattr(y_test, 'values') else np.asarray(y_test)
    labels_test_arr = labels_test.values if hasattr(labels_test, 'values') else np.asarray(labels_test)

    phase2_metrics = compute_supervised_metrics(y_test_arr, y_pred_test, scores=test_scores)
    phase2_attack_rates = per_attack_detection_rate(labels_test_arr, y_pred_test)
    print_metrics('Phase 2 (tuned AE) - reference', phase2_metrics, phase2_attack_rates)
    write_metrics_to_file(
        'Phase 2 (tuned AE) - reference',
        phase2_metrics, phase2_attack_rates,
        extras={
            'ae_dims': str(best_dims),
            'threshold': f'{threshold:.6f}',
            'ae_train_time_seconds': f'{ae_train_time:.2f}',
            'ae_test_inference_seconds': f'{ae_infer_time:.2f}',
        },
    )

    return {
        'df_preprocessed': df_preprocessed,
        'x_test': x_test,
        'labels_test': labels_test_arr,
        'y_test': y_test_arr,
        'y_pred_test': y_pred_test,
        'test_scores': test_scores,
        'phase2_metrics': phase2_metrics,
        'ae_train_time': ae_train_time,
        'ae_infer_time': ae_infer_time,
    }


def run_supervised_experiment(model_name, train_fn, predict_fn,
                              x_train, y_train, x_val, y_val, x_test, y_test, labels_test):
    print(f'\n=== Training {model_name} on flow data ===')
    model, train_time = train_fn(x_train, y_train, x_val=x_val, y_val=y_val)

    y_pred_test, scores_test, infer_time = predict_fn(model, x_test)
    metrics = compute_supervised_metrics(y_test, y_pred_test, scores=scores_test)
    attack_rates = per_attack_detection_rate(labels_test, y_pred_test)
    print_metrics(f'{model_name} on flow test set', metrics, attack_rates)
    write_metrics_to_file(
        f'{model_name} on flow test set',
        metrics, attack_rates,
        extras={
            'train_time_seconds': f'{train_time:.2f}',
            'test_inference_seconds': f'{infer_time:.4f}',
            'test_rows': len(y_test),
        },
    )

    slug = model_name.lower().replace(' ', '_')
    plot_confusion_matrix(y_test, y_pred_test, model_name, f'{slug}_cm.png')
    plot_roc_curve(y_test, scores_test, model_name, f'{slug}_roc.png')
    plot_feature_importance(model, list(x_train.columns), model_name, f'{slug}_fi.png')

    return model, train_time, infer_time


MODEL_REGISTRY = {
    'gb': ('Gradient Boosting', train_gradient_boosting, predict_gradient_boosting),
    'nn': ('Neural Net', train_neural_net, predict_neural_net),
}


def run_two_stage(model, predict_fn, model_name, phase2_bundle, df_flow_agg, flow_scaler):
    print(f'\n=== Two-stage refinement with {model_name} ===')

    df_preprocessed = phase2_bundle['df_preprocessed']
    x_test = phase2_bundle['x_test']
    y_test = phase2_bundle['y_test']
    y_pred_test = phase2_bundle['y_pred_test']
    labels_test = phase2_bundle['labels_test']

    alert_indices_in_xtest = np.where(y_pred_test == 1)[0]
    alert_row_indices = x_test.index[alert_indices_in_xtest]
    print(f'Phase 2 raised {len(alert_row_indices)} alerts on the test set')

    if len(alert_row_indices) == 0:
        print('No Phase 2 alerts to refine — skipping two-stage evaluation')
        return

    t_match_start = time.time()
    x_alert_flow, matched_mask, matched_flow_rows = match_packets_to_flows(
        df_preprocessed, alert_row_indices, df_flow_agg, flow_scaler
    )
    match_time = time.time() - t_match_start

    if matched_mask.sum() == 0:
        print('No Phase 2 alerts could be matched to any flow — cannot refine')
        return

    y_phase3_pred, _, phase3_infer_time = predict_fn(model, x_alert_flow)

    y_two_stage, refinement_stats = compute_two_stage_predictions(
        y_phase2_pred_all=y_pred_test,
        y_phase3_pred_on_matched=y_phase3_pred,
        matched_mask=matched_mask,
    )

    two_stage_metrics = compute_supervised_metrics(y_test, y_two_stage)
    attack_rates_two_stage = per_attack_detection_rate(labels_test, y_two_stage)
    print_metrics(f'Two-stage (Phase 2 AE + Phase 3 {model_name})',
                  two_stage_metrics, attack_rates_two_stage)

    fp_reduction = compute_fp_reduction(phase2_bundle['phase2_metrics'], two_stage_metrics)
    print(f'FP reduction: {fp_reduction["phase2_fp"]} -> {fp_reduction["two_stage_fp"]} '
          f'({fp_reduction["fp_reduction_pct"]:.2f}%)')

    write_metrics_to_file(
        f'Two-stage (Phase 2 AE + Phase 3 {model_name})',
        two_stage_metrics, attack_rates_two_stage,
        extras={
            'phase2_fp': fp_reduction['phase2_fp'],
            'two_stage_fp': fp_reduction['two_stage_fp'],
            'fp_reduction_pct': f'{fp_reduction["fp_reduction_pct"]:.2f}',
            'alerts_matched': int(matched_mask.sum()),
            'alerts_total': int(len(matched_mask)),
            'match_time_seconds': f'{match_time:.2f}',
            'phase3_inference_seconds': f'{phase3_infer_time:.4f}',
            'refinement_kept': refinement_stats['kept'],
            'refinement_dropped': refinement_stats['dropped'],
            'refinement_unmatched_kept': refinement_stats['unmatched_kept'],
        },
    )

    slug = model_name.lower().replace(' ', '_')
    plot_confusion_matrix(y_test, y_two_stage, f'Two-stage {model_name}',
                          f'two_stage_{slug}_cm.png')


def main():
    all_task_choices = ['flow_prep'] + list(MODEL_REGISTRY.keys()) + ['two_stage', 'all']
    parser = argparse.ArgumentParser(description='Phase 3 supervised refinement')
    parser.add_argument(
        'task',
        choices=all_task_choices,
        help=(
            'flow_prep: load/aggregate/sample flow data only; '
            'gb/nn: train that single model on flow data; '
            'two_stage: Phase 2 (AE) + Phase 3 refinement with every trained model; '
            'all: train every model and run two-stage refinement for each'
        ),
    )
    args = parser.parse_args()

    if args.task == 'flow_prep':
        prepare_flow_data()
        print('Phase 3 flow_prep finished')
        return

    print('=== Flow-level dataset preparation ===')
    df_flow_pre, df_flow_agg, flow_scaler = prepare_flow_data()

    (x_train, x_val, x_test,
     y_train, y_val, y_test,
     labels_train, labels_val, labels_test) = make_train_val_test_splits(df_flow_pre)

    labels_test_arr = np.asarray(labels_test).astype(str)

    trained_models = {}

    def train_one(task_key):
        display_name, train_fn, predict_fn = MODEL_REGISTRY[task_key]
        model, _, _ = run_supervised_experiment(
            display_name, train_fn, predict_fn,
            x_train, y_train, x_val, y_val, x_test, y_test, labels_test_arr,
        )
        trained_models[task_key] = (model, predict_fn, display_name)

    if args.task in MODEL_REGISTRY:
        train_one(args.task)

    elif args.task in ('two_stage', 'all'):
        for task_key in MODEL_REGISTRY:
            train_one(task_key)

        phase2_bundle = run_phase2_get_alerts()

        for task_key, (model, predict_fn, display_name) in trained_models.items():
            run_two_stage(model, predict_fn, display_name,
                          phase2_bundle, df_flow_agg, flow_scaler)

    print('\nPhase 3 run finished')


if __name__ == '__main__':
    main()
