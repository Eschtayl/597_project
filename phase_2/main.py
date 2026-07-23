import argparse

import numpy as np

from config import ENSEMBLE_METHOD, ENSEMBLE_WEIGHT_AE
from data_prep import (
    prepare_packet_data,
    split_features_and_labels,
    split_train_test,
    choose_threshold,
    generate_alerts,
)
from eda import run_eda
from models.isolation_forest import train_isolation_forest, get_anomaly_scores as get_if_scores
from models.autoencoder import (
    train_autoencoder,
    tune_autoencoder,
    get_anomaly_scores as get_ae_scores,
    get_latent_embedding,
)
from models.ensemble import fuse_scores
from models.kmeans import train_kmeans, get_anomaly_scores as get_km_scores
from evaluation import (
    compute_detection_metrics,
    per_attack_detection_rate,
    write_metrics_to_file,
    write_tuning_table,
    print_metrics,
    visualise_confusion_matrix,
    visualise_roc_curve,
    visualise_score_distribution,
)


def prepare_splits():
    print('Preparing packet data')
    df_preprocessed, _ = prepare_packet_data()

    print('Splitting features and labels')
    x_features, labels, y_true, identifier_cols = split_features_and_labels(df_preprocessed)
    print(f'Feature matrix: {x_features.shape}, identifiers held out: {len(identifier_cols)}')

    print('Creating train/test split')
    x_train, x_test, labels_train, labels_test, y_train, y_test = split_train_test(
        x_features, labels, y_true
    )
    print(f'Train: {x_train.shape}, Test: {x_test.shape}')
    return x_train, x_test, labels_train, labels_test, y_train, y_test


def evaluate_and_report(y_test, y_pred_test, test_scores, labels_test,
                        threshold, model_name, fig_prefix, plots=()):
    # Shared metrics + write-out for any detector producing anomaly scores.
    # `plots` selects which figures to save (subset of 'roc', 'score', 'confusion');
    # metrics are always computed and written, figures only when explicitly requested.
    metrics = compute_detection_metrics(y_test, y_pred_test, anomaly_scores=test_scores)
    attack_rates = per_attack_detection_rate(labels_test, y_test, y_pred_test)
    print_metrics(metrics, attack_rates)
    write_metrics_to_file(metrics, attack_rates, threshold, model_name=model_name)

    if plots:
        print(f'Visualising {model_name} results: {", ".join(plots)}')
    if 'confusion' in plots:
        visualise_confusion_matrix(
            y_test, y_pred_test, model_name=model_name, save_name=f'{fig_prefix}_confusion_matrix.png'
        )
    if 'roc' in plots:
        visualise_roc_curve(
            y_test, test_scores, model_name=model_name, save_name=f'{fig_prefix}_roc_curve.png'
        )
    if 'score' in plots:
        visualise_score_distribution(
            test_scores, y_test, model_name=model_name, save_name=f'{fig_prefix}_score_distribution.png'
        )


def run_eda_step():
    print('Running EDA')
    df_preprocessed, _ = prepare_packet_data()
    x_features, labels, y_true, identifier_cols = split_features_and_labels(df_preprocessed)
    print(f'Feature matrix: {x_features.shape}, identifiers held out: {len(identifier_cols)}')
    run_eda(x_features, labels)


def run_isolation_forest(x_train, x_test, labels_test, y_train, y_test):
    print('Training Isolation Forest')
    model = train_isolation_forest(x_train, y_train=y_train)

    print('Computing anomaly scores')
    train_scores = get_if_scores(model, x_train)
    test_scores = get_if_scores(model, x_test)

    print('Choosing threshold and generating alerts')
    threshold = choose_threshold(train_scores, y_train)
    y_pred_test = generate_alerts(test_scores, threshold)

    print('Evaluating Isolation Forest')
    # Only the ROC is used in the report (baseline vs tuned AE comparison)
    evaluate_and_report(
        y_test, y_pred_test, test_scores, labels_test,
        threshold, 'Isolation Forest', 'if', plots=('roc',)
    )
    return model, train_scores, test_scores


def run_kmeans(x_train, x_test, labels_test, y_train, y_test):
    print('Training K-Means (PCA + cluster attack-purity scoring)')
    model = train_kmeans(x_train, y_train)

    print('Computing anomaly scores')
    train_scores = get_km_scores(model, x_train)
    test_scores = get_km_scores(model, x_test)

    print('Choosing threshold and generating alerts')
    threshold = choose_threshold(train_scores, y_train)
    y_pred_test = generate_alerts(test_scores, threshold)

    print('Evaluating K-Means')
    evaluate_and_report(
        y_test, y_pred_test, test_scores, labels_test,
        threshold, 'K-Means (PCA + purity)', 'km', plots=('roc', 'score', 'confusion')
    )
    return model, train_scores, test_scores


def run_autoencoder(x_train, x_test, labels_test, y_train, y_test):
    print('Training autoencoder (baseline architecture)')
    model = train_autoencoder(x_train, y_train=y_train)
    # Baseline AE is a reference point only; no figures needed
    return _score_and_report_ae(model, x_train, x_test, labels_test, y_train, y_test,
                                model_name='Autoencoder', fig_prefix='ae', plots=())


def run_tuned_autoencoder(x_train, x_test, labels_test, y_train, y_test):
    print('Tuning autoencoder architecture')
    model, best_dims, results = tune_autoencoder(x_train, y_train=y_train)
    write_tuning_table(results, best_dims)
    _score_and_report_ae(model, x_train, x_test, labels_test, y_train, y_test,
                         model_name=f'Autoencoder (tuned {best_dims})', fig_prefix='ae_tuned',
                         plots=('roc', 'score', 'confusion'))
    return model


def _score_and_report_ae(model, x_train, x_test, labels_test, y_train, y_test,
                         model_name, fig_prefix, plots=()):
    print('Computing reconstruction errors')
    train_scores = get_ae_scores(model, x_train)
    test_scores = get_ae_scores(model, x_test)

    print('Choosing threshold and generating alerts')
    threshold = choose_threshold(train_scores, y_train)
    y_pred_test = generate_alerts(test_scores, threshold)

    print(f'Evaluating {model_name}')
    evaluate_and_report(
        y_test, y_pred_test, test_scores, labels_test,
        threshold, model_name, fig_prefix, plots=plots
    )
    return model, train_scores, test_scores


def run_ensemble(if_train_scores, if_test_scores, ae_train_scores, ae_test_scores,
                 labels_test, y_train, y_test):
    # Option 1: rank-normalized score fusion of IF + AE
    print(f'Fusing IF + AE scores (method={ENSEMBLE_METHOD})')
    fused_train = fuse_scores(
        if_train_scores, ae_train_scores, if_train_scores, ae_train_scores,
        method=ENSEMBLE_METHOD, weight_ae=ENSEMBLE_WEIGHT_AE,
    )
    fused_test = fuse_scores(
        if_test_scores, ae_test_scores, if_train_scores, ae_train_scores,
        method=ENSEMBLE_METHOD, weight_ae=ENSEMBLE_WEIGHT_AE,
    )

    print('Choosing threshold and generating alerts')
    threshold = choose_threshold(fused_train, y_train)
    y_pred_test = generate_alerts(fused_test, threshold)

    print('Evaluating ensemble')
    evaluate_and_report(
        y_test, y_pred_test, fused_test, labels_test,
        threshold, 'Ensemble (IF+AE score fusion)', 'ens'
    )


def run_latent(ae_model, x_train, x_test, labels_test, y_train, y_test):
    # Option 4: Isolation Forest on [AE latent embedding + reconstruction error]
    print('Building AE latent + reconstruction-error features')
    latent_train = get_latent_embedding(ae_model, x_train)
    latent_test = get_latent_embedding(ae_model, x_test)
    recon_train = get_ae_scores(ae_model, x_train).reshape(-1, 1)
    recon_test = get_ae_scores(ae_model, x_test).reshape(-1, 1)
    feat_train = np.hstack([latent_train, recon_train])
    feat_test = np.hstack([latent_test, recon_test])
    print(f'Latent feature matrix: train {feat_train.shape}, test {feat_test.shape}')

    print('Training Isolation Forest on latent features')
    model = train_isolation_forest(feat_train, y_train=y_train)
    train_scores = get_if_scores(model, feat_train)
    test_scores = get_if_scores(model, feat_test)

    print('Choosing threshold and generating alerts')
    threshold = choose_threshold(train_scores, y_train)
    y_pred_test = generate_alerts(test_scores, threshold)

    print('Evaluating IF-on-latent')
    evaluate_and_report(
        y_test, y_pred_test, test_scores, labels_test,
        threshold, 'IF on AE latent + recon error', 'lat'
    )


def main():
    parser = argparse.ArgumentParser(description='Phase 2 packet anomaly detection')
    parser.add_argument(
        'task',
        choices=['eda', 'if', 'ae', 'tune', 'ensemble', 'latent', 'kmeans', 'all'],
        help=(
            'eda: plots only; if: Isolation Forest; ae: baseline autoencoder; '
            'tune: AE architecture search; ensemble: IF+AE score fusion; '
            'latent: IF on AE latent; kmeans: K-Means (PCA + purity); all: run everything'
        ),
    )
    args = parser.parse_args()

    if args.task == 'eda':
        run_eda_step()
        return

    x_train, x_test, labels_train, labels_test, y_train, y_test = prepare_splits()

    if args.task == 'if':
        run_isolation_forest(x_train, x_test, labels_test, y_train, y_test)

    elif args.task == 'ae':
        run_autoencoder(x_train, x_test, labels_test, y_train, y_test)

    elif args.task == 'tune':
        run_tuned_autoencoder(x_train, x_test, labels_test, y_train, y_test)

    elif args.task == 'ensemble':
        _, if_train, if_test = run_isolation_forest(x_train, x_test, labels_test, y_train, y_test)
        _, ae_train, ae_test = _score_and_report_ae(
            tune_and_get_best(x_train, y_train),
            x_train, x_test, labels_test, y_train, y_test,
            model_name='Autoencoder (tuned)', fig_prefix='ae_tuned',
            plots=('roc', 'score', 'confusion'),
        )
        run_ensemble(if_train, if_test, ae_train, ae_test, labels_test, y_train, y_test)

    elif args.task == 'latent':
        ae_model = tune_and_get_best(x_train, y_train)
        run_latent(ae_model, x_train, x_test, labels_test, y_train, y_test)

    elif args.task == 'kmeans':
        run_kmeans(x_train, x_test, labels_test, y_train, y_test)

    elif args.task == 'all':
        _, if_train, if_test = run_isolation_forest(x_train, x_test, labels_test, y_train, y_test)
        run_autoencoder(x_train, x_test, labels_test, y_train, y_test)
        ae_model = tune_and_get_best(x_train, y_train)
        _, ae_train, ae_test = _score_and_report_ae(
            ae_model, x_train, x_test, labels_test, y_train, y_test,
            model_name='Autoencoder (tuned)', fig_prefix='ae_tuned',
            plots=('roc', 'score', 'confusion'),
        )
        run_ensemble(if_train, if_test, ae_train, ae_test, labels_test, y_train, y_test)
        run_latent(ae_model, x_train, x_test, labels_test, y_train, y_test)

    print('Phase 2 run finished')


def tune_and_get_best(x_train, y_train):
    # Run the architecture search and record the table, return the best model
    model, best_dims, results = tune_autoencoder(x_train, y_train=y_train)
    write_tuning_table(results, best_dims)
    return model


# Call the main function
if __name__ == '__main__':
    main()
