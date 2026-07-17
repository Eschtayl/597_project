import argparse

from data_prep import (
    prepare_packet_data,
    split_features_and_labels,
    split_train_test,
    choose_threshold,
    generate_alerts,
)
from eda import run_eda
from models.isolation_forest import train_isolation_forest, get_anomaly_scores as get_if_scores
from models.autoencoder import train_autoencoder, get_anomaly_scores as get_ae_scores
from evaluation import (
    compute_detection_metrics,
    per_attack_detection_rate,
    write_metrics_to_file,
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


def run_eda_step():
    print('Running EDA')
    df_preprocessed, _ = prepare_packet_data()
    x_features, labels, y_true, identifier_cols = split_features_and_labels(df_preprocessed)
    print(f'Feature matrix: {x_features.shape}, identifiers held out: {len(identifier_cols)}')
    run_eda(x_features, labels)
    return


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
    metrics = compute_detection_metrics(y_test, y_pred_test, anomaly_scores=test_scores)
    attack_rates = per_attack_detection_rate(labels_test, y_test, y_pred_test)
    print_metrics(metrics, attack_rates)
    write_metrics_to_file(metrics, attack_rates, threshold, model_name='Isolation Forest')

    print('Visualising Isolation Forest results')
    visualise_confusion_matrix(
        y_test, y_pred_test, model_name='Isolation Forest', save_name='if_confusion_matrix.png'
    )
    visualise_roc_curve(
        y_test, test_scores, model_name='Isolation Forest', save_name='if_roc_curve.png'
    )
    visualise_score_distribution(
        test_scores, y_test, model_name='Isolation Forest', save_name='if_score_distribution.png'
    )
    return


def run_autoencoder(x_train, x_test, labels_test, y_train, y_test):
    print('Training autoencoder')
    model = train_autoencoder(x_train, y_train=y_train)

    print('Computing reconstruction errors')
    train_scores = get_ae_scores(model, x_train)
    test_scores = get_ae_scores(model, x_test)

    print('Choosing threshold and generating alerts')
    threshold = choose_threshold(train_scores, y_train)
    y_pred_test = generate_alerts(test_scores, threshold)

    print('Evaluating autoencoder')
    metrics = compute_detection_metrics(y_test, y_pred_test, anomaly_scores=test_scores)
    attack_rates = per_attack_detection_rate(labels_test, y_test, y_pred_test)
    print_metrics(metrics, attack_rates)
    write_metrics_to_file(metrics, attack_rates, threshold, model_name='Autoencoder')

    print('Visualising autoencoder results')
    visualise_confusion_matrix(
        y_test, y_pred_test, model_name='Autoencoder', save_name='ae_confusion_matrix.png'
    )
    visualise_roc_curve(
        y_test, test_scores, model_name='Autoencoder', save_name='ae_roc_curve.png'
    )
    visualise_score_distribution(
        test_scores, y_test, model_name='Autoencoder', save_name='ae_score_distribution.png'
    )
    return


def main():
    parser = argparse.ArgumentParser(description='Phase 2 packet anomaly detection')
    parser.add_argument(
        'task',
        choices=['eda', 'if', 'ae', 'all'],
        help='eda: plots only; if: Isolation Forest; ae: autoencoder; all: if then ae',
    )
    args = parser.parse_args()

    if args.task == 'eda':
        run_eda_step()
        return

    x_train, x_test, labels_train, labels_test, y_train, y_test = prepare_splits()

    if args.task in ('if', 'all'):
        run_isolation_forest(x_train, x_test, labels_test, y_train, y_test)

    if args.task in ('ae', 'all'):
        run_autoencoder(x_train, x_test, labels_test, y_train, y_test)

    print('Phase 2 run finished')
    return


# Call the main function
if __name__ == '__main__':
    main()
