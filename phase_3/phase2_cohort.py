"""Build finalized Autoencoder packet alerts for the XGBoost cascade.

This stage keeps the original memory-bounded packet sampling approach, applies
behavioral Phase 1 feature cleaning/scaling, and fits the finalized tuned
Autoencoder on benign training packets.  It persists raw packet identifiers
beside the scores so Phase 3 can map alerts to logical flows.

Output: ``data/phase2_cohort.parquet`` with one row per sampled packet.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from .config import BENIGN_LABEL, DATA_DIR, PACKET_DIR, RANDOM_SEED
    from .flow_aggregation import label_from_filename
except ImportError:  # Preserve ``python phase_3/phase2_cohort.py``.
    from config import BENIGN_LABEL, DATA_DIR, PACKET_DIR, RANDOM_SEED
    from flow_aggregation import label_from_filename
from ids_pipeline.components import create_anomaly_detector, create_preprocessor
from ids_pipeline.config import AutoencoderConfig
from phase_2.data_prep import choose_threshold, generate_alerts

SEED = RANDOM_SEED
ID_COLS = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'l4_tcp', 'l4_udp']
BENIGN_TARGET = 200_000
ATTACK_TARGET_PER_TYPE = 1_200

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def min_key_sample(files, n_target, seed, chunksize=100_000):
    """Uniformly sample rows without loading all source CSVs at once."""
    if n_target <= 0:
        raise ValueError('n_target must be positive.')
    rng = np.random.default_rng(seed)
    kept = None
    for file_path in files:
        for chunk in pd.read_csv(file_path, chunksize=chunksize, low_memory=False):
            chunk = chunk.copy()
            chunk['__k'] = rng.random(len(chunk))
            kept = chunk if kept is None else pd.concat([kept, chunk], ignore_index=True)
            if len(kept) > n_target:
                kept = kept.nsmallest(n_target, '__k')
    if kept is None:
        raise ValueError('Cannot sample from an empty file list.')
    return kept


def load_packet_cohort(
    path=PACKET_DIR,
    seed=SEED,
    benign_target=BENIGN_TARGET,
    attack_target_per_type=ATTACK_TARGET_PER_TYPE,
):
    """Draw a memory-bounded packet cohort with balanced attack classes."""
    all_csvs = sorted(glob.glob(os.path.join(path, '*.csv')))
    benign_files = [
        file_path
        for file_path in all_csvs
        if os.path.basename(file_path).startswith('Benign')
    ]
    attack_files = [
        file_path
        for file_path in all_csvs
        if not os.path.basename(file_path).startswith('Benign')
    ]
    if not benign_files or not attack_files:
        raise FileNotFoundError(
            f'Expected benign and attack packet CSV files in: {path}'
        )
    print(f'Packet captures: {len(benign_files)} benign, {len(attack_files)} attack')

    print('Sampling benign packets (memory-bounded)...')
    benign = min_key_sample(benign_files, benign_target, seed)
    benign['label'] = BENIGN_LABEL

    print('Sampling attack packets per type...')
    by_label = {}
    for file_path in attack_files:
        by_label.setdefault(label_from_filename(file_path), []).append(file_path)
    attack_parts = []
    for offset, (label, files) in enumerate(sorted(by_label.items()), start=1):
        # Python hashes are process-randomized; a stable offset is reproducible.
        sample = min_key_sample(files, attack_target_per_type, seed + offset)
        sample['label'] = label
        attack_parts.append(sample)

    attacks = pd.concat(attack_parts, ignore_index=True)
    frame = pd.concat([benign, attacks], ignore_index=True).drop(
        columns=['__k'], errors='ignore'
    )
    frame = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    print(
        f'  packet cohort: {len(frame):,} '
        f'({(frame.label == BENIGN_LABEL).sum():,} benign, '
        f'{(frame.label != BENIGN_LABEL).sum():,} attack)'
    )
    return frame


def build_phase2_cohort(
    *,
    packet_dir=PACKET_DIR,
    output_path=None,
    seed=SEED,
    benign_target=BENIGN_TARGET,
    attack_target_per_type=ATTACK_TARGET_PER_TYPE,
    autoencoder_config=None,
    preprocessor_name='behavioral_packet',
    detector_name='autoencoder',
):
    """Run behavioral preprocessing and the finalized AE, then persist alerts."""
    ae_config = autoencoder_config or AutoencoderConfig()
    frame = load_packet_cohort(
        path=packet_dir,
        seed=seed,
        benign_target=benign_target,
        attack_target_per_type=attack_target_per_type,
    )
    labels = frame['label'].reset_index(drop=True)

    print('Preprocessing behavioral packet features (Phase 1 pipeline)...')
    preprocessor = create_preprocessor(preprocessor_name)
    batch = preprocessor.prepare(
        frame.drop(columns=['label']),
        labels,
        identifier_columns=ID_COLS,
    )
    features = batch.features
    y_true = (labels != BENIGN_LABEL).astype(int).to_numpy()
    print(
        f'  feature matrix: {features.shape}, '
        f'identifiers held out: {len(ID_COLS)}'
    )

    indices = np.arange(len(features))
    train_idx, remainder_idx = train_test_split(
        indices,
        test_size=0.30,
        random_state=seed,
        stratify=y_true,
    )
    val_idx, test_idx = train_test_split(
        remainder_idx,
        test_size=0.50,
        random_state=seed,
        stratify=y_true[remainder_idx],
    )
    split = np.full(len(features), 'train', dtype=object)
    split[val_idx] = 'val'
    split[test_idx] = 'test'

    print(
        'Training finalized Phase 2 Autoencoder on benign train packets '
        f'(dims={ae_config.hidden_dims})...'
    )
    detector = create_anomaly_detector(detector_name, ae_config, seed)
    detector.fit(features.iloc[train_idx], y_true[train_idx])
    scores = detector.score(features)
    threshold = choose_threshold(scores[train_idx], y_true[train_idx])
    alerts = generate_alerts(scores, threshold)

    cohort = batch.identifiers.copy()
    cohort['label'] = labels.values
    cohort['y_true'] = y_true
    cohort['phase2_split'] = split
    cohort['anomaly_score'] = scores
    cohort['y2_alert'] = alerts

    print('\nPhase 2 packet performance (alerts):')
    for part in ('val', 'test'):
        subset = cohort[cohort['phase2_split'] == part]
        tp = int(((subset.y2_alert == 1) & (subset.y_true == 1)).sum())
        fp = int(((subset.y2_alert == 1) & (subset.y_true == 0)).sum())
        fn = int(((subset.y2_alert == 0) & (subset.y_true == 1)).sum())
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        print(
            f'  {part}: packets={len(subset):,} '
            f'alerts={int((subset.y2_alert == 1).sum()):,} '
            f'TP={tp:,} FP={fp:,} recall={recall:.3f} '
            f'precision={precision:.3f}'
        )

    destination = Path(
        output_path or os.path.join(DATA_DIR, 'phase2_cohort.parquet')
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(destination)
    print(f'\nSaved Phase 2 cohort -> {destination}')
    print(
        'Next: cascade_alignment.py maps val/test alerts to the '
        'unified-flow index and runs the leakage guard.'
    )
    return cohort


def build_parser():
    """Build command-line options for packet cohort generation."""
    parser = argparse.ArgumentParser(
        description=(
            "Run behavioral packet preprocessing and the finalized Autoencoder."
        )
    )
    parser.add_argument('--packet-dir', default=PACKET_DIR)
    parser.add_argument(
        '--output', default=os.path.join(DATA_DIR, 'phase2_cohort.parquet')
    )
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--preprocessor', default='behavioral_packet')
    parser.add_argument('--detector', default='autoencoder')
    parser.add_argument('--benign-target', type=int, default=BENIGN_TARGET)
    parser.add_argument(
        '--attack-target-per-type', type=int, default=ATTACK_TARGET_PER_TYPE
    )
    parser.add_argument(
        '--hidden-dims', type=int, nargs='+', default=[128, 64, 32]
    )
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--validation-fraction', type=float, default=0.1)
    return parser


def main():
    """Run the packet cohort stage from the command line."""
    args = build_parser().parse_args()
    build_phase2_cohort(
        packet_dir=args.packet_dir,
        output_path=args.output,
        seed=args.seed,
        benign_target=args.benign_target,
        attack_target_per_type=args.attack_target_per_type,
        preprocessor_name=args.preprocessor,
        detector_name=args.detector,
        autoencoder_config=AutoencoderConfig(
            hidden_dims=tuple(args.hidden_dims),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            validation_fraction=args.validation_fraction,
        ),
    )


if __name__ == '__main__':
    main()
