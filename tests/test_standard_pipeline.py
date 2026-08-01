"""Fast tests for the standardized pipeline components."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ids_pipeline.components import (
    BehavioralPacketPreprocessor,
    FinalAutoencoderDetector,
)
from ids_pipeline.config import AutoencoderConfig, DataConfig, PipelineConfig
from ids_pipeline.orchestrator import StandardPipeline
from phase_3.phase2_cohort import build_phase2_cohort


def _packet_frame(n_rows=24):
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            'src_ip': ['10.0.0.1'] * n_rows,
            'dst_ip': ['10.0.0.2'] * n_rows,
            'src_port': [12345] * n_rows,
            'dst_port': [443] * n_rows,
            'l4_tcp': [1] * n_rows,
            'l4_udp': [0] * n_rows,
            'packet_size': rng.integers(40, 1500, size=n_rows),
            'jitter': rng.random(n_rows),
            'http_request_method': ['GET'] * n_rows,
        }
    )


def test_behavioral_preprocessor_retains_identifiers_and_aligns_rows():
    """The preprocessor must keep mapping fields out of model features."""
    frame = _packet_frame()
    labels = pd.Series(['benign'] * len(frame), name='label')
    identifiers = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'l4_tcp', 'l4_udp']

    batch = BehavioralPacketPreprocessor().prepare(frame, labels, identifiers)

    assert list(batch.identifiers.columns) == identifiers
    assert len(batch.features) == len(frame)
    assert 'src_ip' not in batch.features
    assert not batch.features.isna().any().any()


def test_final_autoencoder_scores_every_row():
    """The configurable AE wrapper should support a lightweight smoke fit."""
    frame = _packet_frame()
    labels = pd.Series(['benign'] * 20 + ['attack'] * 4, name='label')
    identifiers = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'l4_tcp', 'l4_udp']
    batch = BehavioralPacketPreprocessor().prepare(frame, labels, identifiers)
    y_binary = (labels != 'benign').astype(int).to_numpy()
    detector = FinalAutoencoderDetector(
        AutoencoderConfig(
            hidden_dims=(4, 2),
            epochs=1,
            batch_size=4,
            validation_fraction=0.2,
        ),
        seed=5,
    )

    detector.fit(batch.features, y_binary)
    scores = detector.score(batch.features)

    assert scores.shape == (len(frame),)
    assert np.isfinite(scores).all()
    assert (scores >= 0).all()


def test_standard_pipeline_dry_run_does_not_require_data(tmp_path):
    """Dry-run mode should validate and compose the complete stage sequence."""
    config = PipelineConfig(
        project_root=tmp_path,
        data=DataConfig(
            packet_dir=tmp_path / 'packets',
            flow_dir=tmp_path / 'flows',
            artifacts_dir=tmp_path / 'artifacts',
        ),
        autoencoder=AutoencoderConfig(epochs=1),
    )

    result = StandardPipeline(config).run(dry_run=True)

    assert result.packet_cohort == tmp_path / 'artifacts' / 'phase2_cohort.parquet'
    assert result.cascade_results == tmp_path / 'artifacts' / 'cascade_results.json'


def test_packet_cohort_stage_runs_preprocessing_then_autoencoder(tmp_path):
    """The integrated Phase 1/2 stage should persist the cascade schema."""
    packet_dir = tmp_path / 'packets'
    packet_dir.mkdir()
    files = {
        'BenignTraffic.csv': 10,
        'DDoS-HTTP_Flood.csv': 2,
        'DoS-HTTP_Flood.csv': 2,
        'DNS_Spoofing.csv': 2,
        'XSS.csv': 2,
        'DictionaryBruteForce.csv': 2,
    }
    for offset, (name, count) in enumerate(files.items()):
        frame = _packet_frame(count)
        frame['packet_size'] += offset * 100
        frame.to_csv(packet_dir / name, index=False)

    output = tmp_path / 'phase2_cohort.parquet'
    cohort = build_phase2_cohort(
        packet_dir=packet_dir,
        output_path=output,
        seed=11,
        benign_target=10,
        attack_target_per_type=2,
        autoencoder_config=AutoencoderConfig(
            hidden_dims=(4, 2),
            epochs=1,
            batch_size=4,
            validation_fraction=0.2,
        ),
    )

    assert output.exists()
    assert len(cohort) == 20
    assert {
        'label',
        'y_true',
        'phase2_split',
        'anomaly_score',
        'y2_alert',
    }.issubset(cohort.columns)
    assert set(cohort['phase2_split']) == {'train', 'val', 'test'}
