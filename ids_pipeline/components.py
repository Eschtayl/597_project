"""Adapters for behavioral packet preprocessing and anomaly detection."""
from __future__ import annotations

import numpy as np
import pandas as pd

from phase_1.helpers import feature_cleaner, log_and_scale
from phase_2.models.autoencoder import get_anomaly_scores, train_autoencoder

from .config import AutoencoderConfig
from .contracts import PreprocessedPacketBatch


class BehavioralPacketPreprocessor:
    """Prepare identity-free behavioral packet features for model training."""

    def prepare(
        self,
        frame: pd.DataFrame,
        labels: pd.Series,
        identifier_columns: list[str],
    ) -> PreprocessedPacketBatch:
        """Clean and scale packet features while retaining raw mapping fields."""
        aligned_frame = frame.reset_index(drop=True)
        aligned_labels = labels.reset_index(drop=True)
        missing_identifiers = [
            column for column in identifier_columns if column not in aligned_frame.columns
        ]
        if missing_identifiers:
            raise ValueError(
                f"Packet data is missing mapping columns: {missing_identifiers}"
            )

        identifiers = aligned_frame[identifier_columns].copy()
        numeric, _ = feature_cleaner(aligned_frame)
        final, fitted_scaler = log_and_scale(numeric, aligned_labels)
        features = final.drop(columns=["label"]).select_dtypes(include=[np.number])
        if features.empty:
            raise ValueError(
                "Behavioral packet preprocessing produced no numeric model features."
            )
        if features.isna().any().any():
            bad = features.columns[features.isna().any()].tolist()
            raise ValueError(
                "Behavioral packet preprocessing produced NaNs. "
                "Check negative/non-numeric "
                f"source values in: {bad[:8]}"
            )
        return PreprocessedPacketBatch(
            features=features,
            labels=aligned_labels,
            identifiers=identifiers,
            fitted_state=fitted_scaler,
        )


class FinalAutoencoderDetector:
    """Finalized tuned autoencoder used as the default Phase 2 detector."""

    def __init__(self, config: AutoencoderConfig, seed: int = 23) -> None:
        """Create an unfitted detector."""
        self.config = config
        self.seed = seed
        self.model = None

    def fit(
        self, features: pd.DataFrame, y_binary: np.ndarray | None = None
    ) -> None:
        """Fit on benign rows using the configured finalized architecture."""
        self.model = train_autoencoder(
            features,
            y_train=y_binary,
            hidden_dims=self.config.hidden_dims,
            seed=self.seed,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            val_fraction=self.config.validation_fraction,
        )

    def score(self, features: pd.DataFrame) -> np.ndarray:
        """Return reconstruction-error anomaly scores."""
        if self.model is None:
            raise RuntimeError("The autoencoder must be fitted before scoring.")
        return get_anomaly_scores(
            self.model, features, batch_size=self.config.batch_size
        )


PREPROCESSORS = {
    'behavioral_packet': BehavioralPacketPreprocessor,
    # Compatibility for configurations created before the descriptive rename.
    'taylor': BehavioralPacketPreprocessor,
}
ANOMALY_DETECTORS = {
    'autoencoder': FinalAutoencoderDetector,
}


def create_preprocessor(name: str):
    """Create a registered packet preprocessor by configuration name."""
    try:
        return PREPROCESSORS[name]()
    except KeyError as exc:
        raise ValueError(
            f"Unknown preprocessor '{name}'. Available: {sorted(PREPROCESSORS)}"
        ) from exc


def create_anomaly_detector(
    name: str, config: AutoencoderConfig, seed: int
):
    """Create a registered packet anomaly detector by configuration name."""
    try:
        factory = ANOMALY_DETECTORS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown anomaly detector '{name}'. "
            f"Available: {sorted(ANOMALY_DETECTORS)}"
        ) from exc
    return factory(config, seed=seed)
