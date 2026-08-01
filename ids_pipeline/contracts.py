"""Small component contracts used by the standardized pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd


@dataclass
class PreprocessedPacketBatch:
    """Aligned packet features, labels, identifiers, and fitted preprocessing state."""

    features: pd.DataFrame
    labels: pd.Series
    identifiers: pd.DataFrame
    fitted_state: Any


class PacketPreprocessor(Protocol):
    """Contract for packet preprocessing implementations."""

    def prepare(
        self,
        frame: pd.DataFrame,
        labels: pd.Series,
        identifier_columns: list[str],
    ) -> PreprocessedPacketBatch:
        """Prepare aligned packet features while retaining mapping identifiers."""


class AnomalyDetector(Protocol):
    """Contract for unsupervised packet anomaly detectors."""

    def fit(self, features: pd.DataFrame, y_binary: np.ndarray | None = None) -> None:
        """Fit the detector, optionally using labels only to select benign rows."""

    def score(self, features: pd.DataFrame) -> np.ndarray:
        """Return one anomaly score per row, with higher values more anomalous."""


class ClassifierStage(Protocol):
    """Contract for a complete supervised flow-refinement stage."""

    def run(self, runner: Any) -> None:
        """Train and evaluate the configured flow classifier and cascade."""
