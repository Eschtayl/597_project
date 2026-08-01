"""Configuration objects for the behavioral packet → AE → XGBoost pipeline."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _tuple_of_ints(value: Any) -> tuple[int, ...]:
    return tuple(int(item) for item in value)


@dataclass(frozen=True)
class DataConfig:
    """Input locations, artifact location, and Phase 1 sampling sizes."""

    packet_dir: Path
    flow_dir: Path
    artifacts_dir: Path
    benign_target: int = 200_000
    attack_target_per_type: int = 1_200


@dataclass(frozen=True)
class AutoencoderConfig:
    """Hyperparameters for the finalized packet autoencoder."""

    hidden_dims: tuple[int, ...] = (128, 64, 32)
    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    validation_fraction: float = 0.1


@dataclass(frozen=True)
class XGBoostConfig:
    """Configuration for XGBoost tuning and artifact reuse."""

    tuning_candidates: int = 12
    reuse_tuned_parameters: bool = True


@dataclass(frozen=True)
class PipelineConfig:
    """All settings required to execute the standardized pipeline."""

    project_root: Path
    data: DataConfig
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)
    random_seed: int = 23
    preprocessing: str = "behavioral_packet"
    feature_extractor: str = "autoencoder"
    classifier: str = "xgboost"
    python_executable: str = sys.executable

    @classmethod
    def default(cls, project_root: Path | None = None) -> "PipelineConfig":
        """Build the repository's default behavioral packet pipeline."""
        root = (project_root or Path(__file__).resolve().parents[1]).resolve()
        nested = root / "flow_and_packet"
        packet_dir = (
            nested / "packet_based"
            if (nested / "packet_based").is_dir()
            else root / "packet_based"
        )
        flow_dir = (
            nested / "flow_based"
            if (nested / "flow_based").is_dir()
            else root / "flow_based"
        )
        return cls(
            project_root=root,
            data=DataConfig(
                packet_dir=packet_dir,
                flow_dir=flow_dir,
                artifacts_dir=root / "phase_3" / "data",
            ),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "PipelineConfig":
        """Load a pipeline configuration from a JSON file."""
        config_path = Path(path).resolve()
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        root = Path(raw.get("project_root", config_path.parent))
        if not root.is_absolute():
            root = (config_path.parent / root).resolve()

        def resolve(value: str) -> Path:
            candidate = Path(value)
            return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()

        data_raw = raw.get("data", {})
        defaults = cls.default(root)
        data = DataConfig(
            packet_dir=resolve(str(data_raw.get("packet_dir", defaults.data.packet_dir))),
            flow_dir=resolve(str(data_raw.get("flow_dir", defaults.data.flow_dir))),
            artifacts_dir=resolve(
                str(data_raw.get("artifacts_dir", defaults.data.artifacts_dir))
            ),
            benign_target=int(data_raw.get("benign_target", 200_000)),
            attack_target_per_type=int(
                data_raw.get("attack_target_per_type", 1_200)
            ),
        )
        ae_raw = raw.get("autoencoder", {})
        autoencoder = AutoencoderConfig(
            hidden_dims=_tuple_of_ints(ae_raw.get("hidden_dims", (128, 64, 32))),
            epochs=int(ae_raw.get("epochs", 20)),
            batch_size=int(ae_raw.get("batch_size", 256)),
            learning_rate=float(ae_raw.get("learning_rate", 1e-3)),
            weight_decay=float(ae_raw.get("weight_decay", 1e-5)),
            validation_fraction=float(ae_raw.get("validation_fraction", 0.1)),
        )
        xgb_raw = raw.get("xgboost", {})
        xgboost = XGBoostConfig(
            tuning_candidates=int(xgb_raw.get("tuning_candidates", 12)),
            reuse_tuned_parameters=bool(
                xgb_raw.get("reuse_tuned_parameters", True)
            ),
        )
        return cls(
            project_root=root,
            data=data,
            autoencoder=autoencoder,
            xgboost=xgboost,
            random_seed=int(raw.get("random_seed", 23)),
            preprocessing=str(raw.get("preprocessing", "behavioral_packet")),
            feature_extractor=str(raw.get("feature_extractor", "autoencoder")),
            classifier=str(raw.get("classifier", "xgboost")),
            python_executable=str(raw.get("python_executable", sys.executable)),
        )

    def validate(self, require_data: bool = True) -> None:
        """Validate component names, numeric settings, and optionally data paths."""
        for field_name, value in (
            ("preprocessing", self.preprocessing),
            ("feature_extractor", self.feature_extractor),
            ("classifier", self.classifier),
        ):
            if not value.strip():
                raise ValueError(f"{field_name} must name a registered component.")
        if not self.autoencoder.hidden_dims or any(
            dim <= 0 for dim in self.autoencoder.hidden_dims
        ):
            raise ValueError("autoencoder.hidden_dims must contain positive integers.")
        if self.autoencoder.epochs <= 0 or self.autoencoder.batch_size <= 0:
            raise ValueError("Autoencoder epochs and batch_size must be positive.")
        if not 0 < self.autoencoder.validation_fraction < 1:
            raise ValueError("autoencoder.validation_fraction must be between 0 and 1.")
        if self.xgboost.tuning_candidates <= 0:
            raise ValueError("xgboost.tuning_candidates must be positive.")
        if require_data:
            for label, directory in (
                ("packet", self.data.packet_dir),
                ("flow", self.data.flow_dir),
            ):
                if not directory.is_dir():
                    raise FileNotFoundError(f"{label.title()} data directory not found: {directory}")
