"""End-to-end orchestration of the standard three-phase pipeline."""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .config import PipelineConfig


@dataclass(frozen=True)
class PipelineRunResult:
    """Locations of the primary artifacts produced by a standard run."""

    packet_cohort: Path
    cascade_scores: Path
    cascade_results: Path


class CommandRunner:
    """Run repository stages in isolated Python processes."""

    def __init__(self, cwd: Path, dry_run: bool = False) -> None:
        """Create a runner rooted at the project directory."""
        self.cwd = cwd
        self.dry_run = dry_run

    def run(self, command: Sequence[str], env: dict[str, str]) -> None:
        """Run one stage and fail immediately when it exits unsuccessfully."""
        printable = subprocess.list2cmdline(list(command))
        print(f"\n$ {printable}")
        if not self.dry_run:
            subprocess.run(command, cwd=self.cwd, env=env, check=True)


class XGBoostCascadeStage:
    """Execute the finalized flow preparation, XGBoost, and cascade modules."""

    def __init__(self, config: PipelineConfig, env: dict[str, str]) -> None:
        """Bind the classifier stage to a pipeline configuration."""
        self.config = config
        self.env = env

    def run(self, runner: CommandRunner) -> None:
        """Prepare flow data, tune XGBoost when needed, and evaluate the cascade."""
        root = self.config.project_root
        py = self.config.python_executable
        sample = self.config.data.artifacts_dir / "sampled_unified_flows.parquet"
        tuning = self.config.data.artifacts_dir / "xgb_results.json"

        if not sample.exists():
            runner.run([py, str(root / "phase_3" / "flow_sampling.py")], self.env)
        else:
            print(f"Reusing flow sample: {sample}")

        if not (self.config.xgboost.reuse_tuned_parameters and tuning.exists()):
            runner.run(
                [
                    py,
                    str(root / "phase_3" / "xgboost_model.py"),
                    str(self.config.xgboost.tuning_candidates),
                ],
                self.env,
            )
        else:
            print(f"Reusing tuned XGBoost parameters: {tuning}")

        for script in ("cascade_alignment.py", "cascade_heads.py", "cascade_eval.py"):
            runner.run([py, str(root / "phase_3" / script)], self.env)


CLASSIFIER_STAGES = {
    "xgboost": XGBoostCascadeStage,
}


def create_classifier_stage(
    name: str, config: PipelineConfig, env: dict[str, str]
):
    """Create a registered supervised refinement stage."""
    try:
        factory = CLASSIFIER_STAGES[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown classifier '{name}'. Available: {sorted(CLASSIFIER_STAGES)}"
        ) from exc
    return factory(config, env)


class StandardPipeline:
    """Run behavioral preprocessing, finalized AE, and XGBoost refinement."""

    def __init__(self, config: PipelineConfig) -> None:
        """Create a pipeline from an explicit configuration."""
        self.config = config

    def _environment(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update(
            {
                "IDS_PACKET_DIR": str(self.config.data.packet_dir),
                "IDS_FLOW_DIR": str(self.config.data.flow_dir),
                "IDS_ARTIFACTS_DIR": str(self.config.data.artifacts_dir),
                "IDS_RANDOM_SEED": str(self.config.random_seed),
                "PYTHONUTF8": "1",
            }
        )
        return env

    def run(self, dry_run: bool = False) -> PipelineRunResult:
        """Execute the complete default pipeline in dependency order."""
        self.config.validate(require_data=not dry_run)
        runner = CommandRunner(self.config.project_root, dry_run=dry_run)
        env = self._environment()
        ae = self.config.autoencoder
        cohort_command = [
            self.config.python_executable,
            str(self.config.project_root / "phase_3" / "phase2_cohort.py"),
            "--packet-dir",
            str(self.config.data.packet_dir),
            "--output",
            str(self.config.data.artifacts_dir / "phase2_cohort.parquet"),
            "--seed",
            str(self.config.random_seed),
            "--preprocessor",
            self.config.preprocessing,
            "--detector",
            self.config.feature_extractor,
            "--benign-target",
            str(self.config.data.benign_target),
            "--attack-target-per-type",
            str(self.config.data.attack_target_per_type),
            "--hidden-dims",
            *(str(dim) for dim in ae.hidden_dims),
            "--epochs",
            str(ae.epochs),
            "--batch-size",
            str(ae.batch_size),
            "--learning-rate",
            str(ae.learning_rate),
            "--weight-decay",
            str(ae.weight_decay),
            "--validation-fraction",
            str(ae.validation_fraction),
        ]
        runner.run(cohort_command, env)
        create_classifier_stage(self.config.classifier, self.config, env).run(runner)
        artifacts = self.config.data.artifacts_dir
        return PipelineRunResult(
            packet_cohort=artifacts / "phase2_cohort.parquet",
            cascade_scores=artifacts / "cascade_candidate_scores.parquet",
            cascade_results=artifacts / "cascade_results.json",
        )
