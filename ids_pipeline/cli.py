"""Command-line interface for the standardized project pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import PipelineConfig
from .orchestrator import StandardPipeline


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run behavioral packet preprocessing -> finalized Autoencoder "
            "-> XGBoost cascade."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON configuration file (defaults to pipeline_config.json).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and print stage commands without executing them.",
    )
    return parser


def main() -> None:
    """Run the configured standard pipeline."""
    args = build_parser().parse_args()
    config_path = args.config or Path(__file__).resolve().parents[1] / "pipeline_config.json"
    config = (
        PipelineConfig.from_json(config_path)
        if config_path.exists()
        else PipelineConfig.default()
    )
    result = StandardPipeline(config).run(dry_run=args.dry_run)
    print("\nStandard pipeline plan complete." if args.dry_run else "\nStandard pipeline finished.")
    print(f"Packet cohort: {result.packet_cohort}")
    print(f"Cascade scores: {result.cascade_scores}")
    print(f"Cascade results: {result.cascade_results}")


if __name__ == "__main__":
    main()
