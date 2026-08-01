"""Standard, configurable execution pipeline for the intrusion-detection project."""

from .config import PipelineConfig
from .orchestrator import PipelineRunResult, StandardPipeline

__all__ = ["PipelineConfig", "PipelineRunResult", "StandardPipeline"]
