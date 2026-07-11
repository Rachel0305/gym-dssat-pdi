"""Configuration-driven, provenance-aware DQN benchmark framework.

The package is additive: legacy training/evaluation scripts remain untouched
and are connected through adapters by the higher-level runner.
"""

from .checkpoint_manager import CheckpointInfo, CheckpointManager, ResumePlan
from .config_loader import canonical_station_code, deep_merge, load_config
from .experiment_registry import (
    ExperimentRegistry,
    ExperimentSpec,
    ManifestStatus,
    expand_experiments,
)
from .provenance import collect_provenance, compute_config_hash, sha256_file
from .result_registry import ResultRegistry, discover_legacy_results
from .validators import ConfigValidationError, raise_for_invalid_config, validate_config

__all__ = [
    "CheckpointInfo",
    "CheckpointManager",
    "ConfigValidationError",
    "ExperimentRegistry",
    "ExperimentSpec",
    "ManifestStatus",
    "ResultRegistry",
    "ResumePlan",
    "canonical_station_code",
    "collect_provenance",
    "compute_config_hash",
    "deep_merge",
    "discover_legacy_results",
    "expand_experiments",
    "load_config",
    "raise_for_invalid_config",
    "sha256_file",
    "validate_config",
]

