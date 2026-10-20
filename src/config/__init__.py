"""CrossModalLab configuration package.

Contains portable path resolution and shared experiment configuration.
This package must not import ULVO-specific APIs or require CUDA.
"""

from src.config.paths import (
    REPO_ROOT,
    apply_default_cache_environment,
    get_artifacts_dir,
    get_datasets_dir,
    get_hf_home,
    get_smolvlm2_snapshot_dir,
    get_temp_dir,
)
from src.config.experiment import (
    ExperimentConfig,
    get_experiment_config,
    require_gpu_capabilities,
)

__all__ = [
    "REPO_ROOT",
    "apply_default_cache_environment",
    "get_artifacts_dir",
    "get_datasets_dir",
    "get_hf_home",
    "get_smolvlm2_snapshot_dir",
    "get_temp_dir",
    "ExperimentConfig",
    "get_experiment_config",
    "require_gpu_capabilities",
]
