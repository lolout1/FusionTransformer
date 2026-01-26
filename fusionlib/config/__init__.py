"""
Configuration system with inheritance, validation, and ablation sweeps.

See docs/CONFIG_SYSTEM.md for usage guide.
"""

from .hydra_utils import (
    hydra_to_namespace,
    flatten_config,
    config_to_yaml_str,
    load_yaml_config,
    merge_configs,
)
from .loader import (
    load_config,
    validate_config,
    save_config,
    ConfigLoadError,
    ConfigValidationError,
)

# Optional pydantic schema imports
try:
    from .schema import (
        ExperimentConfig,
        ModelConfig,
        DatasetConfig,
        PreprocessingConfig,
        TrainingConfig,
        ModuleToggles,
        ExecutionConfig,
    )
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    ExperimentConfig = None
    ModelConfig = None
    DatasetConfig = None
    PreprocessingConfig = None
    TrainingConfig = None
    ModuleToggles = None
    ExecutionConfig = None

from .ablation import (
    AblationSpec,
    generate_configs,
    run_ablation,
)

__all__ = [
    # Hydra utils
    "hydra_to_namespace",
    "flatten_config",
    "config_to_yaml_str",
    "load_yaml_config",
    "merge_configs",
    # Loader
    "load_config",
    "validate_config",
    "save_config",
    "ConfigLoadError",
    "ConfigValidationError",
    # Schema (may be None if pydantic not available)
    "ExperimentConfig",
    "ModelConfig",
    "DatasetConfig",
    "PreprocessingConfig",
    "TrainingConfig",
    "ModuleToggles",
    "ExecutionConfig",
    # Ablation
    "AblationSpec",
    "generate_configs",
    "run_ablation",
]
