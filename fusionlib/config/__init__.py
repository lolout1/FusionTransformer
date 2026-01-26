"""
Configuration system for FusionTransformer.

Core modules:
- loader: YAML loading with inheritance
- ablation: Sweep and experiment generation
- hydra_utils: Hydra/argparse conversion
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
    resolve_model_class,
    ConfigLoadError,
    ConfigValidationError,
)

from .ablation import (
    AblationSpec,
    AblationRunner,
    generate_configs,
    run_ablation,
)

from .registry import (
    ARCHITECTURES,
    DATASETS,
    INPUT_TYPES,
    STRIDES,
    get_architecture,
    get_dataset,
    get_window_size,
)

__all__ = [
    # Loader
    'load_config',
    'validate_config',
    'save_config',
    'resolve_model_class',
    'ConfigLoadError',
    'ConfigValidationError',
    # Hydra utils
    'hydra_to_namespace',
    'flatten_config',
    'config_to_yaml_str',
    'load_yaml_config',
    'merge_configs',
    # Ablation
    'AblationSpec',
    'AblationRunner',
    'generate_configs',
    'run_ablation',
    # Registry
    'ARCHITECTURES',
    'DATASETS',
    'INPUT_TYPES',
    'STRIDES',
    'get_architecture',
    'get_dataset',
    'get_window_size',
]
