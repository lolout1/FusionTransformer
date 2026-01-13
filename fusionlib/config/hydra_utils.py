"""
Hydra utilities for FusionTransformer.

Bridge between Hydra configs and the existing argparse-based system.
"""

from typing import Any, Dict, Optional
from pathlib import Path
import argparse


def hydra_to_namespace(cfg: Dict[str, Any], parser: argparse.ArgumentParser) -> argparse.Namespace:
    """
    Convert Hydra config dict to argparse.Namespace.

    Flattens nested config to match existing argument structure.

    Args:
        cfg: OmegaConf DictConfig (converted to dict)
        parser: ArgumentParser with existing defaults

    Returns:
        Namespace compatible with Trainer
    """
    # Start with parser defaults
    namespace = parser.parse_args([])

    # Flatten and merge config
    flat = flatten_config(cfg)

    for key, value in flat.items():
        if hasattr(namespace, key):
            setattr(namespace, key, value)

    return namespace


def flatten_config(cfg: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """
    Flatten nested config dict.

    Examples:
        {'model': {'args': {'dim': 64}}} -> {'model_args': {'dim': 64}}
        {'training': {'num_epoch': 80}} -> {'num_epoch': 80}
    """
    result = {}

    for key, value in cfg.items():
        full_key = f"{prefix}{key}" if prefix else key

        if isinstance(value, dict):
            # Handle special nested keys
            if key == 'model_args':
                result['model_args'] = value
            elif key == 'dataset_args':
                result['dataset_args'] = value
            elif key == 'training_args':
                result['training_args'] = value
            elif key in ('train_feeder_args', 'val_feeder_args', 'test_feeder_args'):
                result[key] = value
            else:
                # Recursively flatten
                result.update(flatten_config(value, ""))
        else:
            result[full_key] = value

    return result


def config_to_yaml_str(cfg: Dict[str, Any]) -> str:
    """Convert config dict to YAML string for saving."""
    import yaml
    return yaml.dump(cfg, default_flow_style=False, sort_keys=False)


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load YAML config file."""
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configs with later configs taking precedence.

    Args:
        *configs: Config dicts to merge

    Returns:
        Merged config dict
    """
    result = {}
    for cfg in configs:
        if cfg:
            _deep_merge(result, cfg)
    return result


def _deep_merge(base: Dict, override: Dict) -> None:
    """Deep merge override into base in-place."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
