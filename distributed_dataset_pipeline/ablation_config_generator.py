#!/usr/bin/env python3
"""YAML configuration generator for ablation studies."""

import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from copy import deepcopy
from itertools import product


# Dataset-specific settings
DATASET_CONFIG = {
    'upfall': {
        'name': 'UP-FALL',
        'base_config': 'config/upfall/kalman_optimal.yaml',
        'sampling_rate': 18.0,
        'subjects': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],
        'validation_subjects': [15, 16],
        'num_folds': 15,  # 17 - 2 val subjects
        'window_sizes': [64, 96, 128, 160],  # Longer windows needed for 18Hz
        'filter_fs': 18.0,
    },
    'wedafall': {
        'name': 'WEDA-FALL',
        'base_config': 'config/wedafall/kalman_optimal.yaml',
        'sampling_rate': 50.0,
        'subjects': [1,2,3,4,5,6,7,8,9,10,11,12,13,14],
        'validation_subjects': [13, 14],
        'num_folds': 12,  # 14 - 2 val subjects
        'window_sizes': [64, 96, 128, 192],  # Higher rate allows shorter windows
        'filter_fs': 50.0,
    },
}

# Stride configurations
STRIDE_CONFIGS = {
    'aggressive': {'fall_stride': 8, 'adl_stride': 32, 'ratio': '1:4', 'description': 'Severe imbalance handling'},
    'standard': {'fall_stride': 16, 'adl_stride': 64, 'ratio': '1:4', 'description': 'Moderate imbalance'},
    'moderate': {'fall_stride': 24, 'adl_stride': 96, 'ratio': '1:4', 'description': 'Light oversampling'},
    'equal': {'fall_stride': 32, 'adl_stride': 32, 'ratio': '1:1', 'description': 'Natural distribution baseline'},
}

# Model variants - Kalman fusion models (primary ablation)
MODEL_VARIANTS = {
    'kalman_conv1d_linear': {
        'model': 'Models.encoder_ablation.KalmanConv1dLinear',
        'kalman': True,
        'channels': 7,
        'description': 'Kalman + Conv1D(acc) + Linear(ori)',
        'extra_model_args': {},
    },
    'kalman_conv1d_conv1d': {
        'model': 'Models.encoder_ablation.KalmanConv1dConv1d',
        'kalman': True,
        'channels': 7,
        'description': 'Kalman + Conv1D(acc) + Conv1D(ori)',
        'extra_model_args': {},
    },
}

# Raw model variants (separate study - not included in default ablation)
RAW_MODEL_VARIANTS = {
    'raw_dual_stream': {
        'model': 'Models.dual_stream_baseline.DualStreamBaseline',
        'kalman': False,
        'channels': 7,  # smv + acc(3) + gyro(3) = 7
        'description': 'Raw IMU (smv+acc+gyro) dual-stream',
        'extra_model_args': {
            'acc_in_channels': 4,  # smv + ax + ay + az
            'gyro_in_channels': 3,  # gx + gy + gz
            'acc_dim': 48,
            'gyro_dim': 16,
        },
    },
}

# Embedding dimensions
EMBED_DIMS = [48, 64]

# Default model variants for ablation (Kalman only)
DEFAULT_MODEL_VARIANTS = list(MODEL_VARIANTS.keys())


@dataclass
class AblationConfig:
    """Represents a single ablation configuration."""
    dataset: str
    window_size: int
    stride_name: str
    fall_stride: int
    adl_stride: int
    embed_dim: int
    model_name: str
    model_class: str
    use_kalman: bool
    channels: int

    @property
    def name(self) -> str:
        """Generate unique experiment name."""
        return f"ws{self.window_size}_fs{self.fall_stride}_as{self.adl_stride}_ed{self.embed_dim}_{self.model_name}"

    @property
    def short_name(self) -> str:
        """Shorter name for tables."""
        return f"W{self.window_size}_S{self.stride_name[:3]}_E{self.embed_dim}_{self.model_name[:6]}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'dataset': self.dataset,
            'window_size': self.window_size,
            'stride_name': self.stride_name,
            'fall_stride': self.fall_stride,
            'adl_stride': self.adl_stride,
            'embed_dim': self.embed_dim,
            'model_name': self.model_name,
            'model_class': self.model_class,
            'use_kalman': self.use_kalman,
            'channels': self.channels,
            'name': self.name,
        }


class AblationConfigGenerator:
    """Generates configurations for hyperparameter ablation studies."""

    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize the config generator.

        Args:
            project_root: Path to project root. Auto-detected if None.
        """
        if project_root is None:
            self.project_root = Path(__file__).parent.parent
        else:
            self.project_root = Path(project_root)

    def load_base_config(self, dataset: str) -> Dict:
        """Load base configuration for a dataset."""
        config_path = self.project_root / DATASET_CONFIG[dataset]['base_config']

        if not config_path.exists():
            raise FileNotFoundError(f"Base config not found: {config_path}")

        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _get_model_config(self, model_name: str) -> Dict:
        """Get model configuration from MODEL_VARIANTS or RAW_MODEL_VARIANTS."""
        if model_name in MODEL_VARIANTS:
            return MODEL_VARIANTS[model_name]
        elif model_name in RAW_MODEL_VARIANTS:
            return RAW_MODEL_VARIANTS[model_name]
        else:
            available = list(MODEL_VARIANTS.keys()) + list(RAW_MODEL_VARIANTS.keys())
            raise ValueError(f"Unknown model: {model_name}. Available: {available}")

    def generate_config(
        self,
        dataset: str,
        window_size: int,
        stride_name: str,
        embed_dim: int,
        model_name: str,
    ) -> Tuple[AblationConfig, Dict]:
        """
        Generate a single ablation configuration.

        Args:
            dataset: 'upfall' or 'wedafall'
            window_size: Window size in samples
            stride_name: Stride configuration name
            embed_dim: Embedding dimension
            model_name: Model variant name

        Returns:
            Tuple of (AblationConfig metadata, YAML config dict)
        """
        # Validate inputs
        if dataset not in DATASET_CONFIG:
            raise ValueError(f"Unknown dataset: {dataset}. Available: {list(DATASET_CONFIG.keys())}")
        if stride_name not in STRIDE_CONFIGS:
            raise ValueError(f"Unknown stride: {stride_name}. Available: {list(STRIDE_CONFIGS.keys())}")

        # Load base config (deep copy to avoid mutation)
        config = deepcopy(self.load_base_config(dataset))

        # Get stride settings
        stride_cfg = STRIDE_CONFIGS[stride_name]
        fall_stride = stride_cfg['fall_stride']
        adl_stride = stride_cfg['adl_stride']

        # Get model settings
        model_cfg = self._get_model_config(model_name)

        # Create ablation config metadata
        ablation_cfg = AblationConfig(
            dataset=dataset,
            window_size=window_size,
            stride_name=stride_name,
            fall_stride=fall_stride,
            adl_stride=adl_stride,
            embed_dim=embed_dim,
            model_name=model_name,
            model_class=model_cfg['model'],
            use_kalman=model_cfg['kalman'],
            channels=model_cfg['channels'],
        )

        # Modify config - model
        config['model'] = model_cfg['model']

        # Model args - core parameters
        config['model_args']['imu_frames'] = window_size
        config['model_args']['acc_frames'] = window_size
        config['model_args']['embed_dim'] = embed_dim
        config['model_args']['imu_channels'] = model_cfg['channels']
        config['model_args']['acc_coords'] = model_cfg['channels']

        # Model args - apply extra model-specific args
        extra_args = model_cfg.get('extra_model_args', {})
        for key, value in extra_args.items():
            config['model_args'][key] = value

        # Dataset args - windowing
        config['dataset_args']['max_length'] = window_size
        config['dataset_args']['fall_stride'] = fall_stride
        config['dataset_args']['adl_stride'] = adl_stride

        # Dataset args - Kalman fusion
        config['dataset_args']['enable_kalman_fusion'] = model_cfg['kalman']

        # For raw model, adjust Kalman-specific settings
        if not model_cfg['kalman']:
            # Raw model still uses SMV but not Kalman fusion
            config['dataset_args']['kalman_include_smv'] = True  # Keep SMV for raw
            config['dataset_args']['kalman_output_format'] = None

        return ablation_cfg, config

    def generate_all_configs(
        self,
        dataset: str,
        window_sizes: Optional[List[int]] = None,
        stride_names: Optional[List[str]] = None,
        embed_dims: Optional[List[int]] = None,
        model_names: Optional[List[str]] = None,
    ) -> List[Tuple[AblationConfig, Dict]]:
        """
        Generate all configurations for a dataset.

        Args:
            dataset: 'upfall' or 'wedafall'
            window_sizes: List of window sizes (default: dataset-specific)
            stride_names: List of stride configs (default: all)
            embed_dims: List of embed dims (default: [48, 64])
            model_names: List of model variants (default: all)

        Returns:
            List of (AblationConfig, config_dict) tuples
        """
        # Use defaults if not specified
        if window_sizes is None:
            window_sizes = DATASET_CONFIG[dataset]['window_sizes']
        if stride_names is None:
            stride_names = list(STRIDE_CONFIGS.keys())
        if embed_dims is None:
            embed_dims = EMBED_DIMS
        if model_names is None:
            model_names = DEFAULT_MODEL_VARIANTS

        configs = []

        # Generate all combinations
        for ws, stride, ed, model in product(window_sizes, stride_names, embed_dims, model_names):
            ablation_cfg, config = self.generate_config(
                dataset=dataset,
                window_size=ws,
                stride_name=stride,
                embed_dim=ed,
                model_name=model,
            )
            configs.append((ablation_cfg, config))

        return configs

    def save_configs(
        self,
        configs: List[Tuple[AblationConfig, Dict]],
        output_dir: Path,
    ) -> List[Path]:
        """
        Save configurations to YAML files.

        Args:
            configs: List of (AblationConfig, config_dict) tuples
            output_dir: Directory to save configs

        Returns:
            List of saved config file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []

        for ablation_cfg, config in configs:
            # Create dataset subdirectory
            dataset_dir = output_dir / ablation_cfg.dataset
            dataset_dir.mkdir(parents=True, exist_ok=True)

            # Save config
            config_path = dataset_dir / f"{ablation_cfg.name}.yaml"

            # Add comment header
            header = f"""# Hyperparameter Ablation Configuration
# Dataset: {ablation_cfg.dataset.upper()}
# Window: {ablation_cfg.window_size} samples
# Stride: {ablation_cfg.stride_name} (fall={ablation_cfg.fall_stride}, adl={ablation_cfg.adl_stride})
# Embed Dim: {ablation_cfg.embed_dim}
# Model: {ablation_cfg.model_name}
# Kalman: {ablation_cfg.use_kalman}
# Channels: {ablation_cfg.channels}
#
# Auto-generated for ablation study

"""
            with open(config_path, 'w') as f:
                f.write(header)
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            saved_paths.append(config_path)

        return saved_paths

    def get_experiment_matrix(
        self,
        datasets: Optional[List[str]] = None,
        window_sizes: Optional[Dict[str, List[int]]] = None,
        stride_names: Optional[List[str]] = None,
        embed_dims: Optional[List[int]] = None,
        model_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Get the full experiment matrix summary.

        Returns:
            Dictionary with experiment matrix details
        """
        if datasets is None:
            datasets = ['upfall', 'wedafall']
        if stride_names is None:
            stride_names = list(STRIDE_CONFIGS.keys())
        if embed_dims is None:
            embed_dims = EMBED_DIMS
        if model_names is None:
            model_names = DEFAULT_MODEL_VARIANTS

        total_experiments = 0
        total_folds = 0

        per_dataset = {}

        for dataset in datasets:
            ws = window_sizes.get(dataset, DATASET_CONFIG[dataset]['window_sizes']) if window_sizes else DATASET_CONFIG[dataset]['window_sizes']
            n_configs = len(ws) * len(stride_names) * len(embed_dims) * len(model_names)
            n_folds = DATASET_CONFIG[dataset]['num_folds']
            n_fold_runs = n_configs * n_folds

            per_dataset[dataset] = {
                'name': DATASET_CONFIG[dataset]['name'],
                'window_sizes': ws,
                'num_configs': n_configs,
                'num_folds_per_config': n_folds,
                'total_fold_runs': n_fold_runs,
                'sampling_rate': DATASET_CONFIG[dataset]['sampling_rate'],
            }

            total_experiments += n_configs
            total_folds += n_fold_runs

        return {
            'datasets': datasets,
            'stride_configs': stride_names,
            'embed_dims': embed_dims,
            'model_variants': model_names,
            'per_dataset': per_dataset,
            'total_experiments': total_experiments,
            'total_fold_runs': total_folds,
        }

    def print_experiment_summary(
        self,
        datasets: Optional[List[str]] = None,
        window_sizes: Optional[Dict[str, List[int]]] = None,
        stride_names: Optional[List[str]] = None,
        embed_dims: Optional[List[int]] = None,
        model_names: Optional[List[str]] = None,
    ) -> None:
        """Print a summary of the experiment matrix."""
        matrix = self.get_experiment_matrix(
            datasets=datasets,
            window_sizes=window_sizes,
            stride_names=stride_names,
            embed_dims=embed_dims,
            model_names=model_names,
        )

        print("=" * 70)
        print("HYPERPARAMETER ABLATION EXPERIMENT MATRIX")
        print("=" * 70)

        print(f"\nDimensions:")
        print(f"  Stride Configs: {matrix['stride_configs']}")
        print(f"  Embed Dims: {matrix['embed_dims']}")
        print(f"  Model Variants: {matrix['model_variants']}")

        print(f"\nPer-Dataset Summary:")
        for dataset, info in matrix['per_dataset'].items():
            print(f"\n  {info['name']} ({info['sampling_rate']} Hz):")
            print(f"    Window Sizes: {info['window_sizes']}")
            print(f"    Configurations: {info['num_configs']}")
            print(f"    Folds/Config: {info['num_folds_per_config']}")
            print(f"    Total Fold-Runs: {info['total_fold_runs']}")

        print(f"\nTotal:")
        print(f"  Experiments: {matrix['total_experiments']}")
        print(f"  Fold-Runs: {matrix['total_fold_runs']}")

        # Runtime estimate
        avg_fold_time_min = 2.5
        parallel_exps = 2
        gpus_per_exp = 3
        estimated_hours = (matrix['total_fold_runs'] * avg_fold_time_min) / (parallel_exps * gpus_per_exp * 60)
        print(f"\nEstimated Runtime: {estimated_hours:.1f} hours")
        print(f"  (assuming ~{avg_fold_time_min} min/fold, {parallel_exps} parallel experiments, {gpus_per_exp} GPUs each)")
        print("=" * 70)


def main():
    """Demo the config generator."""
    generator = AblationConfigGenerator()

    # Print experiment summary
    generator.print_experiment_summary()

    # Generate configs for a subset (for demonstration)
    print("\n\nGenerating sample configs for UP-FALL...")
    configs = generator.generate_all_configs(
        dataset='upfall',
        window_sizes=[128],  # Just one window size for demo
        stride_names=['standard'],  # Just one stride for demo
        embed_dims=[48],  # Just one embed dim
        model_names=['kalman_conv1d_linear'],  # Just one model
    )

    print(f"Generated {len(configs)} configs")

    for ablation_cfg, config in configs:
        print(f"\n  {ablation_cfg.name}")
        print(f"    Model: {config['model']}")
        print(f"    Window: {config['model_args']['imu_frames']}")
        print(f"    Embed: {config['model_args']['embed_dim']}")
        print(f"    Stride: {config['dataset_args']['fall_stride']}/{config['dataset_args']['adl_stride']}")


if __name__ == '__main__':
    main()
