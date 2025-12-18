"""
Experiment configuration management.

Defines hyperparameter configs, experiment definitions, and model mappings
for the modular fall detection experiment framework.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path
import yaml


@dataclass
class HyperparameterConfig:
    """Single hyperparameter configuration."""
    name: str
    lr: float
    weight_decay: float
    dropout: float
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'base_lr': self.lr,
            'weight_decay': self.weight_decay,
            'dropout': self.dropout,
        }


# Define 3 standard HP configs for all experiments
HP_CONFIGS = [
    HyperparameterConfig(
        name='hp_baseline',
        lr=0.001,
        weight_decay=0.001,
        dropout=0.5,
        description='Baseline configuration'
    ),
    HyperparameterConfig(
        name='hp_more_reg',
        lr=0.001,
        weight_decay=0.01,
        dropout=0.6,
        description='More regularization to counter overfitting'
    ),
    HyperparameterConfig(
        name='hp_slow_converge',
        lr=0.0005,
        weight_decay=0.01,
        dropout=0.5,
        description='Slower convergence with high regularization'
    ),
]


@dataclass
class ModelConfig:
    """Single model configuration for an experiment."""
    name: str                          # e.g., 'transformer_acc_smv'
    model_class: str                   # e.g., 'Models.transformer.TransModel'
    model_args: Dict[str, Any]         # Model constructor arguments
    dataset_args: Dict[str, Any]       # Dataset configuration
    description: str


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str                          # e.g., 'modality_tests'
    models: List[ModelConfig]
    hp_configs: List[HyperparameterConfig] = field(default_factory=lambda: HP_CONFIGS)
    population: str = 'young_old'      # 'young' or 'young_old'
    num_folds: int = 22
    num_epochs: int = 80
    batch_size: int = 64
    loss_type: str = 'bce'             # 'bce' or 'focal'

    # Subject configuration (22-fold LOSO)
    # Young subjects (29-63) + Old subjects (2-26) for ADL augmentation
    all_subjects: List[int] = field(default_factory=lambda: [
        # Old subjects (ADL only - train only, cannot be test subjects)
        2, 3, 4, 5, 6, 7, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 25, 26,
        # Young subjects (have both falls and ADLs)
        29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 43, 44, 45, 46, 48,
        49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63
    ])
    validation_subjects: List[int] = field(default_factory=lambda: [48, 57])
    # Old subjects (2-26) are train-only because they have no fall data
    # Young train-only subjects that won't be tested
    train_only_subjects: List[int] = field(default_factory=lambda: [
        # Old subjects (ADL only)
        2, 3, 4, 5, 6, 7, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 25, 26,
        # Young train-only
        29, 30, 32, 35, 39, 59
    ])

    # Graph/save options
    save_graphs: bool = True
    save_model_weights: bool = False   # DISABLED BY DEFAULT
    top_k_best: int = 3
    top_k_worst: int = 3
    top_k_least_overfit: int = 2
    top_k_lowest_val_loss: int = 2

    @property
    def test_subjects(self) -> List[int]:
        """Compute test subjects: all - validation - train_only."""
        return [s for s in self.all_subjects
                if s not in self.validation_subjects
                and s not in self.train_only_subjects]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'population': self.population,
            'num_folds': self.num_folds,
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'loss_type': self.loss_type,
            'all_subjects': self.all_subjects,
            'validation_subjects': self.validation_subjects,
            'train_only_subjects': self.train_only_subjects,
            'test_subjects': self.test_subjects,
            'models': [m.name for m in self.models],
            'hp_configs': [hp.name for hp in self.hp_configs],
        }


# Base dataset args template for SmartFallMM
BASE_DATASET_ARGS = {
    'mode': 'sliding_window',
    'max_length': 128,
    'task': 'fd',
    'sensors': ['watch'],
    'stride': 32,
    'use_skeleton': False,
    'enable_normalization': True,
    'normalize_modalities': 'all',
    'enable_filtering': False,
    'enable_motion_filtering': False,
    'enable_class_aware_stride': True,
    'fall_stride': 16,
    'adl_stride': 64,
    'convert_gyro_to_rad': True,
    'enable_simple_truncation': True,
    'max_truncation_diff': 50,
    'discard_mismatched_modalities': False,
    'length_sensitive_modalities': ['accelerometer', 'gyroscope'],
    # Debug mode: prints verbose skip reasons for each trial
    'debug': True,  # Enable during investigation, disable for production
}


def get_dataset_args(
    modalities: List[str],
    include_smv: bool = True,
    include_gyro_mag: bool = False,
    age_group: List[str] = None
) -> Dict[str, Any]:
    """
    Generate dataset args with specified modalities and options.

    Args:
        modalities: List of modalities, e.g., ['accelerometer'] or ['accelerometer', 'gyroscope']
        include_smv: Whether to include SMV (Signal Vector Magnitude) for accelerometer
        include_gyro_mag: Whether to include gyroscope magnitude
        age_group: Age groups to include, default ['young', 'old']

    Returns:
        Dictionary of dataset arguments
    """
    if age_group is None:
        age_group = ['young', 'old']

    args = BASE_DATASET_ARGS.copy()
    args['modalities'] = modalities
    args['age_group'] = age_group
    args['include_smv'] = include_smv
    args['include_gyro_mag'] = include_gyro_mag

    return args


if __name__ == "__main__":
    # Test configuration
    print("=" * 60)
    print("Experiment Configuration Test")
    print("=" * 60)

    print(f"\nHP Configs ({len(HP_CONFIGS)}):")
    for hp in HP_CONFIGS:
        print(f"  - {hp.name}: lr={hp.lr}, weight_decay={hp.weight_decay}, dropout={hp.dropout}")

    # Test model config
    model = ModelConfig(
        name='test_model',
        model_class='Models.transformer.TransModel',
        model_args={'acc_coords': 4},
        dataset_args=get_dataset_args(['accelerometer'], include_smv=True),
        description='Test model'
    )

    # Test experiment config
    exp = ExperimentConfig(
        name='test_experiment',
        models=[model]
    )

    print(f"\nTest Experiment:")
    print(f"  Name: {exp.name}")
    print(f"  Test subjects ({len(exp.test_subjects)}): {exp.test_subjects}")
    print(f"  Validation subjects: {exp.validation_subjects}")
    print(f"  Train-only subjects: {exp.train_only_subjects}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
