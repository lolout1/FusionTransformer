"""
Knowledge Distillation module for skeleton→IMU transfer learning.

Supports alignment-free KD methods robust to missing skeleton timestamps
and irregular IMU sampling.

Components:
- losses: KD loss functions (Embedding, Gram, COMODO)
- resampler: EventTokenResampler for irregular timestamps
- skeleton_encoder: Teacher model for skeleton data
- trainer: KDTrainer for training loop
- data_loader: SmartFallMM data loading with timestamps
- stress_test: Robustness evaluation utilities
- analysis: Dataset characterization tools
"""

__version__ = '0.1.0'

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == 'EmbeddingKDLoss':
        from kd.losses import EmbeddingKDLoss
        return EmbeddingKDLoss
    elif name == 'GramKDLoss':
        from kd.losses import GramKDLoss
        return GramKDLoss
    elif name == 'COMODOLoss':
        from kd.losses import COMODOLoss
        return COMODOLoss
    elif name == 'CombinedKDLoss':
        from kd.losses import CombinedKDLoss
        return CombinedKDLoss
    elif name == 'EventTokenResampler':
        from kd.resampler import EventTokenResampler
        return EventTokenResampler
    elif name == 'TimestampAwareStudent':
        from kd.resampler import TimestampAwareStudent
        return TimestampAwareStudent
    elif name == 'SkeletonTransformer':
        from kd.skeleton_encoder import SkeletonTransformer
        return SkeletonTransformer
    elif name == 'JointSkeletonIMUTeacher':
        from kd.skeleton_encoder import JointSkeletonIMUTeacher
        return JointSkeletonIMUTeacher
    elif name == 'KDTrainer':
        from kd.trainer import KDTrainer
        return KDTrainer
    elif name == 'create_kd_trainer':
        from kd.trainer import create_kd_trainer
        return create_kd_trainer
    elif name == 'StressTestEvaluator':
        from kd.stress_test import StressTestEvaluator
        return StressTestEvaluator
    # Data loading
    elif name == 'TrialMatcher':
        from kd.data_loader import TrialMatcher
        return TrialMatcher
    elif name == 'KDDataset':
        from kd.data_loader import KDDataset
        return KDDataset
    elif name == 'WindowedKDDataset':
        from kd.data_loader import WindowedKDDataset
        return WindowedKDDataset
    elif name == 'create_kd_dataloaders':
        from kd.data_loader import create_kd_dataloaders
        return create_kd_dataloaders
    elif name == 'prepare_loso_fold':
        from kd.data_loader import prepare_loso_fold
        return prepare_loso_fold
    raise AttributeError(f"module 'kd' has no attribute '{name}'")


__all__ = [
    # Losses
    'EmbeddingKDLoss',
    'GramKDLoss',
    'COMODOLoss',
    'CombinedKDLoss',
    # Models
    'EventTokenResampler',
    'TimestampAwareStudent',
    'SkeletonTransformer',
    'JointSkeletonIMUTeacher',
    # Training
    'KDTrainer',
    'create_kd_trainer',
    # Data Loading
    'TrialMatcher',
    'KDDataset',
    'WindowedKDDataset',
    'create_kd_dataloaders',
    'prepare_loso_fold',
    # Evaluation
    'StressTestEvaluator',
]
