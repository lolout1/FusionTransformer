"""Integration helpers for training pipeline."""

import torch
from typing import Dict, Any, Optional
from . import CUDAPreprocessor


class PreprocessingModule(torch.nn.Module):
    """Preprocessing as nn.Module for seamless integration in training."""

    def __init__(
        self,
        device: str = 'cpu',
        normalize_mode: str = 'zscore',
        normalize_modalities: str = 'acc_only',
        apply_filter: bool = False,
        acc_filter_cutoff: float = 5.5,
        gyro_filter_cutoff: float = 0.5,
        fs: float = 30.0,
        include_smv: bool = True
    ):
        super().__init__()
        self.preprocessor = CUDAPreprocessor(device=device)
        self.normalize_mode = normalize_mode
        self.normalize_modalities = normalize_modalities
        self.apply_filter = apply_filter
        self.acc_filter_cutoff = acc_filter_cutoff
        self.gyro_filter_cutoff = gyro_filter_cutoff
        self.fs = fs
        self.include_smv = include_smv

    def forward(
        self,
        acc_data: torch.Tensor,
        gyro_data: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.preprocessor.preprocess_batch(
            acc_data,
            gyro_data,
            normalize_mode=self.normalize_mode,
            normalize_modalities=self.normalize_modalities,
            apply_filter=self.apply_filter,
            acc_filter_cutoff=self.acc_filter_cutoff,
            gyro_filter_cutoff=self.gyro_filter_cutoff,
            fs=self.fs,
            include_smv=self.include_smv
        )


def create_preprocessing_module(config: Dict[str, Any], device: str = 'cpu') -> PreprocessingModule:
    """Create preprocessing module from config dict."""
    return PreprocessingModule(
        device=device,
        normalize_mode='zscore' if config.get('enable_normalization', True) else 'none',
        normalize_modalities=config.get('normalize_modalities', 'acc_only'),
        apply_filter=config.get('enable_filtering', False),
        acc_filter_cutoff=config.get('acc_filter_cutoff', 5.5),
        gyro_filter_cutoff=config.get('gyro_filter_cutoff', 0.5),
        fs=config.get('filter_fs', 30.0),
        include_smv=config.get('include_smv', True)
    )


def apply_gpu_preprocessing(
    batch: Dict[str, torch.Tensor],
    preprocessor: CUDAPreprocessor,
    config: Dict[str, Any]
) -> torch.Tensor:
    """Apply GPU preprocessing to a batch from DataLoader."""
    acc_data = batch.get('accelerometer')
    gyro_data = batch.get('gyroscope')

    if acc_data is None:
        raise ValueError("Batch must contain 'accelerometer' key")

    return preprocessor.preprocess_batch(
        acc_data,
        gyro_data,
        normalize_mode='zscore' if config.get('enable_normalization', True) else 'none',
        normalize_modalities=config.get('normalize_modalities', 'acc_only'),
        apply_filter=config.get('enable_filtering', False),
        acc_filter_cutoff=config.get('acc_filter_cutoff', 5.5),
        gyro_filter_cutoff=config.get('gyro_filter_cutoff', 0.5),
        fs=config.get('filter_fs', 30.0),
        include_smv=config.get('include_smv', True)
    )
