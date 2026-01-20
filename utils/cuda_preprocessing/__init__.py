"""
CUDA-accelerated preprocessing kernels for IMU data.
Hybrid approach: PyTorch GPU ops + custom CUDA kernels with auto-fallback.
"""

import torch
from typing import Optional, Union, Literal

_CUDA_AVAILABLE = torch.cuda.is_available() if hasattr(torch, 'cuda') else False

try:
    from . import cuda_ops
    _CUDA_KERNELS_AVAILABLE = True
except ImportError:
    _CUDA_KERNELS_AVAILABLE = False

from .ops import (
    sliding_window_gpu, normalize_gpu, fir_filter_gpu,
    compute_smv_gpu, butterworth_coeffs,
)
from .fallback import (
    sliding_window_cpu, normalize_cpu, fir_filter_cpu, compute_smv_cpu,
)


class CUDAPreprocessor:
    """Unified interface for GPU-accelerated preprocessing with CPU fallback."""

    def __init__(
        self,
        device: Union[str, torch.device] = 'cpu',
        use_custom_kernels: bool = False,
        dtype: torch.dtype = torch.float32
    ):
        if device == 'auto':
            self.device = torch.device('cuda' if _CUDA_AVAILABLE else 'cpu')
        else:
            self.device = torch.device(device)
        self.use_custom_kernels = use_custom_kernels and _CUDA_KERNELS_AVAILABLE
        self.dtype = dtype
        self._filter_cache = {}

    @property
    def is_cuda(self) -> bool:
        return self.device.type == 'cuda'

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device != self.device:
            return tensor.to(device=self.device, dtype=self.dtype)
        return tensor

    def sliding_window(
        self,
        data: torch.Tensor,
        window_size: int,
        stride: int,
        class_aware: bool = False,
        labels: Optional[torch.Tensor] = None,
        fall_stride: int = 16,
        adl_stride: int = 64
    ) -> torch.Tensor:
        data = self._to_device(data)
        fn = sliding_window_gpu if self.is_cuda else sliding_window_cpu
        return fn(data, window_size, stride, class_aware, labels, fall_stride, adl_stride)

    def normalize(
        self,
        data: torch.Tensor,
        mode: Literal['zscore', 'minmax', 'none'] = 'zscore',
        dim: int = -2,
        eps: float = 1e-8,
        per_channel: bool = True
    ) -> torch.Tensor:
        if mode == 'none':
            return data
        data = self._to_device(data)
        fn = normalize_gpu if self.is_cuda else normalize_cpu
        return fn(data, mode, dim, eps, per_channel)

    def apply_fir_filter(
        self,
        data: torch.Tensor,
        filter_type: Literal['lowpass', 'highpass', 'bandpass'] = 'lowpass',
        cutoff: Union[float, tuple] = 5.0,
        fs: float = 30.0,
        order: int = 4,
        zero_phase: bool = True
    ) -> torch.Tensor:
        data = self._to_device(data)
        cache_key = (filter_type, cutoff, fs, order)
        if cache_key not in self._filter_cache:
            self._filter_cache[cache_key] = butterworth_coeffs(
                filter_type, cutoff, fs, order, device=self.device
            )
        b, a = self._filter_cache[cache_key]
        fn = fir_filter_gpu if self.is_cuda else fir_filter_cpu
        return fn(data, b, a, zero_phase)

    def compute_smv(self, data: torch.Tensor, dim: int = -1) -> torch.Tensor:
        data = self._to_device(data)
        fn = compute_smv_gpu if self.is_cuda else compute_smv_cpu
        return fn(data, dim)

    def compute_smv_zero_mean(self, data: torch.Tensor, dim: int = -1) -> torch.Tensor:
        data = self._to_device(data)
        mean = data.mean(dim=-2, keepdim=True)
        return self.compute_smv(data - mean, dim)

    def preprocess_batch(
        self,
        acc_data: torch.Tensor,
        gyro_data: Optional[torch.Tensor] = None,
        normalize_mode: Literal['zscore', 'minmax', 'none'] = 'zscore',
        normalize_modalities: Literal['all', 'acc_only', 'gyro_only', 'none'] = 'acc_only',
        apply_filter: bool = False,
        acc_filter_cutoff: float = 5.5,
        gyro_filter_cutoff: float = 0.5,
        fs: float = 30.0,
        include_smv: bool = True
    ) -> torch.Tensor:
        acc_data = self._to_device(acc_data)
        if gyro_data is not None:
            gyro_data = self._to_device(gyro_data)

        if apply_filter:
            acc_data = self.apply_fir_filter(acc_data, 'lowpass', acc_filter_cutoff, fs)
            if gyro_data is not None:
                gyro_data = self.apply_fir_filter(gyro_data, 'highpass', gyro_filter_cutoff, fs)

        if normalize_mode != 'none':
            if normalize_modalities in ['all', 'acc_only']:
                acc_data = self.normalize(acc_data, normalize_mode)
            if gyro_data is not None and normalize_modalities in ['all', 'gyro_only']:
                gyro_data = self.normalize(gyro_data, normalize_mode)

        if include_smv:
            smv = self.compute_smv_zero_mean(acc_data)
            acc_data = torch.cat([smv, acc_data], dim=-1)

        if gyro_data is not None:
            return torch.cat([acc_data, gyro_data], dim=-1)
        return acc_data

    def clear_cache(self):
        self._filter_cache.clear()


def create_preprocessor(device: str = 'cpu', **kwargs) -> CUDAPreprocessor:
    return CUDAPreprocessor(device=device, **kwargs)


__all__ = [
    'CUDAPreprocessor', 'create_preprocessor',
    'sliding_window_gpu', 'sliding_window_cpu',
    'normalize_gpu', 'normalize_cpu',
    'fir_filter_gpu', 'fir_filter_cpu',
    'compute_smv_gpu', 'compute_smv_cpu',
    'butterworth_coeffs',
]
