"""JIT loader for CUDA preprocessing kernels."""

import os
import torch

_cuda_ops = None


def _load_cuda_ops():
    global _cuda_ops
    if _cuda_ops is not None:
        return _cuda_ops

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    from torch.utils.cpp_extension import load

    root = os.path.dirname(os.path.abspath(__file__))
    source = os.path.join(root, 'kernels', 'preprocessing_kernels.cu')

    _cuda_ops = load(
        name='cuda_preprocessing_ops',
        sources=[source],
        extra_cuda_cflags=['-O3', '--use_fast_math'],
        verbose=False
    )
    return _cuda_ops


def sliding_window(input: torch.Tensor, window_size: int, stride: int) -> torch.Tensor:
    return _load_cuda_ops().sliding_window(input, window_size, stride)


def normalize(input: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return _load_cuda_ops().normalize(input, eps)


def fir_filter(input: torch.Tensor, kernel: torch.Tensor, zero_phase: bool = True) -> torch.Tensor:
    return _load_cuda_ops().fir_filter(input, kernel, zero_phase)


def compute_smv(input: torch.Tensor) -> torch.Tensor:
    return _load_cuda_ops().compute_smv(input)
