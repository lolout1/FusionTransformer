"""CPU fallback implementations for preprocessing."""

import torch
import numpy as np
from typing import Optional, Literal


def sliding_window_cpu(
    data: torch.Tensor,
    window_size: int,
    stride: int,
    class_aware: bool = False,
    labels: Optional[torch.Tensor] = None,
    fall_stride: int = 16,
    adl_stride: int = 64
) -> torch.Tensor:
    """CPU sliding window implementation."""
    if data.dim() == 2:
        data = data.unsqueeze(0)

    B, T, C = data.shape
    if T < window_size:
        raise ValueError(f"Sequence length {T} < window size {window_size}")

    windows_list = []

    for i in range(B):
        if class_aware and labels is not None:
            s = fall_stride if labels[i] == 1 else adl_stride
        else:
            s = stride

        n_windows = (T - window_size) // s + 1
        indices = torch.arange(n_windows) * s

        for idx in indices:
            windows_list.append(data[i, idx:idx + window_size, :])

    if not windows_list:
        return torch.empty(0, window_size, C, dtype=data.dtype, device=data.device)

    return torch.stack(windows_list, dim=0)


def normalize_cpu(
    data: torch.Tensor,
    mode: Literal['zscore', 'minmax'] = 'zscore',
    dim: int = -2,
    eps: float = 1e-8,
    per_channel: bool = True
) -> torch.Tensor:
    """CPU normalization implementation."""
    if mode == 'zscore':
        if per_channel:
            mean = data.mean(dim=dim, keepdim=True)
            std = data.std(dim=dim, keepdim=True)
        else:
            mean = data.mean()
            std = data.std()
        return (data - mean) / (std + eps)

    elif mode == 'minmax':
        if per_channel:
            min_val = data.min(dim=dim, keepdim=True).values
            max_val = data.max(dim=dim, keepdim=True).values
        else:
            min_val = data.min()
            max_val = data.max()
        return (data - min_val) / (max_val - min_val + eps)

    return data


def fir_filter_cpu(
    data: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    zero_phase: bool = True
) -> torch.Tensor:
    """CPU FIR filter using scipy when available."""
    try:
        from scipy.signal import filtfilt, lfilter
        was_tensor = True
        device = data.device
        dtype = data.dtype

        data_np = data.detach().cpu().numpy()
        b_np = b.detach().cpu().numpy()
        a_np = a.detach().cpu().numpy()

        if zero_phase:
            if data_np.shape[-2] >= 3 * max(len(b_np), len(a_np)):
                filtered = filtfilt(b_np, a_np, data_np, axis=-2)
            else:
                filtered = lfilter(b_np, a_np, data_np, axis=-2)
        else:
            filtered = lfilter(b_np, a_np, data_np, axis=-2)

        return torch.from_numpy(filtered.copy()).to(device=device, dtype=dtype)

    except ImportError:
        return _fir_filter_manual(data, b, a, zero_phase)


def _fir_filter_manual(
    data: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    zero_phase: bool = True
) -> torch.Tensor:
    """Manual FIR filter when scipy unavailable."""
    import torch.nn.functional as F

    original_shape = data.shape
    if data.dim() == 2:
        data = data.unsqueeze(0)

    B, T, C = data.shape
    data = data.permute(0, 2, 1)

    kernel = b.flip(0).view(1, 1, -1).expand(C, 1, -1)
    padding = (len(b) - 1) // 2

    filtered = F.conv1d(data, kernel, padding=padding, groups=C)

    if zero_phase:
        filtered = filtered.flip(-1)
        filtered = F.conv1d(filtered, kernel, padding=padding, groups=C)
        filtered = filtered.flip(-1)

    filtered = filtered.permute(0, 2, 1)

    if len(original_shape) == 2:
        filtered = filtered.squeeze(0)

    return filtered[..., :original_shape[-2], :]


def compute_smv_cpu(data: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """CPU Signal Vector Magnitude computation."""
    return torch.norm(data, dim=dim, keepdim=True)
