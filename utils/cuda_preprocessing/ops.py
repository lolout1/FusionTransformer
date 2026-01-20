"""PyTorch GPU operations for preprocessing."""

import torch
import torch.nn.functional as F
from typing import Optional, Union, Tuple, Literal
import math


def sliding_window_gpu(
    data: torch.Tensor,
    window_size: int,
    stride: int,
    class_aware: bool = False,
    labels: Optional[torch.Tensor] = None,
    fall_stride: int = 16,
    adl_stride: int = 64
) -> torch.Tensor:
    """Extract sliding windows using unfold (GPU-optimized)."""
    if data.dim() == 2:
        data = data.unsqueeze(0)

    B, T, C = data.shape
    if T < window_size:
        raise ValueError(f"Sequence length {T} < window size {window_size}")

    if class_aware and labels is not None:
        windows_list = []
        for i in range(B):
            s = fall_stride if labels[i] == 1 else adl_stride
            w = data[i].unfold(0, window_size, s)
            windows_list.append(w.permute(0, 2, 1))
        return torch.cat(windows_list, dim=0)

    windows = data.unfold(1, window_size, stride)
    return windows.permute(0, 1, 3, 2).reshape(-1, window_size, C)


def normalize_gpu(
    data: torch.Tensor,
    mode: Literal['zscore', 'minmax'] = 'zscore',
    dim: int = -2,
    eps: float = 1e-8,
    per_channel: bool = True
) -> torch.Tensor:
    """GPU-accelerated normalization using parallel reduction."""
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


def butterworth_coeffs(
    filter_type: str,
    cutoff: Union[float, Tuple[float, float]],
    fs: float,
    order: int,
    device: torch.device = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Butterworth filter coefficients."""
    nyq = 0.5 * fs

    if filter_type == 'lowpass':
        Wn = cutoff / nyq
    elif filter_type == 'highpass':
        Wn = cutoff / nyq
    elif filter_type == 'bandpass':
        Wn = (cutoff[0] / nyq, cutoff[1] / nyq)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    try:
        from scipy.signal import butter
        b, a = butter(order, Wn, btype=filter_type)
        b = torch.tensor(b, dtype=torch.float32, device=device)
        a = torch.tensor(a, dtype=torch.float32, device=device)
    except ImportError:
        b = _butterworth_manual(order, Wn, filter_type, device)
        a = torch.ones(1, device=device)

    return b, a


def _butterworth_manual(order: int, Wn: float, btype: str, device: torch.device) -> torch.Tensor:
    """Manual Butterworth FIR approximation when scipy unavailable."""
    n_taps = 2 * order + 1
    h = torch.zeros(n_taps, device=device)
    center = n_taps // 2

    for i in range(n_taps):
        if i == center:
            h[i] = 2 * Wn if btype == 'lowpass' else 1 - 2 * Wn
        else:
            n = i - center
            if btype == 'lowpass':
                h[i] = math.sin(2 * math.pi * Wn * n) / (math.pi * n)
            else:
                h[i] = -math.sin(2 * math.pi * Wn * n) / (math.pi * n)

    window = torch.hann_window(n_taps, device=device)
    return h * window


def fir_filter_gpu(
    data: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    zero_phase: bool = True
) -> torch.Tensor:
    """Apply FIR filter using conv1d (GPU-optimized)."""
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


def compute_smv_gpu(data: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute Signal Vector Magnitude on GPU."""
    return torch.norm(data, dim=dim, keepdim=True)


def iir_filter_gpu(
    data: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    zero_phase: bool = True
) -> torch.Tensor:
    """IIR filter implementation (sequential, less GPU-efficient)."""
    if data.dim() == 2:
        data = data.unsqueeze(0)

    B, T, C = data.shape
    output = torch.zeros_like(data)

    nb, na = len(b), len(a)

    for t in range(T):
        for i in range(min(nb, t + 1)):
            output[:, t, :] += b[i] * data[:, t - i, :]
        for i in range(1, min(na, t + 1)):
            output[:, t, :] -= a[i] * output[:, t - i, :]

    if zero_phase:
        output_rev = torch.zeros_like(data)
        data_rev = output.flip(1)
        for t in range(T):
            for i in range(min(nb, t + 1)):
                output_rev[:, t, :] += b[i] * data_rev[:, t - i, :]
            for i in range(1, min(na, t + 1)):
                output_rev[:, t, :] -= a[i] * output_rev[:, t - i, :]
        output = output_rev.flip(1)

    return output.squeeze(0) if B == 1 else output
