"""
Shared utilities for external fall detection datasets (UP-FALL, WEDA-FALL).

Provides common functions for:
- Sliding window extraction
- Kalman fusion integration
- Normalization
- Label mapping
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Sliding Window
# =============================================================================

def sliding_window_numpy(
    data: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    stride: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract sliding windows from continuous data.

    Args:
        data: (T, C) time series data
        labels: (T,) per-timestep labels
        window_size: Window length in samples
        stride: Step size between windows

    Returns:
        windows: (N, window_size, C) windowed data
        window_labels: (N,) label for each window (last timestep)
    """
    T = data.shape[0]
    if T < window_size:
        return np.array([]), np.array([])

    # Calculate number of windows
    n_windows = (T - window_size) // stride + 1

    # Create window indices
    start_indices = np.arange(n_windows) * stride
    window_indices = start_indices[:, None] + np.arange(window_size)

    # Extract windows
    windows = data[window_indices]
    window_labels = labels[start_indices + window_size - 1]  # Last timestep label

    return windows, window_labels


def sliding_window_class_aware(
    data: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    fall_stride: int,
    adl_stride: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract windows with class-aware stride (more overlap for falls).

    For fall detection, we want more fall windows to balance the dataset.

    Args:
        data: (T, C) time series data
        labels: (T,) per-timestep labels (1=fall, 0=ADL)
        window_size: Window length
        fall_stride: Stride for fall segments (smaller = more overlap)
        adl_stride: Stride for ADL segments (larger = less overlap)

    Returns:
        windows: (N, window_size, C)
        window_labels: (N,)
    """
    T = data.shape[0]
    if T < window_size:
        return np.array([]), np.array([])

    windows_list = []
    labels_list = []

    i = 0
    while i + window_size <= T:
        window = data[i:i + window_size]
        window_label = labels[i + window_size - 1]  # Last timestep

        windows_list.append(window)
        labels_list.append(window_label)

        # Class-aware stride
        if window_label == 1:  # Fall
            i += fall_stride
        else:  # ADL
            i += adl_stride

    if not windows_list:
        return np.array([]), np.array([])

    return np.array(windows_list), np.array(labels_list)


# =============================================================================
# Kalman Fusion Integration
# =============================================================================

def get_kalman_config_for_rate(sampling_rate: float) -> dict:
    """
    Get Kalman filter parameters scaled for sampling rate.

    Process noise Q scales linearly with sampling period (dt).
    Measurement noise R is sensor-dependent, not time-dependent.

    Reference: SmartFallMM tuned at 32Hz.

    Args:
        sampling_rate: Data sampling rate in Hz

    Returns:
        Kalman configuration dictionary
    """
    # Reference parameters (SmartFallMM, 32Hz)
    REF_RATE = 32.0
    REF_Q_ORI = 0.005
    REF_Q_RATE = 0.01

    # Scale Q with sampling period
    scale = REF_RATE / sampling_rate

    return {
        'filter_fs': sampling_rate,
        'kalman_filter_type': 'linear',
        'kalman_output_format': 'euler',
        'kalman_include_smv': True,
        'kalman_include_uncertainty': False,
        'kalman_include_innovation': False,
        'kalman_exclude_yaw': False,      # Include yaw by default (7ch output)
        'kalman_include_raw_gyro': False, # Exclude raw gyro by default
        'kalman_Q_orientation': REF_Q_ORI * scale,
        'kalman_Q_rate': REF_Q_RATE * scale,
        'kalman_R_acc': 0.05,  # Sensor noise (unchanged)
        'kalman_R_gyro': 0.1,  # Sensor noise (unchanged)
        'kalman_R_multiplier': 1.0,  # R scale factor (1.0 = no change, >1 = more smoothing)

        # Adaptive Kalman estimation (disabled by default for backward compatibility)
        'adaptive_kalman_enabled': False,
        'adaptive_mode': 'nis',         # 'nis', 'signal', 'hybrid', or 'none'
        'adaptive_alpha': 0.5,          # Sensitivity: 0=no adaptation, 1=full
        'adaptive_scale_min': 0.3,      # Min R scale (max trust in measurements)
        'adaptive_scale_max': 3.0,      # Max R scale (min trust in measurements)
        'adaptive_ema_alpha': 0.1,      # EMA smoothing for scale factor
        'adaptive_warmup_samples': 10,  # Samples before adaptation kicks in
    }


def apply_kalman_fusion(
    acc: np.ndarray,
    gyro: np.ndarray,
    config: dict
) -> np.ndarray:
    """
    Apply Kalman fusion to accelerometer and gyroscope data.

    Uses the existing Kalman preprocessing pipeline.

    Args:
        acc: (T, 3) accelerometer data [ax, ay, az] in m/s² or g
        gyro: (T, 3) gyroscope data [gx, gy, gz] in rad/s
        config: Kalman configuration dict with keys:
            - kalman_include_smv: bool (default True)
            - kalman_exclude_yaw: bool (default False)
            - kalman_include_raw_gyro: bool (default False)
            - kalman_orientation_only: bool (default False)

    Returns:
        fused: (T, C) Kalman features where C depends on config:
            7ch:  [smv, ax, ay, az, roll, pitch, yaw] (default)
            6ch:  [smv, ax, ay, az, roll, pitch] (exclude_yaw=True)
            9ch:  [smv, ax, ay, az, roll, pitch, gx, gy, gz] (exclude_yaw + raw_gyro)
            3ch:  [roll, pitch, yaw] (orientation_only=True)
            2ch:  [roll, pitch] (orientation_only + exclude_yaw)
    """
    from utils.kalman.features import build_kalman_features

    # build_kalman_features handles all config options
    features = build_kalman_features(acc, gyro, config)

    return features


def compute_smv(acc: np.ndarray) -> np.ndarray:
    """
    Compute Signal Magnitude Vector from accelerometer data.

    Args:
        acc: (T, 3) or (N, T, 3) accelerometer data

    Returns:
        smv: (T, 1) or (N, T, 1) signal magnitude vector (zero-mean)
    """
    if acc.ndim == 2:
        smv = np.sqrt(np.sum(acc ** 2, axis=1, keepdims=True))
        smv = smv - smv.mean()
    elif acc.ndim == 3:
        smv = np.sqrt(np.sum(acc ** 2, axis=2, keepdims=True))
        smv = smv - smv.mean(axis=1, keepdims=True)
    else:
        raise ValueError(f"Expected 2D or 3D array, got {acc.ndim}D")
    return smv


# =============================================================================
# Normalization
# =============================================================================

def normalize_data(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    method: str = 'standard'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Normalize data using train statistics.

    Args:
        train: (N, T, C) training data
        val: (N, T, C) validation data
        test: (N, T, C) test data
        method: 'standard' (z-score) or 'minmax'

    Returns:
        Normalized train, val, test arrays and fitted scaler
    """
    N, T, C = train.shape

    # Flatten for scaling
    train_flat = train.reshape(-1, C)

    if method == 'standard':
        scaler = StandardScaler()
    else:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()

    scaler.fit(train_flat)

    train_norm = scaler.transform(train_flat).reshape(N, T, C)
    val_norm = scaler.transform(val.reshape(-1, C)).reshape(val.shape)
    test_norm = scaler.transform(test.reshape(-1, C)).reshape(test.shape)

    return train_norm, val_norm, test_norm, scaler


def normalize_modality(
    data: np.ndarray,
    modality: str,
    normalize_mode: str = 'acc_only'
) -> np.ndarray:
    """
    Per-modality normalization following SmartFallMM pattern.

    Args:
        data: (N, T, C) data array
        modality: 'accelerometer' or 'orientation'
        normalize_mode: 'all', 'acc_only', 'none'

    Returns:
        Normalized data
    """
    if normalize_mode == 'none':
        return data

    if normalize_mode == 'acc_only' and modality != 'accelerometer':
        return data

    N, T, C = data.shape
    data_flat = data.reshape(-1, C)

    scaler = StandardScaler()
    data_norm = scaler.fit_transform(data_flat).reshape(N, T, C)

    return data_norm


# =============================================================================
# Label Maps
# =============================================================================

# UP-FALL activity labels (from paper/dataset)
UPFALL_LABEL_MAP = {
    1: 1,  # Falling forward using hands
    2: 1,  # Falling forward using knees
    3: 1,  # Falling backwards
    4: 1,  # Falling sideward
    5: 1,  # Falling sitting in empty chair
    6: 0,  # Walking
    7: 0,  # Standing
    8: 0,  # Sitting
    9: 0,  # Picking up an object
    10: 0, # Jumping
    11: 0, # Laying
}

# WEDA-FALL labels (determined by folder name)
# F* folders = Fall (1), D* folders = ADL (0)
def wedafall_label_from_folder(folder_name: str) -> int:
    """Get label from WEDA-FALL activity folder name."""
    return 1 if folder_name.startswith('F') else 0


# =============================================================================
# Data Validation
# =============================================================================

def validate_window_data(
    acc: np.ndarray,
    labels: np.ndarray,
    name: str = "data"
) -> None:
    """
    Validate windowed data format.

    Args:
        acc: (N, T, C) windowed accelerometer data
        labels: (N,) labels
        name: Dataset name for error messages

    Raises:
        ValueError if validation fails
    """
    if acc.ndim != 3:
        raise ValueError(f"{name}: acc should be 3D (N,T,C), got {acc.ndim}D")

    if labels.ndim != 1:
        raise ValueError(f"{name}: labels should be 1D, got {labels.ndim}D")

    if acc.shape[0] != labels.shape[0]:
        raise ValueError(
            f"{name}: acc samples ({acc.shape[0]}) != labels ({labels.shape[0]})"
        )

    unique_labels = np.unique(labels)
    if not np.all(np.isin(unique_labels, [0, 1])):
        raise ValueError(f"{name}: labels should be 0/1, got {unique_labels}")


def print_data_summary(
    name: str,
    acc: np.ndarray,
    labels: np.ndarray
) -> None:
    """Print summary statistics for windowed data."""
    n_samples = acc.shape[0]
    n_falls = np.sum(labels == 1)
    n_adls = np.sum(labels == 0)
    fall_ratio = n_falls / n_samples if n_samples > 0 else 0

    print(f"  {name}:")
    print(f"    Samples: {n_samples} (Falls: {n_falls}, ADLs: {n_adls})")
    print(f"    Fall ratio: {fall_ratio:.2%}")
    print(f"    Shape: {acc.shape}")
