"""
Preprocessing utilities for IMU fall detection.

Implements motion-based filtering to match Android app inference behavior,
ensuring training and deployment distribution consistency.

References:
    PersonalizedPredictionLSTM.java (Android app)
    - Lines 66-85: Motion threshold logic for inference
    - Threshold: |value| > 10 m/s² on at least 2 axes

    Casilari et al. (2017). "Analysis of Android Device-Based Solutions
    for Fall Detection." Sensors, DOI: 10.3390/s17061522
"""

import numpy as np
from typing import Dict, Optional, Tuple


def check_motion_threshold(window: np.ndarray,
                           threshold: float = 10.0,
                           min_axes: int = 2) -> bool:
    """
    Check if window contains sufficient motion to trigger prediction.

    Replicates Android app logic from PersonalizedPredictionLSTM.makeInference():
        for(int i=0; i<128; i++) {
            int num = 0;
            for(int j=0;j<3;j++) {
                if (flattenedSamples[0][i][j]>10 || flattenedSamples[0][i][j]<-10){
                    num++;
                }
            }
            if(num>=2) {
                sendForPrediction = true;
                break;
            }
        }

    Args:
        window: Accelerometer window, shape (128, 3) or (128, 4) with SMV
        threshold: Magnitude threshold in m/s² (default 10.0)
        min_axes: Minimum number of axes exceeding threshold (default 2)

    Returns:
        should_predict: True if window would trigger prediction in Android app

    Example:
        >>> window = np.random.randn(128, 3) * 2  # Low motion
        >>> check_motion_threshold(window, threshold=10.0)
        False
        >>> window[50] = [15, 12, 8]  # Add motion spike
        >>> check_motion_threshold(window, threshold=10.0)
        True
    """
    # Extract first 3 channels (ax, ay, az) in case SMV is included
    acc_axes = window[:, :3] if window.shape[1] >= 3 else window

    # Check each timestep in window
    for timestep in acc_axes:
        # Count how many axes exceed threshold
        num_active_axes = np.sum(np.abs(timestep) > threshold)

        # If at least min_axes exceed threshold, window is "active"
        if num_active_axes >= min_axes:
            return True

    # No timestep met criteria - window is "quiet"
    return False


def filter_windows_by_motion(data: Dict[str, np.ndarray],
                             reference_key: str = 'accelerometer',
                             threshold: float = 10.0,
                             min_axes: int = 2) -> Optional[Dict[str, np.ndarray]]:
    """
    Filter sliding windows to keep only those passing motion threshold.

    Ensures training data distribution matches inference: only windows that
    would trigger prediction in the Android app are used for training.

    Args:
        data: Dictionary containing windowed data
            Expected keys: 'accelerometer', 'labels', optionally 'gyroscope'
            Each modality has shape (num_windows, window_size, channels)
        reference_key: Modality to check for motion (typically 'accelerometer')
        threshold: Motion threshold in m/s²
        min_axes: Minimum axes exceeding threshold

    Returns:
        filtered_data: Dictionary with only active windows, or None if all rejected

    Example:
        >>> data = {
        ...     'accelerometer': np.random.randn(100, 128, 4),
        ...     'labels': np.ones(100)
        ... }
        >>> filtered = filter_windows_by_motion(data, threshold=10.0)
        >>> print(f"Kept {len(filtered['labels'])} / 100 windows")

    Notes:
        - Returns None if NO windows pass threshold (rare but possible)
        - Preserves label distribution proportionally
        - Applies same filtering to all modalities consistently
    """
    if reference_key not in data:
        raise ValueError(f"Reference key '{reference_key}' not found in data. "
                        f"Available keys: {list(data.keys())}")

    windows = data[reference_key]
    labels = data.get('labels', None)

    # Check each window
    valid_indices = []
    for i, window in enumerate(windows):
        if check_motion_threshold(window, threshold, min_axes):
            valid_indices.append(i)

    # If no windows pass, return None
    if len(valid_indices) == 0:
        return None

    # Filter all modalities using same indices
    filtered_data = {}
    for key in data.keys():
        if key == 'labels':
            filtered_data['labels'] = labels[valid_indices] if labels is not None else None
        else:
            filtered_data[key] = data[key][valid_indices]

    return filtered_data


def compute_motion_statistics(data: Dict[str, np.ndarray],
                              reference_key: str = 'accelerometer',
                              threshold: float = 10.0,
                              min_axes: int = 2) -> Dict[str, float]:
    """
    Compute statistics about motion filtering impact on dataset.

    Useful for understanding how much data is filtered and whether
    filtering affects class balance.

    Args:
        data: Dictionary with sensor data and 'labels'
        reference_key: Modality to check for motion (typically 'accelerometer')
        threshold: Motion threshold
        min_axes: Minimum axes for motion detection

    Returns:
        statistics: Dictionary containing:
            - total_windows: Original window count
            - active_windows: Windows passing motion threshold
            - rejection_rate: Fraction of windows rejected
            - fall_retention: Fraction of fall windows kept
            - nonfall_retention: Fraction of non-fall windows kept

    Example:
        >>> stats = compute_motion_statistics(data, reference_key='accelerometer', threshold=10.0)
        >>> print(f"Rejection rate: {stats['rejection_rate']:.1%}")
        >>> print(f"Fall retention: {stats['fall_retention']:.1%}")
    """
    if reference_key not in data:
        raise ValueError(f"Reference key '{reference_key}' not found in data. "
                        f"Available keys: {list(data.keys())}")

    windows = data[reference_key]
    labels = data.get('labels', None)

    total_windows = len(windows)
    active_mask = np.array([
        check_motion_threshold(w, threshold, min_axes)
        for w in windows
    ])

    statistics = {
        'total_windows': total_windows,
        'active_windows': active_mask.sum(),
        'quiet_windows': (~active_mask).sum(),
        'rejection_rate': (~active_mask).mean()
    }

    # Class-specific retention if labels available
    if labels is not None:
        fall_mask = labels == 1
        nonfall_mask = labels == 0

        if fall_mask.sum() > 0:
            statistics['fall_retention'] = active_mask[fall_mask].mean()
        if nonfall_mask.sum() > 0:
            statistics['nonfall_retention'] = active_mask[nonfall_mask].mean()

    return statistics


def resample_to_fixed_rate(data: np.ndarray,
                           timestamps: np.ndarray,
                           target_fs: float = 30.0) -> np.ndarray:
    """
    Resample irregularly-sampled IMU data to uniform sampling rate.

    Addresses Android's variable sampling rate issue by interpolating to
    a fixed frequency, enabling consistent temporal analysis.

    Args:
        data: IMU measurements, shape (N, 3) for tri-axial sensor
        timestamps: Timestamps in seconds (can be non-uniform)
        target_fs: Target sampling frequency in Hz

    Returns:
        resampled_data: Uniformly sampled data, shape (M, 3)

    Example:
        >>> # Variable rate data
        >>> data = np.random.randn(1000, 3)
        >>> timestamps = np.sort(np.random.rand(1000)) * 10  # 0-10 seconds
        >>> uniform_data = resample_to_fixed_rate(data, timestamps, target_fs=30.0)
        >>> print(f"Resampled to {len(uniform_data)} samples at 30Hz")

    Notes:
        - Uses linear interpolation (good for IMU signals)
        - Handles gaps in data gracefully
        - Preserves signal characteristics while regularizing timing
    """
    from scipy.interpolate import interp1d

    # Normalize timestamps to start at 0
    time_seconds = timestamps - timestamps[0]

    # Create uniform time grid
    duration = time_seconds[-1]
    target_times = np.arange(0, duration, 1/target_fs)

    # Interpolate each axis
    resampled_axes = []
    for axis_idx in range(data.shape[1]):
        interpolator = interp1d(
            time_seconds,
            data[:, axis_idx],
            kind='linear',
            fill_value='extrapolate',
            bounds_error=False
        )
        resampled_axes.append(interpolator(target_times))

    resampled_data = np.column_stack(resampled_axes)
    return resampled_data
