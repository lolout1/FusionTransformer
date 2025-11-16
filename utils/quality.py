"""
Gyroscope quality assessment module.

Provides timestamp-independent metrics to evaluate IMU sensor quality,
particularly for consumer-grade Android sensors with variable sampling rates.

References:
    Liu et al. (2022). "A hybrid learning-based stochastic noise eliminating
    method with attention-Conv-LSTM network for low-cost MEMS gyroscope."
    Frontiers in Neurorobotics, DOI: 10.3389/fnbot.2022.993936

    IEEE Std 952-2020. "Standard Specification Format Guide and Test Procedure
    for Single-Axis Interferometric Fiber Optic Gyros"
"""

import numpy as np
from typing import Dict, Tuple


def assess_gyro_quality(gyro_data: np.ndarray,
                        method: str = 'simple',
                        threshold: float = 1.0) -> Tuple[bool, Dict[str, float]]:
    """
    Assess gyroscope signal quality using timestamp-independent metrics.

    This function evaluates whether gyroscope data is acceptable for training
    by analyzing signal-to-noise ratio and other statistical properties. All
    metrics are computed from signal values only, making them robust to
    timestamp drift and variable sampling rates common in Android devices.

    Args:
        gyro_data: Gyroscope measurements, shape (N, 3) for [gx, gy, gz] in rad/s
        method: Quality assessment method
            'simple' - SNR threshold only (fast)
            'multi_criteria' - Multiple quality checks (thorough)
        threshold: Minimum acceptable SNR (signal-to-noise ratio)
            Based on empirical analysis showing SNR < 1.0 indicates
            noise-dominated signal for consumer IMUs

    Returns:
        is_acceptable: Boolean indicating if gyroscope quality is acceptable
        metrics: Dictionary containing quality metrics:
            - snr: Signal-to-noise ratio (mean/std of magnitude)
            - mean_magnitude: Average angular velocity magnitude
            - max_bias: Maximum bias across axes
            - variance_ratio: Normalized variance indicator

    Example:
        >>> gyro = np.random.randn(1000, 3) * 0.5  # Noisy gyro
        >>> is_good, metrics = assess_gyro_quality(gyro, threshold=1.0)
        >>> print(f"SNR: {metrics['snr']:.2f}, Acceptable: {is_good}")

    Notes:
        - SNR > 1.0: Signal exceeds noise (acceptable)
        - SNR < 1.0: Noise dominates signal (reject)
        - Consumer-grade IMUs typically have SNR 0.7-1.5
        - Tactical-grade IMUs typically have SNR > 3.0
    """
    # Calculate magnitude (L2 norm) - single aggregate metric
    magnitude = np.linalg.norm(gyro_data, axis=1)

    # Compute quality metrics (all timestamp-independent)
    metrics = {
        # Primary metric: signal-to-noise ratio
        'snr': magnitude.mean() / (magnitude.std() + 1e-8),

        # Average motion level (should be reasonable for wrist-worn)
        'mean_magnitude': magnitude.mean(),

        # Bias per axis (drift indicator)
        'max_bias': np.abs(gyro_data.mean(axis=0)).max(),

        # Variance relative to signal (noise indicator)
        'variance_ratio': magnitude.var() / (magnitude.mean()**2 + 1e-8)
    }

    # Apply quality criteria based on method
    if method == 'simple':
        # Fast check: SNR threshold only
        is_acceptable = metrics['snr'] >= threshold

    elif method == 'multi_criteria':
        # Thorough check: multiple quality criteria
        # Based on empirical analysis of consumer IMU characteristics
        is_acceptable = (
            metrics['snr'] >= threshold and           # Sufficient signal-to-noise
            metrics['mean_magnitude'] < 5.0 and       # Not excessive drift/motion
            metrics['max_bias'] < 0.4 and             # Reasonable bias
            metrics['variance_ratio'] < 2.0           # Variance within bounds
        )
    else:
        raise ValueError(f"Unknown quality assessment method: {method}. "
                        f"Use 'simple' or 'multi_criteria'")

    return is_acceptable, metrics


def compute_quality_statistics(quality_results: Dict[int, Dict]) -> Dict[str, float]:
    """
    Aggregate quality metrics across multiple trials or subjects.

    Args:
        quality_results: Dictionary mapping trial_id to quality metrics
            Example: {0: {'snr': 1.2, 'mean_magnitude': 0.8, ...}, ...}

    Returns:
        aggregate_stats: Summary statistics across all trials
            - mean_snr, std_snr: Average and std dev of SNR
            - acceptance_rate: Fraction of trials passing quality check
            - mean_magnitude: Average motion level across trials
    """
    snr_values = [metrics['snr'] for metrics in quality_results.values()]
    acceptance = [metrics.get('is_acceptable', False)
                  for metrics in quality_results.values()]
    magnitudes = [metrics['mean_magnitude']
                  for metrics in quality_results.values()]

    aggregate_stats = {
        'mean_snr': np.mean(snr_values),
        'std_snr': np.std(snr_values),
        'acceptance_rate': np.mean(acceptance),
        'mean_magnitude': np.mean(magnitudes),
        'num_trials': len(quality_results)
    }

    return aggregate_stats


def detect_static_periods(gyro_data: np.ndarray,
                         threshold: float = 0.1,
                         min_duration: int = 10) -> np.ndarray:
    """
    Detect static (non-moving) periods where gyroscope should read near zero.

    Useful for identifying sensor drift and bias. During static periods,
    a good gyroscope should report values close to zero. Persistent non-zero
    values indicate bias drift or poor calibration.

    Args:
        gyro_data: Gyroscope measurements, shape (N, 3)
        threshold: Maximum magnitude for "static" classification (rad/s)
        min_duration: Minimum consecutive samples to qualify as static period

    Returns:
        is_static: Boolean array, shape (N,), True for static periods

    Example:
        >>> static_mask = detect_static_periods(gyro_data, threshold=0.1)
        >>> static_bias = gyro_data[static_mask].mean(axis=0)
        >>> print(f"Bias during static periods: {static_bias}")
    """
    magnitude = np.linalg.norm(gyro_data, axis=1)
    is_below_threshold = magnitude < threshold

    # Find continuous static periods
    is_static = np.zeros_like(is_below_threshold)
    count = 0

    for i in range(len(is_below_threshold)):
        if is_below_threshold[i]:
            count += 1
            if count >= min_duration:
                is_static[i - min_duration + 1:i + 1] = True
        else:
            count = 0

    return is_static
