"""
Timestamp-Based IMU Alignment Module for SmartFallMM Dataset
============================================================

This module addresses the critical flaw in sample-count-based alignment:
Android IMU API delivers accelerometer and gyroscope at DIFFERENT and VARIABLE
sampling rates, even when timestamps are perfectly synchronized.

Example: S45A10T11
- Accelerometer: 789 samples, effective rate ~87 Hz
- Gyroscope:     467 samples, effective rate ~52 Hz
- Difference:    322 samples (41%) → INCORRECTLY DISCARDED by old logic
- BUT timestamps show: Start offset 1ms, End offset 2ms (synchronized!)

Solution: Parse timestamps, find overlap region, interpolate to common grid.

Author: SmartFallMM Research Team
Date: November 2025
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION AND RESULT DATACLASSES
# =============================================================================

@dataclass
class AlignmentConfig:
    """Configuration for timestamp-based IMU alignment."""

    target_rate: float = 30.0
    """Target sampling rate in Hz. 30Hz is standard for training (128 samples = 4.27s window)."""

    alignment_method: str = 'linear'
    """Interpolation method: 'linear', 'cubic', 'nearest'. Linear recommended for IMU."""

    min_overlap_ratio: float = 0.8
    """Minimum overlap between acc and gyro recordings. Discard if below this threshold."""

    max_time_gap_ms: float = 1000.0
    """Maximum allowed start/end time difference in milliseconds. Indicates different recordings."""

    min_output_samples: int = 64
    """Minimum output samples after alignment. Discard if below (too short for windowing)."""

    length_threshold: int = 10
    """If sample count diff <= this, use as-is without interpolation (backward compat)."""

    handle_duplicates: str = 'offset'
    """How to handle duplicate timestamps: 'offset' (add epsilon), 'drop', 'average'."""

    duplicate_epsilon_ms: float = 0.001
    """Epsilon to add for deduplication (in ms)."""

    # NEW: Conservative interpolation controls
    max_duration_ratio: float = 1.2
    """Maximum allowed ratio between acc/gyro durations for interpolation (e.g., 1.2 = 20% difference max).
    If duration ratio exceeds this AND sample_diff > length_threshold, discard instead of interpolate.
    This prevents interpolation when timestamps have drifted significantly."""

    max_rate_divergence: float = 0.3
    """Maximum allowed relative difference in sampling rates for interpolation.
    E.g., 0.3 means rates must be within 30% of each other.
    If rates diverge more AND sample_diff > length_threshold, discard."""

    def __post_init__(self):
        valid_methods = ['linear', 'cubic', 'nearest', 'quadratic']
        if self.alignment_method not in valid_methods:
            raise ValueError(f"alignment_method must be one of {valid_methods}")


@dataclass
class AlignmentResult:
    """Result of alignment operation for a single trial."""

    success: bool
    """Whether alignment was successful."""

    action: str
    """Action taken: 'use_as_is', 'aligned', 'discarded'."""

    reason: str
    """Human-readable reason for the action."""

    aligned_acc: Optional[np.ndarray] = None
    """Aligned accelerometer data (N, 3) or None if discarded."""

    aligned_gyro: Optional[np.ndarray] = None
    """Aligned gyroscope data (N, 3) or None if discarded."""

    aligned_timestamps: Optional[np.ndarray] = None
    """Common timestamp grid (N,) in milliseconds."""

    output_samples: int = 0
    """Number of output samples."""

    # Input statistics
    acc_samples: int = 0
    gyro_samples: int = 0
    sample_diff: int = 0
    sample_diff_pct: float = 0.0

    # Timestamp statistics
    acc_duration_ms: float = 0.0
    gyro_duration_ms: float = 0.0
    start_offset_ms: float = 0.0
    end_offset_ms: float = 0.0
    overlap_ratio: float = 0.0
    overlap_duration_ms: float = 0.0

    # Sampling rate statistics
    acc_rate_hz: float = 0.0
    gyro_rate_hz: float = 0.0
    output_rate_hz: float = 0.0

    # Quality metrics (computed if aligned)
    has_duplicate_timestamps: bool = False
    has_large_gaps: bool = False
    max_gap_ms: float = 0.0


@dataclass
class AlignmentStats:
    """Aggregate statistics for alignment across multiple trials."""

    total_trials: int = 0
    use_as_is: int = 0
    aligned: int = 0
    discarded: int = 0

    discarded_timestamp_unsync: int = 0
    discarded_insufficient_overlap: int = 0
    discarded_too_short: int = 0
    discarded_duration_drift: int = 0  # NEW: Duration ratio exceeded
    discarded_rate_drift: int = 0      # NEW: Sampling rate divergence exceeded
    discarded_error: int = 0

    mean_interpolation_mae: float = 0.0
    mean_output_rate_hz: float = 0.0

    def update(self, result: AlignmentResult):
        """Update stats from a single result."""
        self.total_trials += 1
        if result.action == 'use_as_is':
            self.use_as_is += 1
        elif result.action == 'aligned':
            self.aligned += 1
        elif result.action == 'discarded':
            self.discarded += 1
            reason_lower = result.reason.lower()
            if 'duration ratio' in reason_lower or 'duration_ratio' in reason_lower:
                self.discarded_duration_drift += 1
            elif 'rate divergence' in reason_lower or 'sampling rate' in reason_lower:
                self.discarded_rate_drift += 1
            elif 'unsync' in reason_lower or 'offset' in reason_lower:
                self.discarded_timestamp_unsync += 1
            elif 'overlap' in reason_lower:
                self.discarded_insufficient_overlap += 1
            elif 'short' in reason_lower:
                self.discarded_too_short += 1
            else:
                self.discarded_error += 1


# =============================================================================
# TIMESTAMP PARSING
# =============================================================================

def parse_imu_csv_with_timestamps(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse IMU CSV file with timestamps.

    Expected formats:
    - Watch: "2022-07-20 15:16:00.862,-4.54,-1.43,8.93" (ISO timestamp, x, y, z)
    - Meta: "epoch_ms,time_str,elapsed_s,x,y,z" (take last 3 columns)

    Args:
        filepath: Path to CSV file

    Returns:
        timestamps_ms: (N,) array of timestamps in milliseconds since epoch
        data: (N, 3) array of x, y, z values

    Raises:
        ValueError: If file format is unrecognized or parsing fails
    """
    try:
        # Read raw file to inspect format
        df = pd.read_csv(filepath, header=None)
        n_cols = df.shape[1]

        if n_cols == 4:
            # Watch format: timestamp, x, y, z
            df.columns = ['timestamp', 'x', 'y', 'z']

            # Parse ISO timestamp
            df['ts'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')

            # Handle parsing failures
            if df['ts'].isna().any():
                # Try alternative formats
                for fmt in ['%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S.%f']:
                    try:
                        df['ts'] = pd.to_datetime(df['timestamp'], format=fmt)
                        break
                    except:
                        continue

            if df['ts'].isna().all():
                raise ValueError(f"Could not parse timestamps in {filepath}")

            timestamps_ms = df['ts'].astype(np.int64) // 10**6  # ns to ms
            data = df[['x', 'y', 'z']].values.astype(np.float32)

        elif n_cols == 6:
            # Meta format: epoch_ms, time_str, elapsed_s, x, y, z
            df.columns = ['epoch_ms', 'time_str', 'elapsed_s', 'x', 'y', 'z']
            timestamps_ms = df['epoch_ms'].values.astype(np.int64)
            data = df[['x', 'y', 'z']].values.astype(np.float32)

        else:
            raise ValueError(f"Unexpected column count {n_cols} in {filepath}")

        # Remove NaN rows
        valid_mask = ~(np.isnan(data).any(axis=1) | np.isnan(timestamps_ms))
        timestamps_ms = timestamps_ms[valid_mask]
        data = data[valid_mask]

        if len(timestamps_ms) == 0:
            raise ValueError(f"No valid data rows in {filepath}")

        return timestamps_ms, data

    except Exception as e:
        raise ValueError(f"Failed to parse {filepath}: {e}")


def deduplicate_timestamps(timestamps: np.ndarray,
                           epsilon: float = 0.001,
                           method: str = 'offset') -> np.ndarray:
    """
    Handle duplicate timestamps from Android sensor batching.

    Android often delivers multiple samples with identical timestamps due to
    batching. This breaks interpolation which requires strictly monotonic x values.

    Args:
        timestamps: (N,) array of timestamps (any unit)
        epsilon: Small offset to add for deduplication
        method: 'offset' (add epsilon), 'drop' (remove duplicates)

    Returns:
        Deduplicated timestamps (N,) or (M,) if dropped
    """
    t = timestamps.astype(np.float64).copy()

    if method == 'offset':
        # Add tiny offsets to make timestamps strictly increasing
        for i in range(1, len(t)):
            if t[i] <= t[i-1]:
                t[i] = t[i-1] + epsilon
        return t

    elif method == 'drop':
        # Keep only first occurrence of each timestamp
        _, unique_idx = np.unique(t, return_index=True)
        return t[np.sort(unique_idx)]

    else:
        raise ValueError(f"Unknown deduplication method: {method}")


def compute_sampling_stats(timestamps: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute sampling rate statistics from timestamps.

    Args:
        timestamps: (N,) array in milliseconds

    Returns:
        mean_rate_hz: Mean sampling rate
        std_rate_hz: Standard deviation of sampling rate
        max_gap_ms: Maximum gap between samples
    """
    if len(timestamps) < 2:
        return 0.0, 0.0, 0.0

    dt = np.diff(timestamps)
    dt_positive = dt[dt > 0]

    if len(dt_positive) == 0:
        return 0.0, 0.0, 0.0

    rates_hz = 1000.0 / dt_positive
    mean_rate = float(np.mean(rates_hz))
    std_rate = float(np.std(rates_hz))
    max_gap = float(np.max(dt))

    return mean_rate, std_rate, max_gap


# =============================================================================
# STRAY SAMPLE TRIMMING
# =============================================================================

def trim_stray_initial_samples(
    acc_times: np.ndarray,
    acc_data: np.ndarray,
    gyro_times: np.ndarray,
    gyro_data: np.ndarray,
    max_stray: int = 5,
    threshold_ms: float = 5000.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Skip initial samples that are wildly out of sync.

    Some files have 1-3 "stray" samples at the start with timestamps far from
    the other modality, but remaining data aligns perfectly.

    Example from S55A10T04:
    - ACC row 1: 11:21:11.018 ← starts here
    - GYRO row 1: 11:20:26.086 ← 45 seconds BEFORE (stray!)
    - GYRO row 2: 11:21:11.018 ← matches ACC perfectly

    Args:
        acc_times: Accelerometer timestamps (ms)
        acc_data: Accelerometer data (N, 3)
        gyro_times: Gyroscope timestamps (ms)
        gyro_data: Gyroscope data (M, 3)
        max_stray: Maximum number of samples to consider as stray (default 5)
        threshold_ms: Threshold for "out of sync" in milliseconds (default 5000)

    Returns:
        Tuple of (acc_times, acc_data, gyro_times, gyro_data, acc_skipped, gyro_skipped)
    """
    acc_skipped = 0
    gyro_skipped = 0

    if len(acc_times) == 0 or len(gyro_times) == 0:
        return acc_times, acc_data, gyro_times, gyro_data, 0, 0

    acc_start = acc_times[0]
    gyro_start = gyro_times[0]

    # Check if gyro starts way before acc
    if gyro_start < acc_start - threshold_ms:
        # Find first gyro sample that's within threshold of acc_start
        for i in range(min(max_stray, len(gyro_times))):
            if gyro_times[i] >= acc_start - threshold_ms:
                gyro_times = gyro_times[i:]
                gyro_data = gyro_data[i:]
                gyro_skipped = i
                logger.debug(f"Trimmed {i} stray gyro samples at start (offset: {acc_start - gyro_start:.0f}ms)")
                break

    # Check if acc starts way before gyro (re-check after potential gyro trim)
    if len(acc_times) > 0 and len(gyro_times) > 0:
        acc_start = acc_times[0]
        gyro_start = gyro_times[0]

        if acc_start < gyro_start - threshold_ms:
            for i in range(min(max_stray, len(acc_times))):
                if acc_times[i] >= gyro_start - threshold_ms:
                    acc_times = acc_times[i:]
                    acc_data = acc_data[i:]
                    acc_skipped = i
                    logger.debug(f"Trimmed {i} stray acc samples at start (offset: {gyro_start - acc_start:.0f}ms)")
                    break

    return acc_times, acc_data, gyro_times, gyro_data, acc_skipped, gyro_skipped


# =============================================================================
# ALIGNMENT FEASIBILITY CHECK
# =============================================================================

def check_alignment_feasibility(
    acc_times: np.ndarray,
    gyro_times: np.ndarray,
    config: AlignmentConfig
) -> Tuple[str, str]:
    """
    Determine if alignment is feasible and what action to take.

    CONSERVATIVE APPROACH: Don't default to interpolation when timestamps may have drifted.

    Decision flowchart:
    1. |start_diff| > max_time_gap_ms → DISCARD (different recordings)
    2. |end_diff| > max_time_gap_ms → DISCARD (different recordings)
    3. overlap_ratio < min_overlap_ratio → DISCARD (insufficient overlap)
    4. |sample_diff| <= length_threshold → USE_AS_IS (truncate to shorter, NO interpolation)
    5. |sample_diff| > length_threshold → Check additional constraints:
       a. Duration ratio within max_duration_ratio → ALIGN (interpolate)
       b. Sampling rates within max_rate_divergence → ALIGN (interpolate)
       c. Otherwise → DISCARD (timestamps drifted, unsafe to interpolate)

    Args:
        acc_times: Accelerometer timestamps (ms)
        gyro_times: Gyroscope timestamps (ms)
        config: Alignment configuration

    Returns:
        action: 'use_as_is', 'align', or 'discard'
        reason: Human-readable explanation
    """
    acc_start, acc_end = acc_times.min(), acc_times.max()
    gyro_start, gyro_end = gyro_times.min(), gyro_times.max()

    # Check start synchronization
    start_diff = abs(acc_start - gyro_start)
    if start_diff > config.max_time_gap_ms:
        return 'discard', f'Start offset {start_diff:.0f}ms > {config.max_time_gap_ms}ms threshold'

    # Check end synchronization
    end_diff = abs(acc_end - gyro_end)
    if end_diff > config.max_time_gap_ms:
        return 'discard', f'End offset {end_diff:.0f}ms > {config.max_time_gap_ms}ms threshold'

    # Compute overlap
    overlap_start = max(acc_start, gyro_start)
    overlap_end = min(acc_end, gyro_end)

    if overlap_end <= overlap_start:
        return 'discard', 'No temporal overlap between modalities'

    overlap_duration = overlap_end - overlap_start
    acc_duration = acc_end - acc_start
    gyro_duration = gyro_end - gyro_start
    min_duration = min(acc_duration, gyro_duration)

    if min_duration <= 0:
        return 'discard', 'Zero duration recording'

    overlap_ratio = overlap_duration / min_duration

    if overlap_ratio < config.min_overlap_ratio:
        return 'discard', f'Overlap {overlap_ratio:.1%} < {config.min_overlap_ratio:.0%} threshold'

    # Check sample count difference
    sample_diff = abs(len(acc_times) - len(gyro_times))

    # RULE 1: If sample diff is small (≤10), use as-is WITHOUT interpolation
    if sample_diff <= config.length_threshold:
        return 'use_as_is', f'Sample diff {sample_diff} <= {config.length_threshold} (truncate, no interpolation)'

    # RULE 2: Sample diff > threshold - need ADDITIONAL checks before interpolating
    # Don't blindly interpolate when timestamps may have drifted!

    # Check duration ratio - if durations differ too much, timestamps have drifted
    max_duration = max(acc_duration, gyro_duration)
    duration_ratio = max_duration / min_duration if min_duration > 0 else float('inf')

    if duration_ratio > config.max_duration_ratio:
        return 'discard', (
            f'Duration ratio {duration_ratio:.2f} > {config.max_duration_ratio} '
            f'(acc={acc_duration:.0f}ms, gyro={gyro_duration:.0f}ms) - timestamps drifted, unsafe to interpolate'
        )

    # Check sampling rate divergence
    acc_rate = len(acc_times) * 1000.0 / acc_duration if acc_duration > 0 else 0
    gyro_rate = len(gyro_times) * 1000.0 / gyro_duration if gyro_duration > 0 else 0
    mean_rate = (acc_rate + gyro_rate) / 2.0

    if mean_rate > 0:
        rate_divergence = abs(acc_rate - gyro_rate) / mean_rate
        if rate_divergence > config.max_rate_divergence:
            return 'discard', (
                f'Sampling rate divergence {rate_divergence:.1%} > {config.max_rate_divergence:.0%} '
                f'(acc={acc_rate:.1f}Hz, gyro={gyro_rate:.1f}Hz) - timestamps drifted, unsafe to interpolate'
            )

    # RULE 3: All checks passed - safe to interpolate
    return 'align', (
        f'Sample diff {sample_diff} > {config.length_threshold}, but timestamps are synchronized '
        f'(duration_ratio={duration_ratio:.2f}, rates: acc={acc_rate:.1f}Hz, gyro={gyro_rate:.1f}Hz) - interpolating'
    )


# =============================================================================
# LINEAR INTERPOLATION
# =============================================================================

def interpolate_to_grid(
    timestamps: np.ndarray,
    data: np.ndarray,
    target_times: np.ndarray,
    method: str = 'linear'
) -> np.ndarray:
    """
    Interpolate multi-channel IMU data to target timestamps.

    Args:
        timestamps: (N,) source timestamps
        data: (N, C) source data with C channels
        target_times: (M,) target timestamps
        method: Interpolation method ('linear', 'cubic', 'nearest')

    Returns:
        Interpolated data (M, C)
    """
    # Deduplicate timestamps for interpolation
    t_unique = deduplicate_timestamps(timestamps)

    n_channels = data.shape[1] if data.ndim > 1 else 1
    result = np.zeros((len(target_times), n_channels), dtype=np.float32)

    for ch in range(n_channels):
        ch_data = data[:, ch] if data.ndim > 1 else data

        # Create interpolator
        interp_func = interp1d(
            t_unique, ch_data,
            kind=method,
            fill_value='extrapolate',
            bounds_error=False
        )

        result[:, ch] = interp_func(target_times)

    return result


# =============================================================================
# MAIN ALIGNMENT FUNCTION
# =============================================================================

def align_imu_arrays(
    acc_data: np.ndarray,
    acc_times: np.ndarray,
    gyro_data: np.ndarray,
    gyro_times: np.ndarray,
    config: AlignmentConfig
) -> AlignmentResult:
    """
    Align accelerometer and gyroscope arrays using timestamp-based interpolation.

    This is the core alignment function that operates on pre-loaded arrays.

    Args:
        acc_data: (N_acc, 3) accelerometer data
        acc_times: (N_acc,) accelerometer timestamps in ms
        gyro_data: (N_gyro, 3) gyroscope data
        gyro_times: (N_gyro,) gyroscope timestamps in ms
        config: Alignment configuration

    Returns:
        AlignmentResult with aligned data or discard reason
    """
    result = AlignmentResult(success=False, action='', reason='')

    # Trim stray initial samples before alignment
    # This handles files where 1-3 timestamps at the start are wildly out of sync
    acc_times, acc_data, gyro_times, gyro_data, acc_skipped, gyro_skipped = \
        trim_stray_initial_samples(acc_times, acc_data, gyro_times, gyro_data)

    if acc_skipped > 0 or gyro_skipped > 0:
        logger.info(f"Trimmed stray samples: acc={acc_skipped}, gyro={gyro_skipped}")

    # Record input statistics (after trimming)
    result.acc_samples = len(acc_data)
    result.gyro_samples = len(gyro_data)
    result.sample_diff = abs(result.acc_samples - result.gyro_samples)
    result.sample_diff_pct = result.sample_diff / max(result.acc_samples, result.gyro_samples) * 100

    # Compute timestamp statistics
    acc_start, acc_end = acc_times.min(), acc_times.max()
    gyro_start, gyro_end = gyro_times.min(), gyro_times.max()

    result.acc_duration_ms = float(acc_end - acc_start)
    result.gyro_duration_ms = float(gyro_end - gyro_start)
    result.start_offset_ms = float(abs(acc_start - gyro_start))
    result.end_offset_ms = float(abs(acc_end - gyro_end))

    # Compute overlap
    overlap_start = max(acc_start, gyro_start)
    overlap_end = min(acc_end, gyro_end)
    result.overlap_duration_ms = float(max(0, overlap_end - overlap_start))

    min_duration = min(result.acc_duration_ms, result.gyro_duration_ms)
    result.overlap_ratio = result.overlap_duration_ms / min_duration if min_duration > 0 else 0.0

    # Compute sampling rates
    result.acc_rate_hz, _, acc_max_gap = compute_sampling_stats(acc_times)
    result.gyro_rate_hz, _, gyro_max_gap = compute_sampling_stats(gyro_times)
    result.max_gap_ms = max(acc_max_gap, gyro_max_gap)
    result.has_large_gaps = result.max_gap_ms > 500  # 500ms gap threshold

    # Check for duplicate timestamps
    result.has_duplicate_timestamps = (
        len(np.unique(acc_times)) < len(acc_times) or
        len(np.unique(gyro_times)) < len(gyro_times)
    )

    # Determine action
    action, reason = check_alignment_feasibility(acc_times, gyro_times, config)
    result.action = action
    result.reason = reason

    if action == 'discard':
        result.success = False
        return result

    if action == 'use_as_is':
        # Truncate to shorter length
        min_len = min(len(acc_data), len(gyro_data))
        result.aligned_acc = acc_data[:min_len].copy()
        result.aligned_gyro = gyro_data[:min_len].copy()
        result.aligned_timestamps = acc_times[:min_len].copy()
        result.output_samples = min_len
        result.output_rate_hz = result.acc_rate_hz
        result.success = True
        return result

    # action == 'align': Perform timestamp-based interpolation

    # Create common time grid at target rate
    n_samples = int(result.overlap_duration_ms / 1000.0 * config.target_rate)

    if n_samples < config.min_output_samples:
        result.action = 'discarded'
        result.reason = f'Output {n_samples} samples < {config.min_output_samples} minimum'
        result.success = False
        return result

    # Generate uniform time grid
    common_times = np.linspace(overlap_start, overlap_end, n_samples)

    # Interpolate both modalities
    result.aligned_acc = interpolate_to_grid(
        acc_times, acc_data, common_times, config.alignment_method
    )
    result.aligned_gyro = interpolate_to_grid(
        gyro_times, gyro_data, common_times, config.alignment_method
    )
    result.aligned_timestamps = common_times

    result.output_samples = n_samples
    result.output_rate_hz = config.target_rate
    result.success = True

    return result


def align_imu_modalities(
    acc_path: str,
    gyro_path: str,
    config: Optional[AlignmentConfig] = None
) -> AlignmentResult:
    """
    Align accelerometer and gyroscope from CSV files.

    High-level function that handles file I/O and calls align_imu_arrays.

    Args:
        acc_path: Path to accelerometer CSV
        gyro_path: Path to gyroscope CSV
        config: Alignment configuration (uses defaults if None)

    Returns:
        AlignmentResult with aligned data or discard reason
    """
    if config is None:
        config = AlignmentConfig()

    result = AlignmentResult(success=False, action='discard', reason='')

    try:
        # Parse CSV files with timestamps
        acc_times, acc_data = parse_imu_csv_with_timestamps(acc_path)
        gyro_times, gyro_data = parse_imu_csv_with_timestamps(gyro_path)

        # Align arrays
        result = align_imu_arrays(acc_data, acc_times, gyro_data, gyro_times, config)

    except Exception as e:
        result.success = False
        result.action = 'discarded'
        result.reason = f'Error: {str(e)}'
        logger.warning(f"Alignment failed for {acc_path}: {e}")

    return result


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def create_alignment_config_from_kwargs(**kwargs) -> AlignmentConfig:
    """
    Create AlignmentConfig from dataset_args kwargs.

    Maps config YAML keys to AlignmentConfig fields.
    """
    return AlignmentConfig(
        target_rate=kwargs.get('alignment_target_rate', 30.0),
        alignment_method=kwargs.get('alignment_method', 'linear'),
        min_overlap_ratio=kwargs.get('min_overlap_ratio', 0.8),
        max_time_gap_ms=kwargs.get('max_time_gap_ms', 1000.0),
        min_output_samples=kwargs.get('min_output_samples', 64),
        length_threshold=kwargs.get('length_threshold', 10),
        handle_duplicates=kwargs.get('handle_duplicates', 'offset'),
        duplicate_epsilon_ms=kwargs.get('duplicate_epsilon_ms', 0.001),
        # Conservative interpolation controls (prevent interpolation when timestamps drifted)
        max_duration_ratio=kwargs.get('max_duration_ratio', 1.2),
        max_rate_divergence=kwargs.get('max_rate_divergence', 0.3)
    )


def align_trial_data(
    trial_data: Dict[str, np.ndarray],
    acc_path: str,
    gyro_path: str,
    config: AlignmentConfig
) -> Tuple[Dict[str, np.ndarray], AlignmentResult]:
    """
    Align trial data using timestamp alignment.

    This function integrates with the DatasetBuilder pipeline.

    Args:
        trial_data: Dictionary with 'accelerometer' and 'gyroscope' arrays
        acc_path: Path to accelerometer CSV (for timestamp access)
        gyro_path: Path to gyroscope CSV (for timestamp access)
        config: Alignment configuration

    Returns:
        aligned_trial_data: Updated trial_data with aligned arrays
        result: AlignmentResult with statistics
    """
    result = align_imu_modalities(acc_path, gyro_path, config)

    if result.success and result.aligned_acc is not None:
        trial_data['accelerometer'] = result.aligned_acc
        trial_data['gyroscope'] = result.aligned_gyro

    return trial_data, result


# =============================================================================
# TESTING
# =============================================================================

if __name__ == '__main__':
    import sys

    print("=" * 60)
    print("IMU Alignment Module Test")
    print("=" * 60)

    # Test with sample files if available
    test_acc = "data/young/accelerometer/watch/S29A01T01.csv"
    test_gyro = "data/young/gyroscope/watch/S29A01T01.csv"

    if os.path.exists(test_acc) and os.path.exists(test_gyro):
        print(f"\nTesting with: {test_acc}")

        config = AlignmentConfig(target_rate=30.0)
        result = align_imu_modalities(test_acc, test_gyro, config)

        print(f"\nResult:")
        print(f"  Action: {result.action}")
        print(f"  Reason: {result.reason}")
        print(f"  Success: {result.success}")
        print(f"\nInput stats:")
        print(f"  Acc samples: {result.acc_samples}")
        print(f"  Gyro samples: {result.gyro_samples}")
        print(f"  Sample diff: {result.sample_diff} ({result.sample_diff_pct:.1f}%)")
        print(f"\nTimestamp stats:")
        print(f"  Acc duration: {result.acc_duration_ms:.0f}ms")
        print(f"  Gyro duration: {result.gyro_duration_ms:.0f}ms")
        print(f"  Start offset: {result.start_offset_ms:.1f}ms")
        print(f"  End offset: {result.end_offset_ms:.1f}ms")
        print(f"  Overlap ratio: {result.overlap_ratio:.1%}")
        print(f"\nSampling rates:")
        print(f"  Acc rate: {result.acc_rate_hz:.1f} Hz")
        print(f"  Gyro rate: {result.gyro_rate_hz:.1f} Hz")
        print(f"  Output rate: {result.output_rate_hz:.1f} Hz")

        if result.success:
            print(f"\nOutput:")
            print(f"  Samples: {result.output_samples}")
            print(f"  Acc shape: {result.aligned_acc.shape}")
            print(f"  Gyro shape: {result.aligned_gyro.shape}")
    else:
        print(f"\nTest files not found. Run from FeatureKD directory.")
        print(f"  Expected: {test_acc}")

    print("\n" + "=" * 60)
