"""
Dataset analysis tools for KD experiments.

Provides timestamp analysis, dropout detection, and skeleton-IMU pairing sanity checks.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def load_imu_with_timestamps(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load IMU CSV file preserving timestamps.

    Returns:
        timestamps: (N,) array of timestamps in seconds since start
        values: (N, 3) array of x, y, z values
    """
    df = pd.read_csv(file_path, header=None, names=['timestamp', 'x', 'y', 'z'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Convert to seconds since start
    t0 = df['timestamp'].iloc[0]
    timestamps = (df['timestamp'] - t0).dt.total_seconds().values
    values = df[['x', 'y', 'z']].values.astype(np.float32)

    return timestamps, values


def load_skeleton(file_path: Path) -> np.ndarray:
    """
    Load skeleton CSV file (no timestamps, just frames).

    Returns:
        skeleton: (T, 96) array of joint positions (32 joints * 3 coords)
    """
    df = pd.read_csv(file_path, header=None)
    return df.values.astype(np.float32)


def compute_delta_stats(timestamps: np.ndarray) -> Dict:
    """Compute statistics about inter-sample time deltas."""
    if len(timestamps) < 2:
        return {'n_samples': len(timestamps), 'error': 'too_few_samples'}

    delta_ms = np.diff(timestamps) * 1000  # Convert to milliseconds

    return {
        'n_samples': len(timestamps),
        'duration_s': float(timestamps[-1] - timestamps[0]),
        'delta_mean_ms': float(np.mean(delta_ms)),
        'delta_median_ms': float(np.median(delta_ms)),
        'delta_std_ms': float(np.std(delta_ms)),
        'delta_min_ms': float(np.min(delta_ms)),
        'delta_max_ms': float(np.max(delta_ms)),
        'zero_delta_count': int(np.sum(delta_ms == 0)),
        'zero_delta_pct': float(np.mean(delta_ms == 0) * 100),
        'gaps_100ms_count': int(np.sum(delta_ms > 100)),
        'gaps_100ms_pct': float(np.mean(delta_ms > 100) * 100),
        'gaps_1s_count': int(np.sum(delta_ms > 1000)),
        'negative_delta_count': int(np.sum(delta_ms < 0)),
        'effective_hz': float(len(timestamps) / max(timestamps[-1] - timestamps[0], 1e-6)),
    }


def detect_dropout_bursts(
    timestamps: np.ndarray,
    gap_threshold_ms: float = 100.0,
    min_burst_len: int = 3
) -> List[Dict]:
    """
    Detect dropout bursts (consecutive large gaps).

    Args:
        timestamps: (N,) array of timestamps in seconds
        gap_threshold_ms: Minimum gap to consider as dropout
        min_burst_len: Minimum consecutive gaps to count as burst

    Returns:
        List of burst info dicts
    """
    if len(timestamps) < 2:
        return []

    delta_ms = np.diff(timestamps) * 1000
    is_gap = delta_ms > gap_threshold_ms

    bursts = []
    burst_start = None
    burst_len = 0

    for i, gap in enumerate(is_gap):
        if gap:
            if burst_start is None:
                burst_start = i
            burst_len += 1
        else:
            if burst_start is not None and burst_len >= min_burst_len:
                bursts.append({
                    'start_idx': burst_start,
                    'end_idx': i,
                    'length': burst_len,
                    'start_time_s': float(timestamps[burst_start]),
                    'total_gap_ms': float(np.sum(delta_ms[burst_start:i])),
                })
            burst_start = None
            burst_len = 0

    # Handle burst at end
    if burst_start is not None and burst_len >= min_burst_len:
        bursts.append({
            'start_idx': burst_start,
            'end_idx': len(is_gap),
            'length': burst_len,
            'start_time_s': float(timestamps[burst_start]),
            'total_gap_ms': float(np.sum(delta_ms[burst_start:])),
        })

    return bursts


def analyze_skeleton_imu_pairing(
    skeleton_path: Path,
    acc_path: Path,
    gyro_path: Optional[Path] = None,
    skeleton_fps: float = 30.0,
) -> Dict:
    """
    Analyze pairing between skeleton and IMU data for the same trial.

    Returns dict with duration comparison, correlation metrics, etc.
    """
    result = {'skeleton_path': str(skeleton_path), 'acc_path': str(acc_path)}

    # Load data
    try:
        skeleton = load_skeleton(skeleton_path)
        acc_ts, acc_vals = load_imu_with_timestamps(acc_path)
    except Exception as e:
        result['error'] = str(e)
        return result

    # Duration comparison
    skel_duration = len(skeleton) / skeleton_fps
    imu_duration = acc_ts[-1] - acc_ts[0] if len(acc_ts) > 1 else 0

    result['skeleton_frames'] = len(skeleton)
    result['skeleton_duration_s'] = float(skel_duration)
    result['imu_samples'] = len(acc_ts)
    result['imu_duration_s'] = float(imu_duration)
    result['duration_diff_s'] = float(abs(skel_duration - imu_duration))
    result['duration_ratio'] = float(skel_duration / max(imu_duration, 1e-6))

    # Compute magnitude signals for correlation
    # Skeleton: use wrist joint (joint 9, indices 24:27)
    wrist_pos = skeleton[:, 24:27]  # Joint 9 is index 8, *3 = 24
    wrist_magnitude = np.linalg.norm(np.diff(wrist_pos, axis=0), axis=1)

    # IMU: acceleration magnitude
    acc_magnitude = np.linalg.norm(acc_vals, axis=1)

    result['wrist_magnitude_mean'] = float(np.mean(wrist_magnitude))
    result['acc_magnitude_mean'] = float(np.mean(acc_magnitude))

    # Load gyro if available
    if gyro_path and gyro_path.exists():
        try:
            gyro_ts, gyro_vals = load_imu_with_timestamps(gyro_path)
            result['gyro_samples'] = len(gyro_ts)
            result['gyro_duration_s'] = float(gyro_ts[-1] - gyro_ts[0])
            result['acc_gyro_len_diff'] = abs(len(acc_ts) - len(gyro_ts))
        except Exception as e:
            result['gyro_error'] = str(e)

    return result


def analyze_subject(
    data_root: Path,
    subject_id: str,
    modalities: List[str] = ['accelerometer', 'gyroscope'],
    device: str = 'watch',
) -> Dict:
    """Analyze all trials for a subject."""
    results = {'subject_id': subject_id, 'trials': []}

    # Find all trials for this subject
    acc_dir = data_root / 'young' / 'accelerometer' / device
    gyro_dir = data_root / 'young' / 'gyroscope' / device
    skel_dir = data_root / 'young' / 'skeleton'

    if not acc_dir.exists():
        results['error'] = f'Accelerometer dir not found: {acc_dir}'
        return results

    # Find all files for this subject
    pattern = f'{subject_id}*.csv'
    acc_files = list(acc_dir.glob(pattern))

    for acc_file in acc_files:
        trial_id = acc_file.stem  # e.g., S29A03T02

        trial_result = {'trial_id': trial_id}

        # Analyze accelerometer timestamps
        try:
            acc_ts, acc_vals = load_imu_with_timestamps(acc_file)
            trial_result['acc'] = compute_delta_stats(acc_ts)
            trial_result['acc']['bursts'] = detect_dropout_bursts(acc_ts)
        except Exception as e:
            trial_result['acc_error'] = str(e)

        # Analyze gyroscope timestamps
        gyro_file = gyro_dir / f'{trial_id}.csv'
        if gyro_file.exists():
            try:
                gyro_ts, gyro_vals = load_imu_with_timestamps(gyro_file)
                trial_result['gyro'] = compute_delta_stats(gyro_ts)
                trial_result['gyro']['bursts'] = detect_dropout_bursts(gyro_ts)
            except Exception as e:
                trial_result['gyro_error'] = str(e)

        # Analyze skeleton-IMU pairing
        skel_file = skel_dir / f'{trial_id}.csv'
        if skel_file.exists():
            trial_result['pairing'] = analyze_skeleton_imu_pairing(
                skel_file, acc_file, gyro_file
            )
            trial_result['has_skeleton'] = True
        else:
            trial_result['has_skeleton'] = False

        results['trials'].append(trial_result)

    return results


def aggregate_statistics(subject_results: List[Dict]) -> Dict:
    """Aggregate statistics across all subjects and trials."""
    all_acc_deltas = []
    all_gyro_deltas = []
    all_pairings = []

    for subj in subject_results:
        for trial in subj.get('trials', []):
            if 'acc' in trial and 'delta_mean_ms' in trial['acc']:
                all_acc_deltas.append(trial['acc'])
            if 'gyro' in trial and 'delta_mean_ms' in trial['gyro']:
                all_gyro_deltas.append(trial['gyro'])
            if 'pairing' in trial and 'duration_ratio' in trial['pairing']:
                all_pairings.append(trial['pairing'])

    def summarize_deltas(deltas: List[Dict], name: str) -> Dict:
        if not deltas:
            return {f'{name}_n_trials': 0}
        return {
            f'{name}_n_trials': len(deltas),
            f'{name}_mean_delta_ms': float(np.mean([d['delta_mean_ms'] for d in deltas])),
            f'{name}_median_delta_ms': float(np.mean([d['delta_median_ms'] for d in deltas])),
            f'{name}_zero_delta_pct': float(np.mean([d['zero_delta_pct'] for d in deltas])),
            f'{name}_gaps_100ms_pct': float(np.mean([d['gaps_100ms_pct'] for d in deltas])),
            f'{name}_effective_hz': float(np.mean([d['effective_hz'] for d in deltas])),
            f'{name}_max_gap_ms': float(np.max([d['delta_max_ms'] for d in deltas])),
        }

    result = {}
    result.update(summarize_deltas(all_acc_deltas, 'acc'))
    result.update(summarize_deltas(all_gyro_deltas, 'gyro'))

    if all_pairings:
        result['pairing_n_trials'] = len(all_pairings)
        result['pairing_duration_ratio_mean'] = float(np.mean([p['duration_ratio'] for p in all_pairings]))
        result['pairing_duration_ratio_std'] = float(np.std([p['duration_ratio'] for p in all_pairings]))
        result['pairing_duration_diff_mean_s'] = float(np.mean([p['duration_diff_s'] for p in all_pairings]))

    return result


def run_full_analysis(
    data_root: Path,
    output_dir: Path,
    subjects: Optional[List[str]] = None,
    device: str = 'watch',
) -> Dict:
    """
    Run complete dataset analysis.

    Args:
        data_root: Path to SmartFallMM dataset root (e.g., data/)
        output_dir: Where to save results and figures
        subjects: List of subject IDs to analyze (None = all)
        device: IMU device to analyze ('watch', 'phone', etc.)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all subjects if not specified
    if subjects is None:
        acc_dir = data_root / 'young' / 'accelerometer' / device
        if not acc_dir.exists():
            raise ValueError(f'Data directory not found: {acc_dir}')
        all_files = list(acc_dir.glob('S*.csv'))
        subjects = sorted(set(f.stem.split('A')[0] for f in all_files))

    print(f'Analyzing {len(subjects)} subjects...')

    # Analyze each subject
    subject_results = []
    for subj_id in subjects:
        print(f'  Analyzing {subj_id}...')
        result = analyze_subject(data_root, subj_id, device=device)
        subject_results.append(result)

    # Aggregate statistics
    aggregate = aggregate_statistics(subject_results)

    # Save results
    full_results = {
        'aggregate': aggregate,
        'subjects': subject_results,
    }

    with open(output_dir / 'analysis_results.json', 'w') as f:
        json.dump(full_results, f, indent=2)

    # Print summary
    print('\n=== Aggregate Statistics ===')
    for k, v in aggregate.items():
        if isinstance(v, float):
            print(f'  {k}: {v:.2f}')
        else:
            print(f'  {k}: {v}')

    return full_results


def main():
    parser = argparse.ArgumentParser(description='Dataset analysis for KD experiments')
    parser.add_argument('--data-root', type=str, default='data',
                        help='Path to SmartFallMM dataset root')
    parser.add_argument('--output-dir', type=str, default='analysis',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='watch',
                        choices=['watch', 'phone', 'meta_wrist', 'meta_hip'],
                        help='IMU device to analyze')
    parser.add_argument('--subjects', type=str, nargs='+', default=None,
                        help='Specific subject IDs to analyze')

    args = parser.parse_args()

    run_full_analysis(
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        subjects=args.subjects,
        device=args.device,
    )


if __name__ == '__main__':
    main()
