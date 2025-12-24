"""
Utility functions for data loading and feature computation.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from pathlib import Path

from .config import DATA_DIR, FALL_ACTIVITIES

def load_trial_data(subject_id, activity_id, trial_id, sensor='watch', data_dir=None):
    """
    Load accelerometer and gyroscope data for a specific trial.

    Args:
        subject_id: Subject number
        activity_id: Activity number (10-14 for falls)
        trial_id: Trial number
        sensor: Sensor type ('watch', 'phone', 'meta_wrist', 'meta_hip')
        data_dir: Override data directory (uses config default if None)

    Returns:
        dict with 'accelerometer' and optionally 'gyroscope' DataFrames, or None if not found
    """
    if data_dir is None:
        data_dir = DATA_DIR
    else:
        data_dir = Path(data_dir)

    result = {'subject': subject_id, 'activity': activity_id, 'trial': trial_id}

    for modality in ['accelerometer', 'gyroscope']:
        for age_group in ['young', 'old']:
            path = data_dir / age_group / modality / sensor / f'S{subject_id}A{activity_id}T{trial_id:02d}.csv'
            if path.exists():
                df = pd.read_csv(path, header=None)
                if len(df.columns) >= 4:
                    df.columns = ['time', 'x', 'y', 'z'][:len(df.columns)]
                    # Convert to numeric
                    for col in ['x', 'y', 'z']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    df = df.dropna()
                if len(df) > 0:
                    result[modality] = df
                break

    return result if 'accelerometer' in result and len(result['accelerometer']) > 0 else None


def get_subject_trials(subject_id, activity_id, sensor='watch', data_dir=None):
    """
    Get list of available trial numbers for a subject/activity.

    Args:
        subject_id: Subject number
        activity_id: Activity number
        sensor: Sensor type
        data_dir: Override data directory

    Returns:
        List of available trial numbers
    """
    if data_dir is None:
        data_dir = DATA_DIR
    else:
        data_dir = Path(data_dir)

    trials = []
    for trial in range(1, 10):
        for age_group in ['young', 'old']:
            path = data_dir / age_group / 'accelerometer' / sensor / f'S{subject_id}A{activity_id}T{trial:02d}.csv'
            if path.exists():
                trials.append(trial)
                break
    return trials


def compute_signal_features(acc_data, gyro_data=None, fs=30.0):
    """
    Extract comprehensive signal features from IMU data.

    Args:
        acc_data: DataFrame with columns ['x', 'y', 'z'] for accelerometer
        gyro_data: Optional DataFrame with columns ['x', 'y', 'z'] for gyroscope
        fs: Sampling frequency in Hz

    Returns:
        dict of computed features
    """
    features = {}

    try:
        ax = pd.to_numeric(acc_data['x'], errors='coerce').values
        ay = pd.to_numeric(acc_data['y'], errors='coerce').values
        az = pd.to_numeric(acc_data['z'], errors='coerce').values

        # Remove NaN values
        valid = ~(np.isnan(ax) | np.isnan(ay) | np.isnan(az))
        ax, ay, az = ax[valid], ay[valid], az[valid]
    except Exception:
        return features

    if len(ax) == 0:
        return features

    # Signal Magnitude Vector
    smv = np.sqrt(ax**2 + ay**2 + az**2)

    # Basic SMV features
    features['smv_max'] = smv.max()
    features['smv_min'] = smv.min()
    features['smv_mean'] = smv.mean()
    features['smv_std'] = smv.std()
    features['smv_range'] = smv.max() - smv.min()
    features['smv_peak_prominence'] = smv.max() - smv.mean()

    # Per-axis features
    for axis, vals in [('ax', ax), ('ay', ay), ('az', az)]:
        features[f'{axis}_max'] = np.abs(vals).max()
        features[f'{axis}_std'] = vals.std()
        features[f'{axis}_range'] = vals.max() - vals.min()

    # Temporal features
    features['duration_s'] = len(smv) / fs
    peak_idx = np.argmax(smv)
    features['peak_time_s'] = peak_idx / fs
    features['peak_time_ratio'] = peak_idx / len(smv)  # 0-1, when peak occurs

    # Impact peak detection
    peaks, properties = find_peaks(smv, height=15, prominence=5)
    features['n_impact_peaks'] = len(peaks)
    features['peak_indices'] = peaks
    features['peak_heights'] = smv[peaks] if len(peaks) > 0 else np.array([])

    # Gyroscope features
    if gyro_data is not None and len(gyro_data) > 0:
        try:
            gx = pd.to_numeric(gyro_data['x'], errors='coerce').values
            gy = pd.to_numeric(gyro_data['y'], errors='coerce').values
            gz = pd.to_numeric(gyro_data['z'], errors='coerce').values

            valid_g = ~(np.isnan(gx) | np.isnan(gy) | np.isnan(gz))
            if valid_g.sum() > 0:
                gx, gy, gz = gx[valid_g], gy[valid_g], gz[valid_g]
                gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)

                features['gyro_max'] = gyro_mag.max()
                features['gyro_min'] = gyro_mag.min()
                features['gyro_mean'] = gyro_mag.mean()
                features['gyro_std'] = gyro_mag.std()
                features['gyro_range'] = gyro_mag.max() - gyro_mag.min()

                # Per-axis gyro
                for axis, vals in [('gx', gx), ('gy', gy), ('gz', gz)]:
                    features[f'{axis}_max'] = np.abs(vals).max()
                    features[f'{axis}_std'] = vals.std()

                # Total rotation (integrated gyro magnitude)
                dt = 1.0 / fs
                features['total_rotation_deg'] = np.sum(np.abs(gyro_mag)) * dt

                # Gyro peak timing
                features['gyro_peak_time_s'] = np.argmax(gyro_mag) / fs
        except Exception:
            pass

    # Jerk (derivative of acceleration)
    if len(ax) > 1:
        jerk_x = np.diff(ax) * fs
        jerk_y = np.diff(ay) * fs
        jerk_z = np.diff(az) * fs
        jerk_mag = np.sqrt(jerk_x**2 + jerk_y**2 + jerk_z**2)
        features['jerk_max'] = jerk_mag.max()
        features['jerk_mean'] = jerk_mag.mean()
        features['jerk_std'] = jerk_mag.std()

    return features


def load_model_comparison(transformer_path, cnn_path):
    """
    Load and merge model results for comparison.

    Args:
        transformer_path: Path to transformer scores.csv
        cnn_path: Path to CNN scores.csv

    Returns:
        DataFrame with merged comparison data
    """
    trans_scores = pd.read_csv(transformer_path)
    cnn_scores = pd.read_csv(cnn_path)

    # Filter out 'Average' row if present
    trans_scores = trans_scores[trans_scores['test_subject'] != 'Average'].copy()
    cnn_scores = cnn_scores[cnn_scores['test_subject'] != 'Average'].copy()
    trans_scores['test_subject'] = trans_scores['test_subject'].astype(int)
    cnn_scores['test_subject'] = cnn_scores['test_subject'].astype(int)

    # Merge
    comparison = trans_scores[['test_subject', 'test_f1_score', 'test_precision', 'test_recall']].copy()
    comparison.columns = ['subject', 'trans_f1', 'trans_prec', 'trans_recall']

    cnn_subset = cnn_scores[['test_subject', 'test_f1_score', 'test_precision', 'test_recall']].copy()
    cnn_subset.columns = ['subject', 'cnn_f1', 'cnn_prec', 'cnn_recall']

    comparison = comparison.merge(cnn_subset, on='subject')
    comparison['delta_f1'] = comparison['trans_f1'] - comparison['cnn_f1']
    comparison['delta_prec'] = comparison['trans_prec'] - comparison['cnn_prec']
    comparison['delta_recall'] = comparison['trans_recall'] - comparison['cnn_recall']
    comparison['winner'] = comparison['delta_f1'].apply(lambda x: 'Transformer' if x > 0 else 'CNN')
    comparison = comparison.sort_values('delta_f1', ascending=False).reset_index(drop=True)

    return comparison


def build_trial_features_dataset(comparison_df, data_dir=None):
    """
    Build a dataset of features for all fall trials across subjects.

    Args:
        comparison_df: DataFrame from load_model_comparison()
        data_dir: Override data directory

    Returns:
        DataFrame with features for each trial
    """
    all_features = []

    for subj in comparison_df['subject'].values:
        subj = int(subj)
        for act_id, act_name in FALL_ACTIVITIES.items():
            trials = get_subject_trials(subj, act_id, data_dir=data_dir)
            for trial in trials:
                data = load_trial_data(subj, act_id, trial, data_dir=data_dir)
                if data and 'accelerometer' in data:
                    features = compute_signal_features(
                        data['accelerometer'],
                        data.get('gyroscope')
                    )
                    # Remove non-scalar features
                    features = {k: v for k, v in features.items()
                               if not isinstance(v, np.ndarray)}

                    features['subject'] = subj
                    features['activity_id'] = act_id
                    features['activity_name'] = act_name
                    features['trial'] = trial
                    all_features.append(features)

    features_df = pd.DataFrame(all_features)

    # Merge with comparison data
    features_df = features_df.merge(
        comparison_df[['subject', 'trans_f1', 'cnn_f1', 'delta_f1', 'winner']],
        on='subject'
    )

    return features_df
