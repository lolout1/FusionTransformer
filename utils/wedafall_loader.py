"""
WEDA-FALL Dataset Loader for FusionTransformer.

WEDA-FALL is a wrist-based fall detection dataset with elderly participants.
Contains accelerometer, gyroscope, and orientation data at multiple frequencies.

Dataset: https://github.com/joaojtmarques/WEDA-FALL
Paper: Marques et al. "WEDA-FALL" (wrist elderly daily activity and fall)

Data format:
- Directory structure: {frequency}/{activity}/{user}_R{trial}_{sensor}.csv
- Sensors: accel, gyro, orientation, vertical_accel
- 25 subjects (14 young + 11 elderly)
- Activities: F01-F08 (falls), D01-D11 (ADLs)
- Note: Elderly subjects (21-31) did NOT perform falls

Frequencies available: 50Hz, 40Hz, 25Hz, 10Hz, 5Hz
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

from .external_dataset_utils import (
    sliding_window_numpy,
    sliding_window_class_aware,
    apply_kalman_fusion,
    get_kalman_config_for_rate,
    normalize_data,
    compute_smv,
    wedafall_label_from_folder,
    validate_window_data,
    print_data_summary,
)


class WEDAFallLoader:
    """
    WEDA-FALL dataset loader compatible with FusionTransformer.

    Loads wrist IMU data from directory structure and prepares
    windowed data for fall detection training.
    """

    # Sampling rates available
    FREQUENCIES = ['50Hz', '40Hz', '25Hz', '10Hz', '5Hz']
    FREQ_TO_HZ = {'50Hz': 50.0, '40Hz': 40.0, '25Hz': 25.0, '10Hz': 10.0, '5Hz': 5.0}

    # Subject categories
    YOUNG_SUBJECTS = list(range(1, 15))  # U01-U14
    ELDERLY_SUBJECTS = list(range(21, 32))  # U21-U31

    # Activities
    FALL_ACTIVITIES = [f'F{i:02d}' for i in range(1, 9)]  # F01-F08
    ADL_ACTIVITIES = [f'D{i:02d}' for i in range(1, 12)]  # D01-D11

    def __init__(
        self,
        base_path: str,
        frequency: str = '50Hz',
        window_size: int = 128,
        stride: int = 32,
        enable_kalman: bool = True,
        enable_class_aware_stride: bool = True,
        fall_stride: int = 16,
        adl_stride: int = 64,
        convert_gyro_to_rad: bool = True,
        normalize: bool = True,
        normalize_modalities: str = 'acc_only',
        include_elderly: bool = False,
        **kwargs
    ):
        """
        Initialize WEDA-FALL loader.

        Args:
            base_path: Path to WEDA-FALL/dataset directory
            frequency: Data frequency ('50Hz', '40Hz', '25Hz', '10Hz', '5Hz')
            window_size: Sliding window length (samples)
            stride: Default stride between windows
            enable_kalman: Apply Kalman fusion to IMU data
            enable_class_aware_stride: Use different strides for falls/ADLs
            fall_stride: Stride for fall windows
            adl_stride: Stride for ADL windows
            convert_gyro_to_rad: Convert gyro from deg/s to rad/s
            normalize: Apply z-score normalization
            normalize_modalities: 'all', 'acc_only', or 'none'
            include_elderly: Include elderly subjects (ADL only, no falls)
            **kwargs: Additional Kalman config options:
                - kalman_include_smv: bool
                - kalman_exclude_yaw: bool
                - kalman_include_raw_gyro: bool
                - kalman_orientation_only: bool
                - kalman_Q_orientation, kalman_Q_rate, kalman_R_acc, kalman_R_gyro: float
        """
        self.base_path = base_path
        self.frequency = frequency
        self.sampling_rate = self.FREQ_TO_HZ[frequency]
        self.window_size = window_size
        self.stride = stride
        self.enable_kalman = enable_kalman
        self.enable_class_aware_stride = enable_class_aware_stride
        self.fall_stride = fall_stride
        self.adl_stride = adl_stride
        self.convert_gyro_to_rad = convert_gyro_to_rad
        self.normalize = normalize
        self.normalize_modalities = normalize_modalities
        self.include_elderly = include_elderly

        # Kalman config: start with defaults, then override with kwargs
        self.kalman_config = get_kalman_config_for_rate(self.sampling_rate)
        # Merge any kalman-related kwargs
        for key, value in kwargs.items():
            if key.startswith('kalman_') or key in ('filter_fs',):
                self.kalman_config[key] = value

        # Load trials
        self.data_path = os.path.join(base_path, frequency)
        self.trials = self._load_trials()
        self.subjects = self._get_subjects()

        print(f"WEDA-FALL Loader initialized:")
        print(f"  Path: {self.data_path}")
        print(f"  Frequency: {frequency} ({self.sampling_rate}Hz)")
        print(f"  Subjects: {len(self.subjects)} ({'including' if include_elderly else 'excluding'} elderly)")
        print(f"  Trials: {len(self.trials)}")
        print(f"  Window: {window_size} samples")
        print(f"  Kalman fusion: {enable_kalman}")

    def _parse_filename(self, filename: str) -> Optional[Tuple[int, int, str]]:
        """
        Parse trial filename to extract subject, trial, and sensor type.

        Format: U{XX}_R{YY}_{sensor}.csv

        Returns:
            (subject_id, trial_id, sensor_type) or None if invalid
        """
        if not filename.endswith('.csv'):
            return None

        parts = filename.replace('.csv', '').split('_')
        if len(parts) != 3:
            return None

        try:
            subject_id = int(parts[0][1:])  # U01 -> 1
            trial_id = int(parts[1][1:])    # R01 -> 1
            sensor_type = parts[2]          # accel, gyro, etc.
            return subject_id, trial_id, sensor_type
        except (ValueError, IndexError):
            return None

    def _load_trials(self) -> Dict[str, Dict]:
        """
        Load all trials from directory structure.

        Returns:
            Dict with trial keys mapping to trial data.
            Key format: "U{subject}_A{activity}_R{trial}"
        """
        trials = {}

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"WEDA-FALL data not found: {self.data_path}")

        # Iterate through activity folders
        for activity in os.listdir(self.data_path):
            activity_path = os.path.join(self.data_path, activity)
            if not os.path.isdir(activity_path):
                continue

            # Determine label from activity folder name
            label = wedafall_label_from_folder(activity)

            # Group files by (subject, trial)
            trial_files = defaultdict(dict)
            for filename in os.listdir(activity_path):
                parsed = self._parse_filename(filename)
                if parsed is None:
                    continue

                subject_id, trial_id, sensor_type = parsed
                file_path = os.path.join(activity_path, filename)
                key = (subject_id, trial_id)
                trial_files[key][sensor_type] = file_path

            # Create trial entries
            for (subject_id, trial_id), files in trial_files.items():
                # Skip elderly subjects if not included
                if not self.include_elderly and subject_id in self.ELDERLY_SUBJECTS:
                    continue

                # Skip if missing accel or gyro
                if 'accel' not in files or 'gyro' not in files:
                    continue

                trial_key = f"U{subject_id:02d}_A{activity}_R{trial_id:02d}"
                trials[trial_key] = {
                    'subject_id': subject_id,
                    'activity': activity,
                    'trial_id': trial_id,
                    'label': label,
                    'accel_file': files['accel'],
                    'gyro_file': files['gyro'],
                    'orientation_file': files.get('orientation'),
                }

        return trials

    def _get_subjects(self) -> List[int]:
        """Get sorted list of unique subject IDs."""
        subjects = set()
        for trial in self.trials.values():
            subjects.add(trial['subject_id'])
        return sorted(subjects)

    def _load_csv(self, file_path: str) -> np.ndarray:
        """
        Load sensor CSV file.

        Format: timestamp, x, y, z

        Returns:
            (T, 3) array of sensor readings
        """
        df = pd.read_csv(file_path)
        # Extract x, y, z columns (skip timestamp)
        data = df.iloc[:, 1:4].values.astype(float)
        return data

    def _process_trial(self, trial: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single trial: load CSVs, Kalman fusion, windowing.

        Args:
            trial: Dict with file paths and metadata

        Returns:
            windows: (N, T, C) windowed data
            labels: (N,) labels for each window
        """
        # Load accelerometer and gyroscope
        acc = self._load_csv(trial['accel_file'])
        gyro = self._load_csv(trial['gyro_file'])
        label = trial['label']

        # Convert gyro to rad/s if needed
        if self.convert_gyro_to_rad:
            gyro = np.deg2rad(gyro)

        # Ensure synchronized lengths
        min_len = min(acc.shape[0], gyro.shape[0])
        acc = acc[:min_len]
        gyro = gyro[:min_len]

        if min_len < self.window_size:
            return np.array([]), np.array([])

        # Apply Kalman fusion if enabled
        if self.enable_kalman:
            try:
                features = apply_kalman_fusion(acc, gyro, self.kalman_config)
            except Exception as e:
                # Fallback: concatenate SMV + acc + gyro
                smv = compute_smv(acc)
                features = np.hstack([smv, acc, gyro])
        else:
            # Raw features: SMV + acc + gyro (7 channels)
            smv = compute_smv(acc)
            features = np.hstack([smv, acc, gyro])

        # Create label array
        labels = np.full(features.shape[0], label)

        # Apply windowing
        if self.enable_class_aware_stride:
            windows, win_labels = sliding_window_class_aware(
                features, labels,
                self.window_size,
                self.fall_stride,
                self.adl_stride
            )
        else:
            windows, win_labels = sliding_window_numpy(
                features, labels,
                self.window_size,
                self.stride
            )

        return windows, win_labels

    def get_subject_trials(self, subject_id: int) -> List[Dict]:
        """
        Get all trials for a subject.

        Args:
            subject_id: Subject ID

        Returns:
            List of trial dicts
        """
        return [
            trial for trial in self.trials.values()
            if trial['subject_id'] == subject_id
        ]

    def get_subject_data(
        self,
        subject_ids: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get windowed data for specified subjects.

        Args:
            subject_ids: List of subject IDs

        Returns:
            data: (N, T, C) windowed features
            labels: (N,) labels
        """
        all_windows = []
        all_labels = []

        for subj in subject_ids:
            trials = self.get_subject_trials(subj)
            for trial in trials:
                windows, labels = self._process_trial(trial)
                if windows.size > 0:
                    all_windows.append(windows)
                    all_labels.append(labels)

        if not all_windows:
            return np.array([]), np.array([])

        return np.concatenate(all_windows), np.concatenate(all_labels)

    def prepare_loso_fold(
        self,
        test_subject: int,
        val_subjects: List[int],
        train_only_subjects: Optional[List[int]] = None
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Prepare train/val/test splits for LOSO cross-validation.

        Args:
            test_subject: Subject ID for testing
            val_subjects: Subject IDs for validation
            train_only_subjects: Subjects used only for training (optional)

        Returns:
            Dict with 'train', 'val', 'test' splits
        """
        # Determine training subjects
        all_subjects = set(self.subjects)
        test_subjects = {test_subject}
        val_subjects_set = set(val_subjects)

        train_subjects = all_subjects - test_subjects - val_subjects_set
        if train_only_subjects:
            train_subjects = train_subjects.union(set(train_only_subjects))
            train_subjects = train_subjects - test_subjects - val_subjects_set

        # Get data for each split
        train_data, train_labels = self.get_subject_data(list(train_subjects))
        val_data, val_labels = self.get_subject_data(list(val_subjects_set))
        test_data, test_labels = self.get_subject_data([test_subject])

        # Validate
        validate_window_data(train_data, train_labels, "train")
        validate_window_data(val_data, val_labels, "val")
        validate_window_data(test_data, test_labels, "test")

        # Normalize if enabled
        if self.normalize and self.normalize_modalities != 'none':
            train_data, val_data, test_data, _ = normalize_data(
                train_data, val_data, test_data
            )

        # Print summary
        print(f"\nLOSO Fold (test={test_subject}):")
        print_data_summary("Train", train_data, train_labels)
        print_data_summary("Val", val_data, val_labels)
        print_data_summary("Test", test_data, test_labels)

        return {
            'train': {'accelerometer': train_data, 'labels': train_labels},
            'val': {'accelerometer': val_data, 'labels': val_labels},
            'test': {'accelerometer': test_data, 'labels': test_labels},
        }

    def get_test_candidates(
        self,
        exclude_validation: List[int] = None,
        young_only: bool = True
    ) -> List[int]:
        """
        Get list of subjects that can be used for LOSO testing.

        Args:
            exclude_validation: Subjects reserved for validation
            young_only: Only include young subjects (with fall data)

        Returns:
            List of test candidate subject IDs
        """
        candidates = set(self.subjects)

        if young_only:
            candidates = candidates.intersection(set(self.YOUNG_SUBJECTS))

        if exclude_validation:
            candidates = candidates - set(exclude_validation)

        return sorted(candidates)


# =============================================================================
# Convenience Functions
# =============================================================================

def prepare_wedafall(arg) -> 'WEDAFallLoader':
    """
    Prepare WEDA-FALL dataset from config arguments.

    Compatible with FusionTransformer's prepare_dataset() interface.

    Args:
        arg: Namespace with dataset_args

    Returns:
        WEDAFallLoader instance
    """
    dataset_args = arg.dataset_args

    base_path = dataset_args.get('base_path', 'other_datasets/WEDA-FALL/dataset')
    frequency = dataset_args.get('frequency', '50Hz')
    window_size = dataset_args.get('max_length', 128)
    stride = dataset_args.get('stride', 32)
    enable_kalman = dataset_args.get('enable_kalman_fusion', True)
    enable_class_aware = dataset_args.get('enable_class_aware_stride', True)
    fall_stride = dataset_args.get('fall_stride', 16)
    adl_stride = dataset_args.get('adl_stride', 64)
    convert_gyro = dataset_args.get('convert_gyro_to_rad', True)
    normalize = dataset_args.get('enable_normalization', True)
    normalize_mode = dataset_args.get('normalize_modalities', 'acc_only')
    include_elderly = dataset_args.get('include_elderly', False)

    return WEDAFallLoader(
        base_path=base_path,
        frequency=frequency,
        window_size=window_size,
        stride=stride,
        enable_kalman=enable_kalman,
        enable_class_aware_stride=enable_class_aware,
        fall_stride=fall_stride,
        adl_stride=adl_stride,
        convert_gyro_to_rad=convert_gyro,
        normalize=normalize,
        normalize_modalities=normalize_mode,
        include_elderly=include_elderly,
    )


# =============================================================================
# Test Script
# =============================================================================

if __name__ == "__main__":
    import sys

    base_path = sys.argv[1] if len(sys.argv) > 1 else "other_datasets/WEDA-FALL/dataset"

    print("=" * 60)
    print("WEDA-FALL Loader Test")
    print("=" * 60)

    loader = WEDAFallLoader(
        base_path=base_path,
        frequency='50Hz',
        window_size=128,
        stride=32,
        enable_kalman=True,
        enable_class_aware_stride=True,
        include_elderly=False,  # Young only for LOSO
    )

    print(f"\nSubjects: {loader.subjects}")
    print(f"Test candidates (young only): {loader.get_test_candidates(exclude_validation=[13, 14])}")

    # Test LOSO fold
    if len(loader.subjects) > 2:
        fold_data = loader.prepare_loso_fold(
            test_subject=1,
            val_subjects=[13, 14]
        )

        print(f"\nTrain shape: {fold_data['train']['accelerometer'].shape}")
        print(f"Val shape: {fold_data['val']['accelerometer'].shape}")
        print(f"Test shape: {fold_data['test']['accelerometer'].shape}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
