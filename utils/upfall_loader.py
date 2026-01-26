"""
UP-FALL Dataset Loader for FusionTransformer.

UP-FALL is a public fall detection dataset with multiple sensors.
This loader extracts wrist accelerometer and gyroscope data.

Dataset: https://sites.google.com/up.edu.mx/har-up/
Paper: Martinez-Villasenor et al. "UP-Fall Detection Dataset" (2019)

Data format:
- Single CSV (CompleteDataSet.csv) with all trials
- Wrist accelerometer: columns 29-31 [x, y, z] in g
- Wrist gyroscope: columns 32-34 [x, y, z] in deg/s
- 17 subjects, 11 activities (5 falls, 6 ADLs)
- ~50Hz sampling rate
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from .external_dataset_utils import (
    sliding_window_numpy,
    sliding_window_class_aware,
    apply_kalman_fusion,
    get_kalman_config_for_rate,
    normalize_data,
    compute_smv,
    UPFALL_LABEL_MAP,
    validate_window_data,
    print_data_summary,
)


class UPFallLoader:
    """
    UP-FALL dataset loader compatible with FusionTransformer.

    Loads wrist IMU data from CompleteDataSet.csv and prepares
    windowed data for fall detection training.
    """

    # Column indices in CompleteDataSet.csv
    WRIST_ACC_COLS = [29, 30, 31]  # x, y, z acceleration (g)
    WRIST_GYRO_COLS = [32, 33, 34]  # x, y, z angular velocity (deg/s)

    # Dataset properties
    # Actual measured rate: mean=19.04Hz, median=20.78Hz (NOT 50Hz as originally assumed)
    # Using 18Hz as conservative estimate based on timestamp analysis
    SAMPLING_RATE = 18.0  # Hz (measured from CompleteDataSet.csv timestamps)
    N_SUBJECTS = 17

    def __init__(
        self,
        csv_path: str,
        window_size: int = 128,
        stride: int = 32,
        enable_kalman: bool = True,
        enable_class_aware_stride: bool = True,
        fall_stride: int = 16,
        adl_stride: int = 64,
        convert_gyro_to_rad: bool = True,
        normalize: bool = True,
        normalize_modalities: str = 'acc_only',
        kalman_warmup_discard: float = 0.0,
        **kwargs
    ):
        """
        Initialize UP-FALL loader.

        Args:
            csv_path: Path to CompleteDataSet.csv
            window_size: Sliding window length (samples)
            stride: Default stride between windows
            enable_kalman: Apply Kalman fusion to IMU data
            enable_class_aware_stride: Use different strides for falls/ADLs
            fall_stride: Stride for fall windows (more overlap)
            adl_stride: Stride for ADL windows (less overlap)
            convert_gyro_to_rad: Convert gyro from deg/s to rad/s
            normalize: Apply z-score normalization
            normalize_modalities: 'all', 'acc_only', or 'none'
            kalman_warmup_discard: Seconds to discard at start of trial after Kalman
                                   (allows filter to converge before windowing)
            **kwargs: Additional Kalman config options:
                - kalman_include_smv: bool
                - kalman_exclude_yaw: bool
                - kalman_include_raw_gyro: bool
                - kalman_orientation_only: bool
                - kalman_Q_orientation, kalman_Q_rate, kalman_R_acc, kalman_R_gyro: float
        """
        self.csv_path = csv_path
        self.window_size = window_size
        self.stride = stride
        self.enable_kalman = enable_kalman
        self.enable_class_aware_stride = enable_class_aware_stride
        self.fall_stride = fall_stride
        self.adl_stride = adl_stride
        self.convert_gyro_to_rad = convert_gyro_to_rad
        self.normalize = normalize
        self.normalize_modalities = normalize_modalities
        # Kalman warm-up discard: convert seconds to samples
        self.kalman_warmup_samples = int(kalman_warmup_discard * self.SAMPLING_RATE)

        # Kalman config: start with defaults for 50Hz, then override with kwargs
        self.kalman_config = get_kalman_config_for_rate(self.SAMPLING_RATE)
        # Merge any kalman-related kwargs
        for key, value in kwargs.items():
            if key.startswith('kalman_') or key in ('filter_fs',):
                self.kalman_config[key] = value

        # Statistics tracking (reset per fold)
        self.fold_stats = {'fall_windows': 0, 'adl_windows': 0, 'fall_trials': 0, 'adl_trials': 0}

        # Load and group data by subject/activity
        self.data = self._load_csv(csv_path)
        self.subjects = sorted(self.data.keys())

        print(f"UP-FALL Loader initialized:")
        print(f"  CSV: {csv_path}")
        print(f"  Subjects: {len(self.subjects)}")
        print(f"  Window: {window_size} samples @ {self.SAMPLING_RATE}Hz")
        print(f"  Kalman fusion: {enable_kalman}")

    def _load_csv(self, csv_path: str) -> Dict[int, Dict[int, Dict]]:
        """
        Load and parse CompleteDataSet.csv.

        Returns:
            Nested dict: {subject_id: {activity_id: {'acc': array, 'gyro': array, 'label': int}}}
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"UP-FALL CSV not found: {csv_path}")

        print(f"Loading UP-FALL CSV: {csv_path}")
        df = pd.read_csv(csv_path, low_memory=False)

        # Skip header row (row 0 has units, row 1 has actual header)
        # The actual data starts from row 2
        df = df.iloc[1:].reset_index(drop=True)

        # Extract columns
        acc_data = df.iloc[:, self.WRIST_ACC_COLS].astype(float).values
        gyro_data = df.iloc[:, self.WRIST_GYRO_COLS].astype(float).values
        subjects = df['Subject'].astype(int).values
        activities = df['Activity'].astype(int).values

        # Convert gyro to rad/s if requested
        if self.convert_gyro_to_rad:
            gyro_data = np.deg2rad(gyro_data)

        # Group by subject and activity
        data = defaultdict(lambda: defaultdict(dict))

        for subj in np.unique(subjects):
            for act in np.unique(activities):
                mask = (subjects == subj) & (activities == act)
                if not np.any(mask):
                    continue

                acc = acc_data[mask]
                gyro = gyro_data[mask]
                label = UPFALL_LABEL_MAP.get(act, 0)

                data[subj][act] = {
                    'acc': acc,
                    'gyro': gyro,
                    'label': label,
                    'n_samples': acc.shape[0],
                }

        print(f"  Loaded {len(data)} subjects, {sum(len(v) for v in data.values())} trials")
        return dict(data)

    def get_subject_trials(self, subject_id: int) -> List[Dict]:
        """
        Get all trials for a subject.

        Args:
            subject_id: Subject ID (1-17)

        Returns:
            List of trial dicts with 'acc', 'gyro', 'label'
        """
        if subject_id not in self.data:
            return []
        return list(self.data[subject_id].values())

    def _process_trial(self, trial: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single trial: Kalman fusion, windowing.

        Args:
            trial: Dict with 'acc', 'gyro', 'label'

        Returns:
            windows: (N, T, C) windowed data
            labels: (N,) labels for each window
        """
        acc = trial['acc']
        gyro = trial['gyro']
        label = trial['label']

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
                print(f"  Warning: Kalman fusion failed: {e}")
                # Fallback: concatenate acc + gyro
                smv = compute_smv(acc)
                features = np.hstack([smv, acc, gyro])

            # Optional: Discard warm-up period if configured
            # Note: Usually not needed since Kalman runs on full trial before windowing
            # Only useful if early-trial windows have poor performance
            if self.kalman_warmup_samples > 0 and features.shape[0] > self.kalman_warmup_samples + self.window_size:
                features = features[self.kalman_warmup_samples:]
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

    def get_subject_data(
        self,
        subject_ids: List[int],
        track_stats: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get windowed data for specified subjects.

        Args:
            subject_ids: List of subject IDs
            track_stats: Whether to accumulate fold statistics

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
                    if track_stats:
                        n_fall = int((labels == 1).sum())
                        n_adl = int((labels == 0).sum())
                        self.fold_stats['fall_windows'] += n_fall
                        self.fold_stats['adl_windows'] += n_adl
                        self.fold_stats['fall_trials'] += 1 if trial['label'] == 1 else 0
                        self.fold_stats['adl_trials'] += 1 if trial['label'] == 0 else 0

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
            Dict with 'train', 'val', 'test' splits, each containing
            'accelerometer' and 'labels' arrays.
        """
        # Reset fold statistics
        self.fold_stats = {'fall_windows': 0, 'adl_windows': 0, 'fall_trials': 0, 'adl_trials': 0}

        # Determine training subjects
        all_subjects = set(self.subjects)
        test_subjects = {test_subject}
        val_subjects_set = set(val_subjects)

        train_subjects = all_subjects - test_subjects - val_subjects_set
        if train_only_subjects:
            # Add train-only subjects that aren't already in train
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
        exclude_validation: List[int] = None
    ) -> List[int]:
        """
        Get list of subjects that can be used for LOSO testing.

        Args:
            exclude_validation: Subjects reserved for validation

        Returns:
            List of test candidate subject IDs
        """
        candidates = set(self.subjects)
        if exclude_validation:
            candidates = candidates - set(exclude_validation)
        return sorted(candidates)


# =============================================================================
# Convenience Functions
# =============================================================================

def prepare_upfall(arg) -> 'UPFallLoader':
    """
    Prepare UP-FALL dataset from config arguments.

    Compatible with FusionTransformer's prepare_dataset() interface.

    Args:
        arg: Namespace with dataset_args

    Returns:
        UPFallLoader instance
    """
    dataset_args = arg.dataset_args

    csv_path = dataset_args.get('csv_path', 'other_datasets/CompleteDataSet.csv')
    window_size = dataset_args.get('max_length', 128)
    stride = dataset_args.get('stride', 32)
    enable_kalman = dataset_args.get('enable_kalman_fusion', True)
    enable_class_aware = dataset_args.get('enable_class_aware_stride', True)
    fall_stride = dataset_args.get('fall_stride', 16)
    adl_stride = dataset_args.get('adl_stride', 64)
    convert_gyro = dataset_args.get('convert_gyro_to_rad', True)
    normalize = dataset_args.get('enable_normalization', True)
    normalize_mode = dataset_args.get('normalize_modalities', 'acc_only')

    return UPFallLoader(
        csv_path=csv_path,
        window_size=window_size,
        stride=stride,
        enable_kalman=enable_kalman,
        enable_class_aware_stride=enable_class_aware,
        fall_stride=fall_stride,
        adl_stride=adl_stride,
        convert_gyro_to_rad=convert_gyro,
        normalize=normalize,
        normalize_modalities=normalize_mode,
    )


# =============================================================================
# Test Script
# =============================================================================

if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "other_datasets/CompleteDataSet.csv"

    print("=" * 60)
    print("UP-FALL Loader Test")
    print("=" * 60)

    loader = UPFallLoader(
        csv_path=csv_path,
        window_size=128,
        stride=32,
        enable_kalman=True,
        enable_class_aware_stride=True,
    )

    print(f"\nSubjects: {loader.subjects}")

    # Test LOSO fold
    fold_data = loader.prepare_loso_fold(
        test_subject=1,
        val_subjects=[15, 16]
    )

    print(f"\nTrain shape: {fold_data['train']['accelerometer'].shape}")
    print(f"Val shape: {fold_data['val']['accelerometer'].shape}")
    print(f"Test shape: {fold_data['test']['accelerometer'].shape}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
