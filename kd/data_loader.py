"""
KD Data Loader for SmartFallMM dataset.

Loads paired skeleton and IMU data for knowledge distillation training.
Compatible with existing utils/loader.py patterns.
"""

import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class TrialMatcher:
    """Matches skeleton and IMU files for the same trial."""

    def __init__(self, data_root: Path, age_group: str = 'young', device: str = 'watch'):
        self.data_root = Path(data_root)
        self.age_group = age_group
        self.device = device

        # Directory paths
        self.skeleton_dir = self.data_root / age_group / 'skeleton'
        self.acc_dir = self.data_root / age_group / 'accelerometer' / device
        self.gyro_dir = self.data_root / age_group / 'gyroscope' / device

    def find_matched_trials(
        self,
        subjects: Optional[List[int]] = None,
        require_skeleton: bool = True,
        require_gyro: bool = True,
    ) -> List[Dict]:
        """
        Find all trials with matched modalities.

        Args:
            subjects: List of subject IDs to include (None = all)
            require_skeleton: If True, only include trials with skeleton
            require_gyro: If True, only include trials with gyroscope

        Returns:
            List of trial dicts with 'subject_id', 'action_id', 'trial_id', 'files'
        """
        trials = []

        # Find all accelerometer files (required for all)
        acc_files = list(self.acc_dir.glob('S*.csv'))

        for acc_file in acc_files:
            # Parse trial ID: S29A03T02 -> subject=29, action=03, trial=02
            trial_id = acc_file.stem
            try:
                parts = trial_id.split('A')
                subject_id = int(parts[0][1:])  # Remove 'S' prefix
                action_trial = parts[1].split('T')
                action_id = int(action_trial[0])
                sequence = int(action_trial[1])
            except (ValueError, IndexError):
                continue

            # Filter by subjects
            if subjects is not None and subject_id not in subjects:
                continue

            # Check for skeleton
            skel_file = self.skeleton_dir / f'{trial_id}.csv'
            if require_skeleton and not skel_file.exists():
                continue

            # Check for gyroscope
            gyro_file = self.gyro_dir / f'{trial_id}.csv'
            if require_gyro and not gyro_file.exists():
                continue

            # Build trial dict
            trial = {
                'subject_id': subject_id,
                'action_id': action_id,
                'sequence': sequence,
                'trial_id': trial_id,
                'label': int(action_id > 9),  # Fall detection: action > 9 is fall
                'files': {
                    'accelerometer': str(acc_file),
                }
            }

            if skel_file.exists():
                trial['files']['skeleton'] = str(skel_file)

            if gyro_file.exists():
                trial['files']['gyroscope'] = str(gyro_file)

            trials.append(trial)

        return sorted(trials, key=lambda t: (t['subject_id'], t['action_id'], t['sequence']))


def load_imu_with_timestamps(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load IMU CSV file preserving timestamps.

    Format: datetime_string,x,y,z (no header)

    Returns:
        timestamps: (N,) array of timestamps in seconds since start
        values: (N, 3) array of x, y, z values
    """
    try:
        df = pd.read_csv(file_path, header=None, names=['timestamp', 'x', 'y', 'z'])

        # Parse timestamps (SmartFallMM format: YYYY-MM-DD HH:MM:SS.fff)
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
        except (ValueError, TypeError):
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        # Drop rows with parsing errors
        df = df.dropna()

        if len(df) == 0:
            return np.zeros(0, dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

        # Convert values to float, coercing errors
        for col in ['x', 'y', 'z']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()

        if len(df) == 0:
            return np.zeros(0, dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

        # Convert to seconds since start
        t0 = df['timestamp'].iloc[0]
        timestamps = (df['timestamp'] - t0).dt.total_seconds().values.astype(np.float32)
        values = df[['x', 'y', 'z']].values.astype(np.float32)

        return timestamps, values
    except Exception:
        return np.zeros(0, dtype=np.float32), np.zeros((0, 3), dtype=np.float32)


def load_skeleton(file_path: str) -> np.ndarray:
    """
    Load skeleton CSV file (no timestamps, sequential frames at 30 FPS).

    Format: 96 values per row (32 joints × 3 coords), no header

    Returns:
        skeleton: (T, 96) array of joint positions
    """
    try:
        df = pd.read_csv(file_path, header=None)
        if df.empty or df.shape[1] < 96:
            return np.zeros((0, 96), dtype=np.float32)
        return df.values.astype(np.float32)
    except Exception:
        return np.zeros((0, 96), dtype=np.float32)


def load_imu_values_only(file_path: str) -> np.ndarray:
    """
    Load IMU CSV file without timestamps (compatible with existing pipeline).

    Format: datetime_string,x,y,z (no header)

    Returns:
        values: (N, 3) array of x, y, z values
    """
    df = pd.read_csv(file_path, header=None, usecols=[1, 2, 3])
    return df.values.astype(np.float32)


class KDDataset(Dataset):
    """
    Dataset for Knowledge Distillation training.

    Provides paired (skeleton, IMU) data for teacher-student training.
    Handles variable-length sequences via padding.
    """

    def __init__(
        self,
        trials: List[Dict],
        max_length: int = 128,
        skeleton_fps: float = 30.0,
        return_timestamps: bool = True,
        normalize_imu: bool = True,
        normalize_skeleton: bool = False,
    ):
        """
        Args:
            trials: List of trial dicts from TrialMatcher
            max_length: Max sequence length (windows are this size)
            skeleton_fps: Skeleton frame rate for synthetic timestamps
            return_timestamps: If True, return IMU timestamps
            normalize_imu: If True, normalize IMU values
            normalize_skeleton: If True, normalize skeleton values
        """
        self.trials = trials
        self.max_length = max_length
        self.skeleton_fps = skeleton_fps
        self.return_timestamps = return_timestamps
        self.normalize_imu = normalize_imu
        self.normalize_skeleton = normalize_skeleton

        # Precompute normalization stats
        self.imu_mean = None
        self.imu_std = None
        self.skeleton_mean = None
        self.skeleton_std = None

        if normalize_imu:
            self._compute_imu_stats()
        if normalize_skeleton:
            self._compute_skeleton_stats()

    def _compute_imu_stats(self):
        """Compute mean/std for IMU normalization from all trials."""
        all_acc = []
        all_gyro = []

        for trial in self.trials:
            if 'accelerometer' in trial['files']:
                acc = load_imu_values_only(trial['files']['accelerometer'])
                all_acc.append(acc)
            if 'gyroscope' in trial['files']:
                gyro = load_imu_values_only(trial['files']['gyroscope'])
                all_gyro.append(gyro)

        if all_acc:
            all_acc = np.concatenate(all_acc, axis=0)
            self.acc_mean = np.mean(all_acc, axis=0)
            self.acc_std = np.std(all_acc, axis=0) + 1e-8
        else:
            self.acc_mean = np.zeros(3)
            self.acc_std = np.ones(3)

        if all_gyro:
            all_gyro = np.concatenate(all_gyro, axis=0)
            self.gyro_mean = np.mean(all_gyro, axis=0)
            self.gyro_std = np.std(all_gyro, axis=0) + 1e-8
        else:
            self.gyro_mean = np.zeros(3)
            self.gyro_std = np.ones(3)

    def _compute_skeleton_stats(self):
        """Compute mean/std for skeleton normalization."""
        all_skel = []

        for trial in self.trials:
            if 'skeleton' in trial['files']:
                skel = load_skeleton(trial['files']['skeleton'])
                all_skel.append(skel)

        if all_skel:
            all_skel = np.concatenate(all_skel, axis=0)
            self.skeleton_mean = np.mean(all_skel, axis=0)
            self.skeleton_std = np.std(all_skel, axis=0) + 1e-8
        else:
            self.skeleton_mean = np.zeros(96)
            self.skeleton_std = np.ones(96)

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        trial = self.trials[idx]
        result = {'label': torch.tensor(trial['label'], dtype=torch.long)}

        # Load accelerometer with timestamps
        if 'accelerometer' in trial['files']:
            acc_ts, acc_vals = load_imu_with_timestamps(trial['files']['accelerometer'])

            if self.normalize_imu:
                acc_vals = (acc_vals - self.acc_mean) / self.acc_std

            result['acc_values'] = torch.from_numpy(acc_vals)
            result['acc_timestamps'] = torch.from_numpy(acc_ts)

        # Load gyroscope with timestamps
        if 'gyroscope' in trial['files']:
            gyro_ts, gyro_vals = load_imu_with_timestamps(trial['files']['gyroscope'])

            if self.normalize_imu:
                gyro_vals = (gyro_vals - self.gyro_mean) / self.gyro_std

            result['gyro_values'] = torch.from_numpy(gyro_vals)
            result['gyro_timestamps'] = torch.from_numpy(gyro_ts)

        # Load skeleton (no timestamps - generate synthetic at 30 FPS)
        if 'skeleton' in trial['files']:
            skeleton = load_skeleton(trial['files']['skeleton'])

            if self.normalize_skeleton:
                skeleton = (skeleton - self.skeleton_mean) / self.skeleton_std

            # Generate synthetic timestamps for skeleton
            n_frames = len(skeleton)
            skel_ts = np.arange(n_frames, dtype=np.float32) / self.skeleton_fps

            result['skeleton'] = torch.from_numpy(skeleton)
            result['skeleton_timestamps'] = torch.from_numpy(skel_ts)

        return result


def collate_kd_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for KD dataset with variable-length sequences.

    Pads sequences to max length in batch and creates attention masks.
    """
    # Filter out None or empty batch items
    batch = [b for b in batch if b is not None and len(b) > 0]
    if len(batch) == 0:
        return {}

    result = {}

    # Collect labels (support both 'label' and 'labels' keys)
    label_key = 'labels' if 'labels' in batch[0] else 'label'
    labels = []
    for b in batch:
        if label_key in b:
            labels.append(b[label_key])
        else:
            # Fallback: try both keys
            if 'label' in b:
                labels.append(b['label'])
            elif 'labels' in b:
                labels.append(b['labels'])
            else:
                labels.append(torch.tensor(0, dtype=torch.long))  # Default
    result['labels'] = torch.stack(labels)

    # Pad and stack each modality
    for key in ['acc_values', 'gyro_values', 'skeleton']:
        if key in batch[0] and all(key in b for b in batch):
            values = [b[key] for b in batch]
            if not values or any(v is None for v in values):
                continue
            max_len = max(v.shape[0] for v in values)

            # Pad to max length
            padded = []
            masks = []
            for v in values:
                pad_len = max_len - v.shape[0]
                if pad_len > 0:
                    v_padded = torch.nn.functional.pad(v, (0, 0, 0, pad_len))
                else:
                    v_padded = v[:max_len]
                padded.append(v_padded)

                # Create mask (True for valid positions)
                mask = torch.zeros(max_len, dtype=torch.bool)
                mask[:v.shape[0]] = True
                masks.append(mask)

            result[key] = torch.stack(padded)
            result[f'{key}_mask'] = torch.stack(masks)

            # Also provide generic 'mask' for convenience (use acc mask as primary)
            if key == 'acc_values' and 'mask' not in result:
                result['mask'] = torch.stack(masks)

    # Pad timestamps similarly
    for key in ['acc_timestamps', 'gyro_timestamps', 'skeleton_timestamps']:
        if key in batch[0] and all(key in b for b in batch):
            timestamps = [b[key] for b in batch]
            if not timestamps or any(t is None for t in timestamps):
                continue
            max_len = max(t.shape[0] for t in timestamps) if timestamps else 1

            padded = []
            for t in timestamps:
                pad_len = max_len - t.shape[0]
                if t.shape[0] == 0:
                    t_padded = torch.zeros(max_len, dtype=t.dtype, device=t.device)
                elif pad_len > 0:
                    t_padded = torch.nn.functional.pad(t, (0, pad_len), value=t[-1].item())
                else:
                    t_padded = t[:max_len]
                padded.append(t_padded)

            result[key] = torch.stack(padded)

    return result


class WindowedKDDataset(Dataset):
    """
    Dataset that pre-windows trials for efficient training.

    Creates fixed-size windows with optional class-aware stride.
    """

    def __init__(
        self,
        trials: List[Dict],
        window_size: int = 128,
        stride: int = 32,
        fall_stride: int = 16,
        adl_stride: int = 64,
        class_aware_stride: bool = True,
        min_window_length: int = 64,
        skeleton_fps: float = 30.0,
        normalize_imu: bool = True,
    ):
        """
        Args:
            trials: List of trial dicts from TrialMatcher
            window_size: Fixed window size
            stride: Default stride (used if class_aware_stride=False)
            fall_stride: Stride for fall windows
            adl_stride: Stride for ADL windows
            class_aware_stride: Use different strides for falls vs ADLs
            min_window_length: Minimum trial length to include
            skeleton_fps: Skeleton frame rate
            normalize_imu: Normalize IMU values
        """
        self.window_size = window_size
        self.skeleton_fps = skeleton_fps
        self.normalize_imu = normalize_imu

        # Build windows
        self.windows = []
        self._build_windows(
            trials, window_size, stride, fall_stride, adl_stride,
            class_aware_stride, min_window_length
        )

        # Compute normalization stats from all data
        if normalize_imu:
            self._compute_stats(trials)

    def _compute_stats(self, trials: List[Dict]):
        """Compute normalization statistics."""
        all_acc = []
        all_gyro = []

        for trial in trials:
            if 'accelerometer' in trial['files']:
                try:
                    acc = load_imu_values_only(trial['files']['accelerometer'])
                    all_acc.append(acc)
                except Exception:
                    pass
            if 'gyroscope' in trial['files']:
                try:
                    gyro = load_imu_values_only(trial['files']['gyroscope'])
                    all_gyro.append(gyro)
                except Exception:
                    pass

        if all_acc:
            all_acc = np.concatenate(all_acc, axis=0)
            self.acc_mean = np.mean(all_acc, axis=0).astype(np.float32)
            self.acc_std = (np.std(all_acc, axis=0) + 1e-8).astype(np.float32)
        else:
            self.acc_mean = np.zeros(3, dtype=np.float32)
            self.acc_std = np.ones(3, dtype=np.float32)

        if all_gyro:
            all_gyro = np.concatenate(all_gyro, axis=0)
            self.gyro_mean = np.mean(all_gyro, axis=0).astype(np.float32)
            self.gyro_std = (np.std(all_gyro, axis=0) + 1e-8).astype(np.float32)
        else:
            self.gyro_mean = np.zeros(3, dtype=np.float32)
            self.gyro_std = np.ones(3, dtype=np.float32)

    def _build_windows(
        self,
        trials: List[Dict],
        window_size: int,
        stride: int,
        fall_stride: int,
        adl_stride: int,
        class_aware_stride: bool,
        min_window_length: int,
    ):
        """Build list of windows from trials."""
        for trial in trials:
            try:
                # Get trial length from accelerometer (reference modality)
                if 'accelerometer' not in trial['files']:
                    continue

                acc_ts, acc_vals = load_imu_with_timestamps(trial['files']['accelerometer'])
                trial_len = len(acc_vals)

                # Skip invalid trials (empty or too short)
                if trial_len < min_window_length or len(acc_ts) == 0:
                    continue

                # Determine stride based on class
                label = trial['label']
                if class_aware_stride:
                    effective_stride = fall_stride if label == 1 else adl_stride
                else:
                    effective_stride = stride

                # Generate window start indices
                n_windows = max(1, (trial_len - window_size) // effective_stride + 1)

                for i in range(n_windows):
                    start = i * effective_stride
                    end = min(start + window_size, trial_len)

                    if end - start < min_window_length:
                        continue

                    self.windows.append({
                        'trial': trial,
                        'start': start,
                        'end': end,
                        'label': label,
                    })

            except Exception:
                continue

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        window = self.windows[idx]
        trial = window['trial']
        start = window['start']
        end = window['end']
        actual_len = end - start

        result = {'label': torch.tensor(window['label'], dtype=torch.long)}

        # Load accelerometer window
        if 'accelerometer' in trial['files']:
            acc_ts, acc_vals = load_imu_with_timestamps(trial['files']['accelerometer'])
            acc_vals = acc_vals[start:end]
            acc_ts = acc_ts[start:end]

            if self.normalize_imu:
                acc_vals = (acc_vals - self.acc_mean) / self.acc_std

            # Pad to window_size if needed
            if actual_len < self.window_size:
                pad_len = self.window_size - actual_len
                acc_vals = np.pad(acc_vals, ((0, pad_len), (0, 0)), mode='constant')
                acc_ts = np.pad(acc_ts, (0, pad_len), mode='edge')

            result['acc_values'] = torch.from_numpy(acc_vals.astype(np.float32))
            result['acc_timestamps'] = torch.from_numpy(acc_ts.astype(np.float32))

        # Load gyroscope window
        if 'gyroscope' in trial['files']:
            try:
                gyro_ts, gyro_vals = load_imu_with_timestamps(trial['files']['gyroscope'])
            except Exception:
                gyro_ts, gyro_vals = np.zeros(0, dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

            # Align gyro to acc indices (may differ slightly)
            gyro_len = len(gyro_vals)
            if gyro_len > 0:
                acc_len_ref = actual_len  # Use actual window length
                g_start = int(start * gyro_len / max(acc_len_ref, 1))
                g_end = min(g_start + (end - start), gyro_len)
                g_start = max(0, min(g_start, gyro_len - 1))
                g_end = max(g_start + 1, min(g_end, gyro_len))

                gyro_vals = gyro_vals[g_start:g_end]
                gyro_ts = gyro_ts[g_start:g_end]

                if self.normalize_imu and len(gyro_vals) > 0:
                    gyro_vals = (gyro_vals - self.gyro_mean) / self.gyro_std

                # Pad to window_size
                if len(gyro_vals) < self.window_size:
                    pad_len = self.window_size - len(gyro_vals)
                    gyro_vals = np.pad(gyro_vals, ((0, pad_len), (0, 0)), mode='constant')
                    if len(gyro_ts) > 0:
                        gyro_ts = np.pad(gyro_ts, (0, pad_len), mode='edge')
                    else:
                        gyro_ts = np.zeros(self.window_size, dtype=np.float32)

                result['gyro_values'] = torch.from_numpy(gyro_vals[:self.window_size].astype(np.float32))
                result['gyro_timestamps'] = torch.from_numpy(gyro_ts[:self.window_size].astype(np.float32))
            else:
                # Empty gyro data - fill with zeros
                result['gyro_values'] = torch.zeros(self.window_size, 3, dtype=torch.float32)
                result['gyro_timestamps'] = torch.zeros(self.window_size, dtype=torch.float32)

        # Load skeleton window
        if 'skeleton' in trial['files']:
            try:
                skeleton = load_skeleton(trial['files']['skeleton'])
            except Exception:
                skeleton = np.zeros((0, 96), dtype=np.float32)

            skel_len = len(skeleton)

            if skel_len > 0:
                # Align skeleton to acc indices using duration ratio
                if 'accelerometer' in trial['files']:
                    try:
                        acc_ts_full, _ = load_imu_with_timestamps(trial['files']['accelerometer'])
                        if len(acc_ts_full) > 1:
                            imu_duration = acc_ts_full[-1] - acc_ts_full[0]
                            skel_duration = skel_len / self.skeleton_fps
                            time_ratio = skel_duration / max(imu_duration, 1e-6)
                            s_start = int(start * time_ratio)
                            s_end = int(end * time_ratio)
                        else:
                            s_start, s_end = start, end
                    except Exception:
                        s_start, s_end = start, end
                else:
                    s_start = int(start * self.skeleton_fps / 30)
                    s_end = int(end * self.skeleton_fps / 30)

                s_start = max(0, min(s_start, skel_len - 1))
                s_end = max(s_start + 1, min(s_end, skel_len))

                skeleton_window = skeleton[s_start:s_end]

                # Generate synthetic timestamps
                skel_ts = np.arange(len(skeleton_window), dtype=np.float32) / self.skeleton_fps

                # Pad to window_size
                if len(skeleton_window) < self.window_size:
                    pad_len = self.window_size - len(skeleton_window)
                    skeleton_window = np.pad(skeleton_window, ((0, pad_len), (0, 0)), mode='constant')
                    if len(skel_ts) > 0:
                        skel_ts = np.pad(skel_ts, (0, pad_len), mode='edge')
                    else:
                        skel_ts = np.zeros(self.window_size, dtype=np.float32)

                result['skeleton'] = torch.from_numpy(skeleton_window[:self.window_size].astype(np.float32))
                result['skeleton_timestamps'] = torch.from_numpy(skel_ts[:self.window_size].astype(np.float32))
            else:
                # Empty skeleton - fill with zeros
                result['skeleton'] = torch.zeros(self.window_size, 96, dtype=torch.float32)
                result['skeleton_timestamps'] = torch.zeros(self.window_size, dtype=torch.float32)

        # Create mask (True for valid positions)
        mask = torch.zeros(self.window_size, dtype=torch.bool)
        mask[:actual_len] = True
        result['mask'] = mask

        return result


def create_kd_dataloaders(
    data_root: str,
    train_subjects: List[int],
    val_subjects: List[int],
    test_subjects: List[int],
    window_size: int = 128,
    batch_size: int = 64,
    num_workers: int = 4,
    class_aware_stride: bool = True,
    fall_stride: int = 16,
    adl_stride: int = 64,
    require_skeleton: bool = True,
    device: str = 'watch',
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders for KD training.

    Args:
        data_root: Path to SmartFallMM data directory
        train_subjects: Subject IDs for training
        val_subjects: Subject IDs for validation
        test_subjects: Subject IDs for testing
        window_size: Fixed window size
        batch_size: Batch size
        num_workers: DataLoader workers
        class_aware_stride: Use different strides for falls vs ADLs
        fall_stride: Stride for fall windows
        adl_stride: Stride for ADL windows
        require_skeleton: Only include trials with skeleton data
        device: IMU device ('watch', 'phone', etc.)

    Returns:
        (train_loader, val_loader, test_loader)
    """
    matcher = TrialMatcher(data_root, device=device)

    # Find trials for each split
    train_trials = matcher.find_matched_trials(
        subjects=train_subjects,
        require_skeleton=require_skeleton,
    )
    val_trials = matcher.find_matched_trials(
        subjects=val_subjects,
        require_skeleton=require_skeleton,
    )
    test_trials = matcher.find_matched_trials(
        subjects=test_subjects,
        require_skeleton=require_skeleton,
    )

    # Create datasets
    train_dataset = WindowedKDDataset(
        train_trials,
        window_size=window_size,
        fall_stride=fall_stride,
        adl_stride=adl_stride,
        class_aware_stride=class_aware_stride,
    )

    val_dataset = WindowedKDDataset(
        val_trials,
        window_size=window_size,
        stride=window_size // 2,  # Fixed stride for validation
        class_aware_stride=False,
    )

    test_dataset = WindowedKDDataset(
        test_trials,
        window_size=window_size,
        stride=window_size // 2,  # Fixed stride for testing
        class_aware_stride=False,
    )

    # Copy normalization stats from train to val/test
    if hasattr(train_dataset, 'acc_mean'):
        val_dataset.acc_mean = train_dataset.acc_mean
        val_dataset.acc_std = train_dataset.acc_std
        val_dataset.gyro_mean = train_dataset.gyro_mean
        val_dataset.gyro_std = train_dataset.gyro_std
        test_dataset.acc_mean = train_dataset.acc_mean
        test_dataset.acc_std = train_dataset.acc_std
        test_dataset.gyro_mean = train_dataset.gyro_mean
        test_dataset.gyro_std = train_dataset.gyro_std

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def prepare_loso_fold(
    data_root: str,
    test_subject: int,
    all_subjects: List[int],
    val_ratio: float = 0.1,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prepare dataloaders for one LOSO fold.

    Args:
        data_root: Path to data directory
        test_subject: Subject ID for testing
        all_subjects: All subject IDs
        val_ratio: Fraction of train subjects for validation
        **kwargs: Passed to create_kd_dataloaders

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Split subjects
    train_subjects = [s for s in all_subjects if s != test_subject]

    # Hold out some for validation
    n_val = max(1, int(len(train_subjects) * val_ratio))
    val_subjects = train_subjects[:n_val]
    train_subjects = train_subjects[n_val:]

    return create_kd_dataloaders(
        data_root=data_root,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=[test_subject],
        **kwargs
    )
