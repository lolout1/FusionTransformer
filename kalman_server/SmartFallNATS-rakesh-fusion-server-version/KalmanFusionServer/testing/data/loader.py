"""Load test data from JSON/Parquet files."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Iterator, Optional, List
import numpy as np

from .schema import TestWindow, TestSession


class TestDataLoader:
    """Load and parse test data from various formats."""

    def __init__(self, data_path: str | Path):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

    def load_windows(self) -> list[TestWindow]:
        """Load all windows from data file."""
        suffix = self.data_path.suffix.lower()

        if suffix == '.json':
            return self._load_json()
        elif suffix == '.parquet':
            return self._load_parquet()
        elif suffix == '.jsonl':
            return self._load_jsonl()
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def _load_json(self) -> list[TestWindow]:
        """Parse prediction-data-couchbase.json format."""
        with open(self.data_path, 'r') as f:
            data = json.load(f)

        windows = []
        records = data.get('prediction_data', data if isinstance(data, list) else [])

        for idx, record in enumerate(records):
            try:
                window = self._parse_record(record, idx)
                windows.append(window)
            except Exception as e:
                print(f"Warning: Skipping record {idx}: {e}")

        return windows

    def _load_jsonl(self) -> list[TestWindow]:
        """Load from JSON Lines format."""
        windows = []
        with open(self.data_path, 'r') as f:
            for idx, line in enumerate(f):
                if line.strip():
                    record = json.loads(line)
                    try:
                        window = self._parse_record(record, idx)
                        windows.append(window)
                    except Exception as e:
                        print(f"Warning: Skipping line {idx}: {e}")
        return windows

    def _load_parquet(self) -> list[TestWindow]:
        """Load from Parquet format."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for Parquet support: pip install pandas pyarrow")

        df = pd.read_parquet(self.data_path)
        windows = []

        for idx, row in df.iterrows():
            try:
                window = self._parse_parquet_row(row, idx)
                windows.append(window)
            except Exception as e:
                print(f"Warning: Skipping row {idx}: {e}")

        return windows

    def _parse_record(self, record: dict, idx: int) -> TestWindow:
        """Parse a single JSON record into TestWindow."""
        # Parse comma-separated accelerometer values
        acc_x = self._parse_csv_array(record.get('watch_accelerometer_x', ''))
        acc_y = self._parse_csv_array(record.get('watch_accelerometer_y', ''))
        acc_z = self._parse_csv_array(record.get('watch_accelerometer_z', ''))

        # Parse comma-separated gyroscope values
        gyro_x = self._parse_csv_array(record.get('watch_gyroscope_x', ''))
        gyro_y = self._parse_csv_array(record.get('watch_gyroscope_y', ''))
        gyro_z = self._parse_csv_array(record.get('watch_gyroscope_z', ''))

        # Validate lengths
        expected_len = 128
        for name, arr in [('acc_x', acc_x), ('acc_y', acc_y), ('acc_z', acc_z),
                          ('gyro_x', gyro_x), ('gyro_y', gyro_y), ('gyro_z', gyro_z)]:
            if len(arr) != expected_len:
                raise ValueError(f"{name} has {len(arr)} samples, expected {expected_len}")

        # Stack into (128, 3) arrays
        acc = np.stack([acc_x, acc_y, acc_z], axis=1).astype(np.float32)
        gyro = np.stack([gyro_x, gyro_y, gyro_z], axis=1).astype(np.float32)

        # Parse label - try 'label' field first, then infer from 'type'
        label = record.get('label')
        error_type = record.get('type', '')

        if label:
            ground_truth = 1 if label.lower() == 'fall' else 0
        elif error_type in ('FN', 'TP'):
            # FN = False Negative (actual Fall, predicted ADL)
            # TP = True Positive (actual Fall, predicted Fall)
            label = 'Fall'
            ground_truth = 1
        elif error_type in ('FP', 'TN'):
            # FP = False Positive (actual ADL, predicted Fall)
            # TN = True Negative (actual ADL, predicted ADL)
            label = 'ADL'
            ground_truth = 0
        else:
            label = 'ADL'
            ground_truth = 0

        return TestWindow(
            uuid=record.get('uuid', f'unknown_{idx}'),
            timestamp_ms=record.get('tsMillis', idx * 4000),  # Estimate if missing
            acc=acc,
            gyro=gyro,
            label=label,
            ground_truth=ground_truth,
            original_prediction=record.get('probability', -1.0),
            original_type=record.get('type', ''),
        )

    def _parse_parquet_row(self, row, idx: int) -> TestWindow:
        """Parse a Parquet row into TestWindow."""
        # Handle different possible column formats
        if 'acc' in row and 'gyro' in row:
            acc = np.array(row['acc'], dtype=np.float32)
            gyro = np.array(row['gyro'], dtype=np.float32)
        else:
            # Try individual columns
            acc = np.stack([row['acc_x'], row['acc_y'], row['acc_z']], axis=1).astype(np.float32)
            gyro = np.stack([row['gyro_x'], row['gyro_y'], row['gyro_z']], axis=1).astype(np.float32)

        label = row.get('label', 'ADL')
        ground_truth = 1 if str(label).lower() == 'fall' else 0

        return TestWindow(
            uuid=row.get('uuid', f'unknown_{idx}'),
            timestamp_ms=row.get('timestamp_ms', idx * 4000),
            acc=acc,
            gyro=gyro,
            label=label,
            ground_truth=ground_truth,
            original_prediction=row.get('probability', -1.0),
            original_type=row.get('type', ''),
        )

    @staticmethod
    def _parse_csv_array(csv_string: str) -> np.ndarray:
        """Parse comma-separated values into numpy array."""
        if not csv_string or csv_string == 'nan':
            return np.zeros(128, dtype=np.float32)
        values = [float(x) for x in csv_string.split(',') if x.strip()]
        return np.array(values, dtype=np.float32)

    def group_into_sessions(
        self,
        windows: list[TestWindow],
        time_gap_threshold_ms: int = 10000
    ) -> list[TestSession]:
        """Group windows into sessions by UUID and time proximity.

        Windows within time_gap_threshold_ms of each other (same UUID) are grouped.
        """
        if not windows:
            return []

        # Sort by UUID then timestamp
        sorted_windows = sorted(windows, key=lambda w: (w.uuid, w.timestamp_ms))

        sessions = []
        current_session: Optional[TestSession] = None

        for window in sorted_windows:
            # Start new session if UUID changes or time gap too large
            start_new = (
                current_session is None or
                current_session.uuid != window.uuid or
                (window.timestamp_ms - current_session.windows[-1].timestamp_ms) > time_gap_threshold_ms
            )

            if start_new:
                if current_session is not None:
                    sessions.append(current_session)

                session_id = f"{window.uuid}_{window.timestamp_ms}"
                current_session = TestSession(
                    uuid=window.uuid,
                    session_id=session_id,
                    ground_truth_label=window.label,
                )

            current_session.add_window(window)

            # Update session label if any window is Fall
            if window.label.lower() == 'fall':
                current_session.ground_truth_label = "Fall"

        if current_session is not None:
            sessions.append(current_session)

        return sessions

    def iter_windows(self) -> Iterator[TestWindow]:
        """Iterate over windows one at a time (memory efficient)."""
        for window in self.load_windows():
            yield window

    def get_stats(self) -> dict:
        """Get statistics about the loaded data."""
        windows = self.load_windows()
        fall_count = sum(1 for w in windows if w.ground_truth == 1)
        adl_count = len(windows) - fall_count

        uuids = set(w.uuid for w in windows)

        return {
            'total_windows': len(windows),
            'fall_windows': fall_count,
            'adl_windows': adl_count,
            'unique_uuids': len(uuids),
            'fall_ratio': fall_count / len(windows) if windows else 0,
        }
