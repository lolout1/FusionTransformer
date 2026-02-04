"""Data schema definitions for testing framework."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np


@dataclass
class TestWindow:
    """Single test window (128 frames of IMU data)."""
    uuid: str
    timestamp_ms: int
    acc: np.ndarray              # (128, 3) m/s²
    gyro: np.ndarray             # (128, 3) rad/s
    label: str                   # "Fall" or "ADL"
    ground_truth: int            # 1 (Fall) or 0 (ADL)
    original_prediction: float   # Original model prediction (if available)
    original_type: str           # FN, FP, TP, TN (if available)
    session_id: str = ""         # For alpha queue grouping
    sequence_idx: int = 0        # Order within session

    def __post_init__(self):
        if self.acc.shape != (128, 3):
            raise ValueError(f"acc shape must be (128, 3), got {self.acc.shape}")
        if self.gyro.shape != (128, 3):
            raise ValueError(f"gyro shape must be (128, 3), got {self.gyro.shape}")


@dataclass
class TestSession:
    """Group of consecutive windows for alpha queue simulation."""
    uuid: str
    windows: list[TestWindow] = field(default_factory=list)
    ground_truth_label: str = "ADL"  # Session-level label
    session_id: str = ""

    @property
    def num_windows(self) -> int:
        return len(self.windows)

    def add_window(self, window: TestWindow) -> None:
        window.session_id = self.session_id
        window.sequence_idx = len(self.windows)
        self.windows.append(window)


@dataclass
class WindowPrediction:
    """Prediction result for a single window."""
    window_idx: int
    probability: float
    preprocessing_time_ms: float
    inference_time_ms: float
    features: Optional[np.ndarray] = None


@dataclass
class SessionPrediction:
    """Prediction result from alpha queue (10 windows averaged)."""
    decision: str                # "FALL" or "ADL"
    avg_probability: float
    window_probabilities: list[float]
    window_indices: list[int]
    ground_truth: str
    is_correct: bool

    @property
    def error_type(self) -> Optional[str]:
        """Return FN, FP, TP, TN or None."""
        if self.decision == "FALL" and self.ground_truth == "Fall":
            return "TP"
        elif self.decision == "FALL" and self.ground_truth == "ADL":
            return "FP"
        elif self.decision == "ADL" and self.ground_truth == "Fall":
            return "FN"
        elif self.decision == "ADL" and self.ground_truth == "ADL":
            return "TN"
        return None


@dataclass
class TestResults:
    """Aggregated test results."""
    config_name: str
    window_predictions: list[WindowPrediction]
    session_predictions: list[SessionPrediction]
    window_metrics: dict
    session_metrics: dict
    total_windows: int
    total_sessions: int
    avg_preprocessing_ms: float
    avg_inference_ms: float
