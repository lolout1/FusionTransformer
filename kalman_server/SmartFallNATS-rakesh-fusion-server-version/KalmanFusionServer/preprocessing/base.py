from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class KalmanState:
    x: np.ndarray  # State vector [roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]
    P: np.ndarray  # Covariance matrix (6x6)


class KalmanFilterProtocol(ABC):
    @abstractmethod
    def predict(self, dt: float) -> None:
        """Prediction step with time delta."""

    @abstractmethod
    def update(self, acc: np.ndarray, gyro: np.ndarray) -> None:
        """Update step with accelerometer and gyroscope measurements."""

    @abstractmethod
    def get_orientation(self) -> np.ndarray:
        """Get current orientation estimate [roll, pitch, yaw]."""

    @abstractmethod
    def get_state(self) -> KalmanState:
        """Get full filter state for persistence."""

    @abstractmethod
    def set_state(self, state: KalmanState) -> None:
        """Restore filter state from persistence."""

    @abstractmethod
    def reset(self, acc: Optional[np.ndarray] = None) -> None:
        """Reset filter, optionally initializing from accelerometer."""


class FeatureExtractor(ABC):
    @property
    @abstractmethod
    def num_channels(self) -> int:
        """Number of output channels."""

    @property
    @abstractmethod
    def requires_kalman(self) -> bool:
        """Whether this extractor requires Kalman filtering."""

    @abstractmethod
    def extract(
        self,
        acc: np.ndarray,
        gyro: np.ndarray,
        orientations: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Extract features from raw sensor data.

        Args:
            acc: Accelerometer data (N, 3) in m/s^2
            gyro: Gyroscope data (N, 3) in rad/s (or deg/s if convert_gyro_to_rad=False)
            orientations: Pre-computed orientations (N, 3) if available, else None

        Returns:
            Features array (N, num_channels)
        """


class Normalizer(ABC):
    @abstractmethod
    def fit(self, data: np.ndarray) -> None:
        """Fit normalizer to data (for training)."""

    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted parameters."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Save normalizer parameters to file."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load normalizer parameters from file."""

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Whether normalizer has been fitted or loaded."""
