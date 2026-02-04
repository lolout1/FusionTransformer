from pathlib import Path
from typing import Optional
import numpy as np

from config import ServerConfig, FeatureMode, NormalizationMode
from .base import FeatureExtractor, Normalizer, KalmanFilterProtocol
from .registry import FEATURE_EXTRACTORS, NORMALIZERS, KALMAN_FILTERS
from .state import UserStateManager
from .filters import LinearKalmanFilter
from .features import KalmanFeatureExtractor, KalmanGyroMagExtractor, RawFeatureExtractor, AccOnlyExtractor
from .normalizers import NoOpNormalizer, AccOnlyNormalizer, StandardNormalizer


class PreprocessingPipeline:
    """Orchestrates feature extraction, normalization, and state management.

    Supports multiple feature modes and normalization strategies with
    per-user Kalman state persistence.
    """

    def __init__(self, config: ServerConfig):
        self.config = config

        # Initialize state manager
        self.state_manager = UserStateManager(
            timeout_ms=config.state.timeout_ms,
            cache_ttl_ms=config.state.cache_ttl_ms,
            window_size=config.window_size,
            default_fs_hz=config.default_fs_hz,
        )

        # Initialize feature extractor
        self.feature_extractor = self._create_feature_extractor()

        # Initialize normalizer
        self.normalizer = self._create_normalizer()

        # Kalman filter template (used to create per-user instances)
        self._kalman_config = {
            "Q_orientation": config.kalman.Q_orientation,
            "Q_rate": config.kalman.Q_rate,
            "R_acc": config.kalman.R_acc,
            "R_gyro": config.kalman.R_gyro,
        }

    def _create_feature_extractor(self) -> FeatureExtractor:
        mode = self.config.preprocessing.feature_mode
        if mode == FeatureMode.KALMAN:
            return KalmanFeatureExtractor()
        elif mode == FeatureMode.KALMAN_GYRO_MAG:
            return KalmanGyroMagExtractor()
        elif mode == FeatureMode.RAW:
            return RawFeatureExtractor()
        elif mode == FeatureMode.ACC_ONLY:
            return AccOnlyExtractor()
        else:
            raise ValueError(f"Unknown feature mode: {mode}")

    def _create_normalizer(self) -> Normalizer:
        mode = self.config.preprocessing.normalization_mode
        scaler_path = self.config.preprocessing.scaler_path

        if mode == NormalizationMode.NONE:
            return NoOpNormalizer()
        elif mode == NormalizationMode.ALL:
            normalizer = StandardNormalizer()
        elif mode == NormalizationMode.ACC_ONLY:
            normalizer = AccOnlyNormalizer()
        else:
            raise ValueError(f"Unknown normalization mode: {mode}")

        # Load scaler if path provided
        if scaler_path:
            scaler_path = Path(scaler_path)
            if scaler_path.exists():
                normalizer.load(str(scaler_path))

        return normalizer

    def _create_kalman_filter(self) -> KalmanFilterProtocol:
        return LinearKalmanFilter(**self._kalman_config)

    def process(
        self,
        user_id: str,
        acc: np.ndarray,
        gyro: np.ndarray,
        timestamps: np.ndarray,
    ) -> np.ndarray:
        """Process sensor data through the full pipeline.

        Args:
            user_id: Unique user/device identifier
            acc: Accelerometer data (N, 3) in m/s^2
            gyro: Gyroscope data (N, 3) in deg/s or rad/s
            timestamps: Per-sample timestamps in ms (N,)

        Returns:
            features: Normalized features ready for model inference (N, C)
        """
        # Convert gyro to rad/s if needed
        if self.config.preprocessing.convert_gyro_to_rad:
            gyro = np.deg2rad(gyro)

        # Compute orientations if needed
        orientations = None
        if self.feature_extractor.requires_kalman:
            kalman_filter = self._create_kalman_filter()
            orientations = self.state_manager.process_window(
                user_id=user_id,
                acc=acc,
                gyro=gyro,
                timestamps=timestamps,
                kalman_filter=kalman_filter,
            )

        # Extract features
        features = self.feature_extractor.extract(acc, gyro, orientations)

        # Normalize
        if self.normalizer.is_fitted:
            features = self.normalizer.transform(features)

        return features

    def process_legacy(
        self,
        user_id: str,
        acc_x: np.ndarray,
        acc_y: np.ndarray,
        acc_z: np.ndarray,
        gyro_x: np.ndarray,
        gyro_y: np.ndarray,
        gyro_z: np.ndarray,
        timestamps: np.ndarray,
    ) -> np.ndarray:
        """Legacy interface matching current server format.

        Converts separate axis arrays to stacked format.
        """
        acc = np.stack([acc_x, acc_y, acc_z], axis=1)
        gyro = np.stack([gyro_x, gyro_y, gyro_z], axis=1)
        return self.process(user_id, acc, gyro, timestamps)

    @property
    def num_output_channels(self) -> int:
        return self.feature_extractor.num_channels

    def cleanup_stale_sessions(self) -> int:
        return self.state_manager.cleanup_stale()
