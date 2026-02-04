import numpy as np
from typing import Optional

from ..base import FeatureExtractor
from ..registry import register_feature_extractor


@register_feature_extractor("kalman_gyro_mag")
class KalmanGyroMagExtractor(FeatureExtractor):
    """Feature extractor with Kalman-filtered roll/pitch and gyro magnitude.

    Replaces yaw (prone to drift) with gyro magnitude.
    Output: [smv, ax, ay, az, roll, pitch, gyro_mag] (7 channels)
    """

    @property
    def num_channels(self) -> int:
        return 7

    @property
    def requires_kalman(self) -> bool:
        return True

    def extract(
        self,
        acc: np.ndarray,
        gyro: np.ndarray,
        orientations: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if orientations is None:
            raise ValueError("KalmanGyroMagExtractor requires pre-computed orientations")

        n_samples = acc.shape[0]
        features = np.zeros((n_samples, 7))

        # SMV
        features[:, 0] = np.sqrt(np.sum(acc**2, axis=1))

        # Raw accelerometer
        features[:, 1:4] = acc

        # Kalman-filtered roll, pitch (no yaw)
        features[:, 4:6] = orientations[:, :2]

        # Gyro magnitude instead of yaw
        features[:, 6] = np.sqrt(np.sum(gyro**2, axis=1))

        return features
