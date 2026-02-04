import numpy as np
from typing import Optional

from ..base import FeatureExtractor
from ..registry import register_feature_extractor


@register_feature_extractor("kalman")
class KalmanFeatureExtractor(FeatureExtractor):
    """Feature extractor with Kalman-filtered Euler orientations.

    Output: [smv, ax, ay, az, roll, pitch, yaw] (7 channels)
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
            raise ValueError("KalmanFeatureExtractor requires pre-computed orientations")

        n_samples = acc.shape[0]
        features = np.zeros((n_samples, 7))

        # SMV (Signal Magnitude Vector)
        features[:, 0] = np.sqrt(np.sum(acc**2, axis=1))

        # Raw accelerometer
        features[:, 1:4] = acc

        # Kalman-filtered orientations [roll, pitch, yaw]
        features[:, 4:7] = orientations

        return features
