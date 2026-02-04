import numpy as np
from typing import Optional

from ..base import FeatureExtractor
from ..registry import register_feature_extractor


@register_feature_extractor("raw")
class RawFeatureExtractor(FeatureExtractor):
    """Feature extractor using raw sensor data without Kalman filtering.

    Output: [smv, ax, ay, az, gx, gy, gz] (7 channels)
    """

    @property
    def num_channels(self) -> int:
        return 7

    @property
    def requires_kalman(self) -> bool:
        return False

    def extract(
        self,
        acc: np.ndarray,
        gyro: np.ndarray,
        orientations: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        n_samples = acc.shape[0]
        features = np.zeros((n_samples, 7))

        # SMV
        features[:, 0] = np.sqrt(np.sum(acc**2, axis=1))

        # Raw accelerometer
        features[:, 1:4] = acc

        # Raw gyroscope
        features[:, 4:7] = gyro

        return features
