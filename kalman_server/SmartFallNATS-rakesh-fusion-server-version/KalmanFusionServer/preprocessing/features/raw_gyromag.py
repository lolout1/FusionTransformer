"""Raw accelerometer + gyro magnitude feature extractor (5 channels).

Added 2026-02-04 for raw_gyromag models from s8_16 ablation.
"""

import numpy as np
from typing import Optional

from ..base import FeatureExtractor
from ..registry import register_feature_extractor


@register_feature_extractor("raw_gyromag")
class RawGyroMagExtractor(FeatureExtractor):
    """Feature extractor using raw accelerometer + gyroscope magnitude.

    Output: [smv, ax, ay, az, gyro_mag] (5 channels)

    Does NOT require Kalman filter - uses raw gyro magnitude directly.
    This collapses the 3 gyro channels into a single magnitude value.
    """

    @property
    def num_channels(self) -> int:
        return 5

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
        features = np.zeros((n_samples, 5), dtype=np.float32)

        # SMV (Signal Magnitude Vector)
        features[:, 0] = np.sqrt(np.sum(acc**2, axis=1))

        # Raw accelerometer [ax, ay, az]
        features[:, 1:4] = acc

        # Gyro magnitude = sqrt(gx^2 + gy^2 + gz^2)
        features[:, 4] = np.sqrt(np.sum(gyro**2, axis=1))

        return features
