import numpy as np
from typing import Optional

from ..base import FeatureExtractor
from ..registry import register_feature_extractor


@register_feature_extractor("acc_only")
class AccOnlyExtractor(FeatureExtractor):
    """Feature extractor using accelerometer data only.

    Output: [smv, ax, ay, az] (4 channels)
    """

    @property
    def num_channels(self) -> int:
        return 4

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
        features = np.zeros((n_samples, 4))

        # SMV
        features[:, 0] = np.sqrt(np.sum(acc**2, axis=1))

        # Raw accelerometer
        features[:, 1:4] = acc

        return features
