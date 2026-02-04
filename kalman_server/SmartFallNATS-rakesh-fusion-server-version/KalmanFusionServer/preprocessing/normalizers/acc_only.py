import pickle
from pathlib import Path
from typing import Optional
import numpy as np

from ..base import Normalizer
from ..registry import register_normalizer


@register_normalizer("acc_only")
class AccOnlyNormalizer(Normalizer):
    """Normalizes only accelerometer channels (0-3: smv, ax, ay, az).

    Channels 4+ (orientation/gyro) are passed through unchanged.
    Uses sklearn-compatible StandardScaler format for the accelerometer scaler.
    """

    ACC_CHANNELS = 4

    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, data: np.ndarray) -> None:
        acc_data = data[:, :self.ACC_CHANNELS]
        self.mean_ = np.mean(acc_data, axis=0)
        self.scale_ = np.std(acc_data, axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        self._fitted = True

    def transform(self, data: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() or load() first.")

        result = data.copy()
        n_channels = data.shape[1]

        # Normalize accelerometer channels (0-3)
        n_acc = min(self.ACC_CHANNELS, n_channels)
        result[:, :n_acc] = (data[:, :n_acc] - self.mean_[:n_acc]) / self.scale_[:n_acc]

        # Channels 4+ remain unchanged
        return result

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"mean_": self.mean_, "scale_": self.scale_}, f)

    def load(self, path: str) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Scaler file not found: {path}")

        with open(path, "rb") as f:
            scaler = pickle.load(f)

        # Handle both sklearn.StandardScaler and dict formats
        if hasattr(scaler, "mean_"):
            self.mean_ = scaler.mean_
            self.scale_ = scaler.scale_
        else:
            self.mean_ = scaler["mean_"]
            self.scale_ = scaler["scale_"]

        self._fitted = True

    @property
    def is_fitted(self) -> bool:
        return self._fitted
