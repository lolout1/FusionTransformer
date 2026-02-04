import numpy as np

from ..base import Normalizer
from ..registry import register_normalizer


@register_normalizer("none")
class NoOpNormalizer(Normalizer):
    """No-op normalizer that passes through data unchanged."""

    def __init__(self):
        self._fitted = True

    def fit(self, data: np.ndarray) -> None:
        pass

    def transform(self, data: np.ndarray) -> np.ndarray:
        return data

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass

    @property
    def is_fitted(self) -> bool:
        return True
