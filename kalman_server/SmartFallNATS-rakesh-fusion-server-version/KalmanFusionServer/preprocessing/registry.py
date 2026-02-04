from typing import Dict, Type, TypeVar, Generic

from .base import FeatureExtractor, Normalizer, KalmanFilterProtocol

T = TypeVar("T")


class ComponentRegistry(Generic[T]):
    def __init__(self):
        self._registry: Dict[str, Type[T]] = {}

    def register(self, name: str, cls: Type[T]) -> None:
        self._registry[name] = cls

    def get(self, name: str) -> Type[T]:
        if name not in self._registry:
            raise KeyError(f"Component '{name}' not registered. Available: {list(self._registry.keys())}")
        return self._registry[name]

    def create(self, name: str, **kwargs) -> T:
        cls = self.get(name)
        return cls(**kwargs)

    def list_names(self) -> list:
        return list(self._registry.keys())


# Global registries
FEATURE_EXTRACTORS: ComponentRegistry[FeatureExtractor] = ComponentRegistry()
NORMALIZERS: ComponentRegistry[Normalizer] = ComponentRegistry()
KALMAN_FILTERS: ComponentRegistry[KalmanFilterProtocol] = ComponentRegistry()


def register_feature_extractor(name: str):
    def decorator(cls: Type[FeatureExtractor]):
        FEATURE_EXTRACTORS.register(name, cls)
        return cls
    return decorator


def register_normalizer(name: str):
    def decorator(cls: Type[Normalizer]):
        NORMALIZERS.register(name, cls)
        return cls
    return decorator


def register_kalman_filter(name: str):
    def decorator(cls: Type[KalmanFilterProtocol]):
        KALMAN_FILTERS.register(name, cls)
        return cls
    return decorator
