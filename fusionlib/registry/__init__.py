"""Component registries for models, encoders, and losses."""

from .base import Registry, MODEL_REGISTRY, ENCODER_REGISTRY, LOSS_REGISTRY

# Trigger auto-registration of built-in components
from . import models  # noqa: F401
from . import encoders  # noqa: F401
from . import losses  # noqa: F401

__all__ = ["Registry", "MODEL_REGISTRY", "ENCODER_REGISTRY", "LOSS_REGISTRY"]
