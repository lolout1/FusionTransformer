"""
FusionLib: Scalable experiment infrastructure for FusionTransformer.

Core modules:
    - registry: Component registration (models, encoders, losses)
    - results: Experiment metrics and comparison tools
    - config: Hydra integration utilities
"""

__version__ = "0.1.0"

from .registry import MODEL_REGISTRY, ENCODER_REGISTRY, LOSS_REGISTRY

__all__ = ["MODEL_REGISTRY", "ENCODER_REGISTRY", "LOSS_REGISTRY"]
