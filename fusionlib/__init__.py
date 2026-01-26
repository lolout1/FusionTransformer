"""
FusionLib: Scalable experiment infrastructure for FusionTransformer.

Core modules:
    - registry: Component registration (models, encoders, losses)
    - results: Experiment metrics and comparison tools
    - config: Config inheritance, validation, ablation sweeps
"""

__version__ = "0.1.0"

# Lazy imports to avoid torch dependency for config-only usage
_registry_loaded = False


def __getattr__(name):
    """Lazy load registry to avoid torch dependency."""
    global _registry_loaded
    if name in ("MODEL_REGISTRY", "ENCODER_REGISTRY", "LOSS_REGISTRY"):
        if not _registry_loaded:
            from .registry import MODEL_REGISTRY, ENCODER_REGISTRY, LOSS_REGISTRY
            globals()["MODEL_REGISTRY"] = MODEL_REGISTRY
            globals()["ENCODER_REGISTRY"] = ENCODER_REGISTRY
            globals()["LOSS_REGISTRY"] = LOSS_REGISTRY
            _registry_loaded = True
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["MODEL_REGISTRY", "ENCODER_REGISTRY", "LOSS_REGISTRY"]
