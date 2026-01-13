"""
Encoder registry for input projection modules.

Supports registration of temporal encoders (Conv1D, Linear, Transformer, etc.)
with factory function for easy instantiation.
"""

import logging
import torch.nn as nn
from typing import Optional
from .base import ENCODER_REGISTRY

_logger = logging.getLogger(__name__)


def register_builtin_encoders() -> None:
    """Register built-in encoder types."""
    try:
        from Models.encoder_ablation import Conv1DEncoder, LinearEncoder
        ENCODER_REGISTRY.register_class(
            name='conv1d',
            cls=Conv1DEncoder,
            description='Conv1D temporal encoder with BatchNorm + SiLU',
            tags=['temporal', 'convolutional']
        )
        ENCODER_REGISTRY.register_class(
            name='linear',
            cls=LinearEncoder,
            description='Linear per-timestep encoder with LayerNorm + SiLU',
            tags=['linear', 'simple']
        )
    except ImportError as e:
        _logger.debug(f"Could not register encoders from encoder_ablation: {e}")

    _logger.info(f"Registered {len(ENCODER_REGISTRY)} encoders")


def create_encoder(
    encoder_type: str,
    in_channels: int,
    out_channels: int,
    kernel_size: int = 8,
    dropout: float = 0.1,
    **kwargs
) -> nn.Module:
    """
    Factory function to create encoder by type.

    Args:
        encoder_type: Registered encoder name ('conv1d', 'linear', etc.)
        in_channels: Input channel dimension
        out_channels: Output channel dimension
        kernel_size: Kernel size for conv encoders (ignored for linear)
        dropout: Dropout rate
        **kwargs: Additional encoder-specific arguments

    Returns:
        Instantiated encoder module

    Raises:
        KeyError: If encoder_type not found in registry
    """
    encoder_cls = ENCODER_REGISTRY.get(encoder_type)

    # Build kwargs based on encoder type
    if encoder_type == 'conv1d':
        return encoder_cls(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            **kwargs
        )
    elif encoder_type == 'linear':
        return encoder_cls(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            **kwargs
        )
    else:
        # Generic instantiation
        return encoder_cls(
            in_channels=in_channels,
            out_channels=out_channels,
            **kwargs
        )


# Auto-register on import
register_builtin_encoders()
