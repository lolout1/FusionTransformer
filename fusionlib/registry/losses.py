"""
Loss function registry.

Registers standard and custom loss functions with factory instantiation.
"""

import logging
import torch.nn as nn
from typing import Optional, Dict, Any
from .base import LOSS_REGISTRY

_logger = logging.getLogger(__name__)


def register_builtin_losses() -> None:
    """Register built-in loss functions."""
    # Standard PyTorch losses
    LOSS_REGISTRY.register_class(
        name='bce',
        cls=nn.BCEWithLogitsLoss,
        aliases=['binary_cross_entropy', 'bce_logits'],
        description='Binary cross-entropy with logits',
        tags=['binary', 'standard']
    )
    LOSS_REGISTRY.register_class(
        name='ce',
        cls=nn.CrossEntropyLoss,
        aliases=['cross_entropy'],
        description='Multi-class cross-entropy',
        tags=['multiclass', 'standard']
    )
    LOSS_REGISTRY.register_class(
        name='mse',
        cls=nn.MSELoss,
        description='Mean squared error',
        tags=['regression', 'standard']
    )

    # Custom losses from utils/loss.py
    try:
        from utils.loss import BinaryFocalLoss
        LOSS_REGISTRY.register_class(
            name='focal',
            cls=BinaryFocalLoss,
            aliases=['binary_focal', 'focal_loss'],
            description='Focal loss for class imbalance (Lin et al. 2017)',
            tags=['binary', 'imbalanced', 'focal']
        )
    except ImportError as e:
        _logger.debug(f"Could not register BinaryFocalLoss: {e}")

    try:
        from utils.loss import ClassBalancedFocalLoss
        LOSS_REGISTRY.register_class(
            name='cb_focal',
            cls=ClassBalancedFocalLoss,
            aliases=['class_balanced_focal', 'cb_loss'],
            description='Class-balanced focal loss (Cui et al. 2019)',
            tags=['binary', 'imbalanced', 'focal', 'class-balanced']
        )
    except ImportError as e:
        _logger.debug(f"Could not register ClassBalancedFocalLoss: {e}")

    _logger.info(f"Registered {len(LOSS_REGISTRY)} losses")


def create_loss(
    loss_type: str,
    pos_weight: Optional[float] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create loss by type.

    Args:
        loss_type: Registered loss name ('bce', 'focal', 'cb_focal', etc.)
        pos_weight: Positive class weight for BCE
        **kwargs: Loss-specific arguments (alpha, gamma, beta, etc.)

    Returns:
        Instantiated loss module
    """
    loss_cls = LOSS_REGISTRY.get(loss_type)

    # Handle pos_weight for BCE
    if loss_type == 'bce' and pos_weight is not None:
        import torch
        return loss_cls(pos_weight=torch.tensor([pos_weight]))

    # For custom losses, pass kwargs directly
    return loss_cls(**kwargs)


# Auto-register on import
register_builtin_losses()
