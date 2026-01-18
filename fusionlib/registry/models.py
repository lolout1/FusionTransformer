"""
Model registry with auto-discovery of existing models.

Registers all FusionTransformer models with:
- Short names for new configs
- Legacy aliases for backward compatibility
- Metadata tags for filtering
"""

import logging
from .base import MODEL_REGISTRY

_logger = logging.getLogger(__name__)


def _safe_register(register_fn):
    """Decorator to catch all import/type errors during registration."""
    def wrapper():
        try:
            register_fn()
        except Exception as e:
            _logger.debug(f"Registration failed for {register_fn.__name__}: {e}")
    return wrapper


def register_builtin_models() -> None:
    """
    Register all built-in models from Models/ directory.

    Called automatically on first import. Safe to call multiple times.
    Catches all errors gracefully to allow partial registration.
    """
    # Kalman Transformer (primary model)
    try:
        from Models.imu_transformer_kalman import KalmanTransformer
        MODEL_REGISTRY.register_class(
            name='kalman_transformer',
            cls=KalmanTransformer,
            aliases=[
                'KalmanTransformer',
                'Models.imu_transformer_kalman.KalmanTransformer'
            ],
            description='Kalman-filtered IMU transformer with dual-stream architecture',
            tags=['kalman', 'transformer', 'dual-stream', 'imu']
        )
    except Exception as e:
        _logger.debug(f"Could not register KalmanTransformer: {e}")

    # Encoder ablation variants
    try:
        from Models.encoder_ablation import (
            KalmanEncoderAblation,
            KalmanConv1dConv1d,
            KalmanConv1dLinear,
            KalmanLinearConv1d,
            KalmanLinearLinear
        )
        MODEL_REGISTRY.register_class(
            name='kalman_encoder_ablation',
            cls=KalmanEncoderAblation,
            aliases=[
                'KalmanEncoderAblation',
                'Models.encoder_ablation.KalmanEncoderAblation'
            ],
            description='Configurable encoder ablation model (conv1d/linear)',
            tags=['kalman', 'ablation', 'encoder', 'configurable']
        )
        MODEL_REGISTRY.register_class(
            name='kalman_conv1d_conv1d',
            cls=KalmanConv1dConv1d,
            aliases=['Models.encoder_ablation.KalmanConv1dConv1d'],
            description='Conv1D encoder for both acc and ori streams',
            tags=['kalman', 'ablation', 'conv1d']
        )
        MODEL_REGISTRY.register_class(
            name='kalman_conv1d_linear',
            cls=KalmanConv1dLinear,
            aliases=['Models.encoder_ablation.KalmanConv1dLinear'],
            description='Conv1D for acc, Linear for ori (hypothesis)',
            tags=['kalman', 'ablation', 'conv1d', 'linear', 'hybrid']
        )
        MODEL_REGISTRY.register_class(
            name='kalman_linear_conv1d',
            cls=KalmanLinearConv1d,
            aliases=['Models.encoder_ablation.KalmanLinearConv1d'],
            description='Linear for acc, Conv1D for ori (control)',
            tags=['kalman', 'ablation', 'conv1d', 'linear']
        )
        MODEL_REGISTRY.register_class(
            name='kalman_linear_linear',
            cls=KalmanLinearLinear,
            aliases=['Models.encoder_ablation.KalmanLinearLinear'],
            description='Linear encoder for both streams (ablation)',
            tags=['kalman', 'ablation', 'linear']
        )
    except Exception as e:
        _logger.debug(f"Could not register encoder ablation models: {e}")

    # Transformer variants from kalman_transformer_variants.py
    try:
        from Models.kalman_transformer_variants import (
            KalmanTransformerBaseline,
            KalmanCrossModalAttention,
            KalmanGatedFusion,
            KalmanDeepNarrow,
            KalmanUncertaintyAware,
            KalmanBalancedRatio,
            KalmanSingleStream,
            KalmanCompact
        )
        MODEL_REGISTRY.register_class(
            name='kalman_baseline',
            cls=KalmanTransformerBaseline,
            aliases=['Models.kalman_transformer_variants.KalmanTransformerBaseline'],
            tags=['kalman', 'baseline']
        )
        MODEL_REGISTRY.register_class(
            name='kalman_cross_modal',
            cls=KalmanCrossModalAttention,
            aliases=['Models.kalman_transformer_variants.KalmanCrossModalAttention'],
            tags=['kalman', 'cross-attention', 'novel']
        )
        MODEL_REGISTRY.register_class(
            name='kalman_gated_fusion',
            cls=KalmanGatedFusion,
            aliases=['Models.kalman_transformer_variants.KalmanGatedFusion'],
            tags=['kalman', 'gated', 'fusion']
        )
        MODEL_REGISTRY.register_class(
            name='kalman_deep_narrow',
            cls=KalmanDeepNarrow,
            aliases=['Models.kalman_transformer_variants.KalmanDeepNarrow'],
            tags=['kalman', 'deep', 'narrow']
        )
        MODEL_REGISTRY.register_class(
            name='kalman_uncertainty_aware',
            cls=KalmanUncertaintyAware,
            aliases=['Models.kalman_transformer_variants.KalmanUncertaintyAware'],
            tags=['kalman', 'uncertainty']
        )
        MODEL_REGISTRY.register_class(
            name='kalman_balanced_ratio',
            cls=KalmanBalancedRatio,
            aliases=['Models.kalman_transformer_variants.KalmanBalancedRatio'],
            tags=['kalman', 'balanced']
        )
        MODEL_REGISTRY.register_class(
            name='kalman_single_stream',
            cls=KalmanSingleStream,
            aliases=['Models.kalman_transformer_variants.KalmanSingleStream'],
            tags=['kalman', 'single-stream']
        )
        MODEL_REGISTRY.register_class(
            name='kalman_compact',
            cls=KalmanCompact,
            aliases=['Models.kalman_transformer_variants.KalmanCompact'],
            tags=['kalman', 'compact', 'lightweight']
        )
    except Exception as e:
        _logger.debug(f"Could not register kalman variants: {e}")

    # IMU Transformer SE (may fail on older Python due to type hints)
    try:
        from Models.imu_transformer_se import IMU_Transformer_SE
        MODEL_REGISTRY.register_class(
            name='imu_transformer_se',
            cls=IMU_Transformer_SE,
            aliases=['Models.imu_transformer_se.IMU_Transformer_SE'],
            tags=['transformer', 'imu', 'squeeze-excitation']
        )
    except Exception as e:
        _logger.debug(f"Could not register IMU_Transformer_SE: {e}")

    # Dual-stream variants
    try:
        from Models.dual_stream_base import DualStreamBase
        MODEL_REGISTRY.register_class(
            name='dual_stream_base',
            cls=DualStreamBase,
            aliases=['Models.dual_stream_base.DualStreamBase'],
            tags=['dual-stream', 'imu']
        )
    except Exception as e:
        _logger.debug(f"Could not register DualStreamBase: {e}")

    try:
        from Models.dual_stream_baseline import DualStreamBaseline
        MODEL_REGISTRY.register_class(
            name='dual_stream_baseline',
            cls=DualStreamBaseline,
            aliases=['Models.dual_stream_baseline.DualStreamBaseline'],
            tags=['dual-stream', 'imu', 'robust']
        )
    except Exception as e:
        _logger.debug(f"Could not register DualStreamBaseline: {e}")

    try:
        from Models.dual_stream_se import DualStreamSE
        MODEL_REGISTRY.register_class(
            name='dual_stream_se',
            cls=DualStreamSE,
            aliases=['Models.dual_stream_se.DualStreamSE'],
            tags=['dual-stream', 'imu', 'squeeze-excitation']
        )
    except Exception as e:
        _logger.debug(f"Could not register DualStreamSE: {e}")

    # Single-stream transformer
    try:
        from Models.single_stream_transformer import SingleStreamTransformer
        MODEL_REGISTRY.register_class(
            name='single_stream_transformer',
            cls=SingleStreamTransformer,
            aliases=['Models.single_stream_transformer.SingleStreamTransformer'],
            tags=['transformer', 'imu', 'single-stream']
        )
    except Exception as e:
        _logger.debug(f"Could not register SingleStreamTransformer: {e}")

    _logger.info(f"Registered {len(MODEL_REGISTRY)} models")


# Auto-register on import
register_builtin_models()
