#!/usr/bin/env python3
"""
Unit tests for Adaptive Kalman Filter implementation.

Verifies:
1. Adaptive scaling computes correctly
2. NIS-based adaptation works as expected
3. Backwards compatibility (disabled by default)
4. Integration with feature extraction pipeline
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.kalman.filters import KalmanFilter, create_filter
from utils.kalman.features import build_kalman_features


def test_adaptive_disabled_by_default():
    """Test that adaptive scaling is disabled by default."""
    kf = KalmanFilter()
    assert kf.adaptive_enabled == False, "Adaptive should be disabled by default"
    assert kf.get_adaptive_scale() == 1.0, "Scale should be 1.0 when disabled"
    print("✓ Adaptive disabled by default")


def test_adaptive_scale_computation():
    """Test that adaptive scale is computed correctly."""
    kf = KalmanFilter(
        adaptive_enabled=True,
        adaptive_alpha=0.5,
        adaptive_scale_min=0.3,
        adaptive_scale_max=3.0,
        adaptive_ema_alpha=1.0,  # No smoothing for testing
        adaptive_warmup=0  # No warmup for testing
    )

    # Simulate low innovation (clean signal) - should reduce R scale
    y_low = np.array([0.01, 0.01, 0.001, 0.001, 0.001])
    S = np.eye(5) * 0.1
    scale = kf._compute_adaptive_scale(y_low, S)
    assert scale < 1.0, f"Low NIS should give scale < 1, got {scale}"
    print(f"✓ Low NIS → scale = {scale:.3f} (< 1.0)")

    # Reset for next test
    kf.scale_ema = 1.0
    kf.sample_count = 0

    # Simulate high innovation (noisy signal) - should increase R scale
    y_high = np.array([0.5, 0.5, 0.3, 0.3, 0.3])
    scale = kf._compute_adaptive_scale(y_high, S)
    assert scale > 1.0, f"High NIS should give scale > 1, got {scale}"
    print(f"✓ High NIS → scale = {scale:.3f} (> 1.0)")


def test_adaptive_warmup():
    """Test that warmup period is respected."""
    kf = KalmanFilter(
        adaptive_enabled=True,
        adaptive_warmup=5
    )

    y = np.array([0.1, 0.1, 0.05, 0.05, 0.05])
    S = np.eye(5) * 0.1

    # During warmup, scale should be 1.0
    for i in range(5):
        scale = kf._compute_adaptive_scale(y, S)
        assert scale == 1.0, f"Scale should be 1.0 during warmup (sample {i})"

    # After warmup, scale should adapt
    scale = kf._compute_adaptive_scale(y, S)
    assert scale != 1.0, "Scale should adapt after warmup"
    print(f"✓ Warmup respected (scale after warmup: {scale:.3f})")


def test_adaptive_ema_smoothing():
    """Test that EMA smoothing works correctly."""
    kf = KalmanFilter(
        adaptive_enabled=True,
        adaptive_alpha=0.5,
        adaptive_ema_alpha=0.1,  # Slow adaptation
        adaptive_warmup=0
    )

    y = np.array([0.5, 0.5, 0.3, 0.3, 0.3])  # High NIS
    S = np.eye(5) * 0.1

    scales = []
    for _ in range(20):
        scale = kf._compute_adaptive_scale(y, S)
        scales.append(scale)

    # Scale should gradually increase due to EMA
    assert scales[-1] > scales[0], "Scale should increase with consistent high NIS"
    assert scales[-1] < 3.0, "Scale should be clamped"
    print(f"✓ EMA smoothing works (scale: {scales[0]:.3f} → {scales[-1]:.3f})")


def test_adaptive_bounds():
    """Test that scale is clamped to bounds."""
    kf = KalmanFilter(
        adaptive_enabled=True,
        adaptive_alpha=1.0,  # Full sensitivity
        adaptive_scale_min=0.3,
        adaptive_scale_max=3.0,
        adaptive_ema_alpha=1.0,  # No smoothing
        adaptive_warmup=0
    )

    # Very low NIS - should hit min bound
    y_low = np.array([0.001, 0.001, 0.0001, 0.0001, 0.0001])
    S = np.eye(5) * 0.1
    scale = kf._compute_adaptive_scale(y_low, S)
    assert scale >= 0.3, f"Scale should be >= min bound, got {scale}"
    print(f"✓ Min bound enforced (scale: {scale:.3f})")

    # Reset
    kf.scale_ema = 1.0
    kf.sample_count = 0

    # Very high NIS - should hit max bound
    y_high = np.array([5.0, 5.0, 3.0, 3.0, 3.0])
    scale = kf._compute_adaptive_scale(y_high, S)
    assert scale <= 3.0, f"Scale should be <= max bound, got {scale}"
    print(f"✓ Max bound enforced (scale: {scale:.3f})")


def test_filter_update_with_adaptive():
    """Test full filter update with adaptive scaling."""
    kf = KalmanFilter(
        Q_orientation=0.01,
        Q_rate=0.01,
        R_acc=0.1,
        R_gyro=0.1,
        adaptive_enabled=True,
        adaptive_warmup=5
    )

    # Simulate some IMU data
    np.random.seed(42)
    T = 50
    acc_data = np.random.randn(T, 3) * 0.1
    acc_data[:, 2] += 9.81  # Add gravity
    gyro_data = np.random.randn(T, 3) * 0.1

    dt = 1/50.0
    scales = []
    orientations = []

    for t in range(T):
        kf.predict(dt)
        kf.update(acc_data[t], gyro_data[t])
        scales.append(kf.get_adaptive_scale())
        orientations.append(kf.get_orientation().copy())

    # Check that filter ran without errors
    assert len(scales) == T
    assert len(orientations) == T

    # Check that adaptation happened after warmup
    assert scales[4] == 1.0, "Scale should be 1.0 during warmup"
    assert any(s != 1.0 for s in scales[10:]), "Scale should adapt after warmup"

    print(f"✓ Full filter update works (final scale: {scales[-1]:.3f})")


def test_create_filter_with_adaptive():
    """Test factory function passes adaptive params."""
    kf = create_filter(
        'linear',
        Q_orientation=0.01,
        R_acc=0.1,
        adaptive_enabled=True,
        adaptive_alpha=0.7,
        adaptive_scale_min=0.5
    )

    assert kf.adaptive_enabled == True
    assert kf.adaptive_alpha == 0.7
    assert kf.adaptive_scale_min == 0.5
    print("✓ Factory function passes adaptive params")


def test_build_kalman_features_adaptive():
    """Test feature extraction with adaptive Kalman."""
    np.random.seed(42)
    T = 100

    # Simulate IMU data
    acc_data = np.random.randn(T, 3) * 0.5
    acc_data[:, 2] += 9.81
    gyro_data = np.random.randn(T, 3) * 0.1  # rad/s

    config = {
        'kalman_filter_type': 'linear',
        'kalman_output_format': 'euler',
        'kalman_include_smv': True,
        'filter_fs': 50.0,
        'kalman_Q_orientation': 0.01,
        'kalman_Q_rate': 0.01,
        'kalman_R_acc': 0.1,
        'kalman_R_gyro': 0.1,
        'adaptive_kalman_enabled': True,
        'adaptive_alpha': 0.5,
        'adaptive_scale_min': 0.3,
        'adaptive_scale_max': 3.0,
        'adaptive_ema_alpha': 0.1,
        'adaptive_warmup_samples': 10,
    }

    features = build_kalman_features(acc_data, gyro_data, config)

    # Check output shape: [smv, ax, ay, az, roll, pitch, yaw] = 7 channels
    assert features.shape == (T, 7), f"Expected shape (100, 7), got {features.shape}"

    # Check no NaN values
    assert not np.isnan(features).any(), "Features contain NaN values"

    print(f"✓ Feature extraction works (shape: {features.shape})")


def test_adaptive_vs_fixed_output():
    """Compare adaptive vs fixed Kalman output (sanity check)."""
    np.random.seed(42)
    T = 100

    acc_data = np.random.randn(T, 3) * 0.3
    acc_data[:, 2] += 9.81
    gyro_data = np.random.randn(T, 3) * 0.05

    base_config = {
        'kalman_filter_type': 'linear',
        'kalman_output_format': 'euler',
        'kalman_include_smv': True,
        'filter_fs': 50.0,
        'kalman_Q_orientation': 0.01,
        'kalman_Q_rate': 0.01,
        'kalman_R_acc': 0.1,
        'kalman_R_gyro': 0.1,
    }

    # Fixed Kalman
    fixed_config = {**base_config, 'adaptive_kalman_enabled': False}
    fixed_features = build_kalman_features(acc_data, gyro_data, fixed_config)

    # Adaptive Kalman
    adaptive_config = {
        **base_config,
        'adaptive_kalman_enabled': True,
        'adaptive_alpha': 0.5,
        'adaptive_scale_min': 0.3,
        'adaptive_scale_max': 3.0,
    }
    adaptive_features = build_kalman_features(acc_data, gyro_data, adaptive_config)

    # Outputs should be similar but not identical
    diff = np.abs(fixed_features - adaptive_features).mean()
    assert diff > 0, "Adaptive should produce different output than fixed"
    assert diff < 1.0, f"Difference too large: {diff}"

    print(f"✓ Adaptive vs Fixed difference: {diff:.6f} (reasonable)")


# ============================================================================
# NEW TESTS FOR SIGNAL-BASED AND HYBRID MODES
# ============================================================================

def test_mode_none_disables_adaptation():
    """Test that mode='none' disables adaptation even if enabled=True."""
    kf = KalmanFilter(
        adaptive_enabled=True,
        adaptive_mode='none',
        adaptive_alpha=0.5
    )

    # Mode 'none' should disable adaptation regardless of adaptive_enabled
    assert kf.adaptive_enabled == False, "mode='none' should disable adaptation"
    assert kf.get_adaptive_mode() == 'none'

    # Run filter and check scale stays at 1.0
    dt = 1/50.0
    for _ in range(20):
        kf.predict(dt)
        acc = np.array([0.1, 0.2, 9.81])
        gyro = np.array([0.01, 0.02, 0.03])
        kf.update(acc, gyro)
        assert kf.get_adaptive_scale() == 1.0

    print("✓ Mode 'none' disables adaptation")


def test_mode_signal_stationary():
    """Test signal-based adaptation with stationary (gravity-only) data."""
    kf = KalmanFilter(
        adaptive_enabled=True,
        adaptive_mode='signal',
        adaptive_alpha=0.5,
        adaptive_warmup=5
    )

    assert kf.get_adaptive_mode() == 'signal'

    dt = 1/50.0
    scales = []

    # Stationary data: acc = gravity
    for t in range(30):
        kf.predict(dt)
        acc = np.array([0, 0, 9.81])  # Pure gravity
        gyro = np.array([0, 0, 0])
        kf.update(acc, gyro)
        scales.append(kf.get_adaptive_scale())

    # After warmup, scale should stay near 1.0 for stationary data
    mean_scale = np.mean(scales[10:])
    assert 0.9 <= mean_scale <= 1.1, f"Stationary scale should be ~1.0, got {mean_scale}"
    print(f"✓ Signal mode stationary: scale = {mean_scale:.3f}")


def test_mode_signal_motion():
    """Test signal-based adaptation with high-motion data."""
    kf = KalmanFilter(
        adaptive_enabled=True,
        adaptive_mode='signal',
        adaptive_alpha=0.5,
        adaptive_warmup=5,
        adaptive_scale_min=0.3,
        adaptive_scale_max=3.0
    )

    dt = 1/50.0
    scales = []

    # Motion data: large acceleration deviations from gravity
    for t in range(30):
        kf.predict(dt)
        # Simulate 2-3g acceleration during motion
        acc = np.array([5.0, 5.0, 9.81 + 10 * np.sin(0.2 * t)])
        gyro = np.array([0.5, 0.5, 0.5])
        kf.update(acc, gyro)
        scales.append(kf.get_adaptive_scale())

    # With high motion, scale should increase (more smoothing)
    mean_scale = np.mean(scales[10:])
    assert mean_scale > 1.0, f"Motion scale should be > 1.0, got {mean_scale}"
    print(f"✓ Signal mode motion: scale = {mean_scale:.3f}")


def test_mode_hybrid():
    """Test hybrid adaptation (NIS + signal)."""
    kf = KalmanFilter(
        adaptive_enabled=True,
        adaptive_mode='hybrid',
        adaptive_alpha=0.5,
        adaptive_warmup=5
    )

    assert kf.get_adaptive_mode() == 'hybrid'

    dt = 1/50.0
    np.random.seed(42)

    scales = []
    for t in range(30):
        kf.predict(dt)
        acc = np.array([0, 0, 9.81]) + np.random.randn(3) * 0.5
        gyro = np.random.randn(3) * 0.1
        kf.update(acc, gyro)
        scales.append(kf.get_adaptive_scale())

    # Hybrid should produce varying scales
    scale_std = np.std(scales[10:])
    assert scale_std > 0, "Hybrid mode should produce varying scales"
    print(f"✓ Hybrid mode: scale std = {scale_std:.4f}")


def test_r_multiplier():
    """Test R multiplier for mis-tuned filter experiments."""
    # Normal filter
    kf_normal = KalmanFilter(R_acc=0.1, R_gyro=0.5, R_multiplier=1.0)

    # Mis-tuned filter (5x R)
    kf_mistuned = KalmanFilter(R_acc=0.1, R_gyro=0.5, R_multiplier=5.0)

    # Check R values are scaled
    assert kf_mistuned.R_multiplier == 5.0
    assert np.allclose(kf_mistuned.R_base, kf_normal.R_base * 5.0)

    # Run both filters and compare
    dt = 1/50.0
    np.random.seed(42)

    ori_normal = []
    ori_mistuned = []

    for t in range(50):
        acc = np.array([0, 0, 9.81]) + np.random.randn(3) * 0.3
        gyro = np.random.randn(3) * 0.05

        kf_normal.predict(dt)
        kf_normal.update(acc, gyro)
        ori_normal.append(kf_normal.get_orientation().copy())

        kf_mistuned.predict(dt)
        kf_mistuned.update(acc, gyro)
        ori_mistuned.append(kf_mistuned.get_orientation().copy())

    # Mistuned (higher R) should be smoother/slower to respond
    ori_normal = np.array(ori_normal)
    ori_mistuned = np.array(ori_mistuned)

    # Check orientations differ
    diff = np.abs(ori_normal - ori_mistuned).mean()
    assert diff > 0, "R multiplier should affect orientation"
    print(f"✓ R multiplier: orientation diff = {diff:.6f}")


def test_invalid_mode_raises_error():
    """Test that invalid adaptive mode raises ValueError."""
    try:
        kf = KalmanFilter(adaptive_mode='invalid_mode')
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "adaptive_mode must be one of" in str(e)
        print("✓ Invalid mode raises ValueError")


def test_signal_scale_getter():
    """Test get_signal_scale() returns signal-based scale."""
    kf = KalmanFilter(
        adaptive_enabled=True,
        adaptive_mode='signal',
        adaptive_warmup=3
    )

    dt = 1/50.0

    # Run with motion data
    for t in range(10):
        kf.predict(dt)
        acc = np.array([5.0, 0, 9.81])  # Off-gravity
        gyro = np.array([0.1, 0.1, 0.1])
        kf.update(acc, gyro)

    signal_scale = kf.get_signal_scale()
    assert signal_scale > 0, "Signal scale should be positive"
    print(f"✓ get_signal_scale() = {signal_scale:.3f}")


def test_create_filter_with_new_params():
    """Test create_filter() passes new parameters correctly."""
    kf = create_filter(
        'linear',
        adaptive_enabled=True,
        adaptive_mode='signal',
        adaptive_alpha=0.7,
        R_multiplier=2.5
    )

    assert kf.get_adaptive_mode() == 'signal'
    assert kf.adaptive_alpha == 0.7
    assert kf.R_multiplier == 2.5
    print("✓ create_filter() passes new params")


def test_build_features_with_signal_mode():
    """Test build_kalman_features with signal-based adaptation."""
    np.random.seed(42)
    T = 100

    acc_data = np.random.randn(T, 3) * 0.3
    acc_data[:, 2] += 9.81
    gyro_data = np.random.randn(T, 3) * 0.1

    config = {
        'kalman_filter_type': 'linear',
        'kalman_output_format': 'euler',
        'kalman_include_smv': True,
        'filter_fs': 50.0,
        'kalman_Q_orientation': 0.01,
        'kalman_Q_rate': 0.1,
        'kalman_R_acc': 0.1,
        'kalman_R_gyro': 0.5,
        'adaptive_kalman_enabled': True,
        'adaptive_mode': 'signal',
        'adaptive_alpha': 0.5,
    }

    features = build_kalman_features(acc_data, gyro_data, config)

    assert features.shape == (T, 7)
    assert not np.isnan(features).any()
    print(f"✓ build_kalman_features with signal mode: shape = {features.shape}")


def test_build_features_with_r_multiplier():
    """Test build_kalman_features with R multiplier."""
    np.random.seed(42)
    T = 100

    acc_data = np.random.randn(T, 3) * 0.3
    acc_data[:, 2] += 9.81
    gyro_data = np.random.randn(T, 3) * 0.1

    config_normal = {
        'kalman_filter_type': 'linear',
        'kalman_output_format': 'euler',
        'kalman_include_smv': True,
        'filter_fs': 50.0,
        'kalman_R_acc': 0.1,
        'kalman_R_gyro': 0.5,
        'kalman_R_multiplier': 1.0,
    }

    config_mistuned = {**config_normal, 'kalman_R_multiplier': 5.0}

    features_normal = build_kalman_features(acc_data.copy(), gyro_data.copy(), config_normal)
    features_mistuned = build_kalman_features(acc_data.copy(), gyro_data.copy(), config_mistuned)

    # Features should differ
    diff = np.abs(features_normal - features_mistuned).mean()
    assert diff > 0, "R multiplier should affect features"
    print(f"✓ build_kalman_features with R multiplier: diff = {diff:.6f}")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("ADAPTIVE KALMAN FILTER UNIT TESTS")
    print("=" * 60)

    tests = [
        # Original NIS-based tests
        test_adaptive_disabled_by_default,
        test_adaptive_scale_computation,
        test_adaptive_warmup,
        test_adaptive_ema_smoothing,
        test_adaptive_bounds,
        test_filter_update_with_adaptive,
        test_create_filter_with_adaptive,
        test_build_kalman_features_adaptive,
        test_adaptive_vs_fixed_output,
        # New signal/hybrid/R_multiplier tests
        test_mode_none_disables_adaptation,
        test_mode_signal_stationary,
        test_mode_signal_motion,
        test_mode_hybrid,
        test_r_multiplier,
        test_invalid_mode_raises_error,
        test_signal_scale_getter,
        test_create_filter_with_new_params,
        test_build_features_with_signal_mode,
        test_build_features_with_r_multiplier,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            print(f"\n{test.__name__}:")
            test()
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
