"""
Kalman filter module for IMU orientation estimation.

Provides multiple filter implementations for fusing accelerometer and
gyroscope data into orientation estimates:
    - Linear Kalman Filter (Euler state)
    - Extended Kalman Filter (quaternion state with gyro bias)
    - Adaptive Kalman Filter (dynamic measurement noise)
    - Madgwick AHRS Filter (gradient descent optimization)
    - VQF Wrapper (state-of-the-art 2023 filter)

Usage:
    from utils.kalman import build_kalman_features, KalmanFilter, ExtendedKalmanFilter

    # Option 1: Direct filter usage
    kf = KalmanFilter(Q_orientation=0.01, R_acc=0.1)
    for acc, gyro in zip(acc_data, gyro_data):
        kf.predict(dt=1/30)
        kf.update(acc, gyro)
        orientation = kf.get_orientation()

    # Option 2: Build features for neural network
    config = {
        'kalman_filter_type': 'linear',
        'kalman_include_smv': True,
        'filter_fs': 30.0
    }
    features = build_kalman_features(acc_data, gyro_data, config)

    # Option 3: Alternative filters (for ablation studies)
    from utils.kalman import MadgwickFilter, AdaptiveKalmanFilter
    mf = MadgwickFilter(beta=0.1, sample_freq=30.0)
    akf = AdaptiveKalmanFilter(threshold_g=2.0, R_scale_max=10.0)
"""

from .filters import KalmanFilter, ExtendedKalmanFilter, create_filter
from .features import (
    KalmanFeatureExtractor,
    build_kalman_features,
    compute_smv,
    normalize_features
)
from .preprocessing import (
    process_trial_kalman,
    kalman_fusion_for_loader,
    assemble_kalman_features,
    validate_gyro_units,
    convert_gyro_to_rads
)
from .tuning import (
    KalmanParameterTuner,
    ParallelKalmanTuner,
    TuningResult,
    default_search_space,
    default_ekf_search_space,
    load_tuned_params,
    get_literature_defaults
)
from .quaternion import (
    quat_multiply,
    quat_normalize,
    quat_to_euler,
    euler_to_quat,
    acc_to_euler
)

# Alternative filter implementations (for ablation studies)
from .madgwick import (
    MadgwickFilter,
    AdaptiveMadgwickFilter,
    process_trial_madgwick
)
from .adaptive_kalman import (
    AdaptiveKalmanFilter,
    AdaptiveEKF,
    create_adaptive_filter,
    process_trial_adaptive
)

# VQF wrapper (optional - requires pip install vqf)
try:
    from .vqf_wrapper import (
        VQFWrapper,
        process_trial_vqf,
        check_vqf_available
    )
    VQF_AVAILABLE = True
except ImportError:
    VQF_AVAILABLE = False

__all__ = [
    # Filters
    'KalmanFilter',
    'ExtendedKalmanFilter',
    'create_filter',

    # Feature extraction
    'KalmanFeatureExtractor',
    'build_kalman_features',
    'compute_smv',
    'normalize_features',

    # Preprocessing
    'process_trial_kalman',
    'kalman_fusion_for_loader',
    'assemble_kalman_features',
    'validate_gyro_units',
    'convert_gyro_to_rads',

    # Tuning
    'KalmanParameterTuner',
    'ParallelKalmanTuner',
    'TuningResult',
    'default_search_space',
    'default_ekf_search_space',
    'load_tuned_params',
    'get_literature_defaults',

    # Quaternion utilities
    'quat_multiply',
    'quat_normalize',
    'quat_to_euler',
    'euler_to_quat',
    'acc_to_euler',

    # Alternative filters (ablation studies)
    'MadgwickFilter',
    'AdaptiveMadgwickFilter',
    'process_trial_madgwick',
    'AdaptiveKalmanFilter',
    'AdaptiveEKF',
    'create_adaptive_filter',
    'process_trial_adaptive',

    # VQF (optional)
    'VQF_AVAILABLE',
]

# Conditionally add VQF exports
if VQF_AVAILABLE:
    __all__.extend(['VQFWrapper', 'process_trial_vqf', 'check_vqf_available'])
