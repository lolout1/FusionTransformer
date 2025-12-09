"""
Kalman filter module for IMU orientation estimation.

Provides Linear Kalman Filter and Extended Kalman Filter implementations
for fusing accelerometer and gyroscope data into orientation estimates.

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
]
