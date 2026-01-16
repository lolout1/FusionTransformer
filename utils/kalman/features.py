"""
Feature extraction from Kalman filter outputs.

Extracts orientation, uncertainty, and innovation features for use
as neural network inputs.

Output Configurations:
    - 7 channels:  [smv, ax, ay, az, roll, pitch, yaw]
    - 10 channels: [smv, ax, ay, az, roll, pitch, yaw, sigma_r, sigma_p, sigma_y]
    - 11 channels: + innovation_magnitude
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from .filters import KalmanFilter, ExtendedKalmanFilter, create_filter


class KalmanFeatureExtractor:
    """Extract features from Kalman filter for fall detection."""

    def __init__(self,
                 filter_type: str = 'linear',
                 output_format: str = 'euler',
                 include_smv: bool = True,
                 include_uncertainty: bool = False,
                 include_innovation: bool = False,
                 Q_params: Optional[Dict] = None,
                 R_params: Optional[Dict] = None):
        """
        Initialize feature extractor.

        Args:
            filter_type: 'linear' or 'ekf'
            output_format: 'euler' (3 angles) or 'quaternion' (4 components)
            include_smv: Include signal magnitude vector as first channel
            include_uncertainty: Include orientation uncertainty as features
            include_innovation: Include innovation magnitude as feature
            Q_params: Process noise parameters
            R_params: Measurement noise parameters
        """
        self.filter_type = filter_type
        self.output_format = output_format
        self.include_smv = include_smv
        self.include_uncertainty = include_uncertainty
        self.include_innovation = include_innovation

        # Build filter parameters
        filter_params = {}
        if Q_params:
            filter_params.update(Q_params)
        if R_params:
            filter_params.update(R_params)
        self.filter_params = filter_params

    def process_trial(self,
                      acc_data: np.ndarray,
                      gyro_data: np.ndarray,
                      dt: float = 1/30.0) -> Dict[str, np.ndarray]:
        """
        Process a single trial through Kalman filter.

        Args:
            acc_data: Accelerometer (T, 3) in m/s²
            gyro_data: Gyroscope (T, 3) in rad/s
            dt: Time step in seconds

        Returns:
            Dict with 'orientation', 'uncertainty', 'innovation'
        """
        T = len(acc_data)
        kf = create_filter(self.filter_type, **self.filter_params)

        # Storage
        if self.output_format == 'quaternion' and self.filter_type == 'ekf':
            ori_dim = 4
        else:
            ori_dim = 3

        orientations = np.zeros((T, ori_dim))
        uncertainties = np.zeros((T, 3))
        innovations = np.zeros(T)

        for t in range(T):
            # Predict
            if self.filter_type == 'linear':
                kf.predict(dt)
                kf.update(acc_data[t], gyro_data[t])
            else:
                kf.predict(gyro_data[t], dt)
                kf.update(acc_data[t])

            # Extract outputs
            if self.output_format == 'quaternion' and self.filter_type == 'ekf':
                orientations[t] = kf.get_orientation_quaternion()
            else:
                if self.filter_type == 'ekf':
                    orientations[t] = kf.get_orientation_euler()
                else:
                    orientations[t] = kf.get_orientation()

            uncertainties[t] = kf.get_uncertainty()
            innovations[t] = kf.get_innovation_magnitude()

        result = {'orientation': orientations}
        if self.include_uncertainty:
            result['uncertainty'] = uncertainties
        if self.include_innovation:
            result['innovation'] = innovations.reshape(-1, 1)

        return result

    def get_feature_channels(self) -> int:
        """Return number of output channels based on config."""
        channels = 0

        # SMV
        if self.include_smv:
            channels += 1

        # Accelerometer (ax, ay, az)
        channels += 3

        # Orientation
        if self.output_format == 'quaternion' and self.filter_type == 'ekf':
            channels += 4
        else:
            channels += 3

        # Uncertainty
        if self.include_uncertainty:
            if self.output_format == 'quaternion':
                channels += 4
            else:
                channels += 3

        # Innovation
        if self.include_innovation:
            channels += 1

        return channels

    def get_channel_names(self) -> List[str]:
        """Return human-readable channel names."""
        names = []

        if self.include_smv:
            names.append('smv')

        names.extend(['ax', 'ay', 'az'])

        if self.output_format == 'quaternion' and self.filter_type == 'ekf':
            names.extend(['q0', 'q1', 'q2', 'q3'])
        else:
            names.extend(['roll', 'pitch', 'yaw'])

        if self.include_uncertainty:
            if self.output_format == 'quaternion':
                names.extend(['sigma_q0', 'sigma_q1', 'sigma_q2', 'sigma_q3'])
            else:
                names.extend(['sigma_roll', 'sigma_pitch', 'sigma_yaw'])

        if self.include_innovation:
            names.append('innovation_mag')

        return names


def compute_smv(acc_data: np.ndarray) -> np.ndarray:
    """
    Compute Signal Magnitude Vector: sqrt(ax² + ay² + az²)

    Args:
        acc_data: (T, 3) accelerometer data

    Returns:
        (T,) SMV values
    """
    return np.linalg.norm(acc_data, axis=1)


def compute_innovation_magnitude(innovation_sequence: np.ndarray) -> np.ndarray:
    """
    Compute L2 norm of innovation at each timestep.

    Args:
        innovation_sequence: (T, D) innovation vectors

    Returns:
        (T,) innovation magnitudes
    """
    if innovation_sequence.ndim == 1:
        return np.abs(innovation_sequence)
    return np.linalg.norm(innovation_sequence, axis=1)


def normalize_features(features: np.ndarray,
                       method: str = 'zscore') -> Tuple[np.ndarray, Dict]:
    """
    Normalize feature array.

    Args:
        features: (T, C) features
        method: 'minmax', 'zscore', or 'none'

    Returns:
        normalized features, normalization parameters
    """
    if method == 'none':
        return features, {}

    if method == 'minmax':
        fmin = features.min(axis=0, keepdims=True)
        fmax = features.max(axis=0, keepdims=True)
        denom = fmax - fmin
        denom[denom < 1e-8] = 1.0
        normalized = (features - fmin) / denom
        return normalized, {'min': fmin, 'max': fmax}

    elif method == 'zscore':
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True)
        std[std < 1e-8] = 1.0
        normalized = (features - mean) / std
        return normalized, {'mean': mean, 'std': std}

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def build_kalman_features(acc_data: np.ndarray,
                          gyro_data: np.ndarray,
                          config: Dict) -> np.ndarray:
    """
    Build feature array for neural network input.

    Main integration point with loader.py.

    Args:
        acc_data: (T, 3) raw accelerometer in m/s²
        gyro_data: (T, 3) gyroscope in rad/s (must be converted from deg/s)
        config: Configuration dict with keys:
            - kalman_filter_type: 'linear' or 'ekf'
            - kalman_output_format: 'euler' or 'quaternion'
            - kalman_include_smv: bool
            - kalman_include_uncertainty: bool
            - kalman_include_innovation: bool
            - kalman_Q_params: dict (optional)
            - kalman_R_params: dict (optional)
            - filter_fs: sampling frequency in Hz

    Returns:
        features: (T, C) where C depends on config
            7ch:  [smv, ax, ay, az, roll, pitch, yaw]
            10ch: [smv, ax, ay, az, roll, pitch, yaw, sigma_r, sigma_p, sigma_y]
            11ch: + innovation_magnitude
    """
    # Validate inputs
    if len(acc_data) != len(gyro_data):
        raise ValueError(f"acc_data ({len(acc_data)}) and gyro_data ({len(gyro_data)}) "
                        "must have same length")

    # Check if gyro might be in deg/s (sanity check)
    # Note: Fall detection can produce extreme angular velocities (10-15 rad/s = 573-860 deg/s)
    # during rapid rotational movements. Using 30 rad/s threshold.
    gyro_max = np.abs(gyro_data).max()
    if gyro_max > 30:
        raise ValueError(f"Gyroscope max value is {gyro_max:.1f}, which suggests deg/s. "
                        "Kalman filter expects rad/s. Set convert_gyro_to_rad=True.")

    # Extract config
    filter_type = config.get('kalman_filter_type', 'linear')
    output_format = config.get('kalman_output_format', 'euler')
    include_smv = config.get('kalman_include_smv', True)
    include_uncertainty = config.get('kalman_include_uncertainty', False)
    include_innovation = config.get('kalman_include_innovation', False)

    # Filter parameters
    Q_params = {}
    R_params = {}

    if 'kalman_Q_orientation' in config:
        Q_params['Q_orientation'] = config['kalman_Q_orientation']
    if 'kalman_Q_rate' in config:
        Q_params['Q_rate'] = config['kalman_Q_rate']
    if 'kalman_Q_quat' in config:
        Q_params['Q_quat'] = config['kalman_Q_quat']
    if 'kalman_Q_bias' in config:
        Q_params['Q_bias'] = config['kalman_Q_bias']
    if 'kalman_R_acc' in config:
        R_params['R_acc'] = config['kalman_R_acc']
    if 'kalman_R_gyro' in config:
        R_params['R_gyro'] = config['kalman_R_gyro']

    # Sampling frequency
    fs = config.get('filter_fs', 30.0)
    dt = 1.0 / fs

    # Create extractor
    extractor = KalmanFeatureExtractor(
        filter_type=filter_type,
        output_format=output_format,
        include_smv=include_smv,
        include_uncertainty=include_uncertainty,
        include_innovation=include_innovation,
        Q_params=Q_params,
        R_params=R_params
    )

    # Process trial
    result = extractor.process_trial(acc_data, gyro_data, dt)

    # Build feature array
    T = len(acc_data)
    feature_list = []

    # SMV
    if include_smv:
        smv = compute_smv(acc_data).reshape(-1, 1)
        feature_list.append(smv)

    # Accelerometer
    feature_list.append(acc_data)

    # Orientation
    feature_list.append(result['orientation'])

    # Uncertainty
    if include_uncertainty:
        feature_list.append(result['uncertainty'])

    # Innovation
    if include_innovation:
        feature_list.append(result['innovation'])

    features = np.hstack(feature_list)

    return features
