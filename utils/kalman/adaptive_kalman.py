"""
Adaptive Kalman Filter with Dynamic Measurement Noise.

Extends the standard Linear Kalman Filter to adaptively adjust
accelerometer measurement noise (R_acc) based on acceleration magnitude.

This addresses Critique 3.1: "Accelerometer-as-gravity assumption breaks
during falls/impacts."

When |a| >> g, the accelerometer measures significant non-gravitational
acceleration, making the gravity-based roll/pitch estimates unreliable.
The adaptive filter increases R_acc during these periods, reducing trust
in accelerometer-derived orientation.

References:
    RAESKF (2024). "Robust Adaptive Error State Kalman Filter"
"""

import numpy as np
from typing import Optional, Dict, Tuple
from .filters import KalmanFilter, ExtendedKalmanFilter
from .quaternion import acc_to_euler, wrap_angle


class AdaptiveKalmanFilter(KalmanFilter):
    """
    Linear Kalman Filter with adaptive accelerometer measurement noise.

    Inflates R_acc when acceleration magnitude deviates from gravity,
    reducing filter reliance on accelerometer-derived orientation during
    high-dynamic events (falls, impacts).
    """

    def __init__(self,
                 threshold_g: float = 2.0,
                 R_scale_max: float = 10.0,
                 R_scale_method: str = 'quadratic',
                 Q_orientation: float = 0.01,
                 Q_rate: float = 0.1,
                 R_acc: float = 0.1,
                 R_gyro: float = 0.5):
        """
        Initialize adaptive Kalman filter.

        Args:
            threshold_g: Acceleration threshold (in multiples of g) for
                        starting to inflate R_acc. Default 2.0g.
            R_scale_max: Maximum R_acc inflation factor. Default 10.0.
            R_scale_method: 'quadratic' or 'linear' scaling.
            Q_orientation: Process noise for orientation (rad²).
            Q_rate: Process noise for angular rates ((rad/s)²).
            R_acc: Base measurement noise for accelerometer (rad²).
            R_gyro: Measurement noise for gyroscope ((rad/s)²).
        """
        super().__init__(
            Q_orientation=Q_orientation,
            Q_rate=Q_rate,
            R_acc=R_acc,
            R_gyro=R_gyro
        )

        self.threshold_g = threshold_g
        self.R_scale_max = R_scale_max
        self.R_scale_method = R_scale_method
        self.base_R_acc = R_acc
        self.gravity = 9.81

        # Statistics tracking
        self.last_acc_magnitude = 0.0
        self.last_R_scale = 1.0

    def update(self, acc: np.ndarray, gyro: np.ndarray) -> None:
        """
        Update with adaptive R_acc based on acceleration magnitude.

        Args:
            acc: Accelerometer [ax, ay, az] in m/s².
            gyro: Gyroscope [gx, gy, gz] in rad/s.
        """
        # Compute acceleration magnitude
        acc_magnitude = np.linalg.norm(acc)
        self.last_acc_magnitude = acc_magnitude

        # Compute adaptive R_acc scale
        if acc_magnitude > self.threshold_g * self.gravity:
            # Deviation from gravity
            deviation = acc_magnitude / self.gravity

            if self.R_scale_method == 'quadratic':
                scale = min(deviation ** 2, self.R_scale_max)
            else:  # linear
                scale = min(deviation, self.R_scale_max)

            self.last_R_scale = scale
        else:
            scale = 1.0
            self.last_R_scale = 1.0

        # Apply adaptive R_acc
        self.R[0, 0] = self.base_R_acc * scale  # roll
        self.R[1, 1] = self.base_R_acc * scale  # pitch

        # Standard Kalman update
        super().update(acc, gyro)

    def get_diagnostics(self) -> Dict:
        """Return diagnostic information."""
        return {
            'acc_magnitude': self.last_acc_magnitude,
            'R_scale': self.last_R_scale,
            'is_high_dynamic': self.last_acc_magnitude > self.threshold_g * self.gravity
        }

    def reset(self) -> None:
        """Reset filter state."""
        super().reset()
        self.last_acc_magnitude = 0.0
        self.last_R_scale = 1.0


class AdaptiveEKF(ExtendedKalmanFilter):
    """
    Extended Kalman Filter with adaptive accelerometer measurement noise.

    Combines quaternion state representation with gyro bias estimation
    and adaptive R_acc based on acceleration magnitude.
    """

    def __init__(self,
                 threshold_g: float = 2.0,
                 R_scale_max: float = 10.0,
                 Q_quat: float = 0.001,
                 Q_bias: float = 0.0001,
                 R_acc: float = 0.1):
        """
        Initialize adaptive EKF.

        Args:
            threshold_g: Threshold for R_acc inflation.
            R_scale_max: Maximum R_acc scale factor.
            Q_quat: Process noise for quaternion.
            Q_bias: Process noise for gyro bias.
            R_acc: Base measurement noise for accelerometer.
        """
        super().__init__(Q_quat=Q_quat, Q_bias=Q_bias, R_acc=R_acc)

        self.threshold_g = threshold_g
        self.R_scale_max = R_scale_max
        self.base_R_acc = R_acc
        self.gravity = 9.81

        self.last_acc_magnitude = 0.0
        self.last_R_scale = 1.0

    def update(self, acc: np.ndarray) -> None:
        """Update with adaptive R_acc."""
        acc_magnitude = np.linalg.norm(acc)
        self.last_acc_magnitude = acc_magnitude

        # Adaptive scaling
        if acc_magnitude > self.threshold_g * self.gravity:
            deviation = acc_magnitude / self.gravity
            scale = min(deviation ** 2, self.R_scale_max)
            self.last_R_scale = scale
        else:
            scale = 1.0
            self.last_R_scale = 1.0

        # Apply adaptive R
        self.R = np.eye(3) * (self.base_R_acc * scale)

        # Standard EKF update
        super().update(acc)

    def get_diagnostics(self) -> Dict:
        """Return diagnostic information."""
        return {
            'acc_magnitude': self.last_acc_magnitude,
            'R_scale': self.last_R_scale,
            'is_high_dynamic': self.last_acc_magnitude > self.threshold_g * self.gravity,
            'gyro_bias': self.get_gyro_bias()
        }


def create_adaptive_filter(filter_type: str = 'linear',
                           threshold_g: float = 2.0,
                           R_scale_max: float = 10.0,
                           **params) -> object:
    """
    Factory function for adaptive Kalman filters.

    Args:
        filter_type: 'linear' or 'ekf'.
        threshold_g: Acceleration threshold for R inflation.
        R_scale_max: Maximum R scale factor.
        **params: Filter-specific parameters.

    Returns:
        AdaptiveKalmanFilter or AdaptiveEKF instance.
    """
    if filter_type == 'linear':
        return AdaptiveKalmanFilter(
            threshold_g=threshold_g,
            R_scale_max=R_scale_max,
            Q_orientation=params.get('Q_orientation', 0.01),
            Q_rate=params.get('Q_rate', 0.1),
            R_acc=params.get('R_acc', 0.1),
            R_gyro=params.get('R_gyro', 0.5)
        )
    elif filter_type == 'ekf':
        return AdaptiveEKF(
            threshold_g=threshold_g,
            R_scale_max=R_scale_max,
            Q_quat=params.get('Q_quat', 0.001),
            Q_bias=params.get('Q_bias', 0.0001),
            R_acc=params.get('R_acc', 0.1)
        )
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


def process_trial_adaptive(acc_data: np.ndarray,
                           gyro_data: np.ndarray,
                           filter_type: str = 'linear',
                           threshold_g: float = 2.0,
                           R_scale_max: float = 10.0,
                           output_format: str = 'euler',
                           dt: float = 1/30.0,
                           return_diagnostics: bool = False,
                           **params) -> Tuple:
    """
    Process trial with adaptive Kalman filter.

    Args:
        acc_data: (T, 3) accelerometer in m/s².
        gyro_data: (T, 3) gyroscope in rad/s.
        filter_type: 'linear' or 'ekf'.
        threshold_g: Threshold for R inflation.
        R_scale_max: Maximum R scale.
        output_format: 'euler' or 'quaternion'.
        dt: Time step.
        return_diagnostics: Return diagnostic info.
        **params: Additional filter parameters.

    Returns:
        orientations: (T, 3) or (T, 4).
        diagnostics: (optional) Dict with per-timestep info.
    """
    T = len(acc_data)
    filt = create_adaptive_filter(
        filter_type=filter_type,
        threshold_g=threshold_g,
        R_scale_max=R_scale_max,
        **params
    )

    # Output dimension
    if output_format == 'quaternion' and filter_type == 'ekf':
        ori_dim = 4
    else:
        ori_dim = 3

    orientations = np.zeros((T, ori_dim))
    diagnostics = {'acc_magnitude': [], 'R_scale': [], 'is_high_dynamic': []}

    for t in range(T):
        if filter_type == 'linear':
            filt.predict(dt)
            filt.update(acc_data[t], gyro_data[t])
            orientations[t] = filt.get_orientation()
        else:
            filt.predict(gyro_data[t], dt)
            filt.update(acc_data[t])
            if output_format == 'quaternion':
                orientations[t] = filt.get_orientation_quaternion()
            else:
                orientations[t] = filt.get_orientation_euler()

        if return_diagnostics:
            diag = filt.get_diagnostics()
            diagnostics['acc_magnitude'].append(diag['acc_magnitude'])
            diagnostics['R_scale'].append(diag['R_scale'])
            diagnostics['is_high_dynamic'].append(diag['is_high_dynamic'])

    if return_diagnostics:
        return orientations, diagnostics
    return orientations
