"""
Madgwick AHRS Filter for IMU Orientation Estimation.

Implementation based on:
    Madgwick, S.O.H. (2010). "An efficient orientation filter for inertial
    and inertial/magnetic sensor arrays"

This implementation supports 6-DOF IMU (accelerometer + gyroscope) without
magnetometer. It uses gradient descent optimization to fuse accelerometer
gravity reference with gyroscope angular velocity.

Advantages over Kalman Filter:
    - No covariance matrices (computationally simpler)
    - Single tuning parameter (beta)
    - Robust to sensor noise
    - No gimbal lock (quaternion representation)
"""

import numpy as np
from typing import Optional, Tuple
from .quaternion import quat_multiply, quat_normalize, quat_to_euler


class MadgwickFilter:
    """
    Madgwick AHRS Filter for 6-DOF IMU (no magnetometer).

    Uses gradient descent to find quaternion orientation that minimizes
    the error between measured and expected gravity direction.

    Attributes:
        beta (float): Filter gain controlling acc/gyro balance.
            - Higher beta = more accelerometer trust (stable but laggy)
            - Lower beta = more gyroscope trust (responsive but drifty)
            - Recommended: 0.1 for 30Hz sampling
        sample_freq (float): IMU sampling frequency in Hz.
        q (np.ndarray): Current quaternion estimate [w, x, y, z].
    """

    def __init__(self,
                 beta: float = 0.1,
                 sample_freq: float = 30.0,
                 initial_quaternion: Optional[np.ndarray] = None):
        """
        Initialize Madgwick filter.

        Args:
            beta: Filter gain (0.01-0.5). Default 0.1 optimal for 30Hz.
            sample_freq: Sampling frequency in Hz.
            initial_quaternion: Initial orientation [w, x, y, z]. Default identity.
        """
        self.beta = beta
        self.sample_freq = sample_freq
        self.dt = 1.0 / sample_freq

        # Initial quaternion (identity = level orientation)
        if initial_quaternion is not None:
            self.q = quat_normalize(initial_quaternion)
        else:
            self.q = np.array([1.0, 0.0, 0.0, 0.0])

    def update(self, acc: np.ndarray, gyro: np.ndarray) -> np.ndarray:
        """
        Update orientation estimate with new IMU measurements.

        Args:
            acc: Accelerometer [ax, ay, az] in m/s² (or any consistent unit).
            gyro: Gyroscope [gx, gy, gz] in rad/s.

        Returns:
            Updated quaternion [w, x, y, z].
        """
        q = self.q
        gx, gy, gz = gyro

        # Normalize accelerometer
        acc_norm = np.linalg.norm(acc)
        if acc_norm < 1e-10:
            # Skip accelerometer update if zero
            return self._gyro_only_update(gyro)

        ax, ay, az = acc / acc_norm

        # Current quaternion components
        q0, q1, q2, q3 = q

        # Auxiliary variables for gradient
        _2q0 = 2.0 * q0
        _2q1 = 2.0 * q1
        _2q2 = 2.0 * q2
        _2q3 = 2.0 * q3
        _4q0 = 4.0 * q0
        _4q1 = 4.0 * q1
        _4q2 = 4.0 * q2
        _8q1 = 8.0 * q1
        _8q2 = 8.0 * q2
        q0q0 = q0 * q0
        q1q1 = q1 * q1
        q2q2 = q2 * q2
        q3q3 = q3 * q3

        # Gradient descent step
        # Objective: minimize difference between measured acc and expected gravity
        # Expected gravity in body frame: [2*(q1q3 - q0q2), 2*(q0q1 + q2q3), q0q0 - q1q1 - q2q2 + q3q3]
        s0 = _4q0 * q2q2 + _2q2 * ax + _4q0 * q1q1 - _2q1 * ay
        s1 = _4q1 * q3q3 - _2q3 * ax + 4.0 * q0q0 * q1 - _2q0 * ay - _4q1 + _8q1 * q1q1 + _8q1 * q2q2 + _4q1 * az
        s2 = 4.0 * q0q0 * q2 + _2q0 * ax + _4q2 * q3q3 - _2q3 * ay - _4q2 + _8q2 * q1q1 + _8q2 * q2q2 + _4q2 * az
        s3 = 4.0 * q1q1 * q3 - _2q1 * ax + 4.0 * q2q2 * q3 - _2q2 * ay

        # Normalize gradient
        norm_s = np.sqrt(s0*s0 + s1*s1 + s2*s2 + s3*s3)
        if norm_s > 1e-10:
            s0 /= norm_s
            s1 /= norm_s
            s2 /= norm_s
            s3 /= norm_s

        # Gyroscope quaternion derivative
        qDot0 = 0.5 * (-q1 * gx - q2 * gy - q3 * gz)
        qDot1 = 0.5 * (q0 * gx + q2 * gz - q3 * gy)
        qDot2 = 0.5 * (q0 * gy - q1 * gz + q3 * gx)
        qDot3 = 0.5 * (q0 * gz + q1 * gy - q2 * gx)

        # Apply feedback (gradient descent correction)
        qDot0 -= self.beta * s0
        qDot1 -= self.beta * s1
        qDot2 -= self.beta * s2
        qDot3 -= self.beta * s3

        # Integrate to get new quaternion
        q0 += qDot0 * self.dt
        q1 += qDot1 * self.dt
        q2 += qDot2 * self.dt
        q3 += qDot3 * self.dt

        # Normalize
        self.q = quat_normalize(np.array([q0, q1, q2, q3]))

        return self.q.copy()

    def _gyro_only_update(self, gyro: np.ndarray) -> np.ndarray:
        """Update using only gyroscope (when acc is zero)."""
        q = self.q
        gx, gy, gz = gyro
        q0, q1, q2, q3 = q

        # Gyroscope quaternion derivative
        qDot0 = 0.5 * (-q1 * gx - q2 * gy - q3 * gz)
        qDot1 = 0.5 * (q0 * gx + q2 * gz - q3 * gy)
        qDot2 = 0.5 * (q0 * gy - q1 * gz + q3 * gx)
        qDot3 = 0.5 * (q0 * gz + q1 * gy - q2 * gx)

        # Integrate
        q0 += qDot0 * self.dt
        q1 += qDot1 * self.dt
        q2 += qDot2 * self.dt
        q3 += qDot3 * self.dt

        self.q = quat_normalize(np.array([q0, q1, q2, q3]))
        return self.q.copy()

    def get_orientation_quaternion(self) -> np.ndarray:
        """Return current quaternion [w, x, y, z]."""
        return self.q.copy()

    def get_orientation_euler(self) -> np.ndarray:
        """Return current Euler angles [roll, pitch, yaw] in radians."""
        return quat_to_euler(self.q)

    def reset(self, quaternion: Optional[np.ndarray] = None) -> None:
        """Reset filter to initial or specified state."""
        if quaternion is not None:
            self.q = quat_normalize(quaternion)
        else:
            self.q = np.array([1.0, 0.0, 0.0, 0.0])


class AdaptiveMadgwickFilter(MadgwickFilter):
    """
    Madgwick filter with adaptive beta based on acceleration magnitude.

    Reduces accelerometer trust during high-dynamic events (falls/impacts)
    when |a| deviates significantly from gravity.
    """

    def __init__(self,
                 beta_static: float = 0.1,
                 beta_dynamic: float = 0.01,
                 acc_threshold: float = 1.5,
                 sample_freq: float = 30.0,
                 **kwargs):
        """
        Initialize adaptive Madgwick filter.

        Args:
            beta_static: Beta when |a| ≈ g (more acc trust).
            beta_dynamic: Beta when |a| >> g (less acc trust).
            acc_threshold: Threshold in multiples of g for switching.
            sample_freq: Sampling frequency in Hz.
        """
        super().__init__(beta=beta_static, sample_freq=sample_freq, **kwargs)
        self.beta_static = beta_static
        self.beta_dynamic = beta_dynamic
        self.acc_threshold = acc_threshold
        self.gravity = 9.81

    def update(self, acc: np.ndarray, gyro: np.ndarray) -> np.ndarray:
        """Update with adaptive beta selection."""
        acc_mag = np.linalg.norm(acc)

        # Adaptive beta based on acceleration magnitude
        if acc_mag > self.acc_threshold * self.gravity:
            self.beta = self.beta_dynamic
        else:
            self.beta = self.beta_static

        return super().update(acc, gyro)


def process_trial_madgwick(acc_data: np.ndarray,
                           gyro_data: np.ndarray,
                           beta: float = 0.1,
                           sample_freq: float = 30.0,
                           output_format: str = 'quaternion',
                           adaptive: bool = False,
                           beta_dynamic: float = 0.01,
                           acc_threshold: float = 1.5) -> np.ndarray:
    """
    Process a trial through Madgwick filter.

    Args:
        acc_data: (T, 3) accelerometer in m/s².
        gyro_data: (T, 3) gyroscope in rad/s.
        beta: Filter gain.
        sample_freq: Sampling frequency.
        output_format: 'quaternion' (4D) or 'euler' (3D).
        adaptive: Use adaptive beta based on acceleration.
        beta_dynamic: Beta during high dynamics (if adaptive=True).
        acc_threshold: Threshold for switching beta (if adaptive=True).

    Returns:
        (T, 4) quaternions or (T, 3) Euler angles.
    """
    T = len(acc_data)

    if adaptive:
        filt = AdaptiveMadgwickFilter(
            beta_static=beta,
            beta_dynamic=beta_dynamic,
            acc_threshold=acc_threshold,
            sample_freq=sample_freq
        )
    else:
        filt = MadgwickFilter(beta=beta, sample_freq=sample_freq)

    if output_format == 'quaternion':
        orientations = np.zeros((T, 4))
        for t in range(T):
            filt.update(acc_data[t], gyro_data[t])
            orientations[t] = filt.get_orientation_quaternion()
    else:
        orientations = np.zeros((T, 3))
        for t in range(T):
            filt.update(acc_data[t], gyro_data[t])
            orientations[t] = filt.get_orientation_euler()

    return orientations
