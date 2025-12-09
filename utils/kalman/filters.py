"""
Kalman filter implementations for IMU orientation estimation.

Implements both standard linear Kalman filter (Euler angle state) and
Extended Kalman Filter (quaternion state with gyro bias estimation).

References:
    Kalman, R.E. (1960). "A New Approach to Linear Filtering and Prediction"
    Sabatini, A.M. (2011). "Estimating Three-Dimensional Orientation of Human
        Body Parts by Inertial/Magnetic Sensing"
    Sola, J. (2017). "Quaternion kinematics for the error-state Kalman filter"
"""

import numpy as np
from typing import Tuple, Optional, Dict
from .quaternion import (
    quat_multiply, quat_normalize, quat_to_euler, euler_to_quat,
    quat_from_gyro, acc_to_euler, wrap_angle
)


class KalmanFilter:
    """
    Standard Linear Kalman Filter for orientation estimation.

    State vector: [roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate] (6D)

    Observation model:
        - roll, pitch from accelerometer (gravity reference)
        - angular rates from gyroscope
        - yaw is NOT directly observable (no magnetometer)
    """

    def __init__(self,
                 Q_orientation: float = 0.01,
                 Q_rate: float = 0.1,
                 R_acc: float = 0.1,
                 R_gyro: float = 0.5):
        """
        Initialize Kalman filter.

        Args:
            Q_orientation: Process noise for orientation (rad²)
            Q_rate: Process noise for angular rates ((rad/s)²)
            R_acc: Measurement noise for accelerometer-derived angles (rad²)
            R_gyro: Measurement noise for gyroscope rates ((rad/s)²)
        """
        self.state_dim = 6
        self.meas_dim = 5  # roll, pitch from acc + 3 gyro rates

        # State: [roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]
        self.x = np.zeros(self.state_dim)

        # State covariance
        self.P = np.eye(self.state_dim) * 0.1

        # Process noise covariance
        self.Q = np.diag([Q_orientation, Q_orientation, Q_orientation,
                         Q_rate, Q_rate, Q_rate])

        # Measurement noise covariance [roll_acc, pitch_acc, gx, gy, gz]
        self.R = np.diag([R_acc, R_acc, R_gyro, R_gyro, R_gyro])

        # State transition matrix (updated per timestep)
        self.F = np.eye(self.state_dim)

        # Measurement matrix: maps state to measurements
        # z = [roll_acc, pitch_acc, gx, gy, gz]
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],  # roll_acc = roll
            [0, 1, 0, 0, 0, 0],  # pitch_acc = pitch
            [0, 0, 0, 1, 0, 0],  # gx = roll_rate
            [0, 0, 0, 0, 1, 0],  # gy = pitch_rate
            [0, 0, 0, 0, 0, 1],  # gz = yaw_rate
        ])

        self.last_innovation = np.zeros(self.meas_dim)
        self.innovation_cov = np.eye(self.meas_dim)

    def predict(self, dt: float) -> None:
        """Propagate state forward in time."""
        # Update state transition matrix
        self.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ])

        # Predict state
        self.x = self.F @ self.x

        # Wrap angles to [-pi, pi]
        self.x[:3] = np.array([wrap_angle(a) for a in self.x[:3]])

        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Enforce symmetry
        self.P = 0.5 * (self.P + self.P.T)

    def update(self, acc: np.ndarray, gyro: np.ndarray) -> None:
        """
        Incorporate new measurements.

        Args:
            acc: Accelerometer [ax, ay, az] in m/s²
            gyro: Gyroscope [gx, gy, gz] in rad/s
        """
        # Extract roll/pitch from accelerometer
        acc_euler = acc_to_euler(acc)
        roll_acc = acc_euler[0]
        pitch_acc = acc_euler[1]

        # Measurement vector
        z = np.array([roll_acc, pitch_acc, gyro[0], gyro[1], gyro[2]])

        # Expected measurement
        z_pred = self.H @ self.x

        # Innovation (wrap angles)
        y = z - z_pred
        y[0] = wrap_angle(y[0])
        y[1] = wrap_angle(y[1])
        self.last_innovation = y

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        self.innovation_cov = S

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ y

        # Wrap angles
        self.x[:3] = np.array([wrap_angle(a) for a in self.x[:3]])

        # Update covariance (Joseph form for numerical stability)
        I_KH = np.eye(self.state_dim) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        # Enforce symmetry and positive definiteness
        self.P = 0.5 * (self.P + self.P.T)
        self.P = np.maximum(self.P, 0)

    def get_orientation(self) -> np.ndarray:
        """Return [roll, pitch, yaw] in radians."""
        return self.x[:3].copy()

    def get_uncertainty(self) -> np.ndarray:
        """Return orientation uncertainty (sqrt of P diagonal)."""
        return np.sqrt(np.diag(self.P)[:3])

    def get_innovation(self) -> np.ndarray:
        """Return last innovation vector."""
        return self.last_innovation.copy()

    def get_innovation_magnitude(self) -> float:
        """Return L2 norm of innovation."""
        return np.linalg.norm(self.last_innovation)

    def reset(self) -> None:
        """Reset to initial state."""
        self.x = np.zeros(self.state_dim)
        self.P = np.eye(self.state_dim) * 0.1
        self.last_innovation = np.zeros(self.meas_dim)


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter with quaternion state and gyro bias estimation.

    State vector: [q0, q1, q2, q3, bias_gx, bias_gy, bias_gz] (7D)

    Advantages over linear KF:
        - No gimbal lock (quaternion representation)
        - Estimates gyroscope bias drift
        - More accurate for large rotations
    """

    def __init__(self,
                 Q_quat: float = 0.001,
                 Q_bias: float = 0.0001,
                 R_acc: float = 0.1):
        """
        Initialize EKF.

        Args:
            Q_quat: Process noise for quaternion (unitless)
            Q_bias: Process noise for gyro bias drift ((rad/s)²)
            R_acc: Measurement noise for accelerometer (m/s²)²
        """
        self.state_dim = 7
        self.meas_dim = 3  # accelerometer

        # State: [q0, q1, q2, q3, bias_gx, bias_gy, bias_gz]
        self.x = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # State covariance
        self.P = np.eye(self.state_dim) * 0.01
        self.P[4:, 4:] *= 0.001  # Lower initial uncertainty for bias

        # Process noise
        self.Q = np.diag([Q_quat]*4 + [Q_bias]*3)

        # Measurement noise
        self.R = np.eye(3) * R_acc

        self.last_innovation = np.zeros(3)
        self.innovation_cov = np.eye(3)
        self.gravity = np.array([0, 0, 9.81])

    def predict(self, gyro: np.ndarray, dt: float) -> None:
        """
        Propagate state using gyroscope.

        Args:
            gyro: Gyroscope [gx, gy, gz] in rad/s
            dt: Time step in seconds
        """
        q = self.x[:4]
        bias = self.x[4:]

        # Correct gyro with bias estimate
        gyro_corrected = gyro - bias

        # Quaternion update from angular velocity
        dq = quat_from_gyro(gyro_corrected, dt)
        q_new = quat_multiply(q, dq)
        q_new = quat_normalize(q_new)

        self.x[:4] = q_new
        # Bias evolves as random walk (no change in prediction)

        # Compute Jacobian F
        F = self._compute_F(gyro_corrected, dt)

        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q
        self.P = 0.5 * (self.P + self.P.T)

    def _compute_F(self, gyro: np.ndarray, dt: float) -> np.ndarray:
        """Compute state transition Jacobian."""
        wx, wy, wz = gyro
        F = np.eye(7)

        # Quaternion kinematics Jacobian (linearized)
        Omega = 0.5 * dt * np.array([
            [0, -wx, -wy, -wz],
            [wx, 0, wz, -wy],
            [wy, -wz, 0, wx],
            [wz, wy, -wx, 0]
        ])
        F[:4, :4] = np.eye(4) + Omega

        # Jacobian of quaternion w.r.t. bias
        q = self.x[:4]
        F[:4, 4:7] = -0.5 * dt * np.array([
            [-q[1], -q[2], -q[3]],
            [q[0], -q[3], q[2]],
            [q[3], q[0], -q[1]],
            [-q[2], q[1], q[0]]
        ])

        return F

    def update(self, acc: np.ndarray) -> None:
        """
        Correct using accelerometer gravity reference.

        Args:
            acc: Accelerometer [ax, ay, az] in m/s²
        """
        # Normalize accelerometer
        acc_norm = np.linalg.norm(acc)
        if acc_norm < 1e-6:
            return
        acc_normalized = acc / acc_norm

        # Expected gravity in body frame
        q = self.x[:4]
        g_expected = self._rotate_gravity(q)

        # Innovation
        y = acc_normalized - g_expected
        self.last_innovation = y

        # Measurement Jacobian
        H = self._compute_H(q)

        # Innovation covariance
        S = H @ self.P @ H.T + self.R / (acc_norm**2)
        self.innovation_cov = S

        # Kalman gain
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return

        # Update state
        dx = K @ y
        self.x = self.x + dx

        # Renormalize quaternion
        self.x[:4] = quat_normalize(self.x[:4])

        # Update covariance (Joseph form)
        I_KH = np.eye(self.state_dim) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ (self.R / (acc_norm**2)) @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    def _rotate_gravity(self, q: np.ndarray) -> np.ndarray:
        """Rotate gravity vector from world to body frame."""
        q0, q1, q2, q3 = q
        # Rotation matrix from quaternion (transpose for world to body)
        R = np.array([
            [1-2*(q2**2+q3**2), 2*(q1*q2+q0*q3), 2*(q1*q3-q0*q2)],
            [2*(q1*q2-q0*q3), 1-2*(q1**2+q3**2), 2*(q2*q3+q0*q1)],
            [2*(q1*q3+q0*q2), 2*(q2*q3-q0*q1), 1-2*(q1**2+q2**2)]
        ])
        # Gravity is [0,0,1] normalized (pointing up)
        return R.T @ np.array([0, 0, 1])

    def _compute_H(self, q: np.ndarray) -> np.ndarray:
        """Compute measurement Jacobian."""
        q0, q1, q2, q3 = q
        H = np.zeros((3, 7))

        # Jacobian of rotated gravity w.r.t. quaternion
        H[:, 0] = 2 * np.array([q2, -q1, q0])
        H[:, 1] = 2 * np.array([q3, q0, q1])
        H[:, 2] = 2 * np.array([q0, -q3, q2])
        H[:, 3] = 2 * np.array([-q1, -q2, q3])

        return H

    def get_orientation_quaternion(self) -> np.ndarray:
        """Return [q0, q1, q2, q3]."""
        return self.x[:4].copy()

    def get_orientation_euler(self) -> np.ndarray:
        """Return [roll, pitch, yaw] in radians."""
        return quat_to_euler(self.x[:4])

    def get_gyro_bias(self) -> np.ndarray:
        """Return estimated [bias_gx, bias_gy, bias_gz]."""
        return self.x[4:].copy()

    def get_uncertainty(self) -> np.ndarray:
        """Return orientation uncertainty from quaternion covariance."""
        # Convert quaternion covariance to Euler angle uncertainty (approximation)
        quat_var = np.diag(self.P[:4])
        # Rough approximation: Euler uncertainty ~ 2 * quaternion uncertainty
        euler_var = 4 * np.mean(quat_var)
        return np.array([np.sqrt(euler_var)] * 3)

    def get_innovation(self) -> np.ndarray:
        """Return last innovation vector."""
        return self.last_innovation.copy()

    def get_innovation_magnitude(self) -> float:
        """Return L2 norm of innovation."""
        return np.linalg.norm(self.last_innovation)

    def reset(self) -> None:
        """Reset to initial state."""
        self.x = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.P = np.eye(self.state_dim) * 0.01
        self.P[4:, 4:] *= 0.001
        self.last_innovation = np.zeros(3)


def create_filter(filter_type: str = 'linear', **params) -> object:
    """
    Factory function to create Kalman filter.

    Args:
        filter_type: 'linear' or 'ekf'
        **params: Filter-specific parameters

    Returns:
        KalmanFilter or ExtendedKalmanFilter instance
    """
    if filter_type == 'linear':
        return KalmanFilter(
            Q_orientation=params.get('Q_orientation', 0.01),
            Q_rate=params.get('Q_rate', 0.1),
            R_acc=params.get('R_acc', 0.1),
            R_gyro=params.get('R_gyro', 0.5)
        )
    elif filter_type == 'ekf':
        return ExtendedKalmanFilter(
            Q_quat=params.get('Q_quat', 0.001),
            Q_bias=params.get('Q_bias', 0.0001),
            R_acc=params.get('R_acc', 0.1)
        )
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
