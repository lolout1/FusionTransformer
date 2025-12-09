"""
Pure numpy quaternion operations for Kalman filtering.

Convention: q = [w, x, y, z] = [q0, q1, q2, q3]
Right-handed coordinate system, Hamilton product convention.

References:
    Sola, J. (2017). "Quaternion kinematics for the error-state Kalman filter"
    arXiv:1711.02508
"""

import numpy as np
from typing import Union

Array = Union[np.ndarray, list]


def quat_multiply(q1: Array, q2: Array) -> np.ndarray:
    """Hamilton product of two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quat_conjugate(q: Array) -> np.ndarray:
    """Quaternion conjugate (inverse for unit quaternion)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_normalize(q: Array) -> np.ndarray:
    """Normalize quaternion to unit length."""
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return np.asarray(q) / norm


def quat_to_euler(q: Array) -> np.ndarray:
    """
    Convert quaternion to Euler angles (roll, pitch, yaw).

    Returns angles in radians, ZYX convention (yaw-pitch-roll).
    """
    w, x, y, z = q

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert Euler angles (radians) to quaternion."""
    cr, sr = np.cos(roll/2), np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)

    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy
    ])


def quat_from_axis_angle(axis: Array, angle: float) -> np.ndarray:
    """Create quaternion from axis-angle representation."""
    axis = np.asarray(axis)
    norm = np.linalg.norm(axis)
    if norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axis / norm
    s = np.sin(angle / 2)
    return np.array([np.cos(angle / 2), axis[0]*s, axis[1]*s, axis[2]*s])


def quat_rotate_vector(q: Array, v: Array) -> np.ndarray:
    """Rotate vector v by quaternion q."""
    q_v = np.array([0.0, v[0], v[1], v[2]])
    q_conj = quat_conjugate(q)
    return quat_multiply(quat_multiply(q, q_v), q_conj)[1:]


def quat_from_gyro(gyro: Array, dt: float) -> np.ndarray:
    """
    Create rotation quaternion from gyroscope angular velocity.

    Args:
        gyro: Angular velocity [wx, wy, wz] in rad/s
        dt: Time step in seconds

    Returns:
        Quaternion representing rotation over dt
    """
    gyro = np.asarray(gyro)
    angle = np.linalg.norm(gyro) * dt
    if angle < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = gyro / np.linalg.norm(gyro)
    return quat_from_axis_angle(axis, angle)


def quat_derivative(q: Array, gyro: Array) -> np.ndarray:
    """
    Compute quaternion time derivative from angular velocity.

    dq/dt = 0.5 * q * [0, wx, wy, wz]
    """
    omega_quat = np.array([0.0, gyro[0], gyro[1], gyro[2]])
    return 0.5 * quat_multiply(q, omega_quat)


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def acc_to_euler(acc: Array) -> np.ndarray:
    """
    Estimate roll and pitch from accelerometer (gravity reference).

    Yaw is unobservable without magnetometer, set to 0.

    Args:
        acc: Accelerometer [ax, ay, az] in m/s² or normalized

    Returns:
        [roll, pitch, 0] in radians
    """
    ax, ay, az = acc
    roll = np.arctan2(ay, az)
    pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
    return np.array([roll, pitch, 0.0])


def rotation_matrix_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Create 3x3 rotation matrix from Euler angles (ZYX convention)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp, cp*sr, cp*cr]
    ])
    return R


def skew_symmetric(v: Array) -> np.ndarray:
    """Create skew-symmetric matrix from vector (for cross product)."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
