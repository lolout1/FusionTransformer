"""
VQF (Versatile Quaternion-based Filter) Wrapper.

Wrapper for the state-of-the-art VQF orientation filter library.

References:
    Laidig, D., & Seel, T. (2023). "VQF: Highly Accurate IMU Orientation
    Estimation with Bias Estimation and Magnetic Disturbance Rejection"
    Information Fusion, Volume 91, Pages 187-204.

    https://github.com/dlaidig/vqf

VQF achieves 2.9° RMSE on standard benchmarks and includes:
    - Automatic gyroscope bias estimation
    - Magnetic disturbance rejection (if magnetometer used)
    - Handles variable sampling rates
    - Optimized C++ implementation with Python bindings
"""

import numpy as np
from typing import Optional, Tuple

try:
    from vqf import VQF
    VQF_AVAILABLE = True
except ImportError:
    VQF_AVAILABLE = False


class VQFWrapper:
    """
    Wrapper for VQF quaternion orientation filter.

    Provides consistent interface with other filters in this module.
    Falls back gracefully if VQF is not installed.

    Attributes:
        sample_freq (float): Sampling frequency in Hz.
        vqf: VQF filter instance.
    """

    def __init__(self,
                 sample_freq: float = 30.0,
                 tau_acc: float = 3.0,
                 tau_mag: float = 9.0,
                 motion_bias_est_enabled: bool = True,
                 rest_bias_est_enabled: bool = True,
                 mag_dist_rejection_enabled: bool = True):
        """
        Initialize VQF wrapper.

        Args:
            sample_freq: Sampling frequency in Hz. Default 30.0.
            tau_acc: Time constant for accelerometer correction in seconds.
                    Higher values = slower correction, more gyro trust.
            tau_mag: Time constant for magnetometer correction (if used).
            motion_bias_est_enabled: Enable bias estimation during motion.
            rest_bias_est_enabled: Enable bias estimation during rest.
            mag_dist_rejection_enabled: Enable magnetic disturbance rejection.
        """
        if not VQF_AVAILABLE:
            raise ImportError(
                "VQF library not installed. Install with: pip install vqf\n"
                "See: https://github.com/dlaidig/vqf"
            )

        self.sample_freq = sample_freq
        self.dt = 1.0 / sample_freq

        # Create VQF instance
        # For 6-DOF IMU without magnetometer
        self.vqf = VQF(gyrTs=self.dt, accTs=self.dt)

        # Configure parameters
        self.vqf.setTauAcc(tau_acc)
        if hasattr(self.vqf, 'setMotionBiasEstEnabled'):
            self.vqf.setMotionBiasEstEnabled(motion_bias_est_enabled)
        if hasattr(self.vqf, 'setRestBiasEstEnabled'):
            self.vqf.setRestBiasEstEnabled(rest_bias_est_enabled)

    def update(self, acc: np.ndarray, gyro: np.ndarray) -> np.ndarray:
        """
        Update orientation estimate with new IMU measurements.

        Args:
            acc: Accelerometer [ax, ay, az] in m/s².
            gyro: Gyroscope [gx, gy, gz] in rad/s.

        Returns:
            Quaternion [w, x, y, z].
        """
        # VQF requires float64 C-contiguous arrays (vqf_real_t = double)
        gyr = np.ascontiguousarray(gyro, dtype=np.float64)
        acc_arr = np.ascontiguousarray(acc, dtype=np.float64)

        # VQF expects gyr, acc order (different from our convention)
        self.vqf.update(gyr=gyr, acc=acc_arr)
        return self.vqf.getQuat6D()

    def get_orientation_quaternion(self) -> np.ndarray:
        """Return current quaternion [w, x, y, z]."""
        return self.vqf.getQuat6D()

    def get_orientation_euler(self) -> np.ndarray:
        """Return current Euler angles [roll, pitch, yaw] in radians."""
        q = self.vqf.getQuat6D()
        return quat_to_euler(q)

    def get_gyro_bias(self) -> np.ndarray:
        """Return estimated gyroscope bias [bx, by, bz] in rad/s."""
        bias = self.vqf.getBiasEstimate()
        return np.array(bias)

    def reset(self, quaternion: Optional[np.ndarray] = None) -> None:
        """Reset filter state."""
        # VQF doesn't have a direct reset with quaternion
        # Recreate the filter
        self.vqf = VQF(gyrTs=self.dt, accTs=self.dt)

    def get_diagnostics(self) -> dict:
        """Return diagnostic information."""
        return {
            'quaternion': self.get_orientation_quaternion(),
            'gyro_bias': self.get_gyro_bias(),
            'filter_type': 'vqf'
        }


def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to Euler angles (roll, pitch, yaw).

    Local implementation to avoid circular import.
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


def process_trial_vqf(acc_data: np.ndarray,
                      gyro_data: np.ndarray,
                      sample_freq: float = 30.0,
                      output_format: str = 'quaternion',
                      tau_acc: float = 3.0,
                      return_bias: bool = False) -> Tuple:
    """
    Process a trial through VQF filter.

    Args:
        acc_data: (T, 3) accelerometer in m/s².
        gyro_data: (T, 3) gyroscope in rad/s.
        sample_freq: Sampling frequency.
        output_format: 'quaternion' (4D) or 'euler' (3D).
        tau_acc: Time constant for accelerometer correction.
        return_bias: Also return gyro bias estimates.

    Returns:
        orientations: (T, 4) quaternions or (T, 3) Euler angles.
        bias: (T, 3) gyro bias estimates (if return_bias=True).
    """
    if not VQF_AVAILABLE:
        raise ImportError("VQF not installed. Run: pip install vqf")

    T = len(acc_data)
    filt = VQFWrapper(sample_freq=sample_freq, tau_acc=tau_acc)

    if output_format == 'quaternion':
        orientations = np.zeros((T, 4))
    else:
        orientations = np.zeros((T, 3))

    if return_bias:
        bias_estimates = np.zeros((T, 3))

    for t in range(T):
        filt.update(acc_data[t], gyro_data[t])

        if output_format == 'quaternion':
            orientations[t] = filt.get_orientation_quaternion()
        else:
            orientations[t] = filt.get_orientation_euler()

        if return_bias:
            bias_estimates[t] = filt.get_gyro_bias()

    if return_bias:
        return orientations, bias_estimates
    return orientations


def check_vqf_available() -> bool:
    """Check if VQF library is available."""
    return VQF_AVAILABLE


class VQFFallback:
    """
    Fallback implementation when VQF is not installed.

    Uses complementary filter approach for basic quaternion estimation.
    NOT a replacement for VQF - just provides minimal functionality.
    """

    def __init__(self, sample_freq: float = 30.0, alpha: float = 0.98):
        """
        Initialize fallback filter.

        Args:
            sample_freq: Sampling frequency.
            alpha: Complementary filter coefficient (gyro weight).
        """
        self.sample_freq = sample_freq
        self.dt = 1.0 / sample_freq
        self.alpha = alpha
        self.q = np.array([1.0, 0.0, 0.0, 0.0])

    def update(self, acc: np.ndarray, gyro: np.ndarray) -> np.ndarray:
        """Update with complementary filter (simplified)."""
        # Gyroscope integration
        omega_mag = np.linalg.norm(gyro)
        if omega_mag > 1e-10:
            axis = gyro / omega_mag
            angle = omega_mag * self.dt
            dq = np.array([
                np.cos(angle / 2),
                axis[0] * np.sin(angle / 2),
                axis[1] * np.sin(angle / 2),
                axis[2] * np.sin(angle / 2)
            ])
            q_gyro = quat_multiply(self.q, dq)
        else:
            q_gyro = self.q

        # Accelerometer correction (tilt only)
        acc_norm = np.linalg.norm(acc)
        if acc_norm > 1e-10:
            acc_n = acc / acc_norm
            # Estimate roll/pitch from accelerometer
            roll = np.arctan2(acc_n[1], acc_n[2])
            pitch = np.arctan2(-acc_n[0], np.sqrt(acc_n[1]**2 + acc_n[2]**2))
            q_acc = euler_to_quat(roll, pitch, 0.0)

            # Complementary blend (simplified - proper would use SLERP)
            self.q = self.alpha * q_gyro + (1 - self.alpha) * q_acc
            self.q = self.q / np.linalg.norm(self.q)
        else:
            self.q = q_gyro / np.linalg.norm(q_gyro)

        return self.q.copy()

    def get_orientation_quaternion(self) -> np.ndarray:
        return self.q.copy()

    def get_orientation_euler(self) -> np.ndarray:
        return quat_to_euler(self.q)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


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
