"""
Unscented Kalman Filter (UKF) for IMU Orientation Estimation.

Simpler and more numerically stable than SR-UKF for initial deployment.

State: [q0, q1, q2, q3, bias_gx, bias_gy, bias_gz] (7D)

Key advantages over EKF:
    1. No Jacobian computation - handles nonlinear quaternion kinematics exactly
    2. Better accuracy for highly nonlinear systems (falls with rapid rotation)
"""

import numpy as np
from typing import Optional, Tuple, Dict


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion to unit length."""
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion multiplication (Hamilton product)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Create quaternion from axis-angle representation."""
    if np.abs(angle) < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    half_angle = angle / 2
    axis_norm = axis / (np.linalg.norm(axis) + 1e-10)
    return np.array([
        np.cos(half_angle),
        axis_norm[0] * np.sin(half_angle),
        axis_norm[1] * np.sin(half_angle),
        axis_norm[2] * np.sin(half_angle)
    ])


def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to Euler angles (roll, pitch, yaw)."""
    w, x, y, z = q
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


def rotate_vector_by_quat(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q."""
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    v_quat = np.array([0.0, v[0], v[1], v[2]])
    result = quat_multiply(quat_multiply(q, v_quat), q_conj)
    return result[1:4]


class UnscentedKalmanFilter:
    """
    Standard Unscented Kalman Filter for IMU orientation estimation.

    State vector: [q0, q1, q2, q3, bias_gx, bias_gy, bias_gz] (7D)

    Parameters:
        Q_quat: Process noise for quaternion (default: 0.005)
        Q_bias: Process noise for gyro bias drift (default: 0.0001)
        R_acc: Measurement noise for accelerometer (default: 0.1)
        alpha: Sigma point spread parameter (default: 0.1)
        beta: Prior distribution parameter (default: 2.0 for Gaussian)
        kappa: Secondary scaling parameter (default: 0.0)
    """

    def __init__(
        self,
        Q_quat: float = 0.005,
        Q_bias: float = 0.0001,
        R_acc: float = 0.1,
        alpha: float = 0.1,
        beta: float = 2.0,
        kappa: float = 0.0,
        initial_quat: Optional[np.ndarray] = None,
        initial_bias: Optional[np.ndarray] = None,
        enable_adaptive_R: bool = True,
        adaptive_threshold_g: float = 2.0,
        adaptive_R_scale_max: float = 20.0
    ):
        self.n = 7  # State dimension

        # Initialize state
        if initial_quat is None:
            initial_quat = np.array([1.0, 0.0, 0.0, 0.0])
        if initial_bias is None:
            initial_bias = np.array([0.0, 0.0, 0.0])

        self.x = np.concatenate([initial_quat, initial_bias])

        # Initialize covariance
        self.P = np.diag([0.01, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001])

        # Process noise
        self.Q = np.diag([
            Q_quat, Q_quat, Q_quat, Q_quat,
            Q_bias, Q_bias, Q_bias
        ])

        # Measurement noise
        self.R_acc_base = R_acc
        self.R_acc = R_acc
        self.R = np.eye(3) * R_acc

        # Adaptive R parameters
        self.enable_adaptive_R = enable_adaptive_R
        self.adaptive_threshold_g = adaptive_threshold_g
        self.adaptive_R_scale_max = adaptive_R_scale_max

        # UKF parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self._compute_weights()

        # Innovation tracking
        self._last_innovation = np.zeros(3)
        self.g = 9.81

    def _compute_weights(self):
        """Compute sigma point weights."""
        n = self.n
        lambda_ = self.alpha**2 * (n + self.kappa) - n

        self.Wm = np.zeros(2 * n + 1)
        self.Wm[0] = lambda_ / (n + lambda_)
        self.Wm[1:] = 1 / (2 * (n + lambda_))

        self.Wc = np.zeros(2 * n + 1)
        self.Wc[0] = lambda_ / (n + lambda_) + (1 - self.alpha**2 + self.beta)
        self.Wc[1:] = 1 / (2 * (n + lambda_))

        self.gamma = np.sqrt(n + lambda_)

    def _generate_sigma_points(self) -> np.ndarray:
        """Generate sigma points using Cholesky decomposition."""
        n = self.n
        sigma = np.zeros((2 * n + 1, n))
        sigma[0] = self.x

        try:
            sqrtP = np.linalg.cholesky(self.P)
        except np.linalg.LinAlgError:
            # Regularize P if not positive definite
            self.P = self.P + np.eye(n) * 1e-6
            sqrtP = np.linalg.cholesky(self.P)

        scaled_sqrtP = self.gamma * sqrtP

        for i in range(n):
            sigma[i + 1] = self.x + scaled_sqrtP[:, i]
            sigma[i + 1 + n] = self.x - scaled_sqrtP[:, i]

        # Normalize quaternion part
        for i in range(2 * n + 1):
            sigma[i, :4] = quat_normalize(sigma[i, :4])

        return sigma

    def _process_model(self, x: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        """Propagate state forward using gyroscope."""
        q = x[:4]
        bias = x[4:7]
        omega = gyro - bias
        angle = np.linalg.norm(omega) * dt

        if angle > 1e-10:
            axis = omega / np.linalg.norm(omega)
            delta_q = quat_from_axis_angle(axis, angle)
        else:
            delta_q = np.array([1.0, 0.0, 0.0, 0.0])

        q_new = quat_multiply(q, delta_q)
        q_new = quat_normalize(q_new)

        return np.concatenate([q_new, bias])

    def _measurement_model(self, x: np.ndarray) -> np.ndarray:
        """Expected accelerometer reading (gravity in body frame)."""
        q = x[:4]
        g_world = np.array([0.0, 0.0, self.g])
        q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
        g_body = rotate_vector_by_quat(q_conj, g_world)
        return g_body

    def predict(self, gyro: np.ndarray, dt: float):
        """Prediction step using gyroscope."""
        n = self.n
        sigma = self._generate_sigma_points()

        # Propagate sigma points
        sigma_pred = np.zeros_like(sigma)
        for i in range(2 * n + 1):
            sigma_pred[i] = self._process_model(sigma[i], gyro, dt)

        # Compute predicted mean
        x_pred = np.zeros(n)
        for i in range(2 * n + 1):
            x_pred += self.Wm[i] * sigma_pred[i]
        x_pred[:4] = quat_normalize(x_pred[:4])

        # Compute predicted covariance
        P_pred = self.Q.copy()
        for i in range(2 * n + 1):
            diff = sigma_pred[i] - x_pred
            # Handle quaternion sign ambiguity
            if np.dot(sigma_pred[i, :4], x_pred[:4]) < 0:
                diff[:4] = -sigma_pred[i, :4] - x_pred[:4]
            P_pred += self.Wc[i] * np.outer(diff, diff)

        self.x = x_pred
        self.P = P_pred

    def update(self, acc: np.ndarray):
        """Update step using accelerometer."""
        n = self.n
        m = 3

        # Adaptive R scaling
        if self.enable_adaptive_R:
            acc_mag = np.linalg.norm(acc)
            if acc_mag > self.adaptive_threshold_g * self.g:
                scale = min((acc_mag / self.g)**2, self.adaptive_R_scale_max)
                self.R = np.eye(m) * self.R_acc_base * scale
            else:
                self.R = np.eye(m) * self.R_acc_base

        # Generate sigma points
        sigma = self._generate_sigma_points()

        # Transform through measurement model
        sigma_z = np.zeros((2 * n + 1, m))
        for i in range(2 * n + 1):
            sigma_z[i] = self._measurement_model(sigma[i])

        # Predicted measurement mean
        z_pred = np.zeros(m)
        for i in range(2 * n + 1):
            z_pred += self.Wm[i] * sigma_z[i]

        # Innovation covariance
        Pzz = self.R.copy()
        for i in range(2 * n + 1):
            diff_z = sigma_z[i] - z_pred
            Pzz += self.Wc[i] * np.outer(diff_z, diff_z)

        # Cross covariance
        Pxz = np.zeros((n, m))
        for i in range(2 * n + 1):
            diff_x = sigma[i] - self.x
            diff_z = sigma_z[i] - z_pred
            Pxz += self.Wc[i] * np.outer(diff_x, diff_z)

        # Kalman gain
        K = Pxz @ np.linalg.inv(Pzz)

        # Update
        innovation = acc - z_pred
        self._last_innovation = innovation

        self.x = self.x + K @ innovation
        self.x[:4] = quat_normalize(self.x[:4])
        self.P = self.P - K @ Pzz @ K.T

        # Ensure P stays symmetric and positive definite
        self.P = (self.P + self.P.T) / 2
        min_eig = np.min(np.linalg.eigvalsh(self.P))
        if min_eig < 1e-10:
            self.P += np.eye(n) * (1e-10 - min_eig)

    def get_orientation_quaternion(self) -> np.ndarray:
        return self.x[:4].copy()

    def get_orientation_euler(self) -> np.ndarray:
        return quat_to_euler(self.x[:4])

    def get_gravity_vector(self) -> np.ndarray:
        return self._measurement_model(self.x) / self.g

    def get_gyro_bias(self) -> np.ndarray:
        return self.x[4:7].copy()

    def get_uncertainty(self) -> np.ndarray:
        return np.sqrt(np.diag(self.P)[1:4])

    def get_innovation_magnitude(self) -> float:
        return np.linalg.norm(self._last_innovation)


def process_trial_ukf(
    acc_data: np.ndarray,
    gyro_data: np.ndarray,
    output_format: str = 'euler',
    dt: float = 1/30.0,
    **kwargs
) -> Dict:
    """
    Process a trial through UKF.

    Args:
        acc_data: (T, 3) accelerometer in m/s^2
        gyro_data: (T, 3) gyroscope in rad/s
        output_format: 'euler', 'quaternion', or 'gravity_vector'
        dt: Time step in seconds
        **kwargs: UKF parameters

    Returns:
        dict with 'orientation', 'uncertainty', 'innovation', 'gyro_bias'
    """
    T = len(acc_data)
    ukf = UnscentedKalmanFilter(**kwargs)

    ori_dim = 4 if output_format == 'quaternion' else 3
    orientations = np.zeros((T, ori_dim))
    uncertainties = np.zeros((T, 3))
    innovations = np.zeros(T)

    for t in range(T):
        ukf.predict(gyro_data[t], dt)
        ukf.update(acc_data[t])

        if output_format == 'quaternion':
            orientations[t] = ukf.get_orientation_quaternion()
        elif output_format == 'gravity_vector':
            orientations[t] = ukf.get_gravity_vector()
        else:
            orientations[t] = ukf.get_orientation_euler()

        uncertainties[t] = ukf.get_uncertainty()
        innovations[t] = ukf.get_innovation_magnitude()

    return {
        'orientation': orientations,
        'uncertainty': uncertainties,
        'innovation': innovations.reshape(-1, 1),
        'gyro_bias': ukf.get_gyro_bias()
    }


if __name__ == "__main__":
    print("=" * 60)
    print("UKF Test")
    print("=" * 60)

    np.random.seed(42)
    T = 50
    dt = 1/30.0

    acc = np.tile([0, 0, 9.81], (T, 1)) + np.random.randn(T, 3) * 0.1
    gyro = np.random.randn(T, 3) * 0.01

    result = process_trial_ukf(acc, gyro, output_format='euler', dt=dt)

    print(f"Output shape: {result['orientation'].shape}")
    euler = result['orientation'][-1]
    print(f"Final Euler (deg): roll={np.degrees(euler[0]):.1f}, pitch={np.degrees(euler[1]):.1f}, yaw={np.degrees(euler[2]):.1f}")
    print(f"Gyro bias (mrad/s): {result['gyro_bias'] * 1000}")
    print("UKF works!")
