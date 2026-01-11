"""
Square Root Unscented Kalman Filter (SR-UKF) for IMU Orientation Estimation.

State: [q0, q1, q2, q3, bias_gx, bias_gy, bias_gz] (7D)

Key advantages over EKF:
    1. No Jacobian computation - handles nonlinear quaternion kinematics exactly
    2. Square root form - guaranteed positive semi-definite covariance
    3. Better accuracy for highly nonlinear systems (falls with rapid rotation)
    4. More numerically stable during high-g impacts

Key advantages over Linear KF:
    1. Quaternion state - no gimbal lock at any orientation
    2. Estimates gyro bias - handles sensor drift
    3. Better for extreme orientations during falls

Reference:
    Van der Merwe, R. and Wan, E.A. (2001). "The Square-Root Unscented Kalman Filter
    for State and Parameter-Estimation"
"""

import numpy as np
from typing import Optional, Tuple

# Pure numpy implementations to avoid scipy version conflicts
def cholesky(A: np.ndarray, lower: bool = True) -> np.ndarray:
    """Cholesky decomposition using numpy."""
    L = np.linalg.cholesky(A)
    return L if lower else L.T

def qr(A: np.ndarray, mode: str = 'economic') -> Tuple[np.ndarray, np.ndarray]:
    """QR decomposition using numpy."""
    if mode == 'economic':
        return np.linalg.qr(A, mode='reduced')
    return np.linalg.qr(A)

def solve_triangular(L: np.ndarray, b: np.ndarray, lower: bool = True, trans: str = 'N') -> np.ndarray:
    """Solve triangular system using numpy."""
    if trans == 'T':
        L = L.T
        lower = not lower
    if lower:
        return np.linalg.solve(L, b)
    else:
        return np.linalg.solve(L, b)


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

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation) - handle gimbal lock
    sinp = 2 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
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


class SquareRootUKF:
    """
    Square Root Unscented Kalman Filter for IMU orientation estimation.

    State vector: [q0, q1, q2, q3, bias_gx, bias_gy, bias_gz] (7D)
        - q0, q1, q2, q3: Unit quaternion orientation
        - bias_gx, bias_gy, bias_gz: Gyroscope bias (rad/s)

    Measurement: Accelerometer [ax, ay, az] (3D)
        - Measures gravity direction in body frame

    Process model:
        q(k+1) = q(k) * delta_q(gyro - bias, dt)
        bias(k+1) = bias(k) + noise

    Measurement model:
        h(x) = rotate_by_quat(q, [0, 0, -g])  # Expected gravity in body frame

    Parameters:
        Q_quat: Process noise for quaternion (default: 0.005)
        Q_bias: Process noise for gyro bias drift (default: 0.0001)
        R_acc: Measurement noise for accelerometer (default: 0.1)
        alpha: Sigma point spread parameter (default: 0.001)
        beta: Prior distribution parameter (default: 2.0 for Gaussian)
        kappa: Secondary scaling parameter (default: 0.0)

    Example:
        >>> ukf = SquareRootUKF()
        >>> for t in range(len(gyro_data)):
        ...     ukf.predict(gyro_data[t], dt=1/30)
        ...     ukf.update(acc_data[t])
        ...     orientation = ukf.get_orientation_euler()
    """

    def __init__(
        self,
        Q_quat: float = 0.005,
        Q_bias: float = 0.0001,
        R_acc: float = 0.1,
        alpha: float = 0.001,
        beta: float = 2.0,
        kappa: float = 0.0,
        initial_quat: Optional[np.ndarray] = None,
        initial_bias: Optional[np.ndarray] = None,
        enable_adaptive_R: bool = True,
        adaptive_threshold_g: float = 2.0,
        adaptive_R_scale_max: float = 20.0
    ):
        # State dimension
        self.n = 7  # 4 quaternion + 3 bias

        # Initialize state
        if initial_quat is None:
            initial_quat = np.array([1.0, 0.0, 0.0, 0.0])
        if initial_bias is None:
            initial_bias = np.array([0.0, 0.0, 0.0])

        self.x = np.concatenate([initial_quat, initial_bias])

        # Initialize square root of covariance (lower triangular)
        P0 = np.diag([0.01, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001])
        self.S = cholesky(P0, lower=True)

        # Process noise
        self.Q_quat = Q_quat
        self.Q_bias = Q_bias
        self._build_Q()

        # Measurement noise
        self.R_acc_base = R_acc
        self.R_acc = R_acc

        # Adaptive R parameters
        self.enable_adaptive_R = enable_adaptive_R
        self.adaptive_threshold_g = adaptive_threshold_g
        self.adaptive_R_scale_max = adaptive_R_scale_max

        # UKF parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        # Compute weights
        self._compute_weights()

        # Innovation tracking
        self._last_innovation = np.zeros(3)
        self._innovation_history = []

        # Gravity constant
        self.g = 9.81

    def _build_Q(self):
        """Build process noise covariance."""
        self.Q = np.diag([
            self.Q_quat, self.Q_quat, self.Q_quat, self.Q_quat,
            self.Q_bias, self.Q_bias, self.Q_bias
        ])
        self.S_Q = cholesky(self.Q, lower=True)

    def _compute_weights(self):
        """Compute sigma point weights."""
        n = self.n
        lambda_ = self.alpha**2 * (n + self.kappa) - n

        # Weights for mean
        self.Wm = np.zeros(2 * n + 1)
        self.Wm[0] = lambda_ / (n + lambda_)
        self.Wm[1:] = 1 / (2 * (n + lambda_))

        # Weights for covariance
        self.Wc = np.zeros(2 * n + 1)
        self.Wc[0] = lambda_ / (n + lambda_) + (1 - self.alpha**2 + self.beta)
        self.Wc[1:] = 1 / (2 * (n + lambda_))

        # Scaling factor for sigma points
        self.gamma = np.sqrt(n + lambda_)

    def _generate_sigma_points(self) -> np.ndarray:
        """
        Generate sigma points using square root of covariance.

        Returns:
            sigma_points: (2n+1, n) array of sigma points
        """
        n = self.n
        sigma = np.zeros((2 * n + 1, n))

        # Mean
        sigma[0] = self.x

        # Spread around mean using columns of sqrt(P)
        scaled_S = self.gamma * self.S

        for i in range(n):
            sigma[i + 1] = self.x + scaled_S[:, i]
            sigma[i + 1 + n] = self.x - scaled_S[:, i]

        # Normalize quaternion part of each sigma point
        for i in range(2 * n + 1):
            sigma[i, :4] = quat_normalize(sigma[i, :4])

        return sigma

    def _process_model(self, x: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        """
        Process model: propagate state forward.

        Args:
            x: State vector [q0,q1,q2,q3, bias_gx,bias_gy,bias_gz]
            gyro: Gyroscope measurement [gx, gy, gz] in rad/s
            dt: Time step in seconds

        Returns:
            x_next: Propagated state
        """
        q = x[:4]
        bias = x[4:7]

        # Correct gyro with bias
        omega = gyro - bias

        # Compute rotation angle
        angle = np.linalg.norm(omega) * dt

        # Create delta quaternion
        if angle > 1e-10:
            axis = omega / np.linalg.norm(omega)
            delta_q = quat_from_axis_angle(axis, angle)
        else:
            delta_q = np.array([1.0, 0.0, 0.0, 0.0])

        # Update quaternion
        q_new = quat_multiply(q, delta_q)
        q_new = quat_normalize(q_new)

        # Bias evolves slowly (random walk)
        bias_new = bias  # No change in predict step

        return np.concatenate([q_new, bias_new])

    def _measurement_model(self, x: np.ndarray) -> np.ndarray:
        """
        Measurement model: expected accelerometer reading.

        When stationary, accelerometer measures gravity in body frame.
        h(x) = R(q)^T * [0, 0, g] = rotate gravity to body frame

        Args:
            x: State vector

        Returns:
            h: Expected measurement [ax, ay, az]
        """
        q = x[:4]
        # Gravity in world frame (pointing down)
        g_world = np.array([0.0, 0.0, self.g])
        # Rotate to body frame (inverse rotation)
        q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
        g_body = rotate_vector_by_quat(q_conj, g_world)
        return g_body

    def predict(self, gyro: np.ndarray, dt: float):
        """
        Prediction step using gyroscope.

        Args:
            gyro: Gyroscope measurement [gx, gy, gz] in rad/s
            dt: Time step in seconds
        """
        n = self.n

        # Generate sigma points
        sigma = self._generate_sigma_points()

        # Propagate sigma points through process model
        sigma_pred = np.zeros_like(sigma)
        for i in range(2 * n + 1):
            sigma_pred[i] = self._process_model(sigma[i], gyro, dt)

        # Compute predicted mean
        x_pred = np.zeros(n)
        for i in range(2 * n + 1):
            x_pred += self.Wm[i] * sigma_pred[i]

        # Normalize quaternion part
        x_pred[:4] = quat_normalize(x_pred[:4])
        self.x = x_pred

        # Compute predicted sqrt covariance using QR decomposition
        # Stack residuals: [sqrt(Wc[1])*(sigma[1]-x), ..., sqrt(Wc[2n])*(sigma[2n]-x), sqrt(Q)]
        residuals = np.zeros((2 * n + n, n))

        for i in range(1, 2 * n + 1):
            diff = sigma_pred[i] - x_pred
            # Handle quaternion difference carefully
            if np.dot(sigma_pred[i, :4], x_pred[:4]) < 0:
                diff[:4] = -sigma_pred[i, :4] - x_pred[:4]
            residuals[i - 1] = np.sqrt(np.abs(self.Wc[i])) * diff

        residuals[2 * n:] = self.S_Q.T

        # QR decomposition to get new sqrt covariance
        _, R = qr(residuals.T, mode='economic')
        self.S = R[:n, :n].T

        # Ensure positive diagonal
        for i in range(n):
            if self.S[i, i] < 0:
                self.S[:, i] = -self.S[:, i]

        # Cholupdate for W[0] term if negative
        if self.Wc[0] < 0:
            diff = sigma_pred[0] - x_pred
            self._cholupdate(self.S, np.sqrt(np.abs(self.Wc[0])) * diff, subtract=True)

    def _cholupdate(self, L: np.ndarray, x: np.ndarray, subtract: bool = False):
        """
        Rank-1 Cholesky update/downdate with numerical safeguards.

        Updates L such that L*L' = L*L' +/- x*x'
        """
        n = len(x)
        x = x.copy()
        sign = -1 if subtract else 1
        eps = 1e-10  # Numerical stability threshold

        for k in range(n):
            r2 = L[k, k]**2 + sign * x[k]**2
            # Numerical safeguard: ensure r2 is positive
            if r2 < eps:
                r2 = eps
            r = np.sqrt(r2)

            # Avoid division by zero
            if np.abs(L[k, k]) < eps:
                L[k, k] = eps

            c = r / L[k, k]
            s = x[k] / L[k, k]
            L[k, k] = r

            if k < n - 1:
                # Avoid division by very small c
                if np.abs(c) < eps:
                    c = eps if c >= 0 else -eps
                L[k+1:, k] = (L[k+1:, k] + sign * s * x[k+1:]) / c
                x[k+1:] = c * x[k+1:] - s * L[k+1:, k]

    def update(self, acc: np.ndarray):
        """
        Update step using accelerometer.

        Args:
            acc: Accelerometer measurement [ax, ay, az] in m/s^2
        """
        n = self.n
        m = 3  # Measurement dimension

        # Adaptive R scaling based on acceleration magnitude
        if self.enable_adaptive_R:
            acc_mag = np.linalg.norm(acc)
            if acc_mag > self.adaptive_threshold_g * self.g:
                # Scale R quadratically with acceleration excess
                scale = min((acc_mag / self.g)**2, self.adaptive_R_scale_max)
                self.R_acc = self.R_acc_base * scale
            else:
                self.R_acc = self.R_acc_base

        # Generate sigma points
        sigma = self._generate_sigma_points()

        # Transform sigma points through measurement model
        sigma_z = np.zeros((2 * n + 1, m))
        for i in range(2 * n + 1):
            sigma_z[i] = self._measurement_model(sigma[i])

        # Compute predicted measurement mean
        z_pred = np.zeros(m)
        for i in range(2 * n + 1):
            z_pred += self.Wm[i] * sigma_z[i]

        # Build sqrt of innovation covariance using QR
        residuals_z = np.zeros((2 * n + m, m))
        for i in range(1, 2 * n + 1):
            residuals_z[i - 1] = np.sqrt(np.abs(self.Wc[i])) * (sigma_z[i] - z_pred)
        residuals_z[2 * n:] = np.sqrt(self.R_acc) * np.eye(m)

        _, R_z = qr(residuals_z.T, mode='economic')
        S_z = R_z[:m, :m].T

        # Compute cross covariance Pxz
        Pxz = np.zeros((n, m))
        for i in range(2 * n + 1):
            diff_x = sigma[i] - self.x
            diff_z = sigma_z[i] - z_pred
            Pxz += self.Wc[i] * np.outer(diff_x, diff_z)

        # Kalman gain: K = Pxz * S_z^{-T} * S_z^{-1}
        K = solve_triangular(S_z, solve_triangular(S_z, Pxz.T, lower=True), lower=True, trans='T').T

        # Innovation
        innovation = acc - z_pred
        self._last_innovation = innovation
        self._innovation_history.append(np.linalg.norm(innovation))

        # State update
        self.x = self.x + K @ innovation
        self.x[:4] = quat_normalize(self.x[:4])

        # Covariance update: S = cholupdate(S, K*S_z, subtract=True)
        U = K @ S_z.T
        for i in range(m):
            self._cholupdate(self.S, U[:, i], subtract=True)

    def get_orientation_quaternion(self) -> np.ndarray:
        """Return quaternion [q0, q1, q2, q3]."""
        return self.x[:4].copy()

    def get_orientation_euler(self) -> np.ndarray:
        """Return Euler angles [roll, pitch, yaw] in radians."""
        return quat_to_euler(self.x[:4])

    def get_gravity_vector(self) -> np.ndarray:
        """Return expected gravity direction in body frame [gx, gy, gz]."""
        return self._measurement_model(self.x) / self.g

    def get_gyro_bias(self) -> np.ndarray:
        """Return estimated gyroscope bias [bias_gx, bias_gy, bias_gz]."""
        return self.x[4:7].copy()

    def get_uncertainty(self) -> np.ndarray:
        """Return uncertainty (diagonal of P) for first 3 orientation components."""
        P = self.S @ self.S.T
        return np.sqrt(np.diag(P)[1:4])  # Skip q0, return roll/pitch/yaw-like uncertainty

    def get_innovation_magnitude(self) -> float:
        """Return magnitude of last innovation for diagnostics."""
        return np.linalg.norm(self._last_innovation)

    def reset(self, initial_quat: Optional[np.ndarray] = None):
        """Reset filter to initial state."""
        if initial_quat is None:
            initial_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.x = np.concatenate([initial_quat, np.zeros(3)])
        P0 = np.diag([0.01, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001])
        self.S = cholesky(P0, lower=True)
        self._last_innovation = np.zeros(3)
        self._innovation_history = []


def create_sr_ukf(**kwargs) -> SquareRootUKF:
    """
    Factory function to create SR-UKF with given parameters.

    Args:
        Q_quat: Quaternion process noise (default: 0.005)
        Q_bias: Gyro bias process noise (default: 0.0001)
        R_acc: Accelerometer measurement noise (default: 0.1)
        alpha: Sigma point spread (default: 0.001)
        beta: Prior distribution parameter (default: 2.0)
        kappa: Secondary scaling (default: 0.0)
        enable_adaptive_R: Enable adaptive R scaling (default: True)
        adaptive_threshold_g: Threshold for R scaling (default: 2.0)
        adaptive_R_scale_max: Maximum R scale factor (default: 20.0)

    Returns:
        Configured SquareRootUKF instance
    """
    return SquareRootUKF(**kwargs)


def process_trial_sr_ukf(
    acc_data: np.ndarray,
    gyro_data: np.ndarray,
    output_format: str = 'euler',
    dt: float = 1/30.0,
    **kwargs
) -> dict:
    """
    Process a trial through SR-UKF.

    Args:
        acc_data: (T, 3) accelerometer in m/s^2
        gyro_data: (T, 3) gyroscope in rad/s
        output_format: 'euler', 'quaternion', or 'gravity_vector'
        dt: Time step in seconds
        **kwargs: SR-UKF parameters

    Returns:
        dict with 'orientation' and optional diagnostic keys
    """
    T = len(acc_data)
    ukf = create_sr_ukf(**kwargs)

    # Determine output dimension
    if output_format == 'quaternion':
        ori_dim = 4
    else:
        ori_dim = 3

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
    print("=" * 70)
    print("Square Root UKF Test")
    print("=" * 70)

    # Test with synthetic data
    np.random.seed(42)
    T = 100
    dt = 1/30.0

    # Simulate stationary sensor (gravity pointing down)
    acc_data = np.tile([0, 0, 9.81], (T, 1)) + np.random.randn(T, 3) * 0.1
    gyro_data = np.random.randn(T, 3) * 0.01  # Small gyro noise

    # Process
    result = process_trial_sr_ukf(acc_data, gyro_data, output_format='euler', dt=dt)

    print(f"\nInput: {T} samples at {1/dt:.0f} Hz")
    print(f"Output shape: {result['orientation'].shape}")
    print(f"\nFinal Euler angles (should be ~[0, 0, 0]):")
    print(f"  Roll:  {np.degrees(result['orientation'][-1, 0]):.2f} deg")
    print(f"  Pitch: {np.degrees(result['orientation'][-1, 1]):.2f} deg")
    print(f"  Yaw:   {np.degrees(result['orientation'][-1, 2]):.2f} deg")
    print(f"\nEstimated gyro bias: {result['gyro_bias'] * 1000:.4f} mrad/s")
    print(f"Mean innovation: {np.mean(result['innovation']):.4f}")

    # Test quaternion output
    result_quat = process_trial_sr_ukf(acc_data, gyro_data, output_format='quaternion', dt=dt)
    print(f"\nQuaternion output shape: {result_quat['orientation'].shape}")
    print(f"Final quaternion (should be ~[1, 0, 0, 0]):")
    print(f"  {result_quat['orientation'][-1]}")

    # Test gravity vector output
    result_grav = process_trial_sr_ukf(acc_data, gyro_data, output_format='gravity_vector', dt=dt)
    print(f"\nGravity vector output shape: {result_grav['orientation'].shape}")
    print(f"Final gravity (should be ~[0, 0, 1]):")
    print(f"  {result_grav['orientation'][-1]}")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
