# utils/kalman_preprocessing.py
"""
Linear Kalman Filter for IMU preprocessing.
Handles variable sampling rates from Android IMU API.

This is a SMOOTHING filter (denoises raw IMU signals) NOT a sensor fusion filter.
Input: 6ch [ax, ay, az, gx, gy, gz]
Output: 6ch [ax, ay, az, gx, gy, gz] (same channels, denoised)

References:
    - Sabatini 2011: Kalman filter for orientation estimation
    - Adapted for signal smoothing with constant-velocity model
"""

import numpy as np
from typing import Tuple, Optional
import torch


class LinearKalmanFilter:
    """
    Linear Kalman Filter with constant-velocity motion model.

    State: [position, velocity] for each axis
    Observation: direct measurement of position (acceleration/angular velocity)

    This filter smooths noisy IMU signals by assuming continuous motion
    with slowly-changing velocity.
    """

    def __init__(
        self,
        dim: int = 3,
        process_noise_pos: float = 0.01,
        process_noise_vel: float = 0.1,
        measurement_noise: float = 0.5,
        adaptive_dt: bool = True
    ):
        """
        Initialize Kalman filter.

        Args:
            dim: Number of dimensions to filter (3 for xyz)
            process_noise_pos: Process noise for position state (Q_pos)
            process_noise_vel: Process noise for velocity state (Q_vel)
            measurement_noise: Measurement noise (R)
            adaptive_dt: Whether to scale Q by actual dt
        """
        self.dim = dim
        self.state_dim = dim * 2  # [pos, vel] for each dimension
        self.process_noise_pos = process_noise_pos
        self.process_noise_vel = process_noise_vel
        self.measurement_noise = measurement_noise
        self.adaptive_dt = adaptive_dt

        # Observation matrix H: observe position only
        # H = [I_dim | 0_dim] -> (dim x state_dim)
        self.H = np.zeros((self.dim, self.state_dim))
        self.H[:self.dim, :self.dim] = np.eye(self.dim)

        # Measurement noise R: diagonal matrix
        self.R = np.eye(self.dim) * self.measurement_noise

    def _get_transition_matrix(self, dt: float) -> np.ndarray:
        """
        State transition matrix F for given dt.

        Constant velocity model:
        [pos_new]   [1  dt] [pos]
        [vel_new] = [0  1 ] [vel]

        Args:
            dt: Time step in seconds

        Returns:
            F: (state_dim x state_dim) transition matrix
        """
        F = np.eye(self.state_dim)
        # Position update: pos_new = pos + vel * dt
        F[:self.dim, self.dim:] = np.eye(self.dim) * dt
        return F

    def _get_process_noise(self, dt: float) -> np.ndarray:
        """
        Process noise Q scaled by dt.

        Args:
            dt: Time step in seconds

        Returns:
            Q: (state_dim x state_dim) process noise matrix
        """
        Q = np.zeros((self.state_dim, self.state_dim))
        scale = dt if self.adaptive_dt else 1.0

        # Position process noise
        Q[:self.dim, :self.dim] = np.eye(self.dim) * self.process_noise_pos * scale

        # Velocity process noise
        Q[self.dim:, self.dim:] = np.eye(self.dim) * self.process_noise_vel * scale

        return Q

    def filter(
        self,
        observations: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        return_covariance: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply Kalman filter to observation sequence.

        Args:
            observations: (T, dim) array of measurements
            timestamps: (T,) array of timestamps in seconds (optional)
            return_covariance: whether to return state covariance

        Returns:
            filtered: (T, dim) filtered observations
            covariance: (T, dim) diagonal of P (optional)
        """
        T = observations.shape[0]

        if T == 0:
            return observations, None

        # Initialize state with first observation
        x = np.zeros(self.state_dim)
        x[:self.dim] = observations[0]

        # Initialize covariance
        P = np.eye(self.state_dim) * 1.0

        # Output arrays
        filtered = np.zeros_like(observations)
        covariances = np.zeros_like(observations) if return_covariance else None

        for t in range(T):
            # Compute dt
            if timestamps is not None and t > 0:
                dt = max(timestamps[t] - timestamps[t-1], 0.001)
            else:
                # Default: assume 30Hz sampling
                dt = 1.0 / 30.0

            # === PREDICT STEP ===
            F = self._get_transition_matrix(dt)
            Q = self._get_process_noise(dt)

            x_pred = F @ x
            P_pred = F @ P @ F.T + Q

            # === UPDATE STEP ===
            z = observations[t]

            # Innovation (measurement residual)
            y = z - self.H @ x_pred

            # Innovation covariance
            S = self.H @ P_pred @ self.H.T + self.R

            # Kalman gain
            K = P_pred @ self.H.T @ np.linalg.inv(S)

            # State update
            x = x_pred + K @ y

            # Covariance update (Joseph form for numerical stability)
            I_KH = np.eye(self.state_dim) - K @ self.H
            P = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T

            # Store filtered position (not velocity)
            filtered[t] = x[:self.dim]

            if return_covariance:
                covariances[t] = np.diag(P)[:self.dim]

        return filtered, covariances


def apply_kalman_to_imu(
    acc_data: np.ndarray,
    gyro_data: np.ndarray,
    timestamps_acc: Optional[np.ndarray] = None,
    timestamps_gyro: Optional[np.ndarray] = None,
    config: dict = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Linear Kalman Filter to IMU data.

    Filters accelerometer and gyroscope separately with appropriate
    noise parameters for each sensor type.

    Args:
        acc_data: (T, 3) accelerometer [ax, ay, az]
        gyro_data: (T, 3) gyroscope [gx, gy, gz]
        timestamps_acc: (T,) timestamps for acc (optional)
        timestamps_gyro: (T,) timestamps for gyro (optional)
        config: dict with filter parameters

    Returns:
        acc_filtered: (T, 3)
        gyro_filtered: (T, 3)
    """
    if config is None:
        config = {
            'process_noise_pos': 0.01,
            'process_noise_vel': 0.1,
            'measurement_noise_acc': 0.5,
            'measurement_noise_gyro': 1.0
        }

    # Filter accelerometer (relatively clean signal)
    acc_kf = LinearKalmanFilter(
        dim=3,
        process_noise_pos=config.get('process_noise_pos', 0.01),
        process_noise_vel=config.get('process_noise_vel', 0.1),
        measurement_noise=config.get('measurement_noise_acc', 0.5),
        adaptive_dt=config.get('adaptive_dt', True)
    )
    acc_filtered, _ = acc_kf.filter(acc_data, timestamps_acc)

    # Filter gyroscope (noisier signal, higher measurement noise)
    gyro_kf = LinearKalmanFilter(
        dim=3,
        process_noise_pos=config.get('process_noise_pos', 0.01),
        process_noise_vel=config.get('process_noise_vel', 0.1),
        measurement_noise=config.get('measurement_noise_gyro', 1.0),
        adaptive_dt=config.get('adaptive_dt', True)
    )
    gyro_filtered, _ = gyro_kf.filter(gyro_data, timestamps_gyro)

    return acc_filtered.astype(np.float32), gyro_filtered.astype(np.float32)


def kalman_preprocess_batch(
    batch_data: torch.Tensor,
    enable_kalman: bool = True,
    kalman_config: dict = None
) -> torch.Tensor:
    """
    Apply Kalman preprocessing to batch of IMU data.

    This function is designed to be called during training/inference
    to apply Kalman smoothing to each sample in the batch.

    Args:
        batch_data: (B, T, C) where C=6 [ax,ay,az,gx,gy,gz] or C=8 with SMV
        enable_kalman: whether to apply filter
        kalman_config: filter parameters

    Returns:
        filtered_batch: (B, T, C)
    """
    if not enable_kalman:
        return batch_data

    if kalman_config is None:
        kalman_config = {
            'process_noise_pos': 0.01,
            'process_noise_vel': 0.1,
            'measurement_noise_acc': 0.5,
            'measurement_noise_gyro': 1.0
        }

    # Convert to numpy for processing
    device = batch_data.device
    dtype = batch_data.dtype
    batch_np = batch_data.cpu().numpy()
    B, T, C = batch_np.shape
    filtered = np.zeros_like(batch_np)

    for b in range(B):
        if C == 6:
            # [ax, ay, az, gx, gy, gz]
            acc = batch_np[b, :, :3]
            gyro = batch_np[b, :, 3:6]
            acc_f, gyro_f = apply_kalman_to_imu(acc, gyro, config=kalman_config)
            filtered[b, :, :3] = acc_f
            filtered[b, :, 3:6] = gyro_f

        elif C == 8:
            # [smv, ax, ay, az, gyro_mag, gx, gy, gz]
            acc = batch_np[b, :, 1:4]  # [ax, ay, az]
            gyro = batch_np[b, :, 5:8]  # [gx, gy, gz]
            acc_f, gyro_f = apply_kalman_to_imu(acc, gyro, config=kalman_config)

            # Recompute magnitude features from filtered data
            filtered[b, :, 0] = np.linalg.norm(acc_f, axis=1)  # SMV
            filtered[b, :, 1:4] = acc_f
            filtered[b, :, 4] = np.linalg.norm(gyro_f, axis=1)  # gyro_mag
            filtered[b, :, 5:8] = gyro_f

        elif C == 4:
            # [smv, ax, ay, az] - accelerometer only
            acc = batch_np[b, :, 1:4]
            acc_kf = LinearKalmanFilter(
                dim=3,
                process_noise_pos=kalman_config.get('process_noise_pos', 0.01),
                process_noise_vel=kalman_config.get('process_noise_vel', 0.1),
                measurement_noise=kalman_config.get('measurement_noise_acc', 0.5),
                adaptive_dt=True
            )
            acc_f, _ = acc_kf.filter(acc)
            filtered[b, :, 0] = np.linalg.norm(acc_f, axis=1)
            filtered[b, :, 1:4] = acc_f

        elif C == 3:
            # [ax, ay, az] - raw accelerometer only
            acc = batch_np[b, :, :3]
            acc_kf = LinearKalmanFilter(
                dim=3,
                process_noise_pos=kalman_config.get('process_noise_pos', 0.01),
                process_noise_vel=kalman_config.get('process_noise_vel', 0.1),
                measurement_noise=kalman_config.get('measurement_noise_acc', 0.5),
                adaptive_dt=True
            )
            acc_f, _ = acc_kf.filter(acc)
            filtered[b, :, :3] = acc_f

        else:
            raise ValueError(f"Unexpected channel count: {C}. Expected 3, 4, 6, or 8.")

    return torch.from_numpy(filtered).to(device=device, dtype=dtype)


if __name__ == "__main__":
    # Quick test
    print("Testing Kalman preprocessing...")

    # Test 6-channel input
    batch_6ch = torch.randn(4, 128, 6)
    filtered_6ch = kalman_preprocess_batch(batch_6ch, enable_kalman=True)
    print(f"6ch - Input shape: {batch_6ch.shape}, Output shape: {filtered_6ch.shape}")
    assert filtered_6ch.shape == batch_6ch.shape

    # Test 8-channel input
    batch_8ch = torch.randn(4, 128, 8)
    filtered_8ch = kalman_preprocess_batch(batch_8ch, enable_kalman=True)
    print(f"8ch - Input shape: {batch_8ch.shape}, Output shape: {filtered_8ch.shape}")
    assert filtered_8ch.shape == batch_8ch.shape

    # Test disabled
    filtered_disabled = kalman_preprocess_batch(batch_6ch, enable_kalman=False)
    assert torch.equal(filtered_disabled, batch_6ch)

    # Verify smoothing effect
    noise_level_before = batch_6ch.std()
    noise_level_after = filtered_6ch.std()
    print(f"Std before: {noise_level_before:.4f}, after: {noise_level_after:.4f}")

    print("Kalman preprocessing test passed!")
