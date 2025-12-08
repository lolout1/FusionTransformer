"""
Kalman Smoothing for IMU Signal Denoising

Unlike Kalman FUSION (which combines acc+gyro -> orientation), Kalman SMOOTHING
applies a 1D Kalman filter to each channel independently to reduce noise while
preserving dynamics.

State model (per channel):
    x_k = [position, velocity]^T  (2D state)
    z_k = raw_signal              (1D measurement)

Process model: x_k = F * x_{k-1} + w
    F = [[1, dt], [0, 1]]  # Constant velocity model

Measurement model: z_k = H * x_k + v
    H = [1, 0]  # We observe position only

Tunable parameters:
    - Q (process noise): Higher = trust dynamics more, smoother output
    - R (measurement noise): Higher = trust measurements less, smoother output

This is DIFFERENT from utils/kalman.py which does sensor FUSION.
"""

import numpy as np
from typing import Dict, Tuple, Optional


class KalmanSmoother1D:
    """1D Kalman filter for signal smoothing with constant velocity model."""

    def __init__(self,
                 dt: float = 1/30.0,
                 process_noise: float = 0.01,
                 measurement_noise: float = 0.1):
        """
        Initialize Kalman smoother for 1D signal.

        Args:
            dt: Time step (1/sample_rate)
            process_noise: Q - higher = smoother (trust dynamics)
            measurement_noise: R - higher = smoother (distrust measurements)
        """
        self.dt = dt

        # State transition matrix (constant velocity model)
        # x_k = [position, velocity]
        self.F = np.array([[1.0, dt],
                          [0.0, 1.0]], dtype=np.float64)

        # Measurement matrix (observe position only)
        self.H = np.array([[1.0, 0.0]], dtype=np.float64)

        # Process noise covariance (discretized continuous white noise)
        self.Q = process_noise * np.array([
            [dt**4/4, dt**3/2],
            [dt**3/2, dt**2]
        ], dtype=np.float64)

        # Measurement noise covariance
        self.R = np.array([[measurement_noise]], dtype=np.float64)

    def smooth(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply Kalman smoothing to 1D signal.

        Args:
            signal: (T,) raw signal

        Returns:
            smoothed: (T,) smoothed signal
        """
        T = len(signal)
        if T == 0:
            return signal.copy()

        # Initialize state with first measurement
        x = np.array([[signal[0]], [0.0]], dtype=np.float64)  # [position, velocity]
        P = np.eye(2, dtype=np.float64) * 1.0  # Initial covariance

        smoothed = np.zeros(T, dtype=np.float64)

        # Identity matrix for Joseph form update
        I = np.eye(2, dtype=np.float64)

        for t in range(T):
            # Predict step
            x_pred = self.F @ x
            P_pred = self.F @ P @ self.F.T + self.Q

            # Update step
            z = np.array([[signal[t]]], dtype=np.float64)
            y = z - self.H @ x_pred  # Innovation (measurement residual)
            S = self.H @ P_pred @ self.H.T + self.R  # Innovation covariance

            # Kalman gain
            K = P_pred @ self.H.T @ np.linalg.inv(S)

            # State update
            x = x_pred + K @ y

            # Covariance update (Joseph form for numerical stability)
            IKH = I - K @ self.H
            P = IKH @ P_pred @ IKH.T + K @ self.R @ K.T

            smoothed[t] = x[0, 0]

        return smoothed

    def smooth_bidirectional(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply bidirectional (forward-backward) Kalman smoothing.

        This provides better smoothing by using future information.
        Also known as Rauch-Tung-Striebel (RTS) smoother.

        Args:
            signal: (T,) raw signal

        Returns:
            smoothed: (T,) smoothed signal
        """
        T = len(signal)
        if T == 0:
            return signal.copy()

        # Forward pass - store all states and covariances
        x_forward = np.zeros((T, 2, 1), dtype=np.float64)
        P_forward = np.zeros((T, 2, 2), dtype=np.float64)
        x_pred_all = np.zeros((T, 2, 1), dtype=np.float64)
        P_pred_all = np.zeros((T, 2, 2), dtype=np.float64)

        # Initialize
        x = np.array([[signal[0]], [0.0]], dtype=np.float64)
        P = np.eye(2, dtype=np.float64) * 1.0
        I = np.eye(2, dtype=np.float64)

        for t in range(T):
            # Predict
            x_pred = self.F @ x
            P_pred = self.F @ P @ self.F.T + self.Q

            x_pred_all[t] = x_pred
            P_pred_all[t] = P_pred

            # Update
            z = np.array([[signal[t]]], dtype=np.float64)
            y = z - self.H @ x_pred
            S = self.H @ P_pred @ self.H.T + self.R
            K = P_pred @ self.H.T @ np.linalg.inv(S)

            x = x_pred + K @ y
            IKH = I - K @ self.H
            P = IKH @ P_pred @ IKH.T + K @ self.R @ K.T

            x_forward[t] = x
            P_forward[t] = P

        # Backward pass (RTS smoother)
        x_smooth = np.zeros((T, 2, 1), dtype=np.float64)
        x_smooth[T-1] = x_forward[T-1]

        for t in range(T-2, -1, -1):
            # Smoother gain
            G = P_forward[t] @ self.F.T @ np.linalg.inv(P_pred_all[t+1])

            # Smoothed state
            x_smooth[t] = x_forward[t] + G @ (x_smooth[t+1] - x_pred_all[t+1])

        return x_smooth[:, 0, 0]


def kalman_smooth_imu(acc_data: np.ndarray,
                      gyro_data: np.ndarray,
                      config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Kalman smoothing to accelerometer and gyroscope data.

    Args:
        acc_data: (T, 3) accelerometer [ax, ay, az] in m/s^2
        gyro_data: (T, 3) gyroscope [gx, gy, gz] in rad/s
        config: dict with smoothing parameters:
            - kalman_smooth_fs: sampling frequency (default 30 Hz)
            - kalman_smooth_Q_acc: process noise for accelerometer
            - kalman_smooth_R_acc: measurement noise for accelerometer
            - kalman_smooth_Q_gyro: process noise for gyroscope
            - kalman_smooth_R_gyro: measurement noise for gyroscope
            - kalman_smooth_bidirectional: use RTS smoother (default False)

    Returns:
        acc_smoothed: (T, 3) smoothed accelerometer
        gyro_smoothed: (T, 3) smoothed gyroscope
    """
    # Get configuration with defaults
    fs = config.get('kalman_smooth_fs', 30.0)
    dt = 1.0 / fs

    # Separate Q/R for acc and gyro (acc is typically cleaner)
    acc_Q = config.get('kalman_smooth_Q_acc', 0.01)
    acc_R = config.get('kalman_smooth_R_acc', 0.05)
    gyro_Q = config.get('kalman_smooth_Q_gyro', 0.05)  # Higher for noisier gyro
    gyro_R = config.get('kalman_smooth_R_gyro', 0.1)   # Trust gyro less

    bidirectional = config.get('kalman_smooth_bidirectional', False)

    # Create smoothers
    acc_smoother = KalmanSmoother1D(dt, acc_Q, acc_R)
    gyro_smoother = KalmanSmoother1D(dt, gyro_Q, gyro_R)

    # Initialize output arrays
    acc_smoothed = np.zeros_like(acc_data, dtype=np.float64)
    gyro_smoothed = np.zeros_like(gyro_data, dtype=np.float64)

    # Smooth each channel independently
    smooth_fn = acc_smoother.smooth_bidirectional if bidirectional else acc_smoother.smooth
    gyro_smooth_fn = gyro_smoother.smooth_bidirectional if bidirectional else gyro_smoother.smooth

    for i in range(3):
        acc_smoothed[:, i] = smooth_fn(acc_data[:, i].astype(np.float64))
        gyro_smoothed[:, i] = gyro_smooth_fn(gyro_data[:, i].astype(np.float64))

    return acc_smoothed.astype(acc_data.dtype), gyro_smoothed.astype(gyro_data.dtype)


def kalman_smoothing_for_loader(trial_data: Dict, config: Dict) -> Dict:
    """
    Integration function for utils/loader.py

    Apply Kalman smoothing to trial data. This should be called
    BEFORE normalization in the data loading pipeline.

    Args:
        trial_data: dict with 'accelerometer' and 'gyroscope' keys
            - accelerometer: (T, 3) array
            - gyroscope: (T, 3) array
        config: smoothing configuration dict

    Returns:
        trial_data: modified dict with smoothed signals
    """
    if 'accelerometer' not in trial_data or 'gyroscope' not in trial_data:
        return trial_data

    acc = trial_data['accelerometer']  # (T, 3)
    gyro = trial_data['gyroscope']      # (T, 3)

    # Validate input shapes
    if acc.ndim != 2 or gyro.ndim != 2:
        return trial_data
    if acc.shape[1] != 3 or gyro.shape[1] != 3:
        return trial_data

    # Apply smoothing
    acc_smooth, gyro_smooth = kalman_smooth_imu(acc, gyro, config)

    # Update trial data
    trial_data['accelerometer'] = acc_smooth
    trial_data['gyroscope'] = gyro_smooth

    return trial_data


# Unit tests
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("KALMAN SMOOTHING UNIT TESTS")
    print("=" * 60)

    # Test 1: Basic 1D smoothing
    print("\nTest 1: Basic 1D smoothing")
    np.random.seed(42)

    # Generate noisy sine wave
    t = np.linspace(0, 2*np.pi, 200)
    clean = np.sin(t)
    noisy = clean + np.random.normal(0, 0.3, len(t))

    smoother = KalmanSmoother1D(dt=1/30.0, process_noise=0.01, measurement_noise=0.1)
    smoothed = smoother.smooth(noisy)
    smoothed_bidir = smoother.smooth_bidirectional(noisy)

    # Calculate errors
    mse_noisy = np.mean((noisy - clean)**2)
    mse_smoothed = np.mean((smoothed - clean)**2)
    mse_bidir = np.mean((smoothed_bidir - clean)**2)

    print(f"  MSE (noisy):      {mse_noisy:.4f}")
    print(f"  MSE (smoothed):   {mse_smoothed:.4f} ({100*(1-mse_smoothed/mse_noisy):.1f}% reduction)")
    print(f"  MSE (bidir):      {mse_bidir:.4f} ({100*(1-mse_bidir/mse_noisy):.1f}% reduction)")

    assert mse_smoothed < mse_noisy, "Smoothing should reduce MSE"
    assert mse_bidir < mse_noisy, "Bidirectional smoothing should reduce MSE"
    print("  PASSED")

    # Test 2: IMU smoothing function
    print("\nTest 2: IMU smoothing function")
    T = 100
    acc_data = np.random.randn(T, 3).astype(np.float32)
    gyro_data = np.random.randn(T, 3).astype(np.float32)

    config = {
        'kalman_smooth_fs': 30.0,
        'kalman_smooth_Q_acc': 0.01,
        'kalman_smooth_R_acc': 0.05,
        'kalman_smooth_Q_gyro': 0.05,
        'kalman_smooth_R_gyro': 0.1,
    }

    acc_smooth, gyro_smooth = kalman_smooth_imu(acc_data, gyro_data, config)

    assert acc_smooth.shape == acc_data.shape, "Shape should be preserved"
    assert gyro_smooth.shape == gyro_data.shape, "Shape should be preserved"
    assert acc_smooth.dtype == acc_data.dtype, "Dtype should be preserved"

    # Check smoothing reduces variance
    assert np.std(acc_smooth) < np.std(acc_data), "Smoothing should reduce variance"
    print(f"  Acc variance: {np.std(acc_data):.4f} -> {np.std(acc_smooth):.4f}")
    print(f"  Gyro variance: {np.std(gyro_data):.4f} -> {np.std(gyro_smooth):.4f}")
    print("  PASSED")

    # Test 3: Loader integration function
    print("\nTest 3: Loader integration function")
    trial_data = {
        'accelerometer': acc_data,
        'gyroscope': gyro_data,
        'label': 1
    }

    result = kalman_smoothing_for_loader(trial_data, config)

    assert 'accelerometer' in result
    assert 'gyroscope' in result
    assert 'label' in result
    assert result['label'] == 1, "Other fields should be preserved"
    print("  PASSED")

    # Test 4: Edge cases
    print("\nTest 4: Edge cases")

    # Empty array
    empty_acc = np.zeros((0, 3))
    empty_gyro = np.zeros((0, 3))
    empty_trial = {'accelerometer': empty_acc, 'gyroscope': empty_gyro}
    result = kalman_smoothing_for_loader(empty_trial, config)
    assert result['accelerometer'].shape == (0, 3)
    print("  Empty array: PASSED")

    # Single sample
    single_acc = np.random.randn(1, 3)
    single_gyro = np.random.randn(1, 3)
    single_trial = {'accelerometer': single_acc, 'gyroscope': single_gyro}
    result = kalman_smoothing_for_loader(single_trial, config)
    assert result['accelerometer'].shape == (1, 3)
    print("  Single sample: PASSED")

    # Missing modality
    missing_trial = {'accelerometer': acc_data}
    result = kalman_smoothing_for_loader(missing_trial, config)
    assert 'gyroscope' not in result
    print("  Missing modality: PASSED")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)

    # Save visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(t, noisy, 'b-', alpha=0.5, label='Noisy')
    axes[0].plot(t, clean, 'g-', linewidth=2, label='Clean')
    axes[0].plot(t, smoothed, 'r-', linewidth=2, label='Kalman Smoothed')
    axes[0].set_title('Forward-only Kalman Smoothing')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, noisy, 'b-', alpha=0.5, label='Noisy')
    axes[1].plot(t, clean, 'g-', linewidth=2, label='Clean')
    axes[1].plot(t, smoothed_bidir, 'r-', linewidth=2, label='Bidirectional Smoothed')
    axes[1].set_title('Bidirectional (RTS) Kalman Smoothing')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('kalman_smoothing_test.png', dpi=150)
    print(f"\nVisualization saved to: kalman_smoothing_test.png")
