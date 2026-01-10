"""
Fall Detection Inference Script

Usage:
    python inference.py --weights weights/best_model.pth --input sample.csv
    python inference.py --test  # Run with synthetic data
"""

import argparse
import numpy as np
import torch

from Models.kalman_transformer_variants import KalmanBalancedFlexible


class LinearKalmanFilter:
    """Linear Kalman Filter for IMU orientation estimation."""

    def __init__(self, Q_orientation=0.005, Q_rate=0.01, R_acc=0.05, R_gyro=0.1):
        self.x = np.zeros(6)  # [roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]
        self.P = np.eye(6) * 0.1
        self.Q = np.diag([Q_orientation, Q_orientation, Q_orientation,
                          Q_rate, Q_rate, Q_rate])
        self.R = np.diag([R_acc, R_acc, R_gyro, R_gyro, R_gyro])
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ])

    def predict(self, dt: float):
        F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ])
        self.x = F @ self.x
        self.x[:3] = np.arctan2(np.sin(self.x[:3]), np.cos(self.x[:3]))
        self.P = F @ self.P @ F.T + self.Q
        self.P = 0.5 * (self.P + self.P.T)

    def update(self, acc: np.ndarray, gyro: np.ndarray):
        ax, ay, az = acc
        roll_acc = np.arctan2(ay, az)
        pitch_acc = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
        z = np.array([roll_acc, pitch_acc, gyro[0], gyro[1], gyro[2]])
        y = z - self.H @ self.x
        y[:2] = np.arctan2(np.sin(y[:2]), np.cos(y[:2]))
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[:3] = np.arctan2(np.sin(self.x[:3]), np.cos(self.x[:3]))
        I_KH = np.eye(6) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    def get_orientation(self) -> np.ndarray:
        return self.x[:3].copy()


def preprocess(acc: np.ndarray, gyro: np.ndarray, fs: float = 30.0) -> np.ndarray:
    """
    Preprocess raw IMU data to 7-channel Kalman features.

    Args:
        acc: (T, 3) accelerometer in m/s²
        gyro: (T, 3) gyroscope in rad/s
        fs: Sampling frequency

    Returns:
        features: (T, 7) [SMV, ax, ay, az, roll, pitch, yaw]
    """
    T = len(acc)
    dt = 1.0 / fs

    # Compute SMV
    smv = np.sqrt(np.sum(acc**2, axis=1, keepdims=True))

    # Kalman filter for orientation
    kf = LinearKalmanFilter()
    orientations = np.zeros((T, 3))
    for t in range(T):
        kf.predict(dt)
        kf.update(acc[t], gyro[t])
        orientations[t] = kf.get_orientation()

    # Assemble: [SMV, ax, ay, az, roll, pitch, yaw]
    features = np.hstack([smv, acc, orientations])

    # Normalize acc channels only (0-3)
    acc_channels = features[:, :4]
    mean = acc_channels.mean(axis=0)
    std = acc_channels.std(axis=0) + 1e-8
    features[:, :4] = (acc_channels - mean) / std

    return features


def load_model(weights_path: str, device: str = 'cpu') -> KalmanBalancedFlexible:
    """Load model with trained weights."""
    model = KalmanBalancedFlexible(
        imu_frames=128,
        imu_channels=7,
        num_heads=4,
        num_layers=2,
        embed_dim=64,
        dropout=0.5,
        activation='relu',
        norm_first=True,
        se_reduction=4,
        acc_ratio=0.5,
        use_se=True,
        use_tap=True,
        use_pos_encoding=False
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def predict(model, features: np.ndarray, device: str = 'cpu') -> dict:
    """
    Run inference on preprocessed features.

    Args:
        model: Loaded model
        features: (128, 7) preprocessed features
        device: 'cpu' or 'cuda'

    Returns:
        {'probability': float, 'is_fall': bool}
    """
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logit, _ = model(x)
        prob = torch.sigmoid(logit).item()
    return {'probability': prob, 'is_fall': prob > 0.5}


def test_with_synthetic():
    """Test inference with synthetic data."""
    print("Testing with synthetic data...")

    # Simulate 128 samples at 30Hz (~4.3 seconds)
    T = 128
    np.random.seed(42)

    # Normal ADL: gravity + small noise
    acc_adl = np.zeros((T, 3))
    acc_adl[:, 2] = 9.8  # gravity on z-axis
    acc_adl += np.random.randn(T, 3) * 0.5
    gyro_adl = np.random.randn(T, 3) * 0.1

    # Fall: spike in acceleration
    acc_fall = acc_adl.copy()
    acc_fall[60:70, :] += np.random.randn(10, 3) * 15  # impact
    gyro_fall = gyro_adl.copy()
    gyro_fall[55:75, :] += np.random.randn(20, 3) * 2  # rotation

    # Load model
    model = load_model('weights/best_model.pth')

    # Test ADL
    features_adl = preprocess(acc_adl, gyro_adl)
    result_adl = predict(model, features_adl)
    print(f"ADL sample:  prob={result_adl['probability']:.3f}, is_fall={result_adl['is_fall']}")

    # Test Fall
    features_fall = preprocess(acc_fall, gyro_fall)
    result_fall = predict(model, features_fall)
    print(f"Fall sample: prob={result_fall['probability']:.3f}, is_fall={result_fall['is_fall']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fall Detection Inference')
    parser.add_argument('--weights', default='weights/best_model.pth', help='Model weights path')
    parser.add_argument('--input', help='CSV file with acc/gyro data')
    parser.add_argument('--test', action='store_true', help='Run test with synthetic data')
    args = parser.parse_args()

    if args.test:
        test_with_synthetic()
    elif args.input:
        print(f"Loading data from {args.input}")
        # CSV format: ax,ay,az,gx,gy,gz (no header)
        data = np.loadtxt(args.input, delimiter=',')
        acc, gyro = data[:, :3], data[:, 3:6]

        if len(acc) < 128:
            print(f"Error: Need at least 128 samples, got {len(acc)}")
            exit(1)

        # Use first 128 samples
        acc, gyro = acc[:128], gyro[:128]

        model = load_model(args.weights)
        features = preprocess(acc, gyro)
        result = predict(model, features)

        print(f"Probability: {result['probability']:.4f}")
        print(f"Prediction:  {'FALL' if result['is_fall'] else 'ADL'}")
    else:
        parser.print_help()
