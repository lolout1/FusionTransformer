"""Pytest configuration and fixtures."""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_imu_data():
    """Generate sample IMU data for testing."""
    np.random.seed(42)
    n_samples = 300  # 10 seconds at 30Hz

    # Simulate stationary IMU (gravity on z-axis)
    acc = np.zeros((n_samples, 3))
    acc[:, 2] = 9.81  # gravity
    acc += np.random.randn(n_samples, 3) * 0.1  # noise

    gyro = np.random.randn(n_samples, 3) * 0.01  # small angular velocity noise

    return {'acc': acc, 'gyro': gyro, 'dt': 1/30.0}


@pytest.fixture
def fall_imu_data():
    """Generate IMU data simulating a fall event."""
    np.random.seed(42)
    n_samples = 300

    acc = np.zeros((n_samples, 3))
    gyro = np.zeros((n_samples, 3))

    # Pre-fall: stationary
    acc[:100, 2] = 9.81

    # Fall impact (samples 100-150)
    acc[100:120, :] = [0, 0, 25]  # high acceleration spike
    acc[120:150, :] = [5, 3, 2]   # tumbling

    # Post-fall: lying down
    acc[150:, :] = [9.81, 0, 0]  # gravity on x-axis (lying down)

    # Gyro during fall
    gyro[100:150, :] = np.random.randn(50, 3) * 2  # rapid rotation

    # Add noise
    acc += np.random.randn(n_samples, 3) * 0.1
    gyro += np.random.randn(n_samples, 3) * 0.01

    return {'acc': acc, 'gyro': gyro, 'dt': 1/30.0}


@pytest.fixture
def sample_windows():
    """Generate sample windows for testing."""
    np.random.seed(42)
    n_windows = 10
    window_size = 128
    n_channels = 7

    windows = np.random.randn(n_windows, window_size, n_channels).astype(np.float32)
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 1])  # 3 falls, 7 ADLs

    return {'windows': windows, 'labels': labels}
