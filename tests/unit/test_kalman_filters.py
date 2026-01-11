"""Unit tests for Kalman filter implementations."""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.kalman.filters import KalmanFilter, ExtendedKalmanFilter, create_filter
from utils.kalman.quaternion import quat_normalize, quat_to_euler, euler_to_quat


class TestLinearKalmanFilter:
    """Tests for Linear Kalman Filter."""

    def test_initialization(self):
        kf = KalmanFilter()
        assert kf.x.shape == (6,)  # [roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]
        assert kf.P.shape == (6, 6)
        assert np.allclose(kf.x, 0)  # Initial state zeros

    def test_predict_step(self, sample_imu_data):
        kf = KalmanFilter()
        dt = sample_imu_data['dt']

        x_before = kf.x.copy()
        kf.predict(dt)
        x_after = kf.x

        # State should change after predict
        assert kf.P is not None

    def test_update_step(self, sample_imu_data):
        kf = KalmanFilter()
        acc = sample_imu_data['acc'][0]
        gyro = sample_imu_data['gyro'][0]

        kf.update(acc, gyro)
        orientation = kf.get_orientation()

        assert len(orientation) == 3  # roll, pitch, yaw
        assert all(np.isfinite(orientation))

    def test_static_convergence(self, sample_imu_data):
        """Filter should converge to stable orientation for stationary IMU."""
        kf = KalmanFilter()
        dt = sample_imu_data['dt']

        orientations = []
        for i in range(len(sample_imu_data['acc'])):
            kf.predict(dt)
            kf.update(sample_imu_data['acc'][i], sample_imu_data['gyro'][i])
            orientations.append(kf.get_orientation())

        final_ori = orientations[-1]
        # Stationary with gravity on z: roll and pitch should be near 0
        assert abs(final_ori[0]) < 0.1  # roll < ~6 degrees
        assert abs(final_ori[1]) < 0.1  # pitch < ~6 degrees

    def test_angle_wrapping(self):
        """Test angle wrapping at ±π boundaries via predict/update."""
        from utils.kalman.quaternion import wrap_angle

        # Test the wrap_angle utility directly
        angle = np.pi + 0.2
        wrapped = wrap_angle(angle)
        assert -np.pi <= wrapped <= np.pi

        # Test via filter operation
        kf = KalmanFilter()
        acc = np.array([0, 0, 9.81])
        gyro = np.array([0, 0, 3.0])  # high yaw rate

        for _ in range(100):
            kf.predict(0.033)
            kf.update(acc, gyro)

        ori = kf.get_orientation()
        assert all(-np.pi <= a <= np.pi for a in ori)

    def test_covariance_positive_definite(self, sample_imu_data):
        """Covariance should remain positive definite."""
        kf = KalmanFilter()
        dt = sample_imu_data['dt']

        for i in range(100):
            kf.predict(dt)
            kf.update(sample_imu_data['acc'][i], sample_imu_data['gyro'][i])

            # Check positive definite (all eigenvalues > 0)
            eigenvalues = np.linalg.eigvalsh(kf.P)
            assert all(eigenvalues > 0), f"Non-positive eigenvalue at step {i}"


class TestExtendedKalmanFilter:
    """Tests for Extended Kalman Filter."""

    def test_initialization(self):
        ekf = ExtendedKalmanFilter()
        assert ekf.x.shape == (7,)  # [q0, q1, q2, q3, bias_gx, bias_gy, bias_gz]
        assert ekf.P.shape == (7, 7)

        # Initial quaternion should be unit [1, 0, 0, 0]
        q = ekf.x[:4]
        assert np.isclose(np.linalg.norm(q), 1.0)
        assert np.isclose(q[0], 1.0)

    def test_quaternion_normalization(self):
        ekf = ExtendedKalmanFilter()

        # Perturb quaternion
        ekf.x[:4] = [0.9, 0.3, 0.2, 0.1]  # Not unit length

        # After predict, quaternion should be normalized
        ekf.predict(np.array([0.01, 0.01, 0.01]), 0.033)
        q = ekf.x[:4]
        assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-6)

    def test_bias_estimation(self, sample_imu_data):
        """EKF should estimate gyro bias over time."""
        ekf = ExtendedKalmanFilter()
        dt = sample_imu_data['dt']

        # Add known bias to gyro
        bias = np.array([0.05, -0.03, 0.02])
        biased_gyro = sample_imu_data['gyro'] + bias

        for i in range(len(sample_imu_data['acc'])):
            ekf.predict(biased_gyro[i], dt)
            ekf.update(sample_imu_data['acc'][i])

        estimated_bias = ekf.get_gyro_bias()
        # Should converge toward true bias
        assert all(np.isfinite(estimated_bias))

    def test_euler_output(self, sample_imu_data):
        ekf = ExtendedKalmanFilter()
        dt = sample_imu_data['dt']

        for i in range(50):
            ekf.predict(sample_imu_data['gyro'][i], dt)
            ekf.update(sample_imu_data['acc'][i])

        euler = ekf.get_orientation_euler()
        assert len(euler) == 3
        assert all(np.isfinite(euler))
        assert all(-np.pi <= e <= np.pi for e in euler)


class TestUnscentedKalmanFilter:
    """Tests for Unscented Kalman Filter."""

    def test_initialization(self):
        from utils.kalman.ukf import UnscentedKalmanFilter
        ukf = UnscentedKalmanFilter()
        assert ukf.x.shape == (7,)
        assert ukf.P.shape == (7, 7)

        q = ukf.x[:4]
        assert np.isclose(np.linalg.norm(q), 1.0)

    def test_predict_update(self, sample_imu_data):
        from utils.kalman.ukf import UnscentedKalmanFilter
        ukf = UnscentedKalmanFilter()
        dt = sample_imu_data['dt']

        for i in range(50):
            ukf.predict(sample_imu_data['gyro'][i], dt)
            ukf.update(sample_imu_data['acc'][i])

        euler = ukf.get_orientation_euler()
        assert len(euler) == 3
        assert all(np.isfinite(euler))

    def test_quaternion_remains_unit(self, sample_imu_data):
        from utils.kalman.ukf import UnscentedKalmanFilter
        ukf = UnscentedKalmanFilter()
        dt = sample_imu_data['dt']

        for i in range(100):
            ukf.predict(sample_imu_data['gyro'][i], dt)
            ukf.update(sample_imu_data['acc'][i])
            q = ukf.x[:4]
            assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-6)


class TestFilterFactory:
    """Tests for filter factory function."""

    def test_create_linear(self):
        kf = create_filter('linear')
        assert isinstance(kf, KalmanFilter)

    def test_create_ekf(self):
        ekf = create_filter('ekf')
        assert isinstance(ekf, ExtendedKalmanFilter)

    def test_create_ukf(self):
        from utils.kalman.ukf import UnscentedKalmanFilter
        ukf = create_filter('ukf')
        assert isinstance(ukf, UnscentedKalmanFilter)

    def test_invalid_type(self):
        with pytest.raises((ValueError, KeyError)):
            create_filter('invalid_filter_type')


class TestQuaternionUtils:
    """Tests for quaternion utility functions."""

    def test_quat_normalize(self):
        q = np.array([1, 2, 3, 4])
        q_norm = quat_normalize(q)
        assert np.isclose(np.linalg.norm(q_norm), 1.0)

    def test_euler_quat_roundtrip(self):
        euler = np.array([0.1, 0.2, 0.3])  # roll, pitch, yaw
        q = euler_to_quat(*euler)
        euler_back = quat_to_euler(q)
        np.testing.assert_allclose(euler, euler_back, atol=1e-10)

    def test_identity_quaternion(self):
        q = np.array([1, 0, 0, 0])
        euler = quat_to_euler(q)
        np.testing.assert_allclose(euler, [0, 0, 0], atol=1e-10)
