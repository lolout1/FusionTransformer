"""Unit tests for quaternion operations."""

import pytest
import numpy as np
from utils.kalman.quaternion import (
    quat_multiply, quat_normalize, quat_conjugate,
    quat_to_euler, euler_to_quat, quat_from_gyro,
    acc_to_euler, wrap_angle, quat_rotate_vector
)


class TestQuaternionBasics:
    def test_identity_quaternion(self):
        q = np.array([1, 0, 0, 0])
        assert np.allclose(quat_normalize(q), q)

    def test_normalize(self):
        q = np.array([1, 1, 1, 1])
        q_norm = quat_normalize(q)
        assert np.isclose(np.linalg.norm(q_norm), 1.0)

    def test_conjugate(self):
        q = np.array([1, 2, 3, 4])
        q_conj = quat_conjugate(q)
        assert np.allclose(q_conj, [1, -2, -3, -4])

    def test_multiply_identity(self):
        q = np.array([0.5, 0.5, 0.5, 0.5])
        identity = np.array([1, 0, 0, 0])
        result = quat_multiply(q, identity)
        assert np.allclose(result, q)

    def test_multiply_inverse(self):
        q = quat_normalize([1, 2, 3, 4])
        q_conj = quat_conjugate(q)
        result = quat_multiply(q, q_conj)
        assert np.allclose(result, [1, 0, 0, 0], atol=1e-10)


class TestEulerConversion:
    def test_identity_euler(self):
        euler = quat_to_euler([1, 0, 0, 0])
        assert np.allclose(euler, [0, 0, 0])

    def test_round_trip(self):
        roll, pitch, yaw = 0.1, 0.2, 0.3
        q = euler_to_quat(roll, pitch, yaw)
        euler = quat_to_euler(q)
        assert np.allclose(euler, [roll, pitch, yaw], atol=1e-10)

    def test_90_degree_rotations(self):
        q = euler_to_quat(np.pi/2, 0, 0)
        euler = quat_to_euler(q)
        assert np.isclose(euler[0], np.pi/2, atol=1e-10)

    def test_gimbal_lock_handling(self):
        q = euler_to_quat(0, np.pi/2, 0)
        euler = quat_to_euler(q)
        assert np.isclose(euler[1], np.pi/2, atol=1e-5)


class TestGyroIntegration:
    def test_zero_gyro(self):
        gyro = [0, 0, 0]
        q = quat_from_gyro(gyro, 0.01)
        assert np.allclose(q, [1, 0, 0, 0])

    def test_small_rotation(self):
        gyro = [0.1, 0, 0]
        dt = 0.01
        q = quat_from_gyro(gyro, dt)
        expected_angle = 0.1 * dt
        assert np.isclose(np.linalg.norm(q), 1.0)
        assert q[0] > 0.99


class TestAccToEuler:
    def test_gravity_aligned(self):
        acc = [0, 0, 9.81]
        euler = acc_to_euler(acc)
        assert np.allclose(euler[:2], [0, 0], atol=1e-10)

    def test_tilted(self):
        acc = [0, 9.81, 0]
        euler = acc_to_euler(acc)
        assert np.isclose(euler[0], np.pi/2, atol=1e-5)


class TestWrapAngle:
    def test_no_wrap_needed(self):
        assert np.isclose(wrap_angle(0.5), 0.5)

    def test_wrap_positive(self):
        assert np.isclose(wrap_angle(np.pi + 0.1), -np.pi + 0.1, atol=1e-10)

    def test_wrap_negative(self):
        assert np.isclose(wrap_angle(-np.pi - 0.1), np.pi - 0.1, atol=1e-10)


class TestQuatRotateVector:
    def test_identity_rotation(self):
        q = [1, 0, 0, 0]
        v = [1, 2, 3]
        result = quat_rotate_vector(q, v)
        assert np.allclose(result, v)

    def test_90_degree_rotation(self):
        q = euler_to_quat(0, 0, np.pi/2)
        v = [1, 0, 0]
        result = quat_rotate_vector(q, v)
        assert np.allclose(result, [0, 1, 0], atol=1e-10)
