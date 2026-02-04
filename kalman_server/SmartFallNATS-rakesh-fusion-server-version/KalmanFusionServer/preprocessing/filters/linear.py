import numpy as np
from typing import Optional

from ..base import KalmanFilterProtocol, KalmanState
from ..registry import register_kalman_filter


@register_kalman_filter("linear")
class LinearKalmanFilter(KalmanFilterProtocol):
    """Linear Kalman Filter for IMU orientation estimation.

    State vector: [roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]
    Measurements: [roll_acc, pitch_acc, gyro_x, gyro_y, gyro_z]
    """

    def __init__(
        self,
        Q_orientation: float = 0.005,
        Q_rate: float = 0.01,
        R_acc: float = 0.05,
        R_gyro: float = 0.1,
    ):
        self.Q_orientation = Q_orientation
        self.Q_rate = Q_rate
        self.R_acc = R_acc
        self.R_gyro = R_gyro

        self.x = np.zeros(6)
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

    def predict(self, dt: float) -> None:
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

    def update(self, acc: np.ndarray, gyro: np.ndarray) -> None:
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

    def get_state(self) -> KalmanState:
        return KalmanState(x=self.x.copy(), P=self.P.copy())

    def set_state(self, state: KalmanState) -> None:
        self.x = state.x.copy()
        self.P = state.P.copy()

    def reset(self, acc: Optional[np.ndarray] = None) -> None:
        self.x = np.zeros(6)
        self.P = np.eye(6) * 0.1

        if acc is not None:
            ax, ay, az = acc
            self.x[0] = np.arctan2(ay, az)
            self.x[1] = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
