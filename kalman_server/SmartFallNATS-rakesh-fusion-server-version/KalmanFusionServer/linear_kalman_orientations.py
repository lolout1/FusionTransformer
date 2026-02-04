import numpy as np

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

