# Kalman Fusion for Real-Time Fall Detection

This document describes the complete preprocessing pipeline for deploying the **best fall detection model** (98.41% Test F1 on Subject 50) on an Android smartwatch.

## Best Model Summary

| Metric | Value |
|--------|-------|
| **Test F1** | 98.41% |
| **Architecture** | Dual-Stream Kalman Transformer |
| **Model File** | `Models/kalman_transformer_variants.py::KalmanBalancedFlexible` |
| **Weights** | `results/normalization_ablation_20251222/B_acc_only/test_50.pth` |
| **Config** | `config/smartfallmm/reproduce_91.yaml` |

---

## 1. Input Data Requirements

### Raw Sensor Data Format

The model expects data from a **smartwatch** with:

| Sensor | Axes | Units | Sampling Rate |
|--------|------|-------|---------------|
| Accelerometer | ax, ay, az | m/s² | ~30 Hz |
| Gyroscope | gx, gy, gz | **rad/s** | ~30 Hz |

**Critical:** Gyroscope MUST be in **radians/second**, not degrees/second.

```python
# If your watch outputs deg/s, convert:
gyro_rad = gyro_deg * (np.pi / 180.0)
```

### Window Size

| Parameter | Value |
|-----------|-------|
| Window length | **128 samples** (~4.3 seconds at 30Hz) |
| Stride (inference) | 32 samples (~1 second) for real-time |

---

## 2. Preprocessing Pipeline

The preprocessing transforms raw 6-channel IMU data into 7-channel Kalman-fused features.

### Step-by-Step Pipeline

```
Raw IMU [6ch]                    Kalman Fused [7ch]
┌─────────────────┐              ┌─────────────────────────────┐
│ ax, ay, az      │              │ SMV                         │ ← sqrt(ax²+ay²+az²)
│ gx, gy, gz      │  ──────────► │ ax, ay, az                  │ ← raw accelerometer
└─────────────────┘              │ roll, pitch, yaw            │ ← Kalman orientation
                                 └─────────────────────────────┘
```

### 2.1 Signal Magnitude Vector (SMV)

```python
def compute_smv(acc: np.ndarray) -> np.ndarray:
    """
    Compute Signal Magnitude Vector from accelerometer.

    Args:
        acc: (T, 3) accelerometer [ax, ay, az] in m/s²

    Returns:
        smv: (T,) magnitude values
    """
    return np.sqrt(acc[:, 0]**2 + acc[:, 1]**2 + acc[:, 2]**2)
```

**Purpose:** SMV captures total acceleration magnitude regardless of orientation. Falls produce characteristic SMV spikes (>20 m/s²) followed by impact patterns.

### 2.2 Kalman Filter Fusion

The Linear Kalman Filter fuses accelerometer (gravity reference) + gyroscope (angular velocity) to estimate device orientation.

#### State Vector (6D)

```
x = [roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]
     └─── angles (rad) ───┘  └─── angular velocities (rad/s) ───┘
```

#### Kalman Filter Parameters (Optimal)

```python
# These are the EXACT parameters used to train the 98.41% model
kalman_params = {
    'Q_orientation': 0.005,  # Process noise - orientation (rad²)
    'Q_rate': 0.01,          # Process noise - angular rates ((rad/s)²)
    'R_acc': 0.05,           # Measurement noise - accelerometer (rad²)
    'R_gyro': 0.1,           # Measurement noise - gyroscope ((rad/s)²)
}
```

#### Kalman Filter Algorithm

```python
import numpy as np

class LinearKalmanFilter:
    """
    Linear Kalman Filter for IMU orientation estimation.

    This is the EXACT algorithm used in training.
    """

    def __init__(self, Q_orientation=0.005, Q_rate=0.01, R_acc=0.05, R_gyro=0.1):
        # State: [roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]
        self.x = np.zeros(6)

        # State covariance
        self.P = np.eye(6) * 0.1

        # Process noise covariance
        self.Q = np.diag([Q_orientation, Q_orientation, Q_orientation,
                          Q_rate, Q_rate, Q_rate])

        # Measurement noise covariance [roll_acc, pitch_acc, gx, gy, gz]
        self.R = np.diag([R_acc, R_acc, R_gyro, R_gyro, R_gyro])

        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],  # roll from acc
            [0, 1, 0, 0, 0, 0],  # pitch from acc
            [0, 0, 0, 1, 0, 0],  # gx = roll_rate
            [0, 0, 0, 0, 1, 0],  # gy = pitch_rate
            [0, 0, 0, 0, 0, 1],  # gz = yaw_rate
        ])

    def predict(self, dt: float):
        """Prediction step using constant velocity model."""
        # State transition matrix
        F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ])

        # Predict state
        self.x = F @ self.x

        # Wrap angles to [-pi, pi]
        self.x[:3] = np.arctan2(np.sin(self.x[:3]), np.cos(self.x[:3]))

        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q
        self.P = 0.5 * (self.P + self.P.T)  # Enforce symmetry

    def update(self, acc: np.ndarray, gyro: np.ndarray):
        """Update step with accelerometer and gyroscope measurements."""
        # Extract roll/pitch from accelerometer (gravity reference)
        ax, ay, az = acc
        roll_acc = np.arctan2(ay, az)
        pitch_acc = np.arctan2(-ax, np.sqrt(ay**2 + az**2))

        # Measurement vector
        z = np.array([roll_acc, pitch_acc, gyro[0], gyro[1], gyro[2]])

        # Innovation
        y = z - self.H @ self.x
        y[:2] = np.arctan2(np.sin(y[:2]), np.cos(y[:2]))  # Wrap angles

        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ y
        self.x[:3] = np.arctan2(np.sin(self.x[:3]), np.cos(self.x[:3]))

        # Update covariance (Joseph form for stability)
        I_KH = np.eye(6) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    def get_orientation(self) -> np.ndarray:
        """Return [roll, pitch, yaw] in radians."""
        return self.x[:3].copy()
```

#### Processing a Trial

```python
def process_trial_kalman(acc_data: np.ndarray,
                         gyro_data: np.ndarray,
                         fs: float = 30.0) -> np.ndarray:
    """
    Process IMU trial through Kalman filter.

    Args:
        acc_data: (T, 3) accelerometer [ax, ay, az] in m/s²
        gyro_data: (T, 3) gyroscope [gx, gy, gz] in rad/s
        fs: Sampling frequency in Hz

    Returns:
        orientation: (T, 3) [roll, pitch, yaw] in radians
    """
    T = len(acc_data)
    dt = 1.0 / fs

    kf = LinearKalmanFilter(
        Q_orientation=0.005,
        Q_rate=0.01,
        R_acc=0.05,
        R_gyro=0.1
    )

    orientations = np.zeros((T, 3))

    for t in range(T):
        kf.predict(dt)
        kf.update(acc_data[t], gyro_data[t])
        orientations[t] = kf.get_orientation()

    return orientations
```

### 2.3 Feature Assembly

```python
def assemble_features(acc: np.ndarray,
                      gyro: np.ndarray,
                      fs: float = 30.0) -> np.ndarray:
    """
    Assemble 7-channel features for model input.

    Args:
        acc: (T, 3) accelerometer in m/s²
        gyro: (T, 3) gyroscope in rad/s
        fs: Sampling frequency

    Returns:
        features: (T, 7) [SMV, ax, ay, az, roll, pitch, yaw]
    """
    # Compute SMV
    smv = compute_smv(acc).reshape(-1, 1)

    # Compute orientation via Kalman
    orientation = process_trial_kalman(acc, gyro, fs)

    # Assemble: [SMV, ax, ay, az, roll, pitch, yaw]
    features = np.hstack([smv, acc, orientation])

    return features
```

---

## 3. Normalization

**CRITICAL:** Only normalize accelerometer channels (0-3), NOT orientation channels (4-6).

### Channel-Aware Normalization (acc_only mode)

```python
def normalize_features(features: np.ndarray,
                       scaler_params: dict = None,
                       fit: bool = False) -> tuple:
    """
    Normalize features with channel-aware scaling.

    IMPORTANT: Only normalize SMV and accelerometer (channels 0-3).
    Orientation (channels 4-6) stays in radians [-π, π].

    Args:
        features: (T, 7) input features
        scaler_params: {'mean': array, 'std': array} for channels 0-3
        fit: If True, compute and return new scaler params

    Returns:
        normalized: (T, 7) normalized features
        scaler_params: Updated scaler params (if fit=True)
    """
    normalized = features.copy()

    # Only normalize channels 0-3 (SMV, ax, ay, az)
    acc_channels = features[:, :4]

    if fit:
        mean = acc_channels.mean(axis=0)
        std = acc_channels.std(axis=0)
        std[std < 1e-8] = 1.0  # Prevent division by zero
        scaler_params = {'mean': mean, 'std': std}

    # Apply normalization to acc channels only
    normalized[:, :4] = (acc_channels - scaler_params['mean']) / scaler_params['std']

    # Channels 4-6 (roll, pitch, yaw) remain in radians
    # This preserves their physical meaning (bounded in [-π, π])

    return normalized, scaler_params
```

### Training Normalization Statistics

The model was trained with per-fold StandardScaler. For deployment, you can either:

1. **Use fold-specific stats** from training (stored in fold results)
2. **Compute running stats** during first few seconds of real-time data

---

## 4. Model Input Format

### Input Tensor Shape

```python
# Model expects: (batch_size, time_steps, channels)
input_shape = (1, 128, 7)  # Single window inference

# Channel order:
# [0] SMV - Signal Magnitude Vector
# [1] ax  - Accelerometer X
# [2] ay  - Accelerometer Y
# [3] az  - Accelerometer Z
# [4] roll  - Kalman roll angle (radians)
# [5] pitch - Kalman pitch angle (radians)
# [6] yaw   - Kalman yaw angle (radians)
```

### Dual-Stream Split

The model internally splits channels:

```python
# Accelerometer stream (channels 0-3): [SMV, ax, ay, az]
acc_input = features[:, :, :4]  # Shape: (B, 128, 4)

# Orientation stream (channels 4-6): [roll, pitch, yaw]
ori_input = features[:, :, 4:]  # Shape: (B, 128, 3)
```

---

## 5. Post-Processing (Inference)

### Model Output

```python
# Model outputs raw logit (single value)
logit = model(input_tensor)  # Shape: (1, 1)

# Convert to probability
probability = torch.sigmoid(logit).item()

# Binary classification
is_fall = probability > 0.5
```

### Confidence Thresholds

For real-time deployment, consider:

| Threshold | Use Case |
|-----------|----------|
| 0.5 | Standard binary classification |
| 0.7 | High-confidence alerts only |
| 0.3 | Sensitive detection (fewer missed falls) |

---

## 6. Complete Real-Time Pipeline (Python)

```python
import torch
import numpy as np
from collections import deque

class FallDetector:
    """Real-time fall detection using Kalman Transformer."""

    def __init__(self, model_path: str, device: str = 'cpu'):
        # Load model
        from Models.kalman_transformer_variants import KalmanBalancedFlexible

        self.model = KalmanBalancedFlexible(
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
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.device = device

        # Kalman filter
        self.kf = LinearKalmanFilter()

        # Sliding window buffer
        self.buffer = deque(maxlen=128)

        # Normalization stats (compute from first window or use training stats)
        self.scaler_params = None
        self.samples_seen = 0

    def process_sample(self, acc: np.ndarray, gyro: np.ndarray,
                       dt: float = 1/30.0) -> dict:
        """
        Process single IMU sample and detect falls.

        Args:
            acc: [ax, ay, az] in m/s²
            gyro: [gx, gy, gz] in rad/s (MUST be rad/s!)
            dt: Time step in seconds

        Returns:
            {
                'probability': float,  # Fall probability [0, 1]
                'is_fall': bool,       # Binary prediction
                'buffer_full': bool    # True when buffer has 128 samples
            }
        """
        # Update Kalman filter
        self.kf.predict(dt)
        self.kf.update(acc, gyro)
        orientation = self.kf.get_orientation()

        # Compute SMV
        smv = np.sqrt(np.sum(acc**2))

        # Assemble feature vector: [SMV, ax, ay, az, roll, pitch, yaw]
        feature = np.array([smv, acc[0], acc[1], acc[2],
                           orientation[0], orientation[1], orientation[2]])

        self.buffer.append(feature)
        self.samples_seen += 1

        # Need 128 samples for inference
        if len(self.buffer) < 128:
            return {
                'probability': 0.0,
                'is_fall': False,
                'buffer_full': False
            }

        # Convert buffer to array
        window = np.array(self.buffer)  # (128, 7)

        # Initialize normalization from first full window
        if self.scaler_params is None:
            self.scaler_params = {
                'mean': window[:, :4].mean(axis=0),
                'std': window[:, :4].std(axis=0) + 1e-8
            }

        # Normalize acc channels only (0-3)
        normalized = window.copy()
        normalized[:, :4] = (window[:, :4] - self.scaler_params['mean']) / self.scaler_params['std']

        # Prepare tensor
        input_tensor = torch.tensor(normalized, dtype=torch.float32)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)  # (1, 128, 7)

        # Inference
        with torch.no_grad():
            logit = self.model(input_tensor)
            probability = torch.sigmoid(logit).item()

        return {
            'probability': probability,
            'is_fall': probability > 0.5,
            'buffer_full': True
        }
```

---

## 7. Android (Kotlin/Java) Implementation Notes

### Sensor Registration

```kotlin
// Register accelerometer and gyroscope
sensorManager.registerListener(
    this,
    sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER),
    SensorManager.SENSOR_DELAY_GAME  // ~50Hz, downsample to 30Hz
)

sensorManager.registerListener(
    this,
    sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE),
    SensorManager.SENSOR_DELAY_GAME
)
```

### Unit Conversion

```kotlin
// Android gyroscope is already in rad/s - no conversion needed!
// Accelerometer is in m/s² - also correct

// IMPORTANT: Verify your specific watch outputs
// Some watches may differ
fun onSensorChanged(event: SensorEvent) {
    when (event.sensor.type) {
        Sensor.TYPE_ACCELEROMETER -> {
            // event.values = [ax, ay, az] in m/s²
            accBuffer.add(event.values.clone())
        }
        Sensor.TYPE_GYROSCOPE -> {
            // event.values = [gx, gy, gz] in rad/s
            gyroBuffer.add(event.values.clone())
        }
    }
}
```

### Inference with TensorFlow Lite or ONNX

1. Export PyTorch model to ONNX:
```python
dummy_input = torch.randn(1, 128, 7)
torch.onnx.export(model, dummy_input, "fall_detector.onnx")
```

2. Convert to TensorFlow Lite (optional):
```bash
pip install onnx-tf
python -m onnx_tf.tool convert -i fall_detector.onnx -o fall_detector.tflite
```

---

## 8. Key Differences: Training vs Inference

| Aspect | Training | Inference |
|--------|----------|-----------|
| Window source | Pre-segmented trials | Sliding buffer |
| Normalization | Per-fold StandardScaler | Running/fixed stats |
| Stride | 16 (falls) / 64 (ADL) | 32 (real-time) |
| Batch size | 64 | 1 |
| Kalman init | Fresh per trial | Continuous |

---

## 9. Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| All predictions = 0 | Gyro in deg/s not rad/s | Multiply by π/180 |
| Random predictions | Wrong normalization | Check acc_only mode |
| Delayed detection | Insufficient buffer | Reduce stride to 16 |
| False positives | Low threshold | Increase to 0.7 |

### Validation Checklist

- [ ] Gyroscope in rad/s (max ~5-10 rad/s during normal motion)
- [ ] Accelerometer in m/s² (should be ~9.8 at rest)
- [ ] Window size = 128 samples
- [ ] Channels in correct order: [SMV, ax, ay, az, roll, pitch, yaw]
- [ ] Only channels 0-3 normalized
- [ ] Model loaded with correct architecture params

---

## 10. References

- **Model Architecture:** `Models/kalman_transformer_variants.py:KalmanBalancedFlexible`
- **Training Config:** `config/smartfallmm/reproduce_91.yaml`
- **Best Weights:** `results/normalization_ablation_20251222/B_acc_only/test_50.pth`
- **Dataset Paper:** SmartFallMM (Texas State University)

---

**Author:** SmartFall Research Group, Texas State University
**Last Updated:** January 2026
