# Dual-Stream Kalman Transformer for Wearable Fall Detection

A state-of-the-art fall detection system achieving **91.10% F1-score** using smartwatch IMU sensors. This repository implements a novel dual-stream transformer architecture with Kalman-fused orientation features, evaluated on the SmartFallMM dataset using rigorous 21-fold Leave-One-Subject-Out Cross-Validation.

## Overview

Fall detection from wearable sensors presents unique challenges: raw accelerometer data conflates gravitational and dynamic acceleration components, while gyroscope measurements suffer from integration drift. Our approach addresses these limitations through sensor fusion and architectural innovations:

1. **Kalman Filter Fusion**: Fuses accelerometer (gravity reference) and gyroscope (angular velocity) into stable orientation estimates
2. **Dual-Stream Processing**: Separate pathways for acceleration dynamics and orientation kinematics
3. **Attention Mechanisms**: Squeeze-Excitation for channel importance, Temporal Attention Pooling for event localization

## Quick Start

### 1. Dataset Setup

```bash
git clone https://github.com/txst-cs-smartfall/SmartFallMM-Dataset.git
ln -s SmartFallMM-Dataset/Young\ Data data/young
ln -s SmartFallMM-Dataset/Old\ Data data/old
```

### 2. Environment Setup

```bash
pip install -r requirements.txt
```

### 3. Reproduce Best Results

```bash
./scripts/best_exps/run_kalman_transformer.sh
```

This script executes full 21-fold LOSO-CV training with the optimal configuration. Expected runtime: 4-8 hours depending on hardware.

---

## Preprocessing Pipeline

### Kalman Filter Sensor Fusion

Raw 6-axis IMU data (accelerometer + gyroscope) is transformed into a 7-channel representation through a Linear Kalman Filter:

```
Input:  [ax, ay, az, gx, gy, gz]  (6 channels)
Output: [SMV, ax, ay, az, roll, pitch, yaw]  (7 channels)
```

**State Vector (6D):**
```
x = [roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]
```

**Observation Model:**
- Roll and pitch derived from accelerometer gravity reference: `roll = atan2(ay, az)`, `pitch = atan2(-ax, sqrt(ay² + az²))`
- Angular rates directly from gyroscope
- Yaw estimated via gyroscope integration (no magnetometer)

**Filter Parameters (Optimized):**
| Parameter | Value | Description |
|-----------|-------|-------------|
| Q_orientation | 0.005 | Process noise for angles (rad²) |
| Q_rate | 0.01 | Process noise for angular rates ((rad/s)²) |
| R_acc | 0.05 | Measurement noise for accelerometer (rad²) |
| R_gyro | 0.1 | Measurement noise for gyroscope ((rad/s)²) |

**Signal Magnitude Vector (SMV):**
```
SMV = sqrt(ax² + ay² + az²)
```
Captures total acceleration magnitude independent of device orientation. Falls produce characteristic SMV spikes (>20 m/s²) during impact.

### Channel-Aware Normalization

Critical insight: orientation angles (roll, pitch, yaw) are bounded in [-π, π] with physical meaning. Standard normalization destroys this structure.

**Solution:** Normalize only accelerometer channels (0-3), preserve orientation in radians (4-6).

```python
# Channels 0-3: StandardScaler normalization
features[:, :4] = (features[:, :4] - mean) / std

# Channels 4-6: Keep raw radians
features[:, 4:7] = orientation  # Unchanged
```

---

## Model Architecture

### Dual-Stream Kalman Transformer

```
Input: (B, 128, 7)
       │
       ├─── Acceleration Stream ────┐
       │    [SMV, ax, ay, az]       │
       │    Conv1D(4→32, k=8)       │
       │    BatchNorm → SiLU        │
       │                            │
       └─── Orientation Stream ─────┤
            [roll, pitch, yaw]      │
            Conv1D(3→32, k=8)       │
            BatchNorm → SiLU        │
                                    │
            ┌───────────────────────┘
            │ Concatenate → LayerNorm
            ▼
       TransformerEncoder
       (2 layers, 4 heads, dim=64)
            │
       Squeeze-Excitation
       (channel attention)
            │
       Temporal Attention Pooling
       (learned sequence aggregation)
            │
       Dropout(0.5) → Linear(64→1)
            │
       Output: fall probability
```

**Key Components:**

| Component | Purpose |
|-----------|---------|
| Dual-Stream Conv | Modality-specific feature extraction with different dropout rates |
| SE Block | Learns channel importance weights via global pooling + MLP |
| TAP | Attention-weighted temporal pooling to focus on impact events |
| Pre-norm Transformer | Stable training with LayerNorm before attention |

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-3 |
| Weight Decay | 5e-4 |
| Batch Size | 64 |
| Epochs | 80 |
| Loss | Focal Loss (α=0.75, γ=2.0) |
| Early Stopping | Patience=15 |

**Class-Aware Sampling:**
- Fall windows: stride=16 (more overlap for rare events)
- ADL windows: stride=64 (standard coverage)

---

## Results

### Model Comparison (21-fold LOSO-CV)

| Model | Test F1 | Architecture |
|-------|---------|--------------|
| **Dual-Stream Kalman Transformer** | **91.10%** | SE + TAP, dual-stream |
| Kalman Gated Hierarchical Fusion | 90.44% | Multi-scale gating |
| CNN-Mamba | 89.11% | CNN + Mamba blocks |
| Dual-Stream LSTM | 88.84% | Bidirectional LSTM |

### Best Fold Performance

| Fold | Test F1 | Test Accuracy |
|------|---------|---------------|
| S50 | 98.41% | 97.78% |
| S58 | 98.38% | 97.35% |
| S55 | 97.90% | 96.45% |

---

## Reproduction Scripts

### Train Best Model
```bash
./scripts/best_exps/run_kalman_transformer.sh
```

### Train Alternative Architectures
```bash
# Kalman Gated Hierarchical Fusion
python main.py --config config/smartfallmm/kalman_gated_hierarchical.yaml

# CNN-Mamba
python main.py --config config/smartfallmm/cnn_mamba_kalman.yaml

# Dual-Stream LSTM
python main.py --config config/smartfallmm/dual_stream_lstm.yaml
```

### Inference with Pretrained Weights
```bash
python inference.py --test
```

---

## Project Structure

```
├── scripts/best_exps/
│   └── run_kalman_transformer.sh    # Reproduction script
├── config/smartfallmm/
│   ├── reproduce_91.yaml            # Best model config
│   ├── kalman_gated_hierarchical.yaml
│   ├── cnn_mamba_kalman.yaml
│   └── dual_stream_lstm.yaml
├── Models/
│   ├── kalman_transformer_variants.py  # KalmanBalancedFlexible
│   ├── kalman_gated_hierarchical.py    # KGHF
│   ├── cnn_mamba.py
│   └── dual_stream_lstm.py
├── utils/kalman/
│   ├── filters.py                   # Kalman filter implementation
│   ├── preprocessing.py             # Feature assembly
│   └── quaternion.py                # Orientation utilities
├── Feeder/
│   └── Make_Dataset.py              # Data loading
├── weights/
│   └── best_model.pth               # Pretrained weights (S50, 98.41% F1)
├── inference.py                     # Standalone inference
├── main.py                          # Training entry point
└── requirements.txt
```

---

## Dataset

**SmartFallMM** is a multimodal fall detection dataset collected at Texas State University:

| Group | Subjects | Ages | Activities |
|-------|----------|------|------------|
| Young | 30 | 18-35 | 5 fall types + 9 ADLs |
| Old | 21 | 65+ | 9 ADLs only |

**Sensors:** Smartwatch (accelerometer + gyroscope at ~30Hz)

**Evaluation Protocol:** 21-fold Leave-One-Subject-Out Cross-Validation on young subjects ensures no data leakage between training and testing.

Dataset: https://github.com/txst-cs-smartfall/SmartFallMM-Dataset

---

## Citation

```bibtex
[Citation pending publication]
```

## License

Copyright © SmartFall Research Group, Texas State University. All rights reserved.

For dataset access inquiries: Dr. Anne Ngu (angu@txstate.edu)
