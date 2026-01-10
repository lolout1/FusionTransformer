# Dual-Stream Kalman Transformer for Fall Detection

Fall detection using wearable IMU sensors with Kalman-fused orientation features.

## Quick Start

### 1. Clone Dataset

```bash
git clone https://github.com/txst-cs-smartfall/SmartFallMM-Dataset.git
ln -s SmartFallMM-Dataset/Young\ Data data/young
ln -s SmartFallMM-Dataset/Old\ Data data/old
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Inference (Pretrained)

```bash
python inference.py --test
```

---

## Reproduce Best Results

All models evaluated using **21-fold Leave-One-Subject-Out Cross-Validation** on young subjects.

### Dual-Stream Kalman Transformer (91.10% F1)

```bash
python main.py --config config/smartfallmm/reproduce_91.yaml --work-dir results/kalman_transformer
```

| Metric | Value |
|--------|-------|
| Test F1 | 91.10% |
| Best Fold | S50 (98.41%) |
| Architecture | Dual-stream SE + TAP |

### Kalman Gated Hierarchical Fusion (90.44% F1)

```bash
python main.py --config config/smartfallmm/kalman_gated_hierarchical.yaml --work-dir results/kghf
```

| Metric | Value |
|--------|-------|
| Test F1 | 90.44% |
| Architecture | Multi-scale gated fusion |

### CNN-Mamba (89.11% F1)

```bash
python main.py --config config/smartfallmm/cnn_mamba_kalman.yaml --work-dir results/cnn_mamba
```

| Metric | Value |
|--------|-------|
| Test F1 | 89.11% |
| Architecture | CNN + Mamba blocks |

### Dual-Stream LSTM (88.84% F1)

```bash
python main.py --config config/smartfallmm/dual_stream_lstm.yaml --work-dir results/lstm
```

| Metric | Value |
|--------|-------|
| Test F1 | 88.84% |
| Architecture | Bi-LSTM dual-stream |

---

## Results Summary

| Model | Test F1 | Config |
|-------|---------|--------|
| **Dual-Stream Kalman Transformer** | **91.10%** | `reproduce_91.yaml` |
| Kalman Gated Hierarchical Fusion | 90.44% | `kalman_gated_hierarchical.yaml` |
| CNN-Mamba | 89.11% | `cnn_mamba_kalman.yaml` |
| Dual-Stream LSTM | 88.84% | `dual_stream_lstm.yaml` |

---

## Pretrained Weights

Best model weights included: `weights/best_model.pth`
- Fold: S50
- Test F1: 98.41%
- Model: KalmanBalancedFlexible

```python
from Models.kalman_transformer_variants import KalmanBalancedFlexible
import torch

model = KalmanBalancedFlexible()
model.load_state_dict(torch.load('weights/best_model.pth'))
```

---

## Key Features

### Kalman Fusion Preprocessing
- Linear Kalman filter fuses accelerometer + gyroscope → orientation (roll, pitch, yaw)
- 7-channel output: [SMV, ax, ay, az, roll, pitch, yaw]
- Channel-aware normalization: StandardScaler on accelerometer only

### Dual-Stream Architecture
- Separate streams for acceleration (4ch) and orientation (3ch)
- Squeeze-Excitation (SE) for channel attention
- Temporal Attention Pooling (TAP) for sequence aggregation

### Training
- Focal Loss (alpha=0.75, gamma=2.0)
- AdamW optimizer
- Early stopping with patience=15

---

## Dataset

The SmartFallMM dataset:
- **Young Group**: 30 subjects (ages 18-35) with fall + ADL activities
- **Old Group**: 21 subjects (ages 65+) with ADL activities only
- **Sensors**: Smartwatch accelerometer + gyroscope
- **Activities**: 5 fall types + 9 ADLs

Dataset: https://github.com/txst-cs-smartfall/SmartFallMM-Dataset

---

## Project Structure

```
├── config/smartfallmm/
│   ├── reproduce_91.yaml              # Best transformer config
│   ├── kalman_gated_hierarchical.yaml # KGHF config
│   ├── cnn_mamba_kalman.yaml          # CNN-Mamba config
│   └── dual_stream_lstm.yaml          # LSTM config
├── Models/
│   ├── kalman_transformer_variants.py # KalmanBalancedFlexible
│   ├── kalman_gated_hierarchical.py   # KGHF
│   ├── cnn_mamba.py                   # CNN-Mamba
│   └── dual_stream_lstm.py            # LSTM
├── utils/kalman/                      # Kalman filter implementation
├── weights/best_model.pth             # Pretrained weights
├── inference.py                       # Inference script
└── main.py                            # Training script
```

---

## Citation

```
[Paper citation pending]
```

## License

Copyright SmartFall Group, Texas State University. All rights reserved.
