# KalmanTransformer

Dual-stream transformer architecture for wearable fall detection using Kalman-fused IMU data.

## Innovation

Accelerometer and gyroscope data are processed in **separate encoding streams** before transformer fusion. This dual-stream design:

1. **Kalman Filter Fusion**: Raw 6-channel IMU (acc + gyro) is fused into 7 channels: signal magnitude (SMV), 3-axis acceleration, and 3-axis orientation (roll, pitch, yaw)
2. **Asymmetric Encoding**: Accelerometer stream uses Conv1D (captures high-frequency fall transients), orientation stream uses Conv1D or Linear (smooth Kalman-filtered signals)
3. **Modality-Specific Capacity**: 65% embedding capacity for accelerometer (reliable), 35% for orientation

## Architecture

```
Raw IMU (6ch) ─► Kalman Filter ─► 7ch [smv, ax, ay, az, roll, pitch, yaw]
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                      ▼
             Accelerometer (4ch)                   Orientation (3ch)
                    │                                      │
              Conv1D (k=8)                           Conv1D/Linear
                    │                                      │
                    └──────────► Concat ◄─────────────────┘
                                    │
                             TransformerEncoder
                                    │
                           Squeeze-Excitation
                                    │
                       Temporal Attention Pooling
                                    │
                              Classifier
```

## Results

All results from Leave-One-Subject-Out (LOSO) cross-validation.

| Dataset | Model | Test F1 | Accuracy | Precision | Recall | AUC | Folds |
|---------|-------|---------|----------|-----------|--------|-----|-------|
| SmartFallMM | Kalman | 91.38% ± 6.67% | 88.44% | 89.22% | 94.14% | 94.30% | 22 |
| UP-FALL | Kalman | **95.11% ± 3.47%** | 96.36% ± 2.72% | - | - | - | 15 |
| UP-FALL | Raw | 94.77% ± 3.41% | 96.26% ± 2.46% | - | - | - | 15 |
| WEDA-FALL | Kalman | **91.53% ± 4.52%** | 88.87% ± 6.27% | - | - | - | 12 |
| WEDA-FALL | Raw | 90.43% ± 2.75% | 87.25% ± 4.10% | - | - | - | 12 |

Kalman fusion provides +0.34% F1 improvement on UP-FALL (research-grade sensors) and +1.10% F1 on WEDA-FALL (consumer-grade sensors).

---

## SmartFallMM

**Dataset**: 51 subjects (30 young + 21 old), Android smartwatch, ~32 Hz

**Best Model**: `KalmanConv1dLinear` (Conv1D for acc, Linear for orientation)

### Configuration

| Parameter | Value |
|-----------|-------|
| Window Size | 128 samples (~4s) |
| Fall Stride | 16 |
| ADL Stride | 64 |
| Embed Dim | 48 |
| Kalman Q_orientation | 0.005 |
| Kalman R_gyro | 0.1 |

### Training

```bash
python ray_train.py --config config/smartfallmm/kalman.yaml --num-gpus 3
```

---

## UP-FALL

**Dataset**: 17 subjects, research-grade IMU, 18 Hz

**Best Model**: `KalmanConv1dConv1d` (Conv1D for both streams)

### Configuration

| Parameter | Value |
|-----------|-------|
| Window Size | 160 samples (~8.9s) |
| Fall Stride | 8 |
| ADL Stride | 32 |
| Embed Dim | 64 |
| Kalman Q_orientation | 0.032 |
| Kalman R_gyro | 0.1074 |

### Training

```bash
# Kalman model (95.11% F1)
python ray_train.py --config config/upfall/kalman.yaml --num-gpus 3

# Raw baseline (94.77% F1)
python ray_train.py --config config/upfall/raw.yaml --num-gpus 3
```

---

## WEDA-FALL

**Dataset**: 14 young subjects, consumer Fitbit, 50 Hz

**Best Model**: `KalmanConv1dConv1d`

### Configuration

| Parameter | Value |
|-----------|-------|
| Window Size | 192 samples (~3.8s) |
| Fall Stride | 8 |
| ADL Stride | 32 |
| Embed Dim | 48 |
| Kalman Q_orientation | 0.0124 |
| Kalman R_gyro | 0.2822 |

### Training

```bash
# Kalman model (91.53% F1)
python ray_train.py --config config/wedafall/kalman.yaml --num-gpus 3

# Raw baseline (90.43% F1)
python ray_train.py --config config/wedafall/raw.yaml --num-gpus 3
```

---

## Repository Structure

```
FusionTransformer/
├── main.py                      # Single-fold training
├── ray_train.py                 # Distributed LOSO training
├── requirements.txt
│
├── Models/
│   ├── encoder_ablation.py      # KalmanConv1dLinear, KalmanConv1dConv1d
│   └── dual_stream_baseline.py  # DualStreamBaseline (raw comparison)
│
├── utils/
│   ├── upfall_loader.py
│   ├── wedafall_loader.py
│   └── kalman/
│       ├── filters.py           # LinearKalmanFilter
│       └── features.py
│
├── Feeder/
│   └── external_datasets.py
│
├── distributed_dataset_pipeline/
│   ├── run_kalman_vs_raw_comparison.py
│   ├── run_hyperparameter_ablation.py
│   └── ablation_analysis.py
│
└── config/
    ├── smartfallmm/kalman.yaml
    ├── upfall/kalman.yaml
    ├── upfall/raw.yaml
    ├── wedafall/kalman.yaml
    └── wedafall/raw.yaml
```

---

## Installation

```bash
pip install -r requirements.txt
```

Requires: PyTorch 2.0+, Ray, NumPy, Pandas, scikit-learn, einops

---

## Citation

```bibtex
@article{,
  title={},
  author={},
  journal={},
  year={}
}
```

---

## License

MIT License
