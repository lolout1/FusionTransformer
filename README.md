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
| SmartFallMM | Dual-Kalman | **91.38% ± 6.67%** | 88.44% | 89.22% | 94.14% | 94.30% | 22 |
| UP-FALL | Dual-Kalman | **94.43% ± 4.60%** | 95.78% | 93.44% | 96.09% | 98.87% | 15 |
| UP-FALL | Dual-Raw | 94.17% ± 4.42% | 95.90% | 94.73% | 94.02% | 98.81% | 15 |
| WEDA-FALL | Dual-Kalman | **94.66% ± 4.09%** | 93.96% | 92.11% | 97.42% | 98.65% | 12 |
| WEDA-FALL | Dual-Raw | 90.43% ± 2.63% | 87.25% | 84.91% | 97.11% | 94.53% | 12 |

### Total Benefit: Single-Stream Raw → Dual-Stream Kalman

| Dataset | Single-Raw | Dual-Kalman | Δ F1 |
|---------|------------|-------------|------|
| SmartFallMM | 88.96% | **91.38%** | **+2.42%** |
| UP-FALL | 91.06% | **94.43%** | **+3.37%** |
| WEDA-FALL | 91.92% | **94.66%** | **+2.74%** |

Dual-stream architecture + Kalman fusion provides **+2.4% to +3.4% F1** improvement over single-stream raw baselines.

---

## SmartFallMM

**Dataset**: 51 subjects (30 young + 21 old), Android smartwatch, ~30 Hz

**Best Results**: 91.53% ± 6.09% F1 | 87.88% Accuracy | 89.04% Precision | 94.78% Recall | 94.30% AUC

**Model**: `Models.encoder_ablation.KalmanConv1dLinear`

### Model Architecture

| Parameter | Value | Description |
|-----------|-------|-------------|
| embed_dim | 48 | Embedding dimension (smaller reduces overfitting) |
| num_heads | 4 | Transformer attention heads |
| num_layers | 2 | Transformer encoder layers |
| dropout | 0.5 | Dropout rate |
| acc_ratio | 0.65 | 65% capacity for accelerometer stream |
| se_reduction | 4 | Squeeze-Excitation reduction factor |
| activation | relu | Activation function |
| norm_first | True | Pre-norm transformer |

### Windowing

| Parameter | Value | Description |
|-----------|-------|-------------|
| Window Size | 128 samples | ~4.3s at 30Hz |
| Fall Stride | 16 | Aggressive overlap for fall windows |
| ADL Stride | 64 | Standard stride for ADL windows |
| Class-Aware Stride | Enabled | Different strides per class |

### Preprocessing

| Parameter | Value | Description |
|-----------|-------|-------------|
| Normalization | acc_only | Z-score normalize accelerometer only |
| Filtering | Disabled | Kalman filter handles noise |
| Motion Filtering | Disabled | - |
| Gyro Conversion | Enabled | Convert deg/s to rad/s |
| Simple Truncation | Enabled | Handle length mismatches |
| Max Truncation Diff | 50 samples | - |

### Kalman Filter

| Parameter | Value | Description |
|-----------|-------|-------------|
| Filter Type | linear | Linear Kalman filter |
| Output Format | euler | Roll, pitch, yaw angles |
| Include SMV | True | Signal Vector Magnitude prepended |
| Sampling Rate | 30 Hz | - |
| Q_orientation | 0.005 | Process noise (orientation) |
| Q_rate | 0.01 | Process noise (angular rate) |
| R_acc | 0.05 | Measurement noise (accelerometer) |
| R_gyro | 0.1 | Measurement noise (gyroscope) |

**Output**: 7 channels `[smv, ax, ay, az, roll, pitch, yaw]`

### Training

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 0.001 |
| Weight Decay | 0.001 |
| Batch Size | 64 |
| Max Epochs | 80 |
| Loss | Focal |
| Seed | 2 |
| Validation Subjects | [48, 57] |
| LOSO Folds | 22 |

```bash
python ray_train.py --config best_config/smartfallmm/kalman.yaml --num-gpus 3
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

**Best Results**: 94.43% ± 4.60% F1 | 95.78% Accuracy | 93.44% Precision | 96.09% Recall | 98.87% AUC

### Training

```bash
# Kalman model (94.43% F1)
python ray_train.py --config best_config/upfall/kalman.yaml --num-gpus 3

# Raw baseline (94.17% F1)
python ray_train.py --config best_config/upfall/raw.yaml --num-gpus 3
```

---

## WEDA-FALL

**Dataset**: 14 young subjects, consumer Fitbit, 50 Hz

**Best Model**: `KalmanConv1dConv1d` (dual-stream concat)

**Best Results**: 94.66% ± 4.09% F1 | 93.96% Accuracy | 92.11% Precision | 97.42% Recall | 98.65% AUC

### Configuration

| Parameter | Value |
|-----------|-------|
| Window Size | 300 samples (6.0s) |
| Fall Stride | 8 |
| ADL Stride | 32 |
| Embed Dim | 24 |
| Dropout | 0.4 |
| Kalman Q_orientation | 0.0124 |
| Kalman R_gyro | 0.2822 |

### Training

```bash
# Kalman model (94.66% F1)
python ray_train.py --config best_config/wedafall/kalman.yaml --num-gpus 3

# Raw baseline (90.43% F1)
python ray_train.py --config best_config/wedafall/raw.yaml --num-gpus 3
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
└── best_config/
    ├── smartfallmm/kalman.yaml
    ├── upfall/kalman.yaml
    ├── upfall/raw.yaml
    ├── wedafall/kalman.yaml
    └── wedafall/raw.yaml
```

---

## Multi-Dataset Training

Run all three datasets sequentially with W&B logging:

```bash
# All datasets (recommended: 3 GPUs)
python scripts/run_kalman_experiments.py --dataset all --num-gpus 3

# Single dataset
python scripts/run_kalman_experiments.py --dataset smartfallmm --num-gpus 3

# Quick test (2 folds only)
python scripts/run_kalman_experiments.py --dataset all --num-gpus 1 --max-folds 2
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
