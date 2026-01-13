# FusionTransformer

**Multimodal Sensor Fusion with Attention-Enhanced Transformers for Real-Time Fall Detection**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A research framework for multimodal transformer architectures that fuse inertial sensor data through Kalman filtering and cross-modal knowledge distillation, targeting deployment on resource-constrained wearable devices.

---

## Overview

Falls represent a leading cause of injury-related mortality among older adults. This research addresses the challenge of **accurate, real-time fall detection on commodity smartwatches** where sensor data exhibits variable sampling rates, asynchronous modalities, and significant noise.

FusionTransformer introduces a principled approach to multimodal sensor fusion that:

- **Extracts orientation estimates** from raw accelerometer and gyroscope data via Kalman filtering
- **Preserves modality-specific representations** through dual-stream transformer architectures  
- **Adapts dynamically** to sensor noise and variable sampling conditions
- **Enables cross-modal knowledge transfer** from rich visual data (skeleton poses) to lightweight IMU-only models

---

## Key Contributions

### 1. Kalman-Based Sensor Fusion

Rather than naively concatenating raw accelerometer and gyroscope signals, we employ a **Kalman filter** to produce robust orientation estimates (roll, pitch, yaw) from noisy, variably-sampled Android IMU data. This approach:

- Handles asynchronous sensor streams gracefully
- Produces physically meaningful features (orientation) from complementary modalities
- Reduces sensitivity to individual sensor noise and dropouts

```
┌─────────────┐     ┌─────────────┐
│ Accelerometer │────▶│             │
│  (ax,ay,az)   │     │   Kalman    │────▶ [smv, ax, ay, az, roll, pitch, yaw]
│               │     │   Filter    │
│  Gyroscope    │────▶│             │
│  (gx,gy,gz)   │     └─────────────┘
└─────────────┘
```

### 2. Dual-Stream Transformer Architecture

We propose a **dual-stream design** that processes accelerometer and orientation features through separate encoding pathways before fusion. This architecture:

- **Handles variable sampling rates** via modality-specific temporal encoders
- **Enables asymmetric capacity allocation** (e.g., 65% for high-SNR accelerometer, 35% for noisier orientation)
- **Supports feature-level knowledge distillation** with explicit per-modality supervision

```
                    ┌──────────────────┐
 Acc Stream ───────▶│ Transformer Enc. │───┐
    [4ch]           └──────────────────┘   │    ┌────────────┐
                                           ├───▶│   Fusion   │───▶ Prediction
                    ┌──────────────────┐   │    │  + Attn    │
 Ori Stream ───────▶│ Transformer Enc. │───┘    └────────────┘
    [3ch]           └──────────────────┘
```

### 3. Attention Mechanisms for Fall Signatures

Falls exhibit distinctive temporal signatures—brief, high-magnitude acceleration spikes followed by stillness. We integrate:

- **Squeeze-and-Excitation (SE) modules** for adaptive channel weighting
- **Temporal Attention Pooling (TAP)** to focus on the critical impact phase
- **No positional encoding**—falls are temporally invariant within the detection window

### 4. Cross-Modal Knowledge Distillation (Future Work)

We are developing a distillation framework to transfer knowledge from a **skeleton-based teacher model** (trained on synchronized video data) to the lightweight IMU-only student:

```
┌─────────────────────────────────────────────────────────────┐
│                      TRAINING PHASE                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐                                           │
│  │   Skeleton   │──────▶ Teacher Model ──▶ Soft Labels     │
│  │   (96 feat)  │              │                            │
│  └──────────────┘              │ Feature Alignment          │
│                                ▼                            │
│  ┌──────────────┐        ┌─────────────┐                    │
│  │   IMU Data   │───────▶│   Student   │──▶ Predictions    │
│  │   (7 feat)   │        │   (Ours)    │                    │
│  └──────────────┘        └─────────────┘                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                     DEPLOYMENT PHASE                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐        ┌─────────────┐                    │
│  │  Watch IMU   │───────▶│   Student   │──▶ Real-time      │
│  │    Only      │        │   (Ours)    │   Fall Detection  │
│  └──────────────┘        └─────────────┘                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Best Results

| Metric | Value |
|--------|-------|
| **Test F1** | 91.38% ± 6.67 |
| **Accuracy** | 88.44% |
| **Precision** | 89.22% |
| **Recall** | 94.14% |
| **AUC** | 94.30% |

Achieved with `KalmanConv1dLinear` model (22 LOSO folds). See `best_config/` for optimal configuration.

---

## Architecture

### Best Architecture: KalmanConv1dLinear

The optimal architecture uses **asymmetric encoders**: Conv1D for accelerometer (captures temporal transients) and Linear for orientation (sufficient for Kalman-smoothed signals).

```
Input: Raw IMU Window (128 samples @ ~32Hz)
           │
           ▼
    ┌──────────────────┐
    │  Kalman Fusion   │──────▶ 7-channel tensor
    │  (Acc + Gyro)    │        [smv, ax, ay, az, roll, pitch, yaw]
    └──────────────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
┌────────┐   ┌────────┐
│  Acc   │   │  Ori   │
│ Stream │   │ Stream │
│ (4ch)  │   │ (3ch)  │
└───┬────┘   └───┬────┘
    │ Conv1D     │ Linear      ◄── Asymmetric encoders
    │ (k=8)      │ Projection
    │ 31d        │ 17d
    ▼            ▼
    └─────┬──────┘
          │ Concatenation (48d)
          ▼
    ┌──────────────┐
    │ LayerNorm    │
    └──────┬───────┘
          ▼
    ┌──────────────┐
    │ Transformer  │  2 layers, 4 heads
    │   Encoder    │
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │   SE Block   │  Channel Attention
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │   Temporal   │  Focus on Impact Phase
    │   Attention  │
    │   Pooling    │
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │  Classifier  │──────▶ Fall / Non-Fall
    └──────────────┘
```

**Key Design Decisions**:
- **Conv1D for accelerometer**: Captures high-frequency transients during falls
- **Linear for orientation**: Kalman-filtered Euler angles are already temporally smooth
- **Asymmetric capacity**: 65% (31d) for acc, 35% (17d) for ori
- **embed_dim=48**: Smaller capacity reduces overfitting

---

## Dataset

This research uses the **SmartFallMM** dataset—a multimodal resource for wearable sensor-based human activity recognition featuring:

| Property | Details |
|----------|---------|
| **Participants** | 51 subjects (30 young + 21 elderly) |
| **Modalities** | Accelerometer, Gyroscope, Skeleton (32 joints) |
| **Sensors** | Smartwatch, Smartphone, Meta sensors |
| **Sampling** | Variable (~32Hz IMU, 30 FPS skeleton) |
| **Activities** | 14 classes including 5 fall types |
| **Challenges** | Variable sampling, sensor noise, age diversity |

### Data Organization

```
SmartFallMM-Dataset/
├── Young/
│   ├── Accelerometer/
│   │   ├── Watch/          # Primary modality
│   │   ├── Phone/
│   │   └── Meta_*/
│   ├── Gyroscope/
│   │   └── Watch/          # Fused with accelerometer
│   └── Skeleton/           # Teacher supervision
└── Old/
    └── [same structure]
```

---

## Repository Structure

```
FusionTransformer/
│
├── Models/
│   ├── encoder_ablation.py        # Best model: KalmanConv1dLinear
│   ├── imu_transformer_se.py      # SE-enhanced transformer
│   └── dual_stream_se.py          # Dual-stream architecture
│
├── Feeder/
│   └── Make_Dataset.py            # PyTorch Dataset (UTD_mm)
│
├── best_config/
│   └── kalman_conv1d_linear_optimal.yaml  # Optimal configuration
│
├── config/
│   └── smartfallmm/               # Experimental configs
│
├── utils/
│   ├── loader.py                  # DatasetBuilder with Kalman fusion
│   ├── ray_distributed.py         # Ray actors for distributed training
│   ├── callbacks.py               # Early stopping
│   └── threshold_analysis.py      # Post-hoc threshold optimization
│
├── ray_train.py                   # Distributed LOSO training entry point
├── main.py                        # Single-fold training
└── CLAUDE.md                      # Codebase documentation
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/[username]/FusionTransformer.git
cd FusionTransformer
pip install -r requirements.txt
```

### Training

```bash
# Distributed LOSO training (recommended, 3 GPUs)
python ray_train.py --config best_config/kalman_conv1d_linear_optimal.yaml --num-gpus 3

# Quick test (2 folds, 1 GPU)
python ray_train.py --config best_config/kalman_conv1d_linear_optimal.yaml --num-gpus 1 --max-folds 2

# Single-fold training
python main.py --config best_config/kalman_conv1d_linear_optimal.yaml
```

### Configuration

Key hyperparameters in `best_config/kalman_conv1d_linear_optimal.yaml`:

```yaml
model: Models.encoder_ablation.KalmanConv1dLinear

model_args:
  imu_channels: 7          # [smv, ax, ay, az, roll, pitch, yaw]
  embed_dim: 48            # Smaller = less overfitting
  num_heads: 4
  num_layers: 2
  acc_ratio: 0.65          # 65% capacity for accelerometer

dataset_args:
  enable_kalman_fusion: True
  enable_normalization: True
  normalize_modalities: acc_only
  loss_type: focal
  enable_class_aware_stride: True
  fall_stride: 16
  adl_stride: 64
```

---

## Evaluation Protocol

All experiments follow a rigorous **Leave-One-Subject-Out (LOSO)** cross-validation protocol:

- **22 folds** for Young+Old evaluation
- **30 folds** for Young-only evaluation
- **Metrics**: F1-score (primary), Accuracy, Precision, Recall, AUC
- **Statistical testing**: Paired t-tests with Bonferroni correction

---



## Citation

```bibtex
@article{fusionformer2025,
  title={FusionTransformer: Multimodal Sensor Fusion with Attention-Enhanced 
         Transformers for Real-Time Wearable Fall Detection},
  author={Pradhan, Abheek},
  journal={[In Preparation]},
  year={2025}
}
```

---

## Acknowledgments

This research is conducted at Texas State University. We thank the SmartFallMM dataset contributors for making multimodal fall detection research possible.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <i>Advancing wearable health monitoring through principled multimodal fusion</i>
</p>
