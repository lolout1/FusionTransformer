# FusionTransformer

[![CI](https://github.com/YOUR_USERNAME/FusionTransformer/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/FusionTransformer/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](Dockerfile)

> **Dual-stream transformer for wearable fall detection using Kalman-fused IMU data.**
>
> Achieving **91-95% F1** across three benchmark datasets with distributed training on 8 GPUs.

---

## Highlights

- **Multi-Dataset Support**: SmartFallMM, UP-FALL, WEDA-FALL
- **Kalman Sensor Fusion**: IMU acc+gyro fused into stable orientation estimates
- **Dual-Stream Architecture**: Separate encoding for accelerometer (65%) and orientation (35%)
- **Distributed Training**: Ray-based parallel LOSO cross-validation on 8 GPUs
- **Production-Ready**: Docker, CI/CD, type-safe configs, comprehensive testing

---

## Results

All results from Leave-One-Subject-Out (LOSO) cross-validation.

| Dataset | Model | Test F1 | Accuracy | Precision | Recall | Config |
|---------|-------|---------|----------|-----------|--------|--------|
| **SmartFallMM** | KalmanConv1dLinear | **91.38%** ± 6.67 | 88.44% | 89.22% | 94.14% | [kalman.yaml](config/best_config/smartfallmm/kalman.yaml) |
| **UP-FALL** | KalmanConv1dConv1d | **95.18%** | 96.53% | 95.21% | 95.55% | [kalman.yaml](config/best_config/upfall/kalman.yaml) |
| **WEDA-FALL** | KalmanConv1dConv1d | **95.41%** ± 2.39 | 94.57% | 94.57% | 96.43% | [kalman.yaml](config/best_config/wedafall/kalman.yaml) |

### Dual-Stream + Kalman Improvement

| Dataset | Raw | Kalman | Improvement |
|---------|-----|--------|-------------|
| SmartFallMM | 88.96% | **91.38%** | **+2.42%** |
| UP-FALL | 92.64% | **95.18%** | **+2.54%** |
| WEDA-FALL | 94.64% | **95.41%** | **+0.77%** |

---

## UP-FALL Ablation Study

Embedding/heads ablation (from `exps/upfall_embed_ablation_20260126_023133/summary.json`).

| Config | Kalman | Embed Dim | Heads | F1 (%) | Acc (%) | Prec (%) | Rec (%) |
|:------:|:------:|:---------:|:-----:|:------:|:-------:|:--------:|:-------:|
| ed48_h4_kalman | True | 48 | 4 | **95.18** | 96.53 | 95.21 | 95.55 |
| ed48_h4_raw | False | 48 | 4 | 92.64 | 95.01 | 93.79 | 92.43 |

---

## Quick Start

### Docker (Recommended)

```bash
# Build image
docker build -t fusiontransformer:latest .

# Train with 8 GPUs
docker-compose up train

# Quick test (2 folds)
docker-compose up train-quick
```

### Local Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Train with 8 GPUs (default)
make train

# Quick test
make train-quick

# Custom configuration
python ray_train.py --config config/best_config/smartfallmm/kalman.yaml --num-gpus 8
```

### Development

```bash
# Install dev dependencies
make install-dev

# Run tests
make test

# Lint code
make lint

# Validate configs
make validate-configs
```

---

## Architecture

```
Raw IMU (6ch) ──► Kalman Filter ──► 7ch [smv, ax, ay, az, roll, pitch, yaw]
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
            Accelerometer (4ch)                      Orientation (3ch)
            [smv, ax, ay, az]                        [roll, pitch, yaw]
                    │                                         │
              Conv1D (k=8)                               Linear
              65% capacity                             35% capacity
                    │                                         │
                    └──────────────► Concat ◄─────────────────┘
                                        │
                                  LayerNorm(48d)
                                        │
                              TransformerEncoder (2L, 4H)
                                        │
                              Squeeze-Excitation
                                        │
                            Temporal Attention Pooling
                                        │
                                  Linear(1) + Sigmoid
                                        │
                                   Fall / ADL
```

### Key Design Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Kalman Fusion** | Linear filter | Handles variable sampling rates, sensor noise |
| **Asymmetric Encoding** | 65/35 split | Accelerometer captures transients, orientation is smooth |
| **Conv1D for Acc** | kernel=8 | Captures temporal patterns in fall signatures |
| **Linear for Ori** | No kernel | Kalman-filtered orientation is already smooth |
| **embed_dim=48** | Small capacity | Prevents overfitting on limited subjects |

---

## Project Structure

```
FusionTransformer/
├── Models/                          # Neural network architectures
│   ├── encoder_ablation.py          # Best: KalmanConv1dLinear
│   └── dual_stream_*.py             # Architecture variants
│
├── utils/                           # Core utilities
│   ├── ray_distributed.py           # Distributed LOSO training
│   ├── loader.py                    # SmartFallMM data loading
│   ├── upfall_loader.py             # UP-FALL loader
│   ├── wedafall_loader.py           # WEDA-FALL loader
│   └── kalman/                      # Kalman filter implementations
│
├── Feeder/                          # PyTorch Dataset classes
├── fusionlib/                       # Reusable library components
│
├── distributed_dataset_pipeline/    # Ablation study scripts
│   ├── run_architecture_ablation.py # Multi-architecture comparison
│   └── run_kalman_vs_raw_ablation.py
│
├── config/                          # Experiment configurations
│   └── _base/                       # Inheritable base configs
├── config/best_config/                     # Validated reference configs
│
├── tests/                           # Test suite
├── Dockerfile                       # Container definition
├── docker-compose.yml               # Orchestration
└── Makefile                         # Build automation
```

---

## Training

### Full LOSO Cross-Validation

```bash
# SmartFallMM (22 folds, ~2 hours on 8 GPUs)
make train CONFIG=config/best_config/smartfallmm/kalman.yaml

# UP-FALL (15 folds)
make train CONFIG=config/best_config/upfall/kalman.yaml

# WEDA-FALL (12 folds)
make train CONFIG=config/best_config/wedafall/kalman.yaml
```

### Resume Interrupted Training

```bash
make train-resume CONFIG=config/best_config/smartfallmm/kalman.yaml
```

### Architecture Ablation

```bash
# Full ablation (all architectures x all datasets x all window sizes)
make ablation

# Quick ablation (2 folds per config)
make ablation-quick
```

---

## Configuration

Configs use YAML with inheritance support:

```yaml
# config/best_config/smartfallmm/kalman.yaml
_base:
  - _base/model/transformer_small.yaml
  - _base/training/default.yaml
  - _base/kalman/smartfallmm.yaml

model:
  name: Models.encoder_ablation.KalmanConv1dLinear

dataset:
  enable_class_aware_stride: true
  fall_stride: 16
  adl_stride: 64
```

### Key Parameters

| Parameter | SmartFallMM | UP-FALL | WEDA-FALL |
|-----------|-------------|---------|-----------|
| Sampling Rate | 30 Hz | 18 Hz | 50 Hz |
| Window Size | 128 (~4s) | 160 (~9s) | 300 (~6s) |
| embed_dim | 48 | 64 | 24 |
| Kalman Q_ori | 0.005 | 0.032 | 0.012 |

---

## Datasets

| Dataset | Subjects | Sensor | Rate | Activities |
|---------|----------|--------|------|------------|
| **SmartFallMM** | 51 (30 young, 21 old) | Smartwatch | 30 Hz | 5 falls, 10 ADLs |
| **UP-FALL** | 17 | Wrist IMU | 18 Hz | 5 falls, 6 ADLs |
| **WEDA-FALL** | 14 (young) | Fitbit | 50 Hz | 15 falls, 8 ADLs |

---

## Development

### Testing

```bash
# Run all tests
make test

# With coverage
make test-cov

# Specific test file
pytest tests/test_ablation_architectures.py -v
```

### Code Quality

```bash
# Lint
make lint

# Format
make format

# Validate configs
make validate-configs
```

### Pre-commit Hooks

```bash
# Install hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

---

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA 12.1+ (for GPU training)
- Ray 2.0+ (for distributed training)

```bash
pip install -r requirements.txt
```

---

## Citation

```bibtex
@article{fusiontransformer2024,
  title={FusionTransformer: Dual-Stream Kalman-Fused Transformer for Wearable Fall Detection},
  author={},
  year={2024}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.
