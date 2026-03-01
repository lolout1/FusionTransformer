# FusionTransformer

[![CI](https://github.com/lolout1/FusionTransformer/actions/workflows/ci.yml/badge.svg)](https://github.com/lolout1/FusionTransformer/actions/workflows/ci.yml)
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
| **SmartFallMM** | KalmanConv1dConv1d | **91.65%** ± 5.36 | 87.81% | 89.66% | 94.27% | [kalman_baseline.yaml](config/best_config/smartfallmm/kalman_baseline.yaml) |
| **UP-FALL** | KalmanConv1dConv1d | **95.61%** ± 4.01 | 96.81% | 95.70% | 95.83% | [upfall/kalman.yaml](config/best_config/upfall/kalman.yaml) |
| **WEDA-FALL** | KalmanConv1dConv1d | **95.41%** ± 2.50 | 94.57% | 92.82% | 98.22% | [wedafall/kalman.yaml](config/best_config/wedafall/kalman.yaml) |

### Dual-Stream + Kalman Improvement

| Dataset | Raw | Kalman | Improvement |
|---------|-----|--------|-------------|
| SmartFallMM | 88.96% | **91.12%** | **+2.16%** |
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
python ray_train.py --config config/best_config/smartfallmm/kalman_baseline.yaml --num-gpus 8
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
│   ├── run_capacity_ablation.py     # Stream and channel ablation
│   └── run_stride_loss_input_ablation.py  # Stride × loss × input
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
make train CONFIG=config/best_config/smartfallmm/kalman_baseline.yaml

# UP-FALL (15 folds)
make train CONFIG=config/best_config/upfall/kalman.yaml

# WEDA-FALL (12 folds)
make train CONFIG=config/best_config/wedafall/kalman.yaml
```

### Resume Interrupted Training

```bash
make train-resume CONFIG=config/best_config/smartfallmm/kalman_baseline.yaml
```

### Ablation Studies

```bash
# Capacity ablation (stream, channel, embedding)
make ablation

# Quick ablation (2 folds per config)
make ablation-quick

# Kalman vs Raw comparison
make ablation-kalman
```

---

## Configuration

Configs use flat YAML format:

```yaml
# config/best_config/smartfallmm/kalman_baseline.yaml
model: Models.encoder_ablation.KalmanConv1dConv1d

model_args:
  imu_frames: 128
  embed_dim: 48
  num_heads: 4
  num_layers: 2

dataset_args:
  enable_class_aware_stride: true
  fall_stride: 8
  adl_stride: 32
  enable_kalman_fusion: true
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

### Queue & Temporal Ablation

Evaluates deployment performance across aggregation methods, queue sizes, strides, and calibration settings. Trains all configs in `configs_tmp/` then runs a full temporal queue sweep (127k+ parameter combinations).

```bash
# Full run: train 7 configs + queue eval (8 GPUs, 4 parallel)
python distributed_dataset_pipeline/run_queue_ablation.py --num-gpus 8 --parallel 4

# Quick run (2 folds per config)
python distributed_dataset_pipeline/run_queue_ablation.py --num-gpus 8 --parallel 4 --quick

# Eval-only (skip training, reuse existing runs)
python distributed_dataset_pipeline/run_queue_ablation.py --eval-only

# Custom configs directory
python distributed_dataset_pipeline/run_queue_ablation.py --configs-dir path/to/configs --num-gpus 4
```

**Sweep parameters**: strides `[2,4,5,8,10,15,20]`, queue sizes `[3,5,8,10,15,20]`, thresholds `[0.3-0.7]`, retain `[0-5]`, methods `[average, majority_k30-k70]`, calibrated `[true, false]`.

**Outputs** (in `exps/queue_ablation_<timestamp>/`):
- `queue_ablation_report.md` — full analysis with 10 sections
- `queue_ablation_full.csv` — all parameter combinations
- `queue_ablation_best_per_config.csv` — best settings per model config
- `queue_ablation_best_per_stride.csv` — best settings per stride
- `queue_ablation_summary.json` — top results and metadata

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

### Git Workflow

```bash
# Check status
git status

# Stage and commit
git add -A
git commit -m "Description of changes"

# Push to remote
git push origin <branch-name>
```

---

## Temporal Queue Evaluation

Queue evaluation is **separate from LOSO training**. `ray_train.py` performs standard per-window test metrics and saves predictions to `fold_results.pkl`. The queue evaluator reads those predictions post-hoc to simulate real-time temporal queueing — it never touches the training or test loop.

```bash
# Step 1: Train (no queue — standard per-window LOSO)
python ray_train.py --config configs_tmp/nosmv_f8_a32.yaml --num-gpus 8

# Step 2: Queue eval (optional — reads fold_results.pkl from step 1)
python distributed_dataset_pipeline/run_queue_ablation.py --eval-only

# Full pipeline: train all configs + queue eval
python distributed_dataset_pipeline/run_queue_ablation.py --num-gpus 8 --parallel 4

# Quick run (2 folds per config)
python distributed_dataset_pipeline/run_queue_ablation.py --num-gpus 8 --parallel 4 --quick
```

**Sweep parameters**: strides `[2,4,5,8,10,15,20]`, queue sizes `[3,5,8,10,15,20]`, thresholds `[0.3-0.7]`, retain `[0-5]`, methods `[average, majority_k30-k70]`, calibrated `[true, false]`.

**Server-side testing** (`kalman_server/`): toggle queue with `--no-queue` (CLI) or "Simulate Alpha Queue" checkbox (web app). Production NATS server does single-window inference only — queueing is handled by the watch app.

**Outputs** (in `exps/queue_ablation_<timestamp>/`):
- `queue_ablation_report.md` — full analysis
- `queue_ablation_full.csv` — all parameter combinations (127k+ rows)
- `queue_ablation_summary.json` — top results and metadata

---

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA 12.1+ (for GPU training)
- Ray 2.0+ (for distributed training)

---

## Installation

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
