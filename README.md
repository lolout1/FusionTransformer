# Kalman-Fused Dual-Stream Transformer for Wearable Fall Detection

[![Paper](https://img.shields.io/badge/Paper-Under%20Review-blue)]()
[![Dataset](https://img.shields.io/badge/Dataset-SmartFallMM-green)](https://github.com/txst-cs-smartfall/SmartFallMM-Dataset)
[![Python](https://img.shields.io/badge/Python-3.9+-yellow)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)]()

Real-time fall detection from smartwatch IMU achieving **91.1% F1-score** via Kalman sensor fusion and dual-stream transformer architecture. Validated on 21-fold Leave-One-Subject-Out cross-validation.

## Contributions

| Contribution | Description |
|--------------|-------------|
| **Kalman Sensor Fusion** | Fuses accelerometer gravity reference with gyroscope angular velocity into drift-corrected orientation (roll, pitch, yaw), decoupling gravitational from dynamic acceleration |
| **Dual-Stream Architecture** | Parallel feature extraction for acceleration dynamics (4ch) and orientation kinematics (3ch) with squeeze-excitation fusion |
| **Channel-Aware Normalization** | StandardScaler on acceleration only; orientation preserved in radians to maintain bounded physical semantics |
| **Class-Aware Windowing** | Asymmetric stride (fall=16, ADL=64) addresses 5:1 class imbalance without synthetic augmentation |

## Results

| Model | Filter | Test F1 | Val F1 | Params |
|-------|--------|---------|--------|--------|
| KalmanBalancedFlexible | EKF | **91.1%** | 96.1% | 294K |
| KalmanBalancedFlexible | LKF | 89.7% | 95.8% | 294K |
| Single-Stream Baseline | — | 87.2% | 93.4% | 198K |

21-fold LOSO-CV on young adults (N=21 test subjects).

---

## Installation

```bash
git clone https://github.com/txst-smartfall/SmartFallMM.git && cd SmartFallMM
pip install torch numpy scikit-learn pyyaml wandb ray

# Link dataset
git clone https://github.com/txst-cs-smartfall/SmartFallMM-Dataset.git
ln -s SmartFallMM-Dataset/Young\ Data data/young
```

---

## Pipeline

```
Raw IMU ──► Kalman Filter ──► Feature Assembly ──► Model ──► P(fall)
[6ch]        [LKF/EKF/UKF]     [7ch normalized]    [dual-stream]
```

### 1. Preprocessing: Kalman Sensor Fusion

Transforms 6-axis IMU into 7-channel orientation-augmented features:

```python
from utils.kalman import process_trial_kalman

result = process_trial_kalman(
    acc_data,               # (T, 3) m/s²
    gyro_data,              # (T, 3) rad/s
    filter_type='ekf',      # 'linear' | 'ekf' | 'ukf'
    output_format='euler'   # 'euler' | 'quaternion' | 'gravity_vector'
)
# Returns: orientation (T, 3), uncertainty, innovation
```

**Output channels**: `[SMV, ax, ay, az, roll, pitch, yaw]`

| Filter | State Dimension | Use Case |
|--------|-----------------|----------|
| LKF | 6D (Euler + rates) | General activities, fast |
| EKF | 7D (quaternion + gyro bias) | Rapid rotations, no gimbal lock |
| UKF | 7D (sigma points) | Tumbling falls, highest accuracy |

### 2. Data Loading

```python
from Feeder.Make_Dataset import DatasetBuilder

dataset = DatasetBuilder(
    window_size=128,          # 4.3s at 30Hz
    stride=32,                # ADL stride
    fall_stride=16,           # Fall stride (2x oversampling)
    kalman_filter_type='ekf',
    normalization_mode='acc_only'  # Preserve orientation radians
)
```

### 3. Model Architecture

```python
from Models.kalman_transformer_variants import KalmanBalancedFlexible

model = KalmanBalancedFlexible(
    imu_channels=7,
    acc_channels=4,           # Channels 0-3 → acc stream
    embed_dim=64,
    num_heads=4,
    num_layers=2,
    dropout=0.5,
    use_se=True,              # Squeeze-excitation attention
    use_tap=True              # Temporal attention pooling
)
```

**Architecture**:
```
Input [B, 7, 128]
    ├── Acc Stream [0:4] ──► Conv1D ──► 32d
    └── Ori Stream [4:7] ──► Conv1D ──► 32d
                               │
                    Concat ──► 64d
                               │
              TransformerEncoder (2L × 4H)
                               │
                    SE Attention ──► TAP ──► MLP ──► σ
```

### 4. Training

```python
from main import Trainer

trainer = Trainer(args)
trainer.start()  # Runs 21-fold LOSO-CV
```

**Loss**: Focal Loss (α=0.75, γ=2.0) for class imbalance
**Optimizer**: AdamW (lr=1e-3, weight_decay=1e-2)
**Early stopping**: patience=15 on validation loss

---

## Usage

### Training

```bash
# Standard training
python main.py --config config/smartfallmm/kalman_balanced_ekf.yaml

# With W&B logging
python main.py --config config/smartfallmm/kalman_balanced_ekf.yaml --enable_wandb

# Single fold (debugging)
python main.py --config config/smartfallmm/kalman_balanced_ekf.yaml --single_fold 0
```

### Configuration

```yaml
model: Models.kalman_transformer_variants.KalmanBalancedFlexible
model_args:
  imu_channels: 7
  embed_dim: 64
  dropout: 0.5
  use_se: True
  use_tap: True

dataset_args:
  kalman_filter_type: ekf
  kalman_output_format: euler
  normalization_mode: acc_only
  window_size: 128
  fall_stride: 16

batch_size: 32
num_epoch: 80
loss: focal
```

---

## Experiment Tracking

### Weights & Biases

```bash
export WANDB_API_KEY="your-key"
python main.py --config config.yaml --enable_wandb
```

**Logged metrics**:
- Per-epoch: train/val loss, F1, accuracy
- Per-fold: test F1, precision, recall, AUC, confusion matrix
- Summary: mean ± std across 21 folds
- Artifacts: model checkpoints, configs

**Offline mode** (cluster jobs):
```bash
WANDB_MODE=offline python main.py --config config.yaml --enable_wandb
wandb sync --sync-all  # After job completes
```

### Distributed Training (Ray)

```bash
# Hyperparameter sweep
python runners/ray_tune_sweep.py --num_samples 50 --gpus_per_trial 1

# SLURM submission
make submit-ablation
```

---

## Project Structure

```
├── main.py                           # Training entry point
├── config/smartfallmm/               # YAML configurations
├── Models/
│   ├── kalman_transformer_variants.py  # KalmanBalancedFlexible
│   ├── kalman_gated_hierarchical.py    # KGHF variant
│   └── flexible_transformer.py         # Single/dual stream ablations
├── utils/kalman/
│   ├── filters.py                    # LKF, EKF implementations
│   ├── ukf_fast.py                   # Optimized UKF
│   ├── preprocessing.py              # Sensor fusion pipeline
│   └── quaternion.py                 # Quaternion operations
├── Feeder/
│   └── Make_Dataset.py               # Data loading, windowing
├── runners/
│   └── ray_tune_sweep.py             # Distributed HPO
└── utils/wandb_integration.py        # W&B logging utilities
```

---

## Citation

```bibtex
@article{smartfallmm2024,
  title={Kalman-Fused Dual-Stream Transformer for Wearable Fall Detection},
  author={SmartFall Research Group},
  journal={Under Review},
  year={2024}
}
```

## License

© SmartFall Research Group, Texas State University.
Contact: Dr. Anne Ngu (angu@txstate.edu)
