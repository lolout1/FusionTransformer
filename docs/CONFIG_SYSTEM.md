# Configuration System

Type-safe, inheritable YAML configs with ablation sweep support.

**Updated: 2026-01-26** - Added Hydra entry point (`train.py`) and `fusionlib.config` module.

## Quick Start

```yaml
# config/my_experiment.yaml
_base:
  - _base/model/transformer_small.yaml
  - _base/dataset/smartfallmm.yaml
  - _base/preprocessing/kalman.yaml

model_args:
  embed_dim: 64  # Override base
```

```bash
# Hydra entry point (new)
python train.py +config=my_experiment

# Ray distributed (existing)
python ray_train.py --config config/my_experiment.yaml --num-gpus 8

# Ablation sweep
python -m fusionlib.config.ablation config/experiment/arch.yaml
```

---

## Config Inheritance

Use `_base` to inherit from parent configs:

```yaml
_base:
  - _base/model/transformer_small.yaml    # Model defaults
  - _base/dataset/smartfallmm.yaml        # Dataset defaults
  - _base/preprocessing/kalman.yaml       # Preprocessing

# Overrides (later values win)
model_args:
  embed_dim: 64
```

**Resolution order**: Base configs merge left-to-right, then child overrides.

---

## Base Templates

| Template | Path | Contents |
|----------|------|----------|
| **Transformer Small** | `_base/model/transformer_small.yaml` | embed_dim=48, 2 layers, 4 heads |
| **Transformer Large** | `_base/model/transformer_large.yaml` | embed_dim=128, 4 layers, 8 heads |
| **SmartFallMM** | `_base/dataset/smartfallmm.yaml` | 128 frames, watch sensor |
| **UP-FALL** | `_base/dataset/upfall.yaml` | 160 frames, wrist IMU |
| **WEDA-FALL** | `_base/dataset/wedafall.yaml` | 300 frames, Fitbit |
| **Kalman** | `_base/preprocessing/kalman.yaml` | Linear Kalman, euler output |
| **Raw** | `_base/preprocessing/raw.yaml` | No Kalman, raw 6-axis |
| **Uniform Stride** | `_base/preprocessing/uniform_stride.yaml` | No class-aware stride (added 2026-01-26) |
| **Default Training** | `_base/training/default.yaml` | AdamW, focal loss, 80 epochs |

---

## Ablation Sweeps

Define sweeps in YAML, generate all combinations:

### Simple Sweeps (Cartesian Product)

```yaml
# config/ablation/architecture_study.yaml
base: best_config/smartfallmm/kalman.yaml

sweep:
  model_args.embed_dim: [32, 48, 64]
  model_args.num_layers: [1, 2, 3]
  dataset_args.enable_kalman_fusion: [true, false]

execution:
  num_gpus: 8
  parallel_experiments: 2
  max_folds: 2  # Quick mode
```

```bash
# Dry run (see what would run)
python -m fusionlib.config.ablation config/ablation/architecture_study.yaml --dry-run

# Execute
python -m fusionlib.config.ablation config/ablation/architecture_study.yaml
```

### Paired Sweeps (Parameters That Vary Together)

Use `paired_sweep` when parameters must be synchronized (e.g., window size in dataset and model):

```yaml
base: best_config/upfall/kalman.yaml

# Paired sweeps: parameters that move together
paired_sweep:
  - name: window
    values:
      - _name: 2s
        dataset_args.max_length: 36
        model_args.imu_frames: 36
        model_args.acc_frames: 36
      - _name: 4s
        dataset_args.max_length: 72
        model_args.imu_frames: 72
        model_args.acc_frames: 72

  - name: stride
    values:
      - _name: tight
        dataset_args.fall_stride: 8
        dataset_args.adl_stride: 32
      - _name: balanced
        dataset_args.fall_stride: 8
        dataset_args.adl_stride: 44

# Independent sweeps (cartesian product with paired)
sweep:
  dataset_args.enable_kalman_fusion: [true, false]

# Total: 2 windows × 2 strides × 2 inputs = 8 experiments
```

### Named Variants (Explicit Configurations)

Use `variants` for explicit named configurations:

```yaml
base: best_config/upfall/kalman.yaml

variants:
  - name: transformer_kalman
    model: Models.encoder_ablation.KalmanConv1dLinear
    dataset_args.enable_kalman_fusion: true

  - name: lstm_raw
    model: Models.dual_stream_cnn_lstm.DualStreamLSTM
    model_args.num_lstm_layers: 2
    dataset_args.enable_kalman_fusion: false
```

### Per-Dataset Sweeps

Different sweep configurations per dataset:

```yaml
base: best_config/smartfallmm/kalman.yaml

datasets:
  - dataset: smartfallmm
    base_config: best_config/smartfallmm/kalman.yaml
    paired_sweep:
      - name: window
        values:
          - _name: 2s
            dataset_args.max_length: 60   # 30Hz × 2s
            model_args.imu_frames: 60

  - dataset: upfall
    base_config: best_config/upfall/kalman.yaml
    paired_sweep:
      - name: window
        values:
          - _name: 2s
            dataset_args.max_length: 36   # 18Hz × 2s
            model_args.imu_frames: 36
```

---

## Config Sections

### model_args

```yaml
model_args:
  embed_dim: 48        # Embedding dimension (must be divisible by num_heads)
  num_heads: 4         # Attention heads
  num_layers: 2        # Transformer layers
  dropout: 0.5         # Dropout probability
  acc_ratio: 0.65      # Accelerometer stream capacity (dual-stream)
  se_reduction: 4      # Squeeze-Excitation reduction
```

### dataset_args

```yaml
dataset_args:
  mode: sliding_window
  max_length: 128              # Window size
  stride: 32                   # Default stride
  enable_class_aware_stride: true
  fall_stride: 16              # More overlap for falls
  adl_stride: 64               # Less overlap for ADL
  sensors: [watch]
  modalities: [accelerometer, gyroscope]
```

### Preprocessing (in dataset_args)

```yaml
dataset_args:
  # Kalman fusion
  enable_kalman_fusion: true
  kalman_filter_type: linear   # linear | ekf
  kalman_output_format: euler  # euler | quaternion
  kalman_include_smv: true     # Signal Vector Magnitude

  # Noise parameters
  kalman_Q_orientation: 0.005
  kalman_R_gyro: 0.1

  # Normalization
  enable_normalization: true
  normalize_modalities: acc_only  # all | acc_only | none
```

### Training

```yaml
batch_size: 64
num_epoch: 80
optimizer: adamw
base_lr: 0.001
weight_decay: 0.001

dataset_args:
  loss_type: focal  # focal | bce | cb_focal
```

---

## Validation

Validate configs before running:

```bash
# Single file
python tools/validate_config.py best_config/smartfallmm/kalman.yaml

# All configs
python tools/validate_config.py "best_config/**/*.yaml"

# Strict mode (fail on warnings)
python tools/validate_config.py --strict config/experiment.yaml
```

---

## Python API

```python
from fusionlib.config.loader import load_config, validate_config
from fusionlib.config.schema import ExperimentConfig

# Load with inheritance
config = load_config('config/experiment.yaml')

# Validate
validated = validate_config(config)

# Access typed fields
print(validated.model.embed_dim)  # 48
print(validated.preprocessing.enable_kalman_fusion)  # True

# Convert to flat format (for backwards compatibility)
flat = validated.to_flat_config()
```

---

## Examples

### Minimal Config

```yaml
model: Models.encoder_ablation.KalmanConv1dLinear
dataset: smartfallmm
model_args:
  embed_dim: 48
dataset_args:
  enable_kalman_fusion: true
```

### Full Config with Inheritance

```yaml
_base:
  - _base/model/transformer_small.yaml
  - _base/dataset/smartfallmm.yaml
  - _base/preprocessing/kalman.yaml
  - _base/training/default.yaml

# Only specify overrides
model_args:
  embed_dim: 64
  num_layers: 3

dataset_args:
  adl_stride: 32  # More ADL samples
```

### Multi-Dataset Ablation

```yaml
base: best_config/smartfallmm/kalman.yaml

sweep:
  dataset: [smartfallmm, upfall, wedafall]
  model_args.embed_dim: [32, 48, 64]

execution:
  num_gpus: 8
```

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `embed_dim must be divisible by num_heads` | Invalid ratio | Use 32, 48, 64, 96, 128 with 4 heads |
| `Config file not found` | Wrong path | Check relative path from config dir |
| `YAML parse error` | Syntax error | Validate YAML syntax |
| `Unknown parameter` | Typo in key | Check spelling against schema |
