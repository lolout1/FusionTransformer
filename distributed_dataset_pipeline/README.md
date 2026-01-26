# Distributed Dataset Pipeline

Cross-dataset ablation framework for fall detection using Ray distributed training.

Supports multiple datasets, architectures, input representations, and window configurations.

---

## Quick Start

```bash
# Full ablation (all datasets, architectures, window sizes)
python distributed_dataset_pipeline/run_architecture_ablation.py \
    --datasets all --num-gpus 4 --parallel 2

# Quick test (2 folds only)
python distributed_dataset_pipeline/run_architecture_ablation.py \
    --datasets wedafall --quick --num-gpus 2

# Specific configurations
python distributed_dataset_pipeline/run_architecture_ablation.py \
    --datasets upfall --architectures lstm transformer \
    --window-sizes 2s 4s --num-gpus 2
```

---

## Supported Configurations

### Datasets

| Dataset | Sampling Rate | Subjects | Sensor |
|---------|---------------|----------|--------|
| **UP-FALL** | 18 Hz | 17 | Wrist IMU |
| **WEDA-FALL** | 50 Hz | 24 (young + elderly) | Wrist IMU |
| **SmartFallMM** | 30 Hz | 59 | Smartwatch IMU |

### Window Sizes

| Dataset | 2s | 3s | 4s | Default |
|---------|----|----|----| --------|
| UP-FALL | 36 | 54 | 72 | 160 |
| WEDA-FALL | 100 | 150 | 200 | 250 |
| SmartFallMM | 60 | 90 | 120 | 128 |

### Architectures

| Name | Description |
|------|-------------|
| `lstm` | Bidirectional LSTM with dual-stream fusion |
| `baseline_transformer` | Conv1D encoder + Transformer + SE attention |
| `deep_cnn_transformer` | Multi-stage CNN + Transformer |
| `mamba` | State-space model with selective scan |

### Input Types

| Type | Channels | Description |
|------|----------|-------------|
| `kalman` | 7 | SMV + Acc XYZ + Orientation (roll, pitch, yaw) |
| `raw` | 7 | SMV + Acc XYZ + Gyro XYZ |

---

## CLI Reference

```bash
python distributed_dataset_pipeline/run_architecture_ablation.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--datasets` | `all` | `all`, `upfall`, `wedafall` |
| `--architectures` | `all` | `all`, `lstm`, `baseline_transformer`, `deep_cnn_transformer`, `mamba` |
| `--input-types` | `all` | `all`, `kalman`, `raw` |
| `--window-sizes` | `all` | `all`, `2s`, `3s`, `4s`, `default` |
| `--num-gpus` | `4` | Total GPUs |
| `--parallel` | `1` | Concurrent experiments |
| `--max-folds` | all | Limit folds per experiment |
| `--quick` | off | Run only 2 folds |
| `--work-dir` | auto | Output directory |

---

## Output Structure

```
exps/architecture_ablation_{timestamp}/
├── configs/
│   ├── upfall/
│   │   ├── upfall_lstm_kalman_2s.yaml
│   │   └── ...
│   └── wedafall/
├── runs/
│   ├── upfall_lstm_kalman_2s/
│   │   ├── fold_results.pkl
│   │   └── summary.json
│   └── ...
├── results.json
└── architecture_ablation_report.md
```

---

## Adding a New Architecture

1. **Create model** in `Models/`:

```python
class MyModel(nn.Module):
    def __init__(self, imu_frames, imu_channels, embed_dim, **kwargs):
        super().__init__()
        # ...

    def forward(self, x):
        # x: (batch, time, channels)
        return logits, features  # (B, 1), (B, T, embed_dim)
```

2. **Register** in `run_architecture_ablation.py`:

```python
ARCHITECTURES = {
    # ...
    'my_model': {
        'model': 'Models.my_module.MyModel',
        'model_args_override': {},
        'training_override': {'base_lr': 0.0005},
    },
}
```

---

## Adding a New Dataset

1. **Create loader** in `utils/` following the `UPFallLoader` or `WEDAFallLoader` pattern.

2. **Create base config** at `config/best_config/{dataset}/kalman.yaml`.

3. **Register** in `run_architecture_ablation.py`:

```python
DATASET_CONFIGS = {
    # ...
    'mydataset': {
        'base_config': 'config/best_config/mydataset/kalman.yaml',
        'sampling_rate': 50,
        'window_sizes': {'2s': 100, '3s': 150, 'default': 200},
        'num_test_subjects': 15,
    },
}
```

---

## Key Implementation Details

### Window Size Configuration

The `max_length` parameter in `dataset_args` controls the actual window size:

```yaml
dataset_args:
  max_length: 100  # Window size in samples
```

This must match `imu_frames` in `model_args`.

### Class-Aware Stride

Falls are rare. To balance the dataset, use different strides:

```yaml
dataset_args:
  enable_class_aware_stride: true
  fall_stride: 16   # More overlap for falls
  adl_stride: 64    # Less overlap for ADLs
```

### Mamba Channel Split

Mamba requires explicit accelerometer/gyroscope channel split:

```python
'mamba': {
    'model_args_override': {
        'acc_coords': 4,   # SMV + acc_xyz
        'gyro_coords': 3,  # Orientation OR raw gyro
    }
}
```

---

## Related Scripts

| Script | Purpose |
|--------|---------|
| `run_architecture_ablation.py` | Main multi-architecture comparison |
| `run_external_ablation.py` | External dataset experiments |
| `run_kalman_vs_raw_ablation.py` | Kalman vs raw input comparison |
| `run_short_window_ablation.py` | Short window optimization |
| `ablation_analysis.py` | Result analysis utilities |

---

## Testing

```bash
pytest tests/test_ablation_architectures.py -v
```
