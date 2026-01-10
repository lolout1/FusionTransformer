# SmartFallMM: Fall Detection with Multimodal Wearable Sensors

This repository contains the model implementations and training code for fall detection using the SmartFallMM dataset.

## Best Model Results

The following table summarizes the best-performing models evaluated using 21-fold Leave-One-Subject-Out Cross-Validation (LOSO-CV) on young subjects:

| Model | Test F1 | Architecture | Config |
|-------|---------|--------------|--------|
| **Dual-Stream Kalman Transformer** | **90.81%** | SE + TAP, Dual-stream | `reproduce_91.yaml` |
| Kalman Gated Hierarchical Fusion (KGHF) | 90.44% | Multi-scale gating | `kalman_gated_hierarchical.yaml` |
| SingleStream Transformer (Raw 7ch) | 89.49% | SE + TAP, Single-stream | - |
| CNN-Mamba | 89.11% | CNN + Mamba blocks | `cnn_mamba_kalman.yaml` |
| Dual-Stream LSTM (Kalman) | 88.84% | Bi-LSTM | `dual_stream_lstm.yaml` |
| CNN Kalman (Focal Loss) | 88.86% | CNN | - |
| Accelerometer-Only Transformer | 85.91% | TransModel | `transformer.yaml` |

## Key Features

### Kalman Fusion Preprocessing
- Linear Kalman filter fuses accelerometer + gyroscope into orientation (roll, pitch, yaw)
- 7-channel output: [SMV, ax, ay, az, roll, pitch, yaw]
- Channel-aware normalization: StandardScaler on accelerometer only, orientation kept in radians

### Dual-Stream Architecture
- Separate processing streams for acceleration (4ch) and orientation (3ch)
- Squeeze-Excitation (SE) blocks for channel attention
- Temporal Attention Pooling (TAP) for sequence summarization

### Training Configuration
- Binary Focal Loss (alpha=0.75, gamma=2.0) for class imbalance
- AdamW optimizer with cosine annealing
- Early stopping with patience=15

## Usage

```bash
# Train best model (Dual-Stream Kalman Transformer)
python main.py --config config/smartfallmm/reproduce_91.yaml

# Train KGHF variant
python main.py --config config/smartfallmm/kalman_gated_hierarchical.yaml

# Train CNN-Mamba
python main.py --config config/smartfallmm/cnn_mamba_kalman.yaml
```

## Model Files

- `Models/kalman_transformer_variants.py` - KalmanBalancedFlexible (best model)
- `Models/kalman_gated_hierarchical.py` - KGHF architecture
- `Models/cnn_mamba.py` - CNN-Mamba hybrid
- `Models/dual_stream_lstm.py` - Dual-Stream LSTM
- `Models/single_stream_transformer.py` - SingleStream Transformer
- `Models/transformer.py` - Accelerometer-only baseline

## Dataset

The SmartFallMM dataset includes:
- **Young Group**: 30 subjects (ages 18-35) with fall activities
- **Old Group**: 21 subjects (ages 65+) with ADL activities only
- **Sensors**: Smartwatch, smartphone, Meta wrist/hip sensors
- **Activities**: 14 activity classes (5 fall types + 9 ADLs)

For dataset access, contact: Dr. Anne Ngu (angu@txstate.edu)

## Citation

If you use this code, please cite:
```
[Paper citation pending]
```

## License

Copyright SmartFall Group, Texas State University. All rights reserved.
