# Best Configuration - Ablation Study Results

## Optimal Model: KalmanConv1dLinear (embed=48)

| Metric | Value |
|--------|-------|
| **Test F1** | 91.38% ± 6.67 |
| **Test Accuracy** | 88.44% |
| **Precision** | 89.22% |
| **Recall** | 94.14% |
| **AUC** | 94.30% |
| **Avg Best Epoch** | 15.2 |
| **Folds** | 22 (LOSO) |

## Ablation Study Summary

**Date**: 2026-01-13
**Total Experiments**: 22
**Total Time**: 96.2 minutes
**Source**: `results/ablation_20260113_172249/`

### Group Results

| Group | Best Experiment | Test F1 | Avg F1 |
|-------|-----------------|---------|--------|
| encoder | enc_conv1d_conv1d | 90.84% | 90.18% |
| kernel | kernel_8 | 90.84% | 90.40% |
| loss | loss_focal | 90.82% | 90.17% |
| embed | embed_64 | 90.87% | 90.41% |
| stride | stride_64 | 90.84% | 89.55% |
| **interaction** | **conv1d_linear_embed48** | **91.38%** | 89.57% |

### Key Findings

1. **Encoder Architecture**
   - Conv1D for accelerometer: Captures temporal patterns in raw acceleration
   - Linear for orientation: Sufficient for Kalman-smoothed Euler angles (already temporally coherent)

2. **Embedding Dimension**
   - embed_dim=48 outperforms embed_dim=64
   - Smaller capacity reduces overfitting on this dataset size

3. **Loss Function**
   - Focal loss handles class imbalance effectively
   - Similar performance to BCE with class weighting

4. **Data Augmentation**
   - adl_stride=64 provides optimal balance
   - Class-aware stride (fall=16, adl=64) improves class balance

## Usage

```bash
# Run with optimal config
python ray_train.py --config best_config/kalman_conv1d_linear_optimal.yaml --num-gpus 3

# Quick test (2 folds)
python ray_train.py --config best_config/kalman_conv1d_linear_optimal.yaml --num-gpus 1 --max-folds 2
```

## Architecture

```
Input: 7 channels [smv, ax, ay, az, roll, pitch, yaw]
       (Kalman-fused accelerometer + Euler orientation)

Accelerometer Stream (4ch):
  [smv, ax, ay, az] → Conv1D(k=8) → 31d embedding

Orientation Stream (3ch):
  [roll, pitch, yaw] → Linear → 17d embedding

Fusion:
  Concat(31d, 17d) → 48d → TransformerEncoder(2 layers, 4 heads)
                        → SE Attention → Temporal Attention Pooling
                        → Linear(1) → Sigmoid
```
