# Weights

Format: `{stride}_{input}_{norm}.pth`

## s8_16 Models (2026-02-04)

| Model | F1 | Arch |
|-------|-----|------|
| kalman_gyromag_norm | **89.32%** | KalmanConv1dConv1d |
| kalman_gyromag_nonorm | 87.88% | KalmanConv1dConv1d |
| kalman_yaw_norm | 87.68% | KalmanConv1dConv1d |
| raw_gyromag_nonorm | 87.51% | SingleStreamTransformerSE |
| raw_gyromag_norm | 87.46% | SingleStreamTransformerSE |
| kalman_yaw_nonorm | 87.44% | KalmanConv1dConv1d |
| raw_gyro_nonorm | 86.04% | SingleStreamTransformerSE |
| raw_gyro_norm | 84.14% | SingleStreamTransformerSE |

## Training Params

- Sampling: 31.25 Hz
- Window: 128 frames
- embed_dim: 48
- 22-fold LOSO

## Scalers

`scalers/{stride}_{input}_norm_scaler.pkl` - Required for `_norm` models.
