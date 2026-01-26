# Best Configurations

Reference configs for fall detection across three datasets.

## Results

| Dataset | Model | Test F1 | Accuracy | Config |
|---------|-------|---------|----------|--------|
| SmartFallMM | Kalman | 91.53% ± 6.09% | 87.88% | `smartfallmm/kalman.yaml` |
| **UP-FALL** | Kalman | **95.18% ± 3.26%** | 96.53% | `upfall/kalman.yaml` |
| UP-FALL | Raw | 94.77% ± 3.29% | 96.26% | `upfall/raw.yaml` |
| **WEDA-FALL** | Kalman | **94.40% ± 3.21%** | 92.85% | `wedafall/kalman.yaml` |
| WEDA-FALL | Raw | 90.43% ± 2.63% | 87.25% | `wedafall/raw.yaml` |

## Quick Start

```bash
python ray_train.py --config config/best_config/smartfallmm/kalman.yaml --num-gpus 3

python ray_train.py --config config/best_config/upfall/kalman.yaml --num-gpus 3

python ray_train.py --config config/best_config/wedafall/kalman.yaml --num-gpus 3
```

## Dataset Locations

```
other_datasets/
├── CompleteDataSet.csv
└── WEDA-FALL/dataset/
```

## Key Hyperparameters

| Dataset | Window | Stride (fall/adl) | Embed | Model |
|---------|--------|-------------------|-------|-------|
| SmartFallMM | 128 (4.3s) | 16/64 | 48 | KalmanConv1dLinear |
| UP-FALL | 160 (8.9s) | 8/32 | 48 | KalmanConv1dConv1d |
| WEDA-FALL | 250 (5.0s) | 8/32 | 48 | KalmanConv1dConv1d |

## Config Files

```
config/best_config/smartfallmm/kalman.yaml
config/best_config/upfall/kalman.yaml
config/best_config/upfall/raw.yaml
config/best_config/wedafall/kalman.yaml
config/best_config/wedafall/raw.yaml
```
