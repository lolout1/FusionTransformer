# KalmanFusionServer - Codebase Guide

**Updated: 2026-02-04**

## Overview

Fall detection inference server with NATS.io messaging, Kalman filtering, and modular preprocessing.

**Note**: The standalone testing framework is now at `FusionTransformer/testing/`. See `docs/plans/testing-implementation.md` for details.

## Directory Structure

```
KalmanFusionServer/
├── server.py                 # Main server (NATS-based)
├── config/                   # Configuration schema
│   └── schema.py
├── configs/                  # YAML configs per model variant
├── models/                   # Model architectures
│   ├── kalman_conv1d.py      # KalmanConv1dConv1d (dual-stream)
│   └── single_stream_se.py   # SingleStreamTransformerSE
├── preprocessing/            # Feature extraction pipeline
│   ├── pipeline.py           # Main orchestrator
│   ├── features/             # Feature extractors (kalman, raw, etc.)
│   ├── filters/              # Kalman filter
│   ├── normalizers/          # Normalization (acc_only, none)
│   └── state/                # Per-user state management
├── weights/                  # Model weights (.pth)
├── logs/                     # Prediction logs
└── docs/                     # Documentation
    └── plans/                # Implementation plans
```

## Key Concepts

### Preprocessing Pipeline

1. **Gyro conversion**: deg/s → rad/s (if needed)
2. **Kalman filter**: Fuse acc+gyro → orientation (roll, pitch, yaw)
3. **Feature extraction**: Mode-dependent (kalman, raw, raw_gyromag)
4. **Normalization**: acc_only or none

### Feature Modes

| Mode | Channels | Format |
|------|----------|--------|
| kalman | 7 | smv, acc_xyz, roll, pitch, yaw |
| kalman_gyro_mag | 7 | smv, acc_xyz, gyro_mag, roll, pitch |
| raw | 7 | smv, acc_xyz, gyro_xyz |
| raw_gyromag | 5 | smv, acc_xyz, gyro_mag |

### Alpha Queue (Watch-side)

- Beta queue: 128 frames → 1 prediction
- Alpha queue: 10 predictions → average
- If avg > 0.5: FALL (flush queue)
- If avg ≤ 0.5: ADL (retain last 5)

## Usage

### Server
```bash
python server.py --config configs/s8_16_kalman_gyromag_norm.yaml
```

### Testing
```bash
# CLI
python -m testing.cli run --config configs/X.yaml --data logs/prediction-data-couchbase.json

# Web UI
streamlit run testing/app.py
```

### Adding Models

1. Add weights to `weights/{stride}_{input}_{norm}.pth`
2. Generate config: `python tools/generate_server_configs.py --stride sX_Y`
3. Test: `python test_models.py --config configs/X.yaml`

## Files to Never Delete

- `server.py` - Main server
- `config/schema.py` - Config dataclasses
- `preprocessing/pipeline.py` - Processing pipeline
- `models/__init__.py` - Model registry

## Dependencies

```
torch>=2.0
numpy
nats-py
pyyaml
scikit-learn
streamlit  # For testing app
plotly     # For visualizations
```
