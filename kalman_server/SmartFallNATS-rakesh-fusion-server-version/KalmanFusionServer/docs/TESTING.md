# Testing Framework

Automated testing for fall detection models with alpha queue simulation and comprehensive analysis.

**Location**: `FusionTransformer/testing/` (standalone package)

See `docs/plans/testing-implementation.md` for full implementation details.

## Quick Start

```bash
# From FusionTransformer root directory:

# CLI - run single config
python -m testing.cli run \
    --config kalman_server/.../configs/CONFIG.yaml \
    --data kalman_server/.../logs/prediction-data-couchbase.json

# CLI - compare configs
python -m testing.cli compare \
    --configs "kalman_server/.../configs/*.yaml" \
    --data kalman_server/.../logs/prediction-data-couchbase.json

# Web UI
python -m testing.cli app
# Or directly: streamlit run testing/app/main.py
```

## CLI Commands

### list
List available configs.
```bash
python -m testing.cli list --configs-dir configs/
```

### run
Run test on single config.
```bash
python -m testing.cli run \
    --config configs/X.yaml \
    --data data.json \
    --output results.json \
    --threshold 0.5 \
    --no-queue  # Disable alpha queue
```

### compare
Compare multiple configs.
```bash
python -m testing.cli compare \
    --configs "configs/s8_16_*.yaml,configs/s16_32_*.yaml" \
    --data data.json \
    --output comparison.json
```

### app
Launch Streamlit web app.
```bash
python -m testing.cli app
```

## Alpha Queue Simulation

The framework replicates watch-side queue logic:

1. Each window prediction goes to alpha queue
2. When queue has 10 predictions:
   - Compute average probability
   - If avg > threshold: FALL → flush queue
   - If avg ≤ threshold: ADL → retain last 5

This affects final metrics vs window-level evaluation.

## Metrics

### Window-Level
- F1, Precision, Recall, Accuracy
- Confusion matrix
- ROC-AUC
- FN/FP indices

### Session-Level (with alpha queue)
- Same metrics but on aggregated decisions
- Total decisions count
- Queue flush events

## Adding Test Data

### From Couchbase Export
Use existing `logs/prediction-data-couchbase.json` format:
```json
{
  "prediction_data": [
    {
      "uuid": "...",
      "watch_accelerometer_x": "val1,val2,...",
      "watch_accelerometer_y": "...",
      "watch_accelerometer_z": "...",
      "watch_gyroscope_x": "...",
      "watch_gyroscope_y": "...",
      "watch_gyroscope_z": "...",
      "label": "Fall",
      "probability": 0.47
    }
  ]
}
```

### Parquet Format
```python
import pandas as pd
df = pd.DataFrame({
    'uuid': [...],
    'timestamp_ms': [...],
    'acc': [np.array shape (128,3), ...],
    'gyro': [np.array shape (128,3), ...],
    'label': ['Fall', 'ADL', ...],
})
df.to_parquet('test_data/my_data.parquet')
```

## Extending

### Custom Feature Extractor
Add to `preprocessing/features/`:
```python
from ..base import FeatureExtractor
from ..registry import register_feature_extractor

@register_feature_extractor("my_mode")
class MyExtractor(FeatureExtractor):
    @property
    def num_channels(self) -> int:
        return 8

    def extract(self, acc, gyro, orientations=None):
        # Return (128, 8) array
        pass
```

### Custom Metrics
Add to `testing/analysis/metrics.py`:
```python
@staticmethod
def my_metric(predictions, windows):
    # Compute and return
    pass
```
