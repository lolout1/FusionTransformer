# Fall Detection Testing & Analysis Framework

Offline analysis platform for evaluating fall detection models using pre-collected data.

## Features

- **Exact server replication**: Uses same preprocessing pipeline as production
- **Multiple interfaces**: Streamlit (current), FastAPI (future), CLI
- **Comprehensive metrics**: F1, precision, recall, ROC-AUC, PR-AUC, confusion matrix
- **Error analysis**: FN/FP breakdown, per-subject analysis, signal characteristics
- **Interactive visualization**: Signal plots, probability distributions, comparisons
- **Modular design**: Service layer enables easy framework migration

## Quick Start

### CLI
```bash
# Run single config
python -m testing.cli run \
    --config path/to/config.yaml \
    --data path/to/data.json

# Compare multiple configs
python -m testing.cli compare \
    --configs "configs/*.yaml" \
    --data data.json

# Launch web app
python -m testing.cli app
```

### Streamlit App
```bash
streamlit run testing/app/main.py
```

Access via `http://localhost:8501` (or server IP after VPN connection).

### Python API
```python
from testing.services import InferenceService, AnalysisService
from testing.services.inference_service import DataLoadRequest, InferenceRequest
from testing.services.analysis_service import AnalysisRequest

# Load data
inference_svc = InferenceService()
data = inference_svc.load_data(DataLoadRequest(path="data.json"))

# Run inference
predictions = inference_svc.run_inference(InferenceRequest(
    windows=data.windows,
    config_path="config.yaml",
    threshold=0.5
))

# Analyze
analysis_svc = AnalysisService()
result = analysis_svc.analyze(AnalysisRequest(
    predictions=predictions.predictions,
    windows=data.windows
))

print(f"F1: {result.metrics['f1']:.4f}")
```

## Architecture

```
testing/
├── data/           # Data loading and schema
├── analysis/       # Metrics and error analysis
├── inference/      # Pipeline and alpha queue
├── visualization/  # Plotly charts
├── services/       # Business logic (framework-agnostic)
├── app/            # Streamlit web app
├── api/            # FastAPI endpoints (future)
└── cli.py          # Command-line interface
```

### Service Layer Design

The `services/` module contains framework-agnostic business logic:

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  Streamlit  │  │   FastAPI   │  │     CLI     │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
       └────────────────┼────────────────┘
                        │
              ┌─────────▼─────────┐
              │   Service Layer   │
              │  (InferenceService│
              │   AnalysisService)│
              └─────────┬─────────┘
                        │
       ┌────────────────┼────────────────┐
       │                │                │
┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐
│    Data     │  │  Analysis   │  │  Inference  │
│   Loaders   │  │   Metrics   │  │  Pipeline   │
└─────────────┘  └─────────────┘  └─────────────┘
```

## Remote Server Access

### Via VPN
1. Connect to VPN
2. Run: `streamlit run testing/app/main.py --server.address 0.0.0.0`
3. Access: `http://<server-ip>:8501`

### Via SSH Tunnel
```bash
ssh -L 8501:localhost:8501 user@server
# Then access http://localhost:8501 locally
```

### Future: FastAPI
```bash
pip install fastapi uvicorn
uvicorn testing.api.main:app --host 0.0.0.0 --port 8000
```

## Data Formats

### JSON (prediction-data-couchbase.json)
```json
{
  "prediction_data": [
    {
      "uuid": "user-123",
      "watch_accelerometer_x": "1.0,2.0,3.0,...",
      "watch_accelerometer_y": "...",
      "watch_accelerometer_z": "...",
      "watch_gyroscope_x": "...",
      "watch_gyroscope_y": "...",
      "watch_gyroscope_z": "...",
      "label": "Fall",
      "probability": 0.85
    }
  ]
}
```

### CSV
```csv
uuid,timestamp_ms,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,label
user-123,1234567890,"[1.0,2.0,...]","[...]","[...]","[...]","[...]","[...]",Fall
```

## Dependencies

```
# Core
numpy
pandas
torch
scikit-learn

# Visualization
streamlit>=1.30.0
plotly>=5.18.0

# Future API
fastapi  # optional
uvicorn  # optional
```

## Documentation

- `docs/plans/testing-implementation.md` - Implementation details
- `docs/plans/web-app-features.md` - Feature roadmap
- `docs/TESTING.md` - Testing framework overview
- `docs/DATA_FORMAT.md` - Data format specification
