# Testing Framework Implementation Plan

## Overview

Standalone testing and analysis platform for fall detection models. Located at `FusionTransformer/testing/`.

## Completed Components

### Phase 1: Core Data Layer ✅
1. `data/schema.py` - TestWindow, TestSession, PredictionResult, AnalysisResult dataclasses
2. `data/loader.py` - JSON/CSV format loaders with auto-detection
3. `data/__init__.py` - Package exports

### Phase 2: Analysis Engine ✅
4. `analysis/metrics.py` - MetricsCalculator with 23+ metrics, curves, threshold sweep
5. `analysis/errors.py` - ErrorAnalyzer with FN/FP analysis, signal characteristics
6. `analysis/__init__.py` - Package exports

### Phase 3: Inference Layer ✅
7. `inference/queue.py` - AlphaQueueSimulator (10-window rolling average)
8. `inference/pipeline.py` - InferencePipeline wrapping server preprocessing
9. `inference/__init__.py` - Package exports

### Phase 4: Visualization ✅
10. `visualization/charts.py` - Plotly charts (confusion matrix, histograms, ROC, PR, radar)
11. `visualization/signals.py` - IMU signal plots (window, session, comparison)
12. `visualization/__init__.py` - Package exports

### Phase 5: Streamlit App ✅
13. `app/main.py` - Multi-page entry point
14. `app/pages/1_Dashboard.py` - Load data, run inference, view metrics
15. `app/pages/2_Compare.py` - Compare multiple configurations
16. `app/pages/3_Errors.py` - Analyze FN/FP errors
17. `app/pages/4_Signals.py` - Interactive signal viewer

### Phase 6: CLI ✅
18. `cli.py` - Command-line interface (run, compare, app, list)

## Usage

### CLI
```bash
# Run single config
python -m testing.cli run --config CONFIG.yaml --data data.json

# Compare multiple configs
python -m testing.cli compare --configs "configs/*.yaml" --data data.json

# Launch web app
python -m testing.cli app --port 8501
```

### Python API
```python
from testing.data import DataLoader, TestWindow
from testing.analysis import MetricsCalculator, ErrorAnalyzer
from testing.inference import InferencePipeline, AlphaQueueSimulator

# Load data
loader = DataLoader("path/to/data.json")
windows = loader.load()

# Run inference
pipeline = InferencePipeline("config.yaml")
predictions = pipeline.predict_batch(windows)

# Compute metrics
calc = MetricsCalculator(threshold=0.5)
metrics = calc.compute_all(predictions, windows)
```

### Streamlit App
```bash
streamlit run testing/app/main.py
```

## Directory Structure

```
FusionTransformer/testing/
├── __init__.py              # Package entry
├── config.py                # Path configuration
├── cli.py                   # CLI interface
├── data/
│   ├── __init__.py
│   ├── schema.py            # Data models
│   └── loader.py            # Format loaders
├── analysis/
│   ├── __init__.py
│   ├── metrics.py           # Metrics computation
│   └── errors.py            # Error analysis
├── inference/
│   ├── __init__.py
│   ├── pipeline.py          # Server preprocessing wrapper
│   └── queue.py             # Alpha queue simulation
├── visualization/
│   ├── __init__.py
│   ├── charts.py            # Plotly metrics charts
│   └── signals.py           # IMU signal plots
└── app/
    ├── __init__.py
    ├── main.py              # Streamlit entry
    └── pages/
        ├── 1_Dashboard.py
        ├── 2_Compare.py
        ├── 3_Errors.py
        └── 4_Signals.py
```

## Dependencies

```
streamlit>=1.30.0
plotly>=5.18.0
pandas>=2.0.0
numpy
scikit-learn>=1.3.0
torch>=2.0.0
```

## Verification

1. Import test: `python -c "from testing.inference import InferencePipeline"`
2. Data loading: Load JSON, verify window count
3. Metrics: Run inference, verify F1/precision/recall
4. App: `streamlit run testing/app/main.py`
