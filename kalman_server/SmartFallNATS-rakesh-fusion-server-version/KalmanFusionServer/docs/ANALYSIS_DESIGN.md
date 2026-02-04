# Analysis Platform Design

## Purpose

Comprehensive analysis platform for fall detection model evaluation. Not just metrics visualization, but deep analysis to understand **why** models behave the way they do.

---

## Analysis Capabilities

### 1. Core Metrics

**Classification Metrics:**
- Accuracy, F1 Score, Precision, Recall
- Specificity (True Negative Rate), Sensitivity (True Positive Rate)
- TP, FP, TN, FN counts
- Matthews Correlation Coefficient
- Balanced Accuracy

**Probability Metrics:**
- ROC-AUC, PR-AUC
- Brier Score (calibration)
- Log Loss

**Per-Class Metrics:**
- Fall-specific: Precision, Recall, F1
- ADL-specific: Precision, Recall, F1

### 2. Error Analysis

**Error Categorization:**
- False Negatives: Actual falls predicted as ADL
- False Positives: Actual ADLs predicted as falls
- Error rate by subject (UUID)
- Error rate by confidence level

**Error Patterns:**
- Common signal characteristics in errors
- Temporal patterns (time of day, session position)
- Subject-specific error patterns

### 3. Signal Analysis

**Characteristic Comparison:**
- Mean/std acceleration by class (Fall vs ADL)
- Mean/std gyroscope by class
- SMV distribution by class
- Peak detection (fall impact signature)

**Feature Distribution:**
- Kalman-filtered orientation distributions
- Pre-fall vs post-fall signal patterns
- ADL subtypes (walking, sitting, etc.) if labeled

### 4. Model Behavior Analysis

**Confidence Analysis:**
- Probability distribution by class
- Confidence calibration curve
- Decision boundary analysis

**Threshold Sensitivity:**
- Metrics vs threshold sweep
- Optimal threshold identification
- Operating point selection (ROC/PR curve)

### 5. Comparative Analysis

**Multi-Config Comparison:**
- Side-by-side metrics table
- Radar chart (multi-metric)
- Statistical significance testing
- Ranking by metric

**Ablation Support:**
- Group configs by parameter (stride, norm, input type)
- Parameter effect analysis

---

## Web App Structure

### Page: Dashboard
- Config selector (single or multi)
- Data file selector (multi-file support)
- Quick metrics summary
- Run/refresh button
- Recent results history

### Page: Metrics
- Full metrics table
- Confusion matrix (interactive)
- ROC curve, PR curve
- Threshold selector with live metrics update
- Export metrics button

### Page: Comparison
- Config multi-selector
- Comparison table (sortable by any metric)
- Radar chart
- Bar chart by metric
- Best config identification

### Page: Error Analysis
- Error filter (FN, FP, by subject, by confidence)
- Error list with expandable details
- Error clustering visualization
- Common patterns summary
- Signal viewer integration

### Page: Signal Analysis
- Sample browser (filter by class, error type, subject)
- Interactive signal plot:
  - Accelerometer (X, Y, Z)
  - Gyroscope (X, Y, Z)
  - SMV
  - Kalman features (roll, pitch, yaw/gyromag)
- Prediction overlay
- Comparison mode (Fall vs ADL side-by-side)

### Page: Feature Analysis
- Feature distribution plots
- Class separation visualization
- Feature importance (if available)
- Correlation analysis

### Page: Threshold Tuning
- Interactive threshold slider
- Live metrics update
- Operating point visualization
- Cost-sensitive optimization
- Recommendation display

### Page: Export
- Report generation (HTML, PDF)
- Results export (JSON, CSV)
- Configuration export
- Comparison report

---

## Data Architecture

### TestWindow
```python
@dataclass
class TestWindow:
    uuid: str
    timestamp_ms: int
    acc: np.ndarray           # (128, 3)
    gyro: np.ndarray          # (128, 3)
    label: str                # Ground truth: "Fall" or "ADL"
    adl_type: str             # Optional: "walking", "sitting", etc.
    metadata: dict            # Additional info
```

### PredictionResult
```python
@dataclass
class PredictionResult:
    window_id: int
    probability: float
    predicted_label: str
    ground_truth: str
    is_correct: bool
    error_type: str           # TP, FP, TN, FN, or None
    features: np.ndarray      # Preprocessed features
    confidence: float         # abs(prob - 0.5) * 2
```

### AnalysisResult
```python
@dataclass
class AnalysisResult:
    config_name: str
    timestamp: datetime

    # Core metrics
    metrics: dict             # All computed metrics
    confusion_matrix: np.ndarray

    # Per-sample results
    predictions: list[PredictionResult]

    # Analysis data
    error_indices: dict       # {'FN': [...], 'FP': [...]}
    per_subject_metrics: dict # {uuid: metrics}
    probability_distribution: dict

    # Curves
    roc_data: dict            # {fpr, tpr, thresholds}
    pr_data: dict             # {precision, recall, thresholds}
    threshold_sweep: dict     # {threshold: metrics}
```

---

## Modular Design

### Adding New Metrics
```python
# analysis/metrics.py
METRIC_REGISTRY = {}

def register_metric(name):
    def decorator(fn):
        METRIC_REGISTRY[name] = fn
        return fn
    return decorator

@register_metric('matthews_cc')
def matthews_correlation(y_true, y_pred):
    from sklearn.metrics import matthews_corrcoef
    return matthews_corrcoef(y_true, y_pred)
```

### Adding New Data Format
```python
# data/loader.py
FORMAT_REGISTRY = {}

def register_format(suffix):
    def decorator(cls):
        FORMAT_REGISTRY[suffix] = cls
        return cls
    return decorator

@register_format('.csv')
class CSVLoader:
    @staticmethod
    def load(path: str) -> list[TestWindow]:
        # Implementation
        pass
```

### Adding New Visualization
```python
# visualization/charts.py
CHART_REGISTRY = {}

def register_chart(name):
    def decorator(fn):
        CHART_REGISTRY[name] = fn
        return fn
    return decorator

@register_chart('probability_histogram')
def probability_histogram(results: AnalysisResult, **kwargs):
    # Return Plotly figure
    pass
```

---

## Configuration

```yaml
# testing_config.yaml
server_path: "kalman_server/SmartFallNATS-rakesh-fusion-server-version/KalmanFusionServer"

data:
  default_path: "logs/prediction-data-couchbase.json"
  supported_formats: [".json", ".csv", ".parquet"]

analysis:
  default_threshold: 0.5
  threshold_sweep_range: [0.1, 0.9]
  threshold_sweep_step: 0.05

visualization:
  color_scheme: "plotly"
  chart_height: 400
  export_format: "html"
```

---

## Implementation Priority

### Phase 1: Foundation
1. Clean up existing code
2. Proper import structure
3. Core data loading (JSON, CSV)
4. Basic inference pipeline

### Phase 2: Analysis Engine
1. All metrics computation
2. Error analysis
3. Signal analysis utilities
4. Threshold optimization

### Phase 3: Web App (Core)
1. Dashboard page
2. Metrics page
3. Error analysis page
4. Signal viewer page

### Phase 4: Web App (Advanced)
1. Comparison page
2. Feature analysis page
3. Threshold tuning page
4. Export page

### Phase 5: Polish
1. Documentation
2. Error handling
3. Performance optimization
4. Testing
