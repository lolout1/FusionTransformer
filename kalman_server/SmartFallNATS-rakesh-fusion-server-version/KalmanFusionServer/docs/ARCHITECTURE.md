# Automated Testing & Analysis System - Architecture

## Purpose

**Offline analysis tool** to evaluate fall detection models using pre-collected data, replicating exactly what happens when using the Android app in real-time. This allows testing model configurations without physically wearing the device and performing falls/ADLs.

## Design Requirements

1. **Exact Server Replication**: Must use the same preprocessing pipeline as the server (Kalman filter, feature extraction, normalization) to match Android app behavior
2. **Standalone Package**: Usable from any directory, not tied to server deployment
3. **Pre-loaded Data**: Primary use case is analyzing collected data files
4. **Real-time Support**: Optional live NATS streaming for development
5. **Comprehensive Analysis**: Deep insights into model behavior, errors, thresholds

---

## Current State Assessment

**What exists:**
- Basic data loader (JSON parsing)
- Alpha queue simulator
- Simple metrics computation
- Minimal Streamlit app (needs work)
- Basic CLI

**Critical gaps:**
1. Not truly standalone (import issues)
2. Limited error analysis
3. No threshold optimization
4. No per-subject deep dive
5. Basic visualizations only
6. No systematic comparison tools

---

## Comprehensive System Architecture

### Core Design Principles

1. **Exact Replication**: Identical preprocessing to production server
2. **Standalone**: Single directory with clear dependencies
3. **Reproducibility**: Every test run versioned and reproducible
4. **Extensibility**: Easy to add new models, metrics, visualizations
5. **Performance**: Efficient batch processing, caching

---

## System Components

### Layer 1: Data Management

```
testing/data/
├── sources/
│   ├── couchbase.py       # Direct Couchbase queries
│   ├── json_logs.py       # Existing log files
│   ├── parquet.py         # Efficient binary storage
│   └── synthetic.py       # Test data generation
├── validation/
│   ├── schema.py          # Pydantic models with validation
│   ├── quality.py         # Data quality checks
│   └── statistics.py      # Distribution analysis
├── transforms/
│   ├── windowing.py       # Sliding window generation
│   ├── augmentation.py    # Data augmentation
│   └── normalization.py   # Feature scaling
└── catalog.py             # Dataset registry and versioning
```

**Key Features:**
- Dataset versioning with checksums
- Quality gates (reject invalid data)
- Statistics tracking (drift detection)
- Lazy loading for large datasets

### Layer 2: Experiment Management

```
testing/experiments/
├── config.py              # Experiment configuration schema
├── runner.py              # Experiment execution engine
├── tracker.py             # MLflow/W&B integration
├── artifacts.py           # Model weights, scalers, configs
└── comparison.py          # Cross-experiment analysis
```

**Experiment Schema:**
```python
@dataclass
class Experiment:
    id: str                          # Unique identifier
    name: str                        # Human-readable name
    timestamp: datetime
    config: ExperimentConfig
    dataset: DatasetReference
    model: ModelReference
    results: ExperimentResults
    artifacts: list[ArtifactReference]
    tags: dict[str, str]
    parent_id: Optional[str]         # For ablations
```

**Features:**
- Automatic experiment tracking
- Parent-child relationships (ablation studies)
- Artifact versioning
- Comparison across experiments

### Layer 3: Analysis Engine

```
testing/analysis/
├── metrics/
│   ├── classification.py  # F1, precision, recall, etc.
│   ├── calibration.py     # Probability calibration
│   ├── latency.py         # Performance metrics
│   └── custom.py          # Domain-specific metrics
├── statistical/
│   ├── significance.py    # Statistical tests
│   ├── confidence.py      # Confidence intervals
│   └── bootstrap.py       # Bootstrap resampling
├── error_analysis/
│   ├── patterns.py        # Error pattern detection
│   ├── clustering.py      # Error clustering
│   └── attribution.py     # Feature attribution
├── threshold/
│   ├── optimization.py    # Optimal threshold search
│   ├── operating_points.py # ROC/PR analysis
│   └── cost_sensitive.py  # Cost-aware thresholds
└── comparative/
    ├── pairwise.py        # Model vs model
    ├── ablation.py        # Ablation analysis
    └── ranking.py         # Model ranking
```

**Key Analyses:**
1. **Per-Subject Analysis**: Identify hard subjects, personalization needs
2. **Error Pattern Mining**: Cluster similar errors, find systematic issues
3. **Threshold Sensitivity**: How metrics change with threshold
4. **Temporal Analysis**: Performance across time
5. **Feature Importance**: Which features drive predictions

### Layer 4: Visualization Library

```
testing/visualization/
├── components/
│   ├── metrics_card.py    # Summary metric displays
│   ├── confusion_matrix.py
│   ├── roc_pr_curves.py
│   ├── probability_dist.py
│   ├── signal_viewer.py   # Interactive IMU signals
│   ├── error_browser.py   # Browse errors with context
│   ├── comparison_table.py
│   ├── radar_chart.py     # Multi-metric comparison
│   └── timeline.py        # Temporal analysis
├── dashboards/
│   ├── overview.py        # High-level summary
│   ├── model_deep_dive.py # Single model analysis
│   ├── comparison.py      # Multi-model comparison
│   ├── error_analysis.py  # Error investigation
│   ├── subject_analysis.py # Per-subject breakdown
│   └── threshold_tuning.py # Interactive threshold
├── reports/
│   ├── html_report.py     # Static HTML export
│   ├── pdf_report.py      # PDF generation
│   └── latex_tables.py    # Paper-ready tables
└── themes.py              # Consistent styling
```

### Layer 5: Web Application

```
testing/app/
├── main.py                # Streamlit entry point
├── pages/
│   ├── 1_overview.py      # Dashboard home
│   ├── 2_run_test.py      # Execute new tests
│   ├── 3_results.py       # View results
│   ├── 4_compare.py       # Compare models
│   ├── 5_errors.py        # Error analysis
│   ├── 6_subjects.py      # Per-subject analysis
│   ├── 7_threshold.py     # Threshold tuning
│   └── 8_export.py        # Reports & export
├── state.py               # Session state management
├── cache.py               # Data caching
└── auth.py                # Optional authentication
```

**Page Descriptions:**

#### 1. Overview Dashboard
- Summary metrics cards (F1, Precision, Recall)
- Recent experiment history
- Model comparison quick view
- Alerts/warnings (data quality, performance regressions)

#### 2. Run Test
- Config selector (multi-select for comparison)
- Data source selector
- Options: threshold, alpha queue, verbose
- Progress tracking
- Real-time metrics update

#### 3. Results Viewer
- Experiment selector
- Full metrics breakdown
- Confusion matrix
- Probability distributions
- ROC/PR curves
- Download results

#### 4. Model Comparison
- Side-by-side metrics table
- Radar chart (multi-metric)
- Statistical significance tests
- Winner determination per metric
- Export comparison report

#### 5. Error Analysis
- Error type filter (FN, FP)
- Error browser with signals
- Error clustering visualization
- Pattern analysis
- Actionable insights

#### 6. Subject Analysis
- Subject selector
- Per-subject metrics
- Hard subject identification
- Subject-level error patterns
- Personalization recommendations

#### 7. Threshold Tuning
- Interactive threshold slider
- Real-time metrics update
- Operating point visualization
- Cost-sensitive optimization
- Optimal threshold recommendation

#### 8. Export & Reports
- Report template selector
- Format selection (HTML, PDF, LaTeX)
- Artifact export (weights, configs)
- Batch report generation

---

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                                 │
├─────────────────────────────────────────────────────────────────────┤
│  Couchbase DB  │  JSON Logs  │  Parquet Files  │  Synthetic Gen    │
└───────┬────────┴──────┬──────┴────────┬────────┴────────┬──────────┘
        │               │               │                 │
        └───────────────┴───────────────┴─────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   DATA VALIDATION     │
                    │  - Schema validation  │
                    │  - Quality checks     │
                    │  - Statistics         │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │   DATASET CATALOG     │
                    │  - Versioning         │
                    │  - Metadata           │
                    │  - Caching            │
                    └───────────┬───────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
┌───────▼───────┐     ┌─────────▼─────────┐    ┌───────▼───────┐
│  EXPERIMENT   │     │   PREPROCESSING   │    │    MODEL      │
│   CONFIG      │────▶│    PIPELINE       │───▶│   INFERENCE   │
│  - Model      │     │  - Kalman filter  │    │  - Batch      │
│  - Threshold  │     │  - Features       │    │  - Caching    │
│  - Options    │     │  - Normalization  │    │               │
└───────────────┘     └───────────────────┘    └───────┬───────┘
                                                       │
                    ┌──────────────────────────────────┘
                    │
        ┌───────────▼───────────┐
        │   ALPHA QUEUE SIM     │
        │  - Window predictions │
        │  - Session decisions  │
        │  - Queue state track  │
        └───────────┬───────────┘
                    │
        ┌───────────▼───────────┐
        │   ANALYSIS ENGINE     │
        │  - Metrics compute    │
        │  - Error analysis     │
        │  - Statistical tests  │
        └───────────┬───────────┘
                    │
        ┌───────────▼───────────┐
        │   EXPERIMENT TRACKER  │
        │  - Results storage    │
        │  - Artifact linking   │
        │  - Comparison data    │
        └───────────┬───────────┘
                    │
        ┌───────────▼───────────┐
        │   VISUALIZATION       │
        │  - Charts             │
        │  - Dashboards         │
        │  - Reports            │
        └───────────┬───────────┘
                    │
        ┌───────────▼───────────┐
        │   WEB APPLICATION     │
        │  - Streamlit UI       │
        │  - Interactive        │
        │  - Export             │
        └───────────────────────┘
```

---

## Key Workflows

### Workflow 1: Single Model Evaluation

```
1. Select config → 2. Select data → 3. Run inference
                                           ↓
4. Compute metrics ← 5. Alpha queue sim ←─┘
        ↓
6. Generate visualizations → 7. Review results → 8. Export report
```

### Workflow 2: Model Comparison

```
1. Select configs (2+) → 2. Select data → 3. Run all in parallel
                                                    ↓
4. Aggregate results → 5. Statistical comparison → 6. Rank models
        ↓
7. Comparison dashboard → 8. Identify winner → 9. Export findings
```

### Workflow 3: Error Investigation

```
1. Load results → 2. Filter errors (FN/FP) → 3. Cluster errors
                                                    ↓
4. Pattern analysis → 5. Signal visualization → 6. Root cause
        ↓
7. Actionable insights → 8. Model improvement recommendations
```

### Workflow 4: Threshold Optimization

```
1. Load results → 2. Sweep thresholds → 3. Compute metrics per threshold
                                                    ↓
4. Plot metrics vs threshold → 5. Find optimal point → 6. Validate
        ↓
7. Update config → 8. Re-evaluate → 9. Confirm improvement
```

---

## Implementation Phases

### Phase 1: Foundation (Current + Fixes)
- [x] Basic data loader
- [x] Alpha queue simulator
- [x] Simple metrics
- [ ] Fix data validation
- [ ] Add proper logging
- [ ] Unit tests

### Phase 2: Analysis Engine
- [ ] Complete metrics suite
- [ ] Statistical tests
- [ ] Error clustering
- [ ] Threshold optimization
- [ ] Per-subject analysis

### Phase 3: Visualization
- [ ] All chart components
- [ ] Interactive signal viewer
- [ ] Error browser
- [ ] Comparison visualizations

### Phase 4: Web Application
- [ ] Multi-page Streamlit app
- [ ] All dashboard pages
- [ ] Session state management
- [ ] Caching optimization

### Phase 5: Reporting & Integration
- [ ] HTML report generation
- [ ] PDF export
- [ ] LaTeX tables
- [ ] CI/CD integration

### Phase 6: Advanced Features
- [ ] Experiment tracking (MLflow)
- [ ] Data versioning (DVC)
- [ ] Automated regression detection
- [ ] Personalization analysis

---

## Technology Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| Web Framework | Streamlit | Rapid development, good for data apps |
| Visualization | Plotly | Interactive, publication-quality |
| Data Processing | Pandas, NumPy | Standard, efficient |
| Metrics | scikit-learn | Comprehensive, reliable |
| Statistics | scipy.stats | Statistical testing |
| Storage | Parquet | Efficient columnar storage |
| Caching | Streamlit cache | Built-in, simple |
| Experiment Tracking | MLflow (optional) | Industry standard |

---

## Quality Assurance

### Testing Strategy

```
tests/
├── unit/
│   ├── test_data_loader.py
│   ├── test_queue_simulator.py
│   ├── test_metrics.py
│   └── test_visualization.py
├── integration/
│   ├── test_pipeline.py
│   ├── test_experiment_flow.py
│   └── test_webapp.py
└── fixtures/
    ├── sample_data.json
    └── expected_results.json
```

### Code Quality
- Type hints throughout
- Docstrings for public APIs
- Pre-commit hooks (ruff, black)
- CI pipeline for tests

---

## File Structure (Final)

```
testing/
├── __init__.py
├── data/
│   ├── __init__.py
│   ├── schema.py          # Data models
│   ├── loader.py          # Data loading
│   ├── validation.py      # Quality checks
│   └── catalog.py         # Dataset registry
├── experiments/
│   ├── __init__.py
│   ├── config.py          # Experiment config
│   ├── runner.py          # Execution engine
│   └── tracker.py         # Result tracking
├── harness/
│   ├── __init__.py
│   ├── queue_simulator.py # Alpha queue
│   ├── evaluator.py       # Metrics
│   └── preprocessor.py    # Feature pipeline
├── analysis/
│   ├── __init__.py
│   ├── metrics.py         # All metrics
│   ├── statistical.py     # Stat tests
│   ├── errors.py          # Error analysis
│   └── threshold.py       # Threshold opt
├── visualization/
│   ├── __init__.py
│   ├── components.py      # Chart components
│   ├── dashboards.py      # Dashboard layouts
│   └── reports.py         # Report generation
├── app/
│   ├── main.py            # Streamlit entry
│   ├── pages/             # Multi-page app
│   └── state.py           # Session state
├── cli.py                 # CLI interface
└── config.py              # Global config
```

---

## Next Steps

1. **Review this architecture** - Get feedback before implementation
2. **Prioritize features** - What's critical vs nice-to-have
3. **Define MVP** - Minimum viable product scope
4. **Implement incrementally** - Phase by phase with testing
5. **Iterate based on usage** - Refine based on actual needs
