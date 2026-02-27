"""Dashboard page: load data, configure model, run inference, view metrics."""

import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from testing.config import DEFAULT_DATA_DIR, DEFAULT_CONFIGS_DIR, DEFAULT_THRESHOLD
from testing.data import DataLoader
from testing.analysis import MetricsCalculator, ErrorAnalyzer
from testing.visualization import (
    confusion_matrix_chart, probability_histogram,
    roc_curve_chart, pr_curve_chart, threshold_sweep_chart
)

st.set_page_config(page_title="Dashboard", layout="wide")
st.title("Fall Detection Analysis Dashboard")

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.windows = []
    st.session_state.predictions = []
    st.session_state.metrics = {}
    st.session_state.threshold = DEFAULT_THRESHOLD

# =============================================================================
# 1. DATA LOADING
# =============================================================================
st.header("1. Load Test Data")

col1, col2 = st.columns([2, 1])

with col1:
    data_path = st.text_input(
        "Data file path",
        value=str(DEFAULT_DATA_DIR / "prediction-data-couchbase.json"),
        help="Path to JSON or CSV data file"
    )

    if st.button("Load Data", type="primary"):
        try:
            loader = DataLoader(data_path)
            windows = loader.load()
            stats = loader.get_stats(windows)

            st.session_state.windows = windows
            st.session_state.data_loaded = True
            st.session_state.predictions = []

            st.success(f"Loaded {stats['total_windows']} windows")
        except Exception as e:
            st.error(f"Failed to load data: {e}")

with col2:
    if st.session_state.data_loaded:
        falls = sum(1 for w in st.session_state.windows if w.ground_truth == 1)
        adls = len(st.session_state.windows) - falls
        st.metric("Total", len(st.session_state.windows))
        st.metric("Falls", falls)
        st.metric("ADLs", adls)

# =============================================================================
# 2. MODEL CONFIGURATION
# =============================================================================
st.header("2. Model Configuration")

# Config options parsed from available configs
PREPROCESSING_MODES = {
    'kalman_gyromag': 'Kalman + Gyro Magnitude (7ch)',
    'kalman_yaw': 'Kalman + Yaw (7ch)',
    'raw_gyro': 'Raw Gyroscope (7ch)',
    'raw_gyromag': 'Raw + Gyro Magnitude (5ch)',
}

NORMALIZATION_MODES = {
    'norm': 'Normalized (acc_only)',
    'nonorm': 'No Normalization',
}

STRIDE_CONFIGS = {
    's8_16': 'Stride 8/16 (more overlap)',
    's16_32': 'Stride 16/32 (less overlap)',
}

col1, col2, col3 = st.columns(3)

with col1:
    preprocess = st.selectbox(
        "Preprocessing",
        options=list(PREPROCESSING_MODES.keys()),
        format_func=lambda x: PREPROCESSING_MODES[x],
        help="Feature extraction mode"
    )

with col2:
    normalization = st.selectbox(
        "Normalization",
        options=list(NORMALIZATION_MODES.keys()),
        format_func=lambda x: NORMALIZATION_MODES[x],
        help="Whether to normalize accelerometer channels"
    )

with col3:
    stride = st.selectbox(
        "Stride Config",
        options=list(STRIDE_CONFIGS.keys()),
        format_func=lambda x: STRIDE_CONFIGS[x],
        help="Window stride configuration"
    )

# Build config filename
config_name = f"{stride}_{preprocess}_{normalization}.yaml"
config_path = Path(DEFAULT_CONFIGS_DIR) / config_name

# Show selected config
st.info(f"**Selected config:** `{config_name}`")

if not config_path.exists():
    st.warning(f"Config file not found: {config_path}")
    # Fallback to manual selection
    config_dir = Path(DEFAULT_CONFIGS_DIR)
    if config_dir.exists():
        available = sorted([c.name for c in config_dir.glob("*.yaml")])
        if available:
            manual_config = st.selectbox("Or select available config:", available)
            config_path = config_dir / manual_config

# Threshold slider
col1, col2 = st.columns(2)
with col1:
    threshold = st.slider(
        "Classification Threshold",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.threshold,
        step=0.05,
        help="Probability threshold for fall classification"
    )
    st.session_state.threshold = threshold

# =============================================================================
# 3. ANALYSIS OPTIONS
# =============================================================================
st.header("3. Analysis Options")

analysis_options = st.multiselect(
    "Select analyses to run",
    options=['Metrics', 'Confusion Matrix', 'ROC Curve', 'PR Curve', 'Threshold Sweep', 'Error Analysis'],
    default=['Metrics', 'Confusion Matrix'],
    help="Choose which analyses to display"
)

# =============================================================================
# 4. RUN INFERENCE
# =============================================================================
st.header("4. Run Inference")

run_disabled = not st.session_state.data_loaded or not config_path.exists()

if st.button("Run Inference", type="primary", disabled=run_disabled):
    with st.spinner(f"Running inference with {config_name}..."):
        try:
            from testing.inference import InferencePipeline

            pipeline = InferencePipeline(
                config_path=str(config_path),
                threshold=threshold
            )
            predictions = pipeline.predict_batch(st.session_state.windows)

            st.session_state.predictions = predictions
            st.session_state.config_path = str(config_path)
            st.session_state.config_name = config_name
            st.success(f"Generated {len(predictions)} predictions")
        except Exception as e:
            st.error(f"Inference failed: {e}")
            st.exception(e)

# =============================================================================
# 5. RESULTS
# =============================================================================
if st.session_state.predictions:
    st.header("5. Results")

    calc = MetricsCalculator(threshold=threshold)
    windows = st.session_state.windows
    predictions = st.session_state.predictions

    # Always compute core metrics
    metrics = calc.compute_all(predictions, windows)
    st.session_state.metrics = metrics

    # Key metrics row
    if 'Metrics' in analysis_options:
        st.subheader("Classification Metrics")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("F1 Score", f"{metrics['f1']:.4f}")
        col2.metric("Precision", f"{metrics['precision']:.4f}")
        col3.metric("Recall", f"{metrics['recall']:.4f}")
        col4.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        col5.metric("Specificity", f"{metrics['specificity']:.4f}")
        if metrics.get('roc_auc'):
            col6.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")

        # Counts
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("TP", metrics['tp'])
        col2.metric("FP", metrics['fp'])
        col3.metric("TN", metrics['tn'])
        col4.metric("FN", metrics['fn'])

    # Confusion Matrix
    if 'Confusion Matrix' in analysis_options:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Confusion Matrix")
            cm = calc.compute_confusion_matrix(predictions, windows)
            fig = confusion_matrix_chart(cm)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Probability Distribution")
            fall_probs = [p.probability for p, w in zip(predictions, windows) if w.ground_truth == 1]
            adl_probs = [p.probability for p, w in zip(predictions, windows) if w.ground_truth == 0]
            fig = probability_histogram(fall_probs, adl_probs, threshold)
            st.plotly_chart(fig, use_container_width=True)

    # ROC Curve
    if 'ROC Curve' in analysis_options:
        st.subheader("ROC Curve")
        roc_data = calc.compute_roc_curve(predictions, windows)
        if roc_data['auc']:
            fig = roc_curve_chart(roc_data['fpr'], roc_data['tpr'], roc_data['auc'])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ROC curve requires both classes present.")

    # PR Curve
    if 'PR Curve' in analysis_options:
        st.subheader("Precision-Recall Curve")
        pr_data = calc.compute_pr_curve(predictions, windows)
        if pr_data['auc']:
            fig = pr_curve_chart(pr_data['precision'], pr_data['recall'], pr_data['auc'])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("PR curve requires both classes present.")

    # Threshold Sweep
    if 'Threshold Sweep' in analysis_options:
        st.subheader("Threshold Sweep")
        sweep = calc.compute_threshold_sweep(predictions, windows)
        fig = threshold_sweep_chart(sweep)
        st.plotly_chart(fig, use_container_width=True)

        # Optimal threshold
        best_thresh, best_f1 = calc.find_optimal_threshold(predictions, windows, metric='f1')
        st.info(f"**Optimal threshold for F1:** {best_thresh:.2f} (F1 = {best_f1:.4f})")

    # Error Analysis
    if 'Error Analysis' in analysis_options:
        st.subheader("Error Analysis")
        analyzer = ErrorAnalyzer(threshold=threshold)
        summary = analyzer.error_summary(predictions, windows)

        col1, col2 = st.columns(2)
        with col1:
            st.write("**False Negatives (Missed Falls)**")
            st.metric("Count", summary['fn_count'])
            st.metric("Subjects affected", summary['subjects_with_fn'])
            if summary['fn_mean_confidence'] > 0:
                st.metric("Mean confidence", f"{summary['fn_mean_confidence']:.3f}")

        with col2:
            st.write("**False Positives (False Alarms)**")
            st.metric("Count", summary['fp_count'])
            st.metric("Subjects affected", summary['subjects_with_fp'])
            if summary['fp_mean_confidence'] > 0:
                st.metric("Mean confidence", f"{summary['fp_mean_confidence']:.3f}")

    # All metrics expander
    with st.expander("All Metrics (Raw)"):
        formatted = calc.format_metrics(metrics)
        cols = st.columns(4)
        items = list(formatted.items())
        for i, (k, v) in enumerate(items):
            cols[i % 4].write(f"**{k}**: {v}")
