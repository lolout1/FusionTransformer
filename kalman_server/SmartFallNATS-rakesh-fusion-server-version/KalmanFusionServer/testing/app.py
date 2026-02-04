#!/usr/bin/env python3
"""Streamlit web app for model testing and analysis.

Run with: streamlit run testing/app.py
"""

import sys
from pathlib import Path

# Add parent to path for imports
_parent = Path(__file__).parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

import streamlit as st
import numpy as np

st.set_page_config(
    page_title="Fall Detection Model Testing",
    page_icon="🔬",
    layout="wide",
)


def get_available_configs():
    """Get list of available config files."""
    configs_dir = Path(__file__).parent.parent / 'configs'
    return sorted(configs_dir.glob("s*_*.yaml"))


def get_available_data():
    """Get list of available test data files."""
    data_dirs = [
        Path(__file__).parent.parent / 'logs',
        Path(__file__).parent.parent / 'test_data',
    ]
    files = []
    for d in data_dirs:
        if d.exists():
            files.extend(d.glob("*.json"))
            files.extend(d.glob("*.parquet"))
    return files


@st.cache_resource
def load_runner(config_path: str):
    """Load and cache test runner."""
    from testing.harness.runner import TestRunner
    runner = TestRunner(config_path)
    runner.initialize()
    return runner


@st.cache_data
def load_test_data(data_path: str):
    """Load and cache test data."""
    from testing.data.loader import TestDataLoader
    loader = TestDataLoader(data_path)
    windows = loader.load_windows()
    sessions = loader.group_into_sessions(windows)
    stats = loader.get_stats()
    return windows, sessions, stats


def main():
    st.title("🔬 Fall Detection Model Testing")

    # Sidebar - Configuration
    st.sidebar.header("Configuration")

    # Config selector
    configs = get_available_configs()
    config_options = {cfg.stem: str(cfg) for cfg in configs}

    if not config_options:
        st.error("No config files found in configs/ directory")
        return

    selected_config = st.sidebar.selectbox(
        "Model Config",
        options=list(config_options.keys()),
        index=0
    )
    config_path = config_options[selected_config]

    # Data selector
    data_files = get_available_data()
    data_options = {f.name: str(f) for f in data_files}

    if not data_options:
        st.error("No test data found in logs/ or test_data/ directories")
        st.info("Expected formats: .json (prediction-data-couchbase format) or .parquet")
        return

    selected_data = st.sidebar.selectbox(
        "Test Data",
        options=list(data_options.keys()),
        index=0
    )
    data_path = data_options[selected_data]

    # Options
    st.sidebar.subheader("Options")
    use_alpha_queue = st.sidebar.checkbox("Simulate Alpha Queue", value=True)
    threshold = st.sidebar.slider("Decision Threshold", 0.0, 1.0, 0.5, 0.05)

    # Run button
    run_button = st.sidebar.button("🚀 Run Test", type="primary", use_container_width=True)

    # Main content
    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("Data Stats")
        try:
            windows, sessions, stats = load_test_data(data_path)
            st.metric("Total Windows", stats['total_windows'])
            st.metric("Fall Windows", stats['fall_windows'])
            st.metric("ADL Windows", stats['adl_windows'])
            st.metric("Unique UUIDs", stats['unique_uuids'])
            st.metric("Sessions", len(sessions))
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return

    with col1:
        st.subheader("Results")

        if run_button or 'results' in st.session_state:
            if run_button:
                with st.spinner("Running inference..."):
                    try:
                        runner = load_runner(config_path)
                        results = runner.run_all(
                            sessions,
                            use_alpha_queue=use_alpha_queue,
                            threshold=threshold,
                            verbose=False
                        )
                        st.session_state['results'] = results
                        st.session_state['windows'] = windows
                        st.session_state['predictions'] = results.window_predictions
                    except Exception as e:
                        st.error(f"Error running inference: {e}")
                        import traceback
                        st.code(traceback.format_exc())
                        return

            results = st.session_state.get('results')
            if results:
                # Metrics cards
                metrics_cols = st.columns(4)
                with metrics_cols[0]:
                    st.metric("F1 Score", f"{results.window_metrics['f1']:.2%}")
                with metrics_cols[1]:
                    st.metric("Precision", f"{results.window_metrics['precision']:.2%}")
                with metrics_cols[2]:
                    st.metric("Recall", f"{results.window_metrics['recall']:.2%}")
                with metrics_cols[3]:
                    st.metric("Accuracy", f"{results.window_metrics['accuracy']:.2%}")

                # Session metrics if alpha queue enabled
                if results.session_metrics and use_alpha_queue:
                    st.subheader("Session-Level (Alpha Queue)")
                    session_cols = st.columns(4)
                    with session_cols[0]:
                        st.metric("Session F1", f"{results.session_metrics['f1']:.2%}")
                    with session_cols[1]:
                        st.metric("Session Precision", f"{results.session_metrics['precision']:.2%}")
                    with session_cols[2]:
                        st.metric("Session Recall", f"{results.session_metrics['recall']:.2%}")
                    with session_cols[3]:
                        st.metric("Decisions", results.session_metrics['total_decisions'])

                # Visualizations
                st.subheader("Visualizations")
                viz_tabs = st.tabs(["Confusion Matrix", "Probability Distribution", "Error Analysis", "Signal Viewer"])

                with viz_tabs[0]:
                    from testing.analysis.visualization import Visualizer
                    try:
                        cm_fig = Visualizer.confusion_matrix_heatmap(
                            results.window_metrics['confusion_matrix'],
                            title=f"Confusion Matrix - {selected_config}"
                        )
                        st.plotly_chart(cm_fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating confusion matrix: {e}")

                with viz_tabs[1]:
                    from testing.analysis.metrics import MetricsCalculator
                    try:
                        dist = MetricsCalculator.probability_distribution(
                            results.window_predictions,
                            st.session_state['windows']
                        )
                        hist_fig = Visualizer.probability_histogram(
                            dist['fall_probs'],
                            dist['adl_probs'],
                            threshold=threshold
                        )
                        st.plotly_chart(hist_fig, use_container_width=True)

                        st.write(f"**Fall probs:** mean={dist['fall_mean']:.3f}, std={dist['fall_std']:.3f}")
                        st.write(f"**ADL probs:** mean={dist['adl_mean']:.3f}, std={dist['adl_std']:.3f}")
                    except Exception as e:
                        st.error(f"Error creating histogram: {e}")

                with viz_tabs[2]:
                    st.subheader("Error Analysis")

                    error_type = st.selectbox("Error Type", ["False Negatives", "False Positives"])
                    error_key = 'false_negatives' if error_type == "False Negatives" else 'false_positives'
                    error_indices = results.window_metrics.get(error_key, [])

                    st.write(f"**{error_type}:** {len(error_indices)} samples")

                    if error_indices:
                        windows_list = st.session_state['windows']
                        preds_list = st.session_state['predictions']

                        for i, idx in enumerate(error_indices[:10]):  # Show first 10
                            with st.expander(f"Sample {i+1}: idx={idx}, prob={preds_list[idx].probability:.3f}"):
                                window = windows_list[idx]
                                st.write(f"UUID: {window.uuid}")
                                st.write(f"Label: {window.label}")
                                st.write(f"Original prediction: {window.original_prediction:.3f}")

                                # Mini signal plot
                                import plotly.graph_objects as go
                                fig = go.Figure()
                                t = np.arange(128)
                                smv = np.linalg.norm(window.acc, axis=1)
                                fig.add_trace(go.Scatter(x=t, y=smv, name='SMV'))
                                fig.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0))
                                st.plotly_chart(fig, use_container_width=True)

                with viz_tabs[3]:
                    st.subheader("Signal Viewer")

                    windows_list = st.session_state.get('windows', [])
                    if windows_list:
                        sample_idx = st.number_input(
                            "Sample Index",
                            min_value=0,
                            max_value=len(windows_list) - 1,
                            value=0
                        )

                        window = windows_list[sample_idx]
                        pred = st.session_state['predictions'][sample_idx]

                        st.write(f"**UUID:** {window.uuid}")
                        st.write(f"**Label:** {window.label} | **Prediction:** {pred.probability:.3f}")

                        try:
                            sig_fig = Visualizer.signal_plot(window, pred)
                            st.plotly_chart(sig_fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error creating signal plot: {e}")

                # Latency info
                st.subheader("Performance")
                st.write(f"**Preprocessing:** {results.avg_preprocessing_ms:.2f} ms")
                st.write(f"**Inference:** {results.avg_inference_ms:.2f} ms")
                st.write(f"**Total:** {results.avg_preprocessing_ms + results.avg_inference_ms:.2f} ms per window")

        else:
            st.info("Click 'Run Test' to evaluate the selected model on the test data.")


if __name__ == "__main__":
    main()
