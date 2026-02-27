"""Compare page: compare multiple model configurations."""

import streamlit as st
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from testing.config import DEFAULT_CONFIGS_DIR
from testing.analysis import MetricsCalculator
from testing.visualization import metrics_bar_chart, metrics_radar_chart

st.set_page_config(page_title="Compare", layout="wide")
st.title("Model Comparison")

if not st.session_state.get('data_loaded', False):
    st.warning("Please load data on the Dashboard page first.")
    st.stop()

# Config selection
st.header("Select Configurations")

config_dir = Path(DEFAULT_CONFIGS_DIR)
if config_dir.exists():
    all_configs = sorted(config_dir.glob("*.yaml"))
    config_names = [c.name for c in all_configs]
else:
    config_names = []

selected_configs = st.multiselect(
    "Select configs to compare",
    config_names,
    default=config_names[:3] if len(config_names) >= 3 else config_names
)

threshold = st.slider(
    "Classification threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)

# Run comparison
if st.button("Run Comparison", type="primary", disabled=len(selected_configs) < 2):
    results = {}

    progress = st.progress(0)
    status = st.empty()

    for i, config_name in enumerate(selected_configs):
        status.text(f"Running {config_name}...")
        progress.progress((i + 1) / len(selected_configs))

        try:
            from testing.inference import InferencePipeline

            config_path = config_dir / config_name
            pipeline = InferencePipeline(str(config_path), threshold=threshold)
            predictions = pipeline.predict_batch(st.session_state.windows)

            calc = MetricsCalculator(threshold=threshold)
            metrics = calc.compute_all(predictions, st.session_state.windows)
            results[config_name] = metrics
        except Exception as e:
            st.error(f"Failed for {config_name}: {e}")
            results[config_name] = {'error': str(e)}

    status.text("Complete!")
    st.session_state.comparison_results = results

# Display results
if 'comparison_results' in st.session_state and st.session_state.comparison_results:
    results = st.session_state.comparison_results

    # Filter out errors
    valid_results = {k: v for k, v in results.items() if 'error' not in v}

    if not valid_results:
        st.error("No valid results to display.")
        st.stop()

    st.header("Comparison Results")

    # Metrics table
    st.subheader("Metrics Table")

    metrics_to_show = ['f1', 'precision', 'recall', 'accuracy', 'specificity', 'roc_auc']
    df_data = []

    for config, metrics in valid_results.items():
        row = {'Config': config}
        for m in metrics_to_show:
            val = metrics.get(m)
            row[m.upper()] = f"{val:.4f}" if val is not None else "N/A"
        df_data.append(row)

    df = pd.DataFrame(df_data)

    # Highlight best values
    st.dataframe(df, use_container_width=True)

    # Bar chart
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Bar Chart Comparison")
        fig = metrics_bar_chart(valid_results, metrics=['f1', 'precision', 'recall', 'accuracy'])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Radar Chart")
        fig = metrics_radar_chart(valid_results)
        st.plotly_chart(fig, use_container_width=True)

    # Best config
    st.subheader("Best Configuration")
    f1_scores = {k: v.get('f1', 0) for k, v in valid_results.items()}
    best_config = max(f1_scores, key=f1_scores.get)
    st.success(f"**{best_config}** has the highest F1 score: {f1_scores[best_config]:.4f}")

    # Detailed comparison expander
    with st.expander("Detailed Metrics"):
        for config, metrics in valid_results.items():
            st.write(f"### {config}")
            col1, col2, col3 = st.columns(3)
            items = [(k, v) for k, v in metrics.items() if not isinstance(v, (dict, list))]
            n = len(items) // 3 + 1
            for i, col in enumerate([col1, col2, col3]):
                for k, v in items[i*n:(i+1)*n]:
                    if isinstance(v, float):
                        col.write(f"**{k}**: {v:.4f}")
                    else:
                        col.write(f"**{k}**: {v}")
