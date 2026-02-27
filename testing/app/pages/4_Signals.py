"""Signals page: visualize IMU signals and predictions."""

import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from testing.visualization import plot_window, plot_comparison

st.set_page_config(page_title="Signals", layout="wide")
st.title("Signal Viewer")

if not st.session_state.get('data_loaded', False):
    st.warning("Please load data on the Dashboard page first.")
    st.stop()

windows = st.session_state.windows
predictions = st.session_state.get('predictions', [])

# Filter options
st.header("Filter")

col1, col2, col3 = st.columns(3)

with col1:
    filter_type = st.selectbox(
        "Class",
        options=['All', 'Fall', 'ADL']
    )

with col2:
    if predictions:
        error_filter = st.selectbox(
            "Error Type",
            options=['All', 'TP', 'TN', 'FP', 'FN']
        )
    else:
        error_filter = 'All'

with col3:
    uuids = list(set(w.uuid for w in windows))
    uuid_filter = st.selectbox(
        "Subject (UUID)",
        options=['All'] + sorted(uuids)
    )

# Apply filters
filtered_indices = []
for i, w in enumerate(windows):
    # Class filter
    if filter_type != 'All':
        if filter_type == 'Fall' and w.ground_truth != 1:
            continue
        if filter_type == 'ADL' and w.ground_truth != 0:
            continue

    # UUID filter
    if uuid_filter != 'All' and w.uuid != uuid_filter:
        continue

    # Error filter
    if error_filter != 'All' and predictions:
        pred = predictions[i]
        if pred.error_type != error_filter:
            continue

    filtered_indices.append(i)

st.write(f"Showing {len(filtered_indices)} of {len(windows)} windows")

if not filtered_indices:
    st.info("No windows match the current filters.")
    st.stop()

# Window selection
st.header("View Window")

selected_idx = st.selectbox(
    "Select window index",
    options=filtered_indices,
    format_func=lambda x: f"Index {x} - {windows[x].label} ({windows[x].uuid[:8]}...)"
)

# Display selected window
if selected_idx is not None:
    window = windows[selected_idx]
    pred = predictions[selected_idx] if predictions else None

    # Metadata
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Index", selected_idx)
    col2.metric("Label", window.label)
    col3.metric("UUID", window.uuid[:12] + "...")
    if pred:
        col4.metric("Probability", f"{pred.probability:.4f}")

    # Signal plot
    st.subheader("IMU Signals")
    show_smv = st.checkbox("Show SMV", value=True)
    fig = plot_window(window, prediction=pred, show_smv=show_smv)
    st.plotly_chart(fig, use_container_width=True)

    # Raw data expander
    with st.expander("Raw Data"):
        import pandas as pd
        import numpy as np

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Accelerometer (m/s²)**")
            df_acc = pd.DataFrame(window.acc, columns=['X', 'Y', 'Z'])
            df_acc['SMV'] = np.linalg.norm(window.acc, axis=1)
            st.dataframe(df_acc.describe())

        with col2:
            st.write("**Gyroscope (rad/s)**")
            df_gyro = pd.DataFrame(window.gyro, columns=['X', 'Y', 'Z'])
            st.dataframe(df_gyro.describe())


# Comparison mode
st.header("Compare Windows")

st.write("Select multiple windows to compare side-by-side")

compare_indices = st.multiselect(
    "Select windows to compare",
    options=filtered_indices[:50],  # Limit to first 50 for performance
    max_selections=4,
    format_func=lambda x: f"Index {x} - {windows[x].label}"
)

if len(compare_indices) >= 2:
    compare_windows = [windows[i] for i in compare_indices]
    compare_labels = [f"[{i}] {windows[i].label}" for i in compare_indices]

    fig = plot_comparison(compare_windows, labels=compare_labels)
    st.plotly_chart(fig, use_container_width=True)

# Navigation helpers
st.header("Quick Navigation")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("← Previous"):
        current_pos = filtered_indices.index(selected_idx)
        if current_pos > 0:
            st.session_state.signal_idx = filtered_indices[current_pos - 1]
            st.rerun()

with col2:
    if st.button("Next →"):
        current_pos = filtered_indices.index(selected_idx)
        if current_pos < len(filtered_indices) - 1:
            st.session_state.signal_idx = filtered_indices[current_pos + 1]
            st.rerun()

with col3:
    if predictions:
        fn_indices = [i for i in filtered_indices if predictions[i].error_type == 'FN']
        if fn_indices and st.button(f"Next FN ({len(fn_indices)})"):
            st.session_state.signal_idx = fn_indices[0]
            st.rerun()

with col4:
    if predictions:
        fp_indices = [i for i in filtered_indices if predictions[i].error_type == 'FP']
        if fp_indices and st.button(f"Next FP ({len(fp_indices)})"):
            st.session_state.signal_idx = fp_indices[0]
            st.rerun()
