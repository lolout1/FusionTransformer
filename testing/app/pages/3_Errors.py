"""Errors page: analyze false negatives and false positives."""

import streamlit as st
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from testing.analysis import ErrorAnalyzer
from testing.visualization import probability_histogram

st.set_page_config(page_title="Errors", layout="wide")
st.title("Error Analysis")

if not st.session_state.get('predictions'):
    st.warning("Please run inference on the Dashboard page first.")
    st.stop()

windows = st.session_state.windows
predictions = st.session_state.predictions
threshold = st.session_state.get('threshold', 0.5)

analyzer = ErrorAnalyzer(threshold=threshold)

# Error summary
st.header("Error Summary")

summary = analyzer.error_summary(predictions, windows)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Errors", summary['total_errors'])
col2.metric("False Negatives", summary['fn_count'], help="Falls predicted as ADL")
col3.metric("False Positives", summary['fp_count'], help="ADLs predicted as Fall")
col4.metric("Error Rate", f"{summary['total_errors'] / len(predictions) * 100:.1f}%")

# Error type filter
st.header("Error Details")

error_type = st.radio(
    "Error Type",
    options=['All', 'FN', 'FP'],
    horizontal=True
)

filter_type = None if error_type == 'All' else error_type
errors = analyzer.get_errors(predictions, windows, error_type=filter_type)

if not errors:
    st.info("No errors of this type.")
else:
    # Convert to dataframe for display
    df_data = []
    for e in errors:
        df_data.append({
            'Index': e['idx'],
            'Type': e['type'],
            'UUID': e['uuid'][:8] + '...' if len(e['uuid']) > 8 else e['uuid'],
            'Probability': f"{e['probability']:.4f}",
            'Confidence': f"{e['confidence']:.4f}",
            'Label': e['label']
        })

    df = pd.DataFrame(df_data)

    # Sorting options
    col1, col2 = st.columns(2)
    with col1:
        sort_by = st.selectbox("Sort by", ['Index', 'Probability', 'Confidence'])
    with col2:
        ascending = st.checkbox("Ascending", value=True)

    df_sorted = df.sort_values(sort_by, ascending=ascending)
    st.dataframe(df_sorted, use_container_width=True)

    # Error details
    st.subheader("View Error Details")

    error_indices = [e['idx'] for e in errors]
    selected_idx = st.selectbox(
        "Select error index",
        options=error_indices,
        format_func=lambda x: f"Index {x} ({next(e['type'] for e in errors if e['idx'] == x)})"
    )

    if selected_idx is not None:
        selected_error = next(e for e in errors if e['idx'] == selected_idx)

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Error Details**")
            st.json({
                'index': selected_error['idx'],
                'type': selected_error['type'],
                'uuid': selected_error['uuid'],
                'probability': selected_error['probability'],
                'confidence': selected_error['confidence'],
                'label': selected_error['label'],
                'adl_type': selected_error.get('adl_type')
            })

        with col2:
            st.write("**Go to Signal Viewer**")
            st.info(f"View this window's signal on the Signals page (Index: {selected_idx})")


# Error patterns
st.header("Error Patterns")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Errors by Subject")
    if error_type == 'FN' or error_type == 'All':
        st.write("**False Negatives by Subject:**")
        for uuid, count in sorted(summary['fn_by_subject'].items(), key=lambda x: -x[1])[:10]:
            st.write(f"- {uuid[:12]}...: {count}")

with col2:
    if error_type == 'FP' or error_type == 'All':
        st.write("**False Positives by Subject:**")
        for uuid, count in sorted(summary['fp_by_subject'].items(), key=lambda x: -x[1])[:10]:
            st.write(f"- {uuid[:12]}...: {count}")

# Confidence analysis
st.subheader("Confidence Distribution")

fn_errors = [e for e in errors if e['type'] == 'FN']
fp_errors = [e for e in errors if e['type'] == 'FP']

if fn_errors or fp_errors:
    st.write(f"FN mean confidence: {summary['fn_mean_confidence']:.4f}")
    st.write(f"FP mean confidence: {summary['fp_mean_confidence']:.4f}")

    # Probability distribution for errors
    fn_probs = [e['probability'] for e in fn_errors]
    fp_probs = [e['probability'] for e in fp_errors]

    if fn_probs and fp_probs:
        fig = probability_histogram(fn_probs, fp_probs, threshold)
        fig.update_layout(title="Error Probability Distribution (FN=Fall, FP=ADL in legend)")
        st.plotly_chart(fig, use_container_width=True)
