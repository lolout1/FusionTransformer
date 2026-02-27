"""Streamlit multi-page application entry point."""

import streamlit as st
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from testing.config import DEFAULT_DATA_DIR, DEFAULT_CONFIGS_DIR, DEFAULT_THRESHOLD


def main():
    st.set_page_config(
        page_title="Fall Detection Analysis",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.windows = []
        st.session_state.predictions = []
        st.session_state.metrics = {}
        st.session_state.threshold = DEFAULT_THRESHOLD
        st.session_state.config_path = None

    st.title("Fall Detection Model Analysis")

    st.markdown("""
    This application provides comprehensive analysis of fall detection models:

    - **Dashboard**: Run inference, view metrics and confusion matrix
    - **Compare**: Compare multiple model configurations
    - **Errors**: Analyze false negatives and false positives
    - **Signals**: Visualize IMU signals and predictions

    Select a page from the sidebar to begin.
    """)

    # Sidebar info
    with st.sidebar:
        st.header("Session Info")

        if st.session_state.data_loaded:
            st.success(f"Data: {len(st.session_state.windows)} windows")
            if st.session_state.predictions:
                st.success(f"Predictions: {len(st.session_state.predictions)}")
            if st.session_state.config_path:
                st.info(f"Config: {Path(st.session_state.config_path).name}")
        else:
            st.info("No data loaded. Go to Dashboard to load data.")

        st.divider()
        st.caption("Fall Detection Analysis v0.1.0")


if __name__ == "__main__":
    main()
