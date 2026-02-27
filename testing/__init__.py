"""Fall detection model testing and analysis framework.

Usage:
    # CLI
    python -m testing.cli run --config CONFIG --data DATA
    python -m testing.cli compare --configs "PATTERN" --data DATA
    python -m testing.cli app

    # Python API
    from testing.data import DataLoader, TestWindow
    from testing.analysis import MetricsCalculator, ErrorAnalyzer
    from testing.inference import InferencePipeline, AlphaQueueSimulator
    from testing.visualization import plot_window, confusion_matrix_chart
"""

__version__ = "0.1.0"
