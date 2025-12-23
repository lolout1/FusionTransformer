"""
PhD-Level Systematic Analysis Package for Fall Detection Models.

This package provides tools for analyzing and comparing Transformer vs CNN
fall detection models on the SmartFallMM dataset.

Usage:
    from analysis import config, utils
    from analysis.config import PROJECT_ROOT, DATA_DIR, FIGURES_DIR
    from analysis.utils import load_trial_data, compute_signal_features

For portable analysis (when raw data is not available):
    - Model scores are stored in model_results/transformer and model_results/cnn
    - Pre-computed trial features can be loaded from tables/

Directory Structure:
    notebooks/analysis/
    ├── __init__.py          # This file
    ├── config.py            # Path configuration
    ├── utils.py             # Data loading & feature computation
    ├── model_results/       # Copied model scores (portable)
    │   ├── transformer/
    │   │   └── scores.csv
    │   └── cnn/
    │       └── scores.csv
    ├── figures/             # Generated figures
    ├── tables/              # Generated CSV tables
    └── trial_plots/         # Individual trial visualizations
"""

from . import config
from . import utils

__version__ = '1.0.0'
__author__ = 'SmartFall Research Group'
