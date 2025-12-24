"""
Configuration and path management for PhD-level analysis.
All paths are relative and portable across systems.
"""

from pathlib import Path
import os
import warnings

def get_project_root():
    """
    Get the project root directory dynamically.
    Works on both cluster and local desktop.
    """
    # Start from this file's directory
    current = Path(__file__).resolve().parent

    # Walk up until we find the project root markers
    for _ in range(10):  # Max 10 levels up
        # Check for common project root markers
        markers = ['main.py', 'CLAUDE.md', 'requirements.txt', 'config']
        if any((current / m).exists() for m in markers):
            return current
        # Also check for data + Models combo
        if (current / 'data').exists() and (current / 'Models').exists():
            return current
        # Check for notebooks directory (we might be inside it)
        if current.name == 'notebooks' or current.name == 'analysis':
            current = current.parent
            continue
        current = current.parent

    # Fallback: try environment variable
    if 'FEATUREKD_ROOT' in os.environ:
        return Path(os.environ['FEATUREKD_ROOT'])

    # Final fallback: assume we're in notebooks/analysis and go up 2 levels
    return Path(__file__).resolve().parent.parent.parent


# Project root - computed dynamically
PROJECT_ROOT = get_project_root()

# ============================================================================
# DATA DIRECTORIES
# ============================================================================
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'results'

# ============================================================================
# ANALYSIS OUTPUT DIRECTORIES (relative to notebooks/analysis)
# ============================================================================
ANALYSIS_DIR = PROJECT_ROOT / 'notebooks' / 'analysis'
FIGURES_DIR = ANALYSIS_DIR / 'figures'
TABLES_DIR = ANALYSIS_DIR / 'tables'
TRIAL_PLOTS_DIR = ANALYSIS_DIR / 'trial_plots'

# Local results directory (for portable analysis - copied model scores)
LOCAL_RESULTS_DIR = ANALYSIS_DIR / 'model_results'

# Dec 22 results (primary for this analysis)
RESULTS_DEC22_DIR = ANALYSIS_DIR / 'results_dec22'

# Ablation study results (portable copies)
RESULTS_ABLATIONS_DIR = ANALYSIS_DIR / 'results_ablations'

# ============================================================================
# ABLATION RESULT PATHS
# ============================================================================
ABLATION_PATHS = {
    'normalization': RESULTS_ABLATIONS_DIR / 'normalization',
    'best_model': RESULTS_ABLATIONS_DIR / 'best_model',
    'cnn_loss': RESULTS_ABLATIONS_DIR / 'cnn_loss',
    'architecture': RESULTS_ABLATIONS_DIR / 'architecture',
    'stream': RESULTS_ABLATIONS_DIR / 'stream',
    'kalman': RESULTS_ABLATIONS_DIR / 'kalman',
    'pos_enc': RESULTS_ABLATIONS_DIR / 'pos_enc',
    'se_module': RESULTS_ABLATIONS_DIR / 'se_module',
}

def get_ablation_path(ablation_name):
    """Get path to ablation results directory."""
    if ablation_name not in ABLATION_PATHS:
        raise ValueError(f"Unknown ablation: {ablation_name}. Available: {list(ABLATION_PATHS.keys())}")
    return ABLATION_PATHS[ablation_name]

def list_ablation_conditions(ablation_name):
    """List all conditions in an ablation study."""
    path = get_ablation_path(ablation_name)
    if not path.exists():
        return []
    return [d.name for d in path.iterdir() if d.is_dir()]

# ============================================================================
# MODEL RESULT PATHS
# ============================================================================
# These are the experiment result directories
# Will fallback to local copies if cluster paths don't exist

def get_transformer_results():
    """Get path to transformer model results."""
    # Try results_dec22 first (portable)
    dec22_path = RESULTS_DEC22_DIR / 'transformer'
    if dec22_path.exists():
        return dec22_path

    # Try cluster path
    cluster_path = RESULTS_DIR / 'normalization_ablation_20251222' / 'B_acc_only'
    if cluster_path.exists():
        return cluster_path

    # Try legacy local copy
    local_path = LOCAL_RESULTS_DIR / 'transformer'
    if local_path.exists():
        return local_path

    warnings.warn(f"Transformer results not found.")
    return dec22_path  # Return dec22 path for creation


def get_cnn_results():
    """Get path to CNN model results."""
    # Try results_dec22 first (portable)
    dec22_path = RESULTS_DEC22_DIR / 'cnn'
    if dec22_path.exists():
        return dec22_path

    # Try cluster path
    cluster_path = RESULTS_DIR / 'cnn_loss_comparison_20251222_073714' / 'cnn_kalman_focal'
    if cluster_path.exists():
        return cluster_path

    # Try legacy local copy
    local_path = LOCAL_RESULTS_DIR / 'cnn'
    if local_path.exists():
        return local_path

    warnings.warn(f"CNN results not found.")
    return dec22_path


# For backward compatibility
TRANSFORMER_RESULTS = get_transformer_results()
CNN_RESULTS = get_cnn_results()

# ============================================================================
# ACTIVITY DEFINITIONS
# ============================================================================
FALL_ACTIVITIES = {
    10: 'Backfall',
    11: 'Frontfall',
    12: 'Leftfall',
    13: 'Rightfall',
    14: 'Rotatefall'
}

ADL_ACTIVITIES = {
    1: 'Drinking',
    2: 'PickUp',
    3: 'Jacket',
    5: 'Sweeping',
    6: 'Washing',
    7: 'Waving',
    8: 'Walking',
    9: 'SitStand'
}

ALL_ACTIVITIES = {**FALL_ACTIVITIES, **ADL_ACTIVITIES}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_directories():
    """Create all necessary output directories."""
    for d in [FIGURES_DIR, TABLES_DIR, TRIAL_PLOTS_DIR, LOCAL_RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def check_data_availability():
    """Check what data is available and print status."""
    print("=" * 60)
    print("DATA AVAILABILITY CHECK")
    print("=" * 60)
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"  Exists: {PROJECT_ROOT.exists()}")

    print(f"\nData Directory: {DATA_DIR}")
    print(f"  Exists: {DATA_DIR.exists()}")
    if DATA_DIR.exists():
        young = DATA_DIR / 'young'
        old = DATA_DIR / 'old'
        print(f"  Young data: {young.exists()}")
        print(f"  Old data: {old.exists()}")

    print(f"\nResults Directory: {RESULTS_DIR}")
    print(f"  Exists: {RESULTS_DIR.exists()}")

    trans_path = get_transformer_results()
    cnn_path = get_cnn_results()

    print(f"\nTransformer Results: {trans_path}")
    print(f"  Exists: {trans_path.exists()}")
    if trans_path.exists():
        scores = trans_path / 'scores.csv'
        print(f"  scores.csv: {scores.exists()}")

    print(f"\nCNN Results: {cnn_path}")
    print(f"  Exists: {cnn_path.exists()}")
    if cnn_path.exists():
        scores = cnn_path / 'scores.csv'
        print(f"  scores.csv: {scores.exists()}")

    print(f"\nLocal Results (for portable analysis): {LOCAL_RESULTS_DIR}")
    print(f"  Exists: {LOCAL_RESULTS_DIR.exists()}")

    print(f"\nAblation Results: {RESULTS_ABLATIONS_DIR}")
    print(f"  Exists: {RESULTS_ABLATIONS_DIR.exists()}")
    if RESULTS_ABLATIONS_DIR.exists():
        for name, path in ABLATION_PATHS.items():
            conditions = list_ablation_conditions(name) if path.exists() else []
            status = f"{len(conditions)} conditions" if conditions else "EMPTY"
            print(f"  {name}: {status}")

    print("=" * 60)


def get_scores_path(model='transformer'):
    """
    Get path to scores.csv for a model, with fallback to local copy.

    Args:
        model: 'transformer' or 'cnn'

    Returns:
        Path to scores.csv
    """
    if model == 'transformer':
        results_dir = get_transformer_results()
    else:
        results_dir = get_cnn_results()

    scores_path = results_dir / 'scores.csv'

    if not scores_path.exists():
        # Check local
        local_path = LOCAL_RESULTS_DIR / model / 'scores.csv'
        if local_path.exists():
            return local_path

    return scores_path


if __name__ == '__main__':
    check_data_availability()
