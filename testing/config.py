"""Configuration for testing framework."""

from pathlib import Path
import sys

# Paths
ROOT_DIR = Path(__file__).parent.parent
SERVER_DIR = ROOT_DIR / "kalman_server/SmartFallNATS-rakesh-fusion-server-version/KalmanFusionServer"
DEFAULT_DATA_DIR = SERVER_DIR / "logs"
DEFAULT_CONFIGS_DIR = SERVER_DIR / "configs"


def setup_server_imports():
    """Add server directory to path for preprocessing imports."""
    if str(SERVER_DIR) not in sys.path:
        sys.path.insert(0, str(SERVER_DIR))


# Analysis defaults
DEFAULT_THRESHOLD = 0.5
THRESHOLD_SWEEP_RANGE = (0.1, 0.9)
THRESHOLD_SWEEP_STEP = 0.05

# Alpha queue settings
ALPHA_QUEUE_SIZE = 10
ALPHA_QUEUE_RETAIN = 5

# Visualization
CHART_HEIGHT = 400
COLOR_FALL = "red"
COLOR_ADL = "blue"
