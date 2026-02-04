from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
import os
import yaml


class FeatureMode(str, Enum):
    KALMAN = "kalman"
    KALMAN_GYRO_MAG = "kalman_gyro_mag"
    RAW = "raw"
    RAW_GYROMAG = "raw_gyromag"  # 5ch: [smv, ax, ay, az, gyro_mag] (added 2026-02-04)
    ACC_ONLY = "acc_only"


class NormalizationMode(str, Enum):
    NONE = "none"
    ALL = "all"
    ACC_ONLY = "acc_only"


class FilterType(str, Enum):
    LINEAR = "linear"


@dataclass
class KalmanConfig:
    filter_type: FilterType = FilterType.LINEAR
    Q_orientation: float = 0.005
    Q_rate: float = 0.01
    R_acc: float = 0.05
    R_gyro: float = 0.1


@dataclass
class PreprocessingConfig:
    feature_mode: FeatureMode = FeatureMode.KALMAN
    normalization_mode: NormalizationMode = NormalizationMode.ACC_ONLY
    scaler_path: Optional[str] = "weights/acc_scaler.pkl"
    convert_gyro_to_rad: bool = True


@dataclass
class StateConfig:
    timeout_ms: int = 10000
    cache_ttl_ms: int = 60000
    enable_incremental: bool = True


@dataclass
class ModelConfig:
    architecture: str = "KalmanBalancedFlexible"  # Model class name (added 2026-02-04)
    weights_path: str = "weights/best_model.pth"
    device: str = "cpu"
    imu_frames: int = 128
    imu_channels: int = 7
    embed_dim: int = 48  # Changed from 64 to match ablation (2026-02-04)
    acc_ratio: float = 0.65  # Changed from 0.5 to match ablation (2026-02-04)
    num_heads: int = 4
    num_layers: int = 2


@dataclass
class ServerConfig:
    nats_url: str = "nats://localhost:4222"
    subject_pattern: str = "m.kalman_transformer.*"
    default_fs_hz: float = 30.0
    window_size: int = 128

    kalman: KalmanConfig = field(default_factory=KalmanConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    state: StateConfig = field(default_factory=StateConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def apply_env_overrides(self) -> "ServerConfig":
        if os.environ.get("DEVICE"):
            self.model.device = os.environ["DEVICE"]
        if os.environ.get("MODEL_PATH"):
            self.model.weights_path = os.environ["MODEL_PATH"]
        if os.environ.get("SCALER_PATH"):
            self.preprocessing.scaler_path = os.environ["SCALER_PATH"]
        if os.environ.get("NATS_URL"):
            self.nats_url = os.environ["NATS_URL"]
        return self


def _dict_to_config(d: dict) -> ServerConfig:
    kalman_dict = d.get("kalman", {})
    kalman = KalmanConfig(
        filter_type=FilterType(kalman_dict.get("filter_type", "linear")),
        Q_orientation=kalman_dict.get("Q_orientation", 0.005),
        Q_rate=kalman_dict.get("Q_rate", 0.01),
        R_acc=kalman_dict.get("R_acc", 0.05),
        R_gyro=kalman_dict.get("R_gyro", 0.1),
    )

    prep_dict = d.get("preprocessing", {})
    preprocessing = PreprocessingConfig(
        feature_mode=FeatureMode(prep_dict.get("feature_mode", "kalman")),
        normalization_mode=NormalizationMode(prep_dict.get("normalization_mode", "acc_only")),
        scaler_path=prep_dict.get("scaler_path", "weights/acc_scaler.pkl"),
        convert_gyro_to_rad=prep_dict.get("convert_gyro_to_rad", True),
    )

    state_dict = d.get("state", {})
    state = StateConfig(
        timeout_ms=state_dict.get("timeout_ms", 10000),
        cache_ttl_ms=state_dict.get("cache_ttl_ms", 60000),
        enable_incremental=state_dict.get("enable_incremental", True),
    )

    model_dict = d.get("model", {})
    model_args = model_dict.get("model_args", {})
    model = ModelConfig(
        architecture=model_dict.get("architecture", "KalmanBalancedFlexible"),
        weights_path=model_dict.get("weights_path", "weights/best_model.pth"),
        device=model_dict.get("device", "cpu"),
        imu_frames=model_args.get("imu_frames", 128),
        imu_channels=model_args.get("imu_channels", 7),
        embed_dim=model_args.get("embed_dim", 48),
        acc_ratio=model_args.get("acc_ratio", 0.65),
        num_heads=model_args.get("num_heads", 4),
        num_layers=model_args.get("num_layers", 2),
    )

    return ServerConfig(
        nats_url=d.get("nats_url", "nats://localhost:4222"),
        subject_pattern=d.get("subject_pattern", "m.kalman_transformer.*"),
        default_fs_hz=d.get("default_fs_hz", 30.0),
        window_size=d.get("window_size", 128),
        kalman=kalman,
        preprocessing=preprocessing,
        state=state,
        model=model,
    )


def load_config(path: Optional[str] = None) -> ServerConfig:
    if path is None:
        config_dir = Path(__file__).parent
        path = config_dir / "default.yaml"

    path = Path(path)
    if not path.exists():
        return ServerConfig().apply_env_overrides()

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    config = _dict_to_config(data)
    return config.apply_env_overrides()
