"""
Modular Kalman Transformer NATS Server for Real-Time Fall Detection

Features:
- Configuration-driven (YAML)
- Multiple architectures: KalmanConv1dConv1d, SingleStreamTransformerSE, KalmanBalancedFlexible
- Multiple feature modes: kalman, kalman_gyro_mag, raw, raw_gyromag, acc_only
- Multiple normalization modes: none, all, acc_only
- Per-user Kalman state persistence with orientation caching
- Backward compatible with existing binary/JSON payloads

Usage:
    python server.py --config config/default.yaml
    python server.py --config configs/s8_16_kalman_gyromag_norm.yaml

Environment overrides:
    DEVICE, MODEL_PATH, SCALER_PATH, NATS_URL
"""

import asyncio
import functools
import json
import os
import struct
import time
from pathlib import Path
from typing import Optional, Tuple
import argparse

import numpy as np
import torch

from nats.aio.client import Client as NATS

from config import load_config, ServerConfig
from preprocessing import PreprocessingPipeline
from models import get_model_class


# Protocol constants
MAGIC = b"SFN1"
BIN_VERSION = 1
SCALE = 1000.0
WINDOW_SIZE = 128
ACC_GYRO_COUNT = WINDOW_SIZE * 3


class FallDetectionServer:
    def __init__(self, config: ServerConfig):
        self.config = config
        self.pipeline = PreprocessingPipeline(config)
        self.model: Optional[torch.nn.Module] = None
        self._cleanup_task: Optional[asyncio.Task] = None

    def initialize(self) -> None:
        self._load_model()
        self._warmup()
        print(f"[OK] Server initialized")
        print(f"     Architecture: {self.config.model.architecture}")
        print(f"     Feature mode: {self.config.preprocessing.feature_mode.value}")
        print(f"     Normalization: {self.config.preprocessing.normalization_mode.value}")
        print(f"     Channels: {self.config.model.imu_channels}")
        print(f"     State caching: {self.config.state.enable_incremental}")

    def _load_model(self) -> None:
        weights_path = self.config.model.weights_path
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights not found: {weights_path}")

        # Dynamic architecture loading based on config
        arch_name = self.config.model.architecture
        ModelClass = get_model_class(arch_name)

        self.model = ModelClass(
            imu_frames=self.config.model.imu_frames,
            imu_channels=self.config.model.imu_channels,
            num_heads=self.config.model.num_heads,
            num_layers=self.config.model.num_layers,
            embed_dim=self.config.model.embed_dim,
            dropout=0.5,
            activation='relu',
            norm_first=True,
            se_reduction=4,
            acc_ratio=self.config.model.acc_ratio,
        )
        self.model.load_state_dict(
            torch.load(weights_path, map_location=self.config.model.device)
        )
        self.model.eval()
        print(f"[OK] Loaded {arch_name}: {weights_path}")

    def _warmup(self) -> None:
        dummy_acc = np.zeros((WINDOW_SIZE, 3), dtype=np.float32)
        dummy_gyro = np.zeros((WINDOW_SIZE, 3), dtype=np.float32)
        dummy_ts = np.arange(WINDOW_SIZE) * (1000.0 / self.config.default_fs_hz)
        features = self.pipeline.process("__warmup__", dummy_acc, dummy_gyro, dummy_ts)
        self._infer(features)
        print("[OK] Warmup complete")

    def _infer(self, features: np.ndarray) -> float:
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        x = x.to(self.config.model.device)
        with torch.no_grad():
            logit, _ = self.model(x)
            prob = torch.sigmoid(logit).item()
        return prob

    def process_request(
        self,
        user_id: str,
        acc: np.ndarray,
        gyro: np.ndarray,
        timestamps: np.ndarray,
    ) -> float:
        features = self.pipeline.process(user_id, acc, gyro, timestamps)
        return self._infer(features)

    async def cleanup_loop(self, interval_s: float = 30.0) -> None:
        while True:
            await asyncio.sleep(interval_s)
            removed = self.pipeline.cleanup_stale_sessions()
            if removed > 0:
                print(f"[CLEANUP] Removed {removed} stale sessions")


def parse_payload_auto(
    data: bytes,
    default_fs: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
    """Parse binary or JSON payload.

    Returns:
        acc (128, 3), gyro (128, 3), timestamps (128,), fs_hz, user_id
    """
    if data[:4] == MAGIC:
        return _parse_binary(data, default_fs)
    return _parse_json(data, default_fs)


def _parse_binary(
    data: bytes,
    default_fs: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
    if len(data) < 21:
        raise ValueError(f"Binary payload too short: {len(data)}")

    if data[4] != BIN_VERSION:
        raise ValueError(f"Unsupported version: {data[4]}")

    ts_millis = struct.unpack_from(">q", data, 8)[0]
    fs = struct.unpack_from(">f", data, 16)[0]
    uuid_len = data[20]
    off = 21

    uuid = data[off:off + uuid_len].decode("utf-8", errors="ignore")
    off += uuid_len

    needed = (ACC_GYRO_COUNT * 2) * 2
    if len(data) < off + needed:
        raise ValueError(f"Missing sensor data")

    acc_i16 = np.frombuffer(data, dtype=">i2", count=ACC_GYRO_COUNT, offset=off)
    off += ACC_GYRO_COUNT * 2
    gyro_i16 = np.frombuffer(data, dtype=">i2", count=ACC_GYRO_COUNT, offset=off)

    acc = (acc_i16.astype(np.float32) / SCALE).reshape(WINDOW_SIZE, 3)
    gyro = (gyro_i16.astype(np.float32) / SCALE).reshape(WINDOW_SIZE, 3)

    # Generate timestamps from ts_millis and fs
    sample_duration = 1000.0 / fs
    timestamps = ts_millis + np.arange(WINDOW_SIZE) * sample_duration

    return acc, gyro, timestamps, float(fs), uuid


def _parse_json(
    data: bytes,
    default_fs: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
    obj = json.loads(data.decode("utf-8"))

    if isinstance(obj, list):
        raise ValueError("Expected JSON object, got array")

    fs = float(obj.get("fsHz", default_fs))
    uuid = str(obj.get("uuid", ""))
    ts_millis = int(obj.get("tsMillis", int(time.time() * 1000)))

    acc = np.asarray(obj.get("acc"), dtype=np.float32)
    gyro = np.asarray(obj.get("gyro"), dtype=np.float32)

    acc = _fix_nans(acc)
    gyro = _fix_nans(gyro)

    acc = _ensure_shape(acc, "acc")
    gyro = _ensure_shape(gyro, "gyro")

    # Convert gyro units if needed
    units = obj.get("unitsGyro", "rad/s").lower()
    if units in ("deg/s", "dps", "degrees/s"):
        gyro = np.deg2rad(gyro).astype(np.float32)

    # Check for per-sample timestamps
    if "timestamps" in obj:
        timestamps = np.asarray(obj["timestamps"], dtype=np.float64)
        if len(timestamps) != WINDOW_SIZE:
            # Interpolate or generate
            timestamps = ts_millis + np.arange(WINDOW_SIZE) * (1000.0 / fs)
    else:
        timestamps = ts_millis + np.arange(WINDOW_SIZE) * (1000.0 / fs)

    return acc, gyro, timestamps, fs, uuid


def _fix_nans(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 2:
        return arr
    col_all_nan = np.all(np.isnan(arr), axis=0)
    if np.any(col_all_nan):
        arr[:, col_all_nan] = 0.0
    nan_mask = np.isnan(arr)
    if np.any(nan_mask):
        col_mean = np.nanmean(arr, axis=0)
        arr[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])
    return arr


def _ensure_shape(arr: np.ndarray, name: str) -> np.ndarray:
    if arr is None:
        raise ValueError(f"Missing '{name}'")
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"'{name}' must be 2D with >= 3 columns")

    arr = arr[:, :3]
    T = arr.shape[0]

    if T == WINDOW_SIZE:
        return arr.astype(np.float32)
    if T > WINDOW_SIZE:
        return arr[-WINDOW_SIZE:].astype(np.float32)

    pad = np.zeros((WINDOW_SIZE - T, 3), dtype=np.float32)
    return np.vstack([pad, arr]).astype(np.float32)


async def run_server(config: ServerConfig) -> None:
    server = FallDetectionServer(config)
    server.initialize()

    nc = NATS()
    print(f"Connecting to NATS: {config.nats_url}")
    await nc.connect(config.nats_url)
    print(f"Subscribed to '{config.subject_pattern}'")

    # Start cleanup task
    cleanup_task = asyncio.create_task(server.cleanup_loop())

    def _process_sync(acc, gyro, timestamps, user_id):
        return server.process_request(user_id, acc, gyro, timestamps)

    async def handler(msg):
        try:
            t0 = time.perf_counter()

            acc, gyro, timestamps, fs, user_id = parse_payload_auto(
                msg.data, config.default_fs_hz
            )

            loop = asyncio.get_running_loop()
            prob = await loop.run_in_executor(
                None,
                functools.partial(_process_sync, acc, gyro, timestamps, user_id),
            )

            dur_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[{msg.subject}] user={user_id[:8]}... prob={prob:.4f} latency={dur_ms:.1f}ms")

            await msg.respond(str(prob).encode("utf-8"))

        except Exception as e:
            print(f"Error: {e}")
            await msg.respond(b"NaN")

    await nc.subscribe(config.subject_pattern, cb=handler)

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup_task.cancel()
        await nc.drain()


def main():
    parser = argparse.ArgumentParser(description="Fall Detection NATS Server")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    args = parser.parse_args()

    config = load_config(args.config)
    asyncio.run(run_server(config))


if __name__ == "__main__":
    main()
