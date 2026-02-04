#!/usr/bin/env python3
"""
NATS worker that serves predictions for multiple personalized models.

- Subjects:  m.<uuid>
- Model dirs: /home/su-kgv34/personalized_models/<uuid>/
- Files:
    - For --model-type tflite: <uuid>.tflite
    - For --model-type pth:    <uuid>_kd.pth

PTH path uses:
    - Global scaler: SCALER_PATH env (default: inertial_scaler.pkl)
    - TransModel architecture
    - Per-UUID weights: /home/su-kgv34/personalized_models/<uuid>/<uuid>_kd.pth
"""

import argparse
import asyncio
import json
import os
from typing import Dict, Any

import numpy as np
from nats.aio.client import Client as NATS

import tensorflow as tf
import torch
import joblib

from ttf_student_model import TransModel

# ------------------ Global state ------------------ #

MODEL_CACHE: Dict[str, Any] = {}   # uuid -> model instance
MODEL_ROOT = os.path.join(os.path.dirname(__file__), "/home/su-kgv34/personalized_models")

MODEL_TYPE: str = "pth"         # overwritten by CLI
device = torch.device(os.environ.get("DEVICE", "cpu"))

inertial_scaler = None             # loaded once for all PTH models


# ---------- Helpers ----------

def compute_smv(acc_data):
    """
    acc_data: (T, 3) numpy array
    returns:  (T, 4) with SMV (zero-mean) in first column then original x,y,z
    """
    acc_array = np.asarray(acc_data)
    mean = np.mean(acc_array, axis=0, keepdims=True)            # (1, 3)
    zero_mean = acc_array - mean                                 # (T, 3)
    smv = np.sqrt(np.sum(zero_mean ** 2, axis=1, keepdims=True)) # (T, 1)
    return np.concatenate([smv, acc_array], axis=-1)             # (T, 4)

def handle_nan_and_scale(data_2d):
    """
    data_2d: (T, 3) array (x,y,z)
    - replace all-NaN columns with zeros
    - mean-impute NaNs per column
    - scale with inertial_scaler
    returns: (T, 3) scaled
    """
    data = np.asarray(data_2d, dtype=np.float32)
    if np.all(np.isnan(data), axis=0).any():
        data[:, np.all(np.isnan(data), axis=0)] = 0
    col_mean = np.nanmean(data, axis=0)
    nan_mask = np.isnan(data)
    data[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])
    return inertial_scaler.transform(data)


# ------------------ TFLite wrapper ------------------ #

class TFLiteModel:
    """Wrapper around a TFLite interpreter with a .predict(np_array) API."""

    def __init__(self, model_path: str):
        print(f"[TFLiteModel] Loading TFLite model from: {model_path}")
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print("[TFLiteModel] Model loaded and tensors allocated.")

    def predict(self, np_array: np.ndarray) -> np.ndarray:
        # Basic handling for dimensions and dtype
        input_info = self.input_details[0]
        x = np_array.astype(input_info["dtype"])

        # If model expects a batch dimension that isn't present, add it
        if x.ndim == len(input_info["shape"]) - 1:
            x = np.expand_dims(x, axis=0)

        self.interpreter.set_tensor(input_info["index"], x)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]["index"])
        return output_data


# ------------------ Helpers ------------------ #

def extract_uuid_from_subject(subject: str, prefix: str = "m.") -> str:
    """
    Extract the UUID from subject "m.<uuid>".
    If subject has more pieces like "m.<uuid>.something", only the first part after prefix is used.
    """
    if not subject.startswith(prefix):
        raise ValueError(f"Subject '{subject}' does not start with '{prefix}'")

    rest = subject[len(prefix):]
    uuid = rest.split(".", 1)[0]
    return uuid


# -------- PTH infrastructure: scaler + per-UUID models -------- #

def initialize_pth_infrastructure():
    """
    Load PyTorch scaler once (global), set device.
    """
    global inertial_scaler

    scaler_path = os.environ.get("SCALER_PATH", "inertial_scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    inertial_scaler = joblib.load(scaler_path)
    print(f"[PTH] inertial_scaler loaded from {scaler_path}")
    print(f"[PTH] Using device: {device}")


def get_or_load_tflite_model(uuid: str) -> TFLiteModel:
    """
    Load/cached TFLite model /home/su-kgv34/personalized_models/<uuid>/<uuid>.tflite
    """
    if uuid in MODEL_CACHE:
        return MODEL_CACHE[uuid]

    filename = f"{uuid}.tflite"
    model_dir = os.path.join(MODEL_ROOT, uuid)
    model_path = os.path.join(model_dir, filename)

    if not os.path.exists(model_path):
        filename = "default.tflite"
        model_dir = os.path.join(MODEL_ROOT, "default")
        model_path = os.path.join(model_dir, filename)
        print(f"TFLite model not found for uuid={uuid} loading base model: {model_path}")

    model = TFLiteModel(model_path)
    MODEL_CACHE[uuid] = model
    print(f"[TFLite] Model for uuid={uuid} cached.")
    return model


def get_or_load_pth_model(uuid: str) -> torch.nn.Module:
    """
    Load/cached PTH model /home/su-kgv34/personalized_models/<uuid>/<uuid>_kd.pth
    Uses TransModel architecture and global inertial_scaler.
    """
    global inertial_scaler

    if inertial_scaler is None:
        initialize_pth_infrastructure()

    if uuid in MODEL_CACHE:
        return MODEL_CACHE[uuid]

    filename = f"{uuid}_kd.pth"
    model_dir = os.path.join(MODEL_ROOT, uuid)
    model_path = os.path.join(model_dir, filename)

    if not os.path.exists(model_path):
        filename = "default_kd.pth"
        model_dir = os.path.join(MODEL_ROOT, "default")
        model_path = os.path.join(model_dir, filename)
        print(f"PTH model not found for uuid={uuid} loading base model: {model_path}")

    # initialize_model()
    model = TransModel(
        num_layers=2,
        norm_first=True,
        embed_dim=32,
        activation='relu',
        num_classes=1,
        acc_coords=3,
        acc_frames=128,
        mocap_frames=128,
        num_heads=4
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    MODEL_CACHE[uuid] = model
    print(f"[PTH] Model for uuid={uuid} loaded from {model_path} and cached.")
    return model


# ------------------ Per-model-type processing ------------------ #

def process_byte_data_tflite(model: TFLiteModel, byte_data: bytes) -> str:
    """
    TFLite: JSON -> np.array -> model.predict -> scalar string.
    """
    data_string = byte_data.decode("utf-8")
    data_list = json.loads(data_string)
    np_array = np.asarray(data_list, dtype=np.float32)

    output_data = model.predict(np_array)
    scalar = float(np.asarray(output_data).reshape(-1)[0])
    return str(scalar)


def process_byte_data_pth(model: torch.nn.Module, byte_data: bytes) -> str:
    """
    PTH: Your more complex pipeline with scaler, NaN handling, SMV, padding/trimming.

    Input: bytes of JSON 3D array: shape (B,T,F) or (T,F), F >= 3
    Output: stringified float prob
    """
    global inertial_scaler

    if model is None or inertial_scaler is None:
        raise RuntimeError("Model/scaler not initialized. Call initialize_pth_infrastructure() first.")

    # ---- JSON -> numpy ----
    data_list = json.loads(byte_data.decode("utf-8"))
    arr = np.asarray(data_list, dtype=np.float32)

    # Accept (T, F) or (B, T, F)
    if arr.ndim == 3:
        acc_tf = arr[0]
    elif arr.ndim == 2:
        acc_tf = arr
    else:
        raise ValueError(f"Unexpected payload shape: {arr.shape}. Expected (B,T,F) or (T,F).")

    if acc_tf.shape[-1] < 3:
        raise ValueError(f"Expected at least 3 features (x,y,z), got {acc_tf.shape[-1]}")

    acc_xyz = acc_tf[..., :3]

    # ---- NaN handling + scaling + SMV ----
    acc_normalized = handle_nan_and_scale(acc_xyz)
    acc_augmented = compute_smv(acc_normalized)  # (T,4)

    # ---- pad/trim to 128 frames ----
    TARGET_T = 128
    T = acc_augmented.shape[0]
    if T < TARGET_T:
        print(f"Target shape is {T} < {TARGET_T}, padding zeros")
        pad = np.zeros((TARGET_T - T, acc_augmented.shape[1]), dtype=acc_augmented.dtype)
        acc_augmented = np.vstack([acc_augmented, pad])
    elif T > TARGET_T:
        print(f"Target shape is {T} > {TARGET_T}, trimming to {TARGET_T}")
        acc_augmented = acc_augmented[:TARGET_T]

    # ---- tensorize & predict ----
    input_tensor = torch.tensor(
        acc_augmented,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0).to("cpu")  # keep inference on CPU

    with torch.no_grad():
        out = model(input_tensor)
        if isinstance(out, tuple):
            out = out[0]
        logits = out.squeeze()
        prob = torch.sigmoid(logits).item()
        print("out prob:", prob)

    return str(prob)


# ------------------ NATS worker ------------------ #

async def run(nats_url: str, subject_pattern: str = "m.*"):
    nc = NATS()

    async def error_cb(e):
        print(f"[NATS error] {e}")

    print(f"[NATS] Connecting to {nats_url} ...")
    await nc.connect(nats_url, error_cb=error_cb)
    print("[NATS] Connected.")

    async def message_handler(msg):
        subject = msg.subject
        data = msg.data

        try:
            uuid = extract_uuid_from_subject(subject, prefix="m.")
            print(f"[NATS] Received message on '{subject}', uuid={uuid}")

            if MODEL_TYPE == "pth":
                model = get_or_load_pth_model(uuid)
                result = await asyncio.to_thread(process_byte_data_pth, model, data)
            else:  # "tflite"
                model = get_or_load_tflite_model(uuid)
                result = await asyncio.to_thread(process_byte_data_tflite, model, data)

            await msg.respond(result.encode("utf8"))
            print(f"[NATS] Responded to '{subject}' with {result}")

        except Exception as e:
            print(f"[NATS] Failed to process message on '{subject}': {e}")


    print(f"[NATS] Subscribing to '{subject_pattern}' ...")
    await nc.subscribe(subject_pattern, cb=message_handler)

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("[NATS] KeyboardInterrupt, draining...")
    finally:
        await nc.drain()
        print("[NATS] Disconnected.")


# ------------------ CLI entry ------------------ #

def main():
    global MODEL_TYPE, MODEL_ROOT

    parser = argparse.ArgumentParser(description="Multi-user prediction server over NATS.")
    parser.add_argument(
        "--model-type",
        choices=["tflite", "pth"],
        # default="pth",
        required=True,
        help="Type of personalized model files to load: 'tflite' or 'pth'.",
    )
    parser.add_argument(
        "--nats-url",
        default=os.getenv("NATS_URL", "nats://chocolatefrog@cssmartfall1.cose.txstate.edu:4224"),
        help="NATS server URL.",
    )
    parser.add_argument(
        "--models-root",
        default=MODEL_ROOT,
        help="Root directory containing per-uuid model folders (default: /home/su-kgv34/personalized_models).",
    )

    args = parser.parse_args()
    MODEL_TYPE = args.model_type
    MODEL_ROOT = args.models_root

    print(f"[Config] MODEL_TYPE={MODEL_TYPE}")
    print(f"[Config] MODEL_ROOT={MODEL_ROOT}")
    print(f"[Config] NATS_URL={args.nats_url}")
    print(f"[Config] DEVICE={device}")

    asyncio.run(run(args.nats_url))


if __name__ == "__main__":
    main()
    # python multi_pred_nats_models.py --model-type pth
    # python multi_pred_nats_models.py --model-type tflite
