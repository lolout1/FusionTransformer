import asyncio
import json
import os
import numpy as np
import torch
from nats.aio.client import Client as NATS
import joblib
from ttf_student_model import TransModel
from scipy.signal import butter, filtfilt
import time

# ---------- Globals ----------
model = None
device = "cpu"
inertial_scaler = None  # loaded in initialize_model()

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

# ---------- Model init ----------
def initialize_model():
    """
    Load PyTorch model (.pth) + scaler, set eval mode.
    """
    global model, inertial_scaler

    scaler_path = os.environ.get("SCALER_PATH", "inertial_scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    inertial_scaler = joblib.load(scaler_path)

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

    pth_path = os.environ.get("MODEL_PATH", "ttfstudent_35_nokd.pth")
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"Model weights .pth not found: {pth_path}")
    state_dict = torch.load(pth_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("PyTorch model + scaler loaded and ready.")

# ---------- Payload processing ----------
def process_byte_data(byte_data: bytes) -> str:
    """
    Input: bytes of JSON 3D array: shape (B,T,F) or (T,F), F=3
    Output: stringified float score
    """
    if model is None or inertial_scaler is None:
        raise RuntimeError("Model/scaler not initialized. Call initialize_model() first.")

    # Parse JSON array
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

    # NaN handle + scale
    acc_normalized = handle_nan_and_scale(acc_xyz)
    acc_augmented = compute_smv(acc_normalized)  # (T,4)

    # ---- pad/trim to 128 frames ----
    TARGET_T = 128
    T = acc_augmented.shape[0]
    if T < TARGET_T:
        print(f"Traget shape is {T} which is less than expected target shape {TARGET_T}, so padding zeros")
        pad = np.zeros((TARGET_T - T, acc_augmented.shape[1]), dtype=acc_augmented.dtype)
        acc_augmented = np.vstack([acc_augmented, pad])
    elif T > TARGET_T:
        print(f"Traget shape is {T} which is more than expected target shape {TARGET_T}, so trimming to expected shape")
        acc_augmented = acc_augmented[:TARGET_T]
    # -------------------------------------

    # Tensorize
    input_tensor = torch.tensor(acc_augmented, dtype=torch.float32, device=device).unsqueeze(0).to('cpu')

    with torch.no_grad():
        out = model(input_tensor)
        if isinstance(out, tuple):
            out = out[0]
        logits = out.squeeze()
        prob = torch.sigmoid(logits).item()
        print("out prob: ",prob)

    return str(prob)

# ---------- NATS server ----------
async def run():
    nc = NATS()
    initialize_model()
    # ---- Warm-up to avoid first-call latency ----
    with torch.no_grad():
        dummy = torch.zeros((1, 128, 4), dtype=torch.float32)
        out = model(dummy)
        if isinstance(out, tuple):
            out = out[0]
        _ = torch.sigmoid(out)
    # --------------------------------------------

    print("Connecting to NATS server...")
    await nc.connect("nats://chocolatefrog@cssmartfall1.cose.txstate.edu:4224")
    print("Subscribed to 'm.*' topic.")

    async def message_handler(msg):
        try:
            start = time.perf_counter()
            result = process_byte_data(msg.data)
            dur_ms = (time.perf_counter() - start) * 1000
            print(f"model handler latency: {dur_ms:.1f} ms")
            await msg.respond(result.encode("utf8"))
        except Exception as e:
            print(f"Failed to process message on '{msg.subject}': {e}")
            await msg.respond(b"NaN")

    await nc.subscribe("m.*", cb=message_handler)

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        await nc.drain()

if __name__ == "__main__":
    asyncio.run(run())
