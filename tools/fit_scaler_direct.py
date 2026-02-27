#!/usr/bin/env python3
"""Fit scaler directly from SmartFallMM data using the training pipeline."""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from argparse import Namespace

# Change to project root so data path works
os.chdir(Path(__file__).parent.parent)

from utils.dataset import prepare_smartfallmm

OUTPUT_DIR = Path("kalman_server/SmartFallNATS-rakesh-fusion-server-version/KalmanFusionServer/weights/scalers")


def load_config():
    """Load kalman config and convert to Namespace."""
    config_path = "config/best_config/smartfallmm/kalman.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Override sampling rate
    config['dataset_args']['filter_fs'] = 31.25

    # Use larger stride for faster loading
    config['dataset_args']['fall_stride'] = 32
    config['dataset_args']['adl_stride'] = 32

    # Ensure kalman settings for best model variant
    config['dataset_args']['enable_kalman_fusion'] = True

    return Namespace(**config)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading SmartFallMM data with training pipeline...")
    print(f"Working dir: {os.getcwd()}")

    arg = load_config()

    # Use prepare_smartfallmm from training
    builder = prepare_smartfallmm(arg)

    # Get data for some subjects (test subjects only)
    test_subjects = [34, 36, 37, 38, 43, 44, 45, 46, 49, 50]
    print(f"Loading data for subjects: {test_subjects}")

    builder.make_dataset(test_subjects, fuse=True)

    # Access data from builder.data
    print(f"Data keys: {list(builder.data.keys())}")

    if 'accelerometer' in builder.data:
        acc_data = builder.data['accelerometer']
        print(f"Accelerometer shape: {acc_data.shape}")

        # Data is (N, T, C) where C includes SMV and xyz
        # For Kalman fusion, first 4 channels are [smv, ax, ay, az]
        acc_flat = acc_data[:, :, :4].reshape(-1, 4)

        print(f"Acc samples: {acc_flat.shape[0]}")
        print(f"Mean: {acc_flat.mean(axis=0).round(4)}")
        print(f"Std: {acc_flat.std(axis=0).round(4)}")

        # Fit scaler
        scaler = StandardScaler()
        scaler.fit(acc_flat)

        print(f"\nScaler fitted:")
        print(f"  mean_: {scaler.mean_.round(4)}")
        print(f"  scale_: {scaler.scale_.round(4)}")

        # Save for all models
        models = ['kalman_gyromag', 'kalman_yaw', 'raw_gyro', 'raw_gyromag']
        for model in models:
            path = OUTPUT_DIR / f"s8_16_{model}_norm_scaler.pkl"
            with open(path, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"Saved: {path}")
    else:
        print("ERROR: No accelerometer data found")
        print(f"Available keys: {list(builder.data.keys())}")


if __name__ == "__main__":
    main()
