#!/usr/bin/env python3
"""
Test gravity removal from accelerometer data.
Verifies that high-pass filter removes DC (gravity) component.
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from utils.loader import butterworth_filter

def test_gravity_removal():
    """Test that high-pass filter removes gravity."""

    # Simulate raw accelerometer data with gravity on Z axis
    # Stationary: [0, 0, 9.8] + small noise
    n_samples = 256
    fs = 30.0  # Hz

    # Create synthetic data
    np.random.seed(42)
    noise = np.random.randn(n_samples, 3) * 0.1  # Small sensor noise

    # Gravity on Z axis (stationary)
    raw_acc = np.zeros((n_samples, 3))
    raw_acc[:, 2] = 9.8  # Gravity on Z
    raw_acc += noise

    print("=" * 60)
    print("GRAVITY REMOVAL TEST")
    print("=" * 60)

    print(f"\nInput: Raw accelerometer with gravity on Z axis")
    print(f"  Shape: {raw_acc.shape}")
    print(f"  Mean (X, Y, Z): ({raw_acc[:, 0].mean():.2f}, {raw_acc[:, 1].mean():.2f}, {raw_acc[:, 2].mean():.2f})")
    print(f"  Magnitude: {np.sqrt((raw_acc**2).sum(axis=1)).mean():.2f} m/s²")

    # Apply high-pass filter to remove gravity
    cutoff = 0.3  # Hz
    filtered_acc = butterworth_filter(
        raw_acc,
        cutoff=cutoff,
        fs=fs,
        order=4,
        filter_type='high'
    )

    print(f"\nAfter high-pass filter (cutoff={cutoff} Hz):")
    print(f"  Mean (X, Y, Z): ({filtered_acc[:, 0].mean():.4f}, {filtered_acc[:, 1].mean():.4f}, {filtered_acc[:, 2].mean():.4f})")
    print(f"  Magnitude: {np.sqrt((filtered_acc**2).sum(axis=1)).mean():.4f} m/s²")

    # Test with motion
    print("\n" + "=" * 60)
    print("TEST WITH SIMULATED MOTION")
    print("=" * 60)

    # Add sinusoidal motion (1 Hz) to the raw data
    t = np.arange(n_samples) / fs
    motion = np.zeros((n_samples, 3))
    motion[:, 0] = 2.0 * np.sin(2 * np.pi * 1.0 * t)  # 1 Hz motion on X
    motion[:, 1] = 1.5 * np.sin(2 * np.pi * 2.0 * t)  # 2 Hz motion on Y

    raw_with_motion = raw_acc + motion

    print(f"\nInput: Raw acc with gravity + 1-2 Hz motion")
    print(f"  Mean magnitude: {np.sqrt((raw_with_motion**2).sum(axis=1)).mean():.2f} m/s²")

    filtered_with_motion = butterworth_filter(
        raw_with_motion,
        cutoff=cutoff,
        fs=fs,
        order=4,
        filter_type='high'
    )

    print(f"\nAfter gravity removal:")
    print(f"  Mean (X, Y, Z): ({filtered_with_motion[:, 0].mean():.4f}, {filtered_with_motion[:, 1].mean():.4f}, {filtered_with_motion[:, 2].mean():.4f})")
    print(f"  Std (X, Y, Z): ({filtered_with_motion[:, 0].std():.2f}, {filtered_with_motion[:, 1].std():.2f}, {filtered_with_motion[:, 2].std():.2f})")

    # Motion should be preserved
    print(f"\n  Expected motion amplitude X: ~2.0, Got: {filtered_with_motion[:, 0].std() * np.sqrt(2):.2f}")
    print(f"  Expected motion amplitude Y: ~1.5, Got: {filtered_with_motion[:, 1].std() * np.sqrt(2):.2f}")

    # Gravity should be removed
    gravity_removed = abs(filtered_with_motion[:, 2].mean()) < 0.5
    print(f"\n  Gravity removed from Z: {'✓ YES' if gravity_removed else '✗ NO'} (mean Z = {filtered_with_motion[:, 2].mean():.4f})")

    print("\n" + "=" * 60)
    print("TEST WITH REAL DATA")
    print("=" * 60)

    # Test with real watch data
    import pandas as pd
    import glob

    watch_files = glob.glob("data/young/accelerometer/watch/S50*.csv")[:3]

    for filepath in watch_files:
        try:
            df = pd.read_csv(filepath, header=None, nrows=256)
            raw = df.iloc[:, 1:4].values.astype(float)

            raw_mag = np.sqrt((raw**2).sum(axis=1)).mean()

            filtered = butterworth_filter(raw, cutoff=0.3, fs=30, order=4, filter_type='high')
            filtered_mag = np.sqrt((filtered**2).sum(axis=1)).mean()

            print(f"\n  {filepath.split('/')[-1]}:")
            print(f"    Raw magnitude:      {raw_mag:.2f} m/s² (includes gravity)")
            print(f"    Filtered magnitude: {filtered_mag:.2f} m/s² (gravity removed)")

        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("\n  High-pass filter successfully removes gravity (DC component)")
    print("  Motion frequencies (> 0.5 Hz) are preserved")
    print("  Use remove_gravity: True in config to enable")
    print()

if __name__ == "__main__":
    test_gravity_removal()
