#!/usr/bin/env python3
"""
Quick verification test for IMU model setup
Tests that all components are properly configured
"""

import torch
import numpy as np
import sys

print("="*80)
print("IMU SETUP VERIFICATION TEST")
print("="*80)

# Test 1: Import models
print("\n[Test 1] Importing models...")
try:
    from Models.transformer import TransModel
    from Models.imu_transformer import IMUTransformer
    print("✓ Models imported successfully")
except Exception as e:
    print(f"✗ Failed to import models: {e}")
    sys.exit(1)

# Test 2: Instantiate TransModel (4 channels)
print("\n[Test 2] Testing TransModel (4 channels)...")
try:
    model_trans = TransModel(
        acc_frames=128,
        num_classes=1,
        num_heads=4,
        num_layer=2,
        embed_dim=64,
        dropout=0.5
    )
    test_input_4ch = torch.randn(2, 128, 4)  # [SMV, ax, ay, az]
    output, features = model_trans(test_input_4ch)
    print(f"✓ TransModel output shape: {output.shape} (expected: [2, 1])")
    print(f"✓ TransModel features shape: {features.shape} (expected: [2, 128, 64])")
    assert output.shape == (2, 1), "TransModel output shape mismatch"
except Exception as e:
    print(f"✗ TransModel test failed: {e}")
    sys.exit(1)

# Test 3: Instantiate IMUTransformer (8 channels)
print("\n[Test 3] Testing IMUTransformer (8 channels)...")
try:
    model_imu = IMUTransformer(
        imu_frames=128,
        imu_channels=8,
        num_classes=1,
        num_heads=4,
        num_layers=2,
        embed_dim=64,
        dropout=0.5
    )
    test_input_8ch = torch.randn(2, 128, 8)  # [acc_smv, ax, ay, az, gyro_mag, gx, gy, gz]
    output, features = model_imu(test_input_8ch)
    print(f"✓ IMUTransformer output shape: {output.shape} (expected: [2, 1])")
    print(f"✓ IMUTransformer features shape: {features.shape} (expected: [2, 128, 64])")
    assert output.shape == (2, 1), "IMUTransformer output shape mismatch"
except Exception as e:
    print(f"✗ IMUTransformer test failed: {e}")
    sys.exit(1)

# Test 4: Test data loader (8-channel output)
print("\n[Test 4] Testing data loader (8-channel IMU data)...")
try:
    from Feeder.Make_Dataset import UTD_mm

    # Create dummy dataset
    dummy_data = {
        'accelerometer': np.random.randn(10, 128, 3).astype(np.float32),
        'gyroscope': np.random.randn(10, 128, 3).astype(np.float32),
        'labels': np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    }

    dataset = UTD_mm(dummy_data, batch_size=2)
    sample_data, sample_label, sample_idx = dataset[0]

    # Check that accelerometer data now has 8 channels
    imu_data = sample_data['accelerometer']
    print(f"✓ IMU data shape: {imu_data.shape} (expected: [128, 8])")
    print(f"  Channels: [acc_smv, ax, ay, az, gyro_mag, gx, gy, gz]")
    assert imu_data.shape == (128, 8), f"Expected 8 channels, got {imu_data.shape}"
except Exception as e:
    print(f"✗ Data loader test failed: {e}")
    sys.exit(1)

# Test 5: Test DTW alignment function
print("\n[Test 5] Testing DTW alignment function...")
try:
    from utils.loader import align_gyro_to_acc

    # Create dummy trial data with slight misalignment
    trial_data = {
        'accelerometer': np.random.randn(130, 3).astype(np.float32),
        'gyroscope': np.random.randn(128, 3).astype(np.float32),  # Different length
    }

    aligned_data = align_gyro_to_acc(trial_data, use_fast_dtw=True)

    acc_shape = aligned_data['accelerometer'].shape
    gyro_shape = aligned_data['gyroscope'].shape

    print(f"✓ After DTW alignment:")
    print(f"  Accelerometer shape: {acc_shape}")
    print(f"  Gyroscope shape: {gyro_shape}")
    print(f"  Lengths match: {acc_shape[0] == gyro_shape[0]}")

    assert acc_shape[0] == gyro_shape[0], "DTW alignment failed - lengths don't match"
except Exception as e:
    print(f"✗ DTW alignment test failed: {e}")
    sys.exit(1)

# Test 6: Verify config files exist
print("\n[Test 6] Verifying config files...")
import os
configs_to_check = [
    'config/smartfallmm/transformer.yaml',
    'config/smartfallmm/imu_8channel.yaml',
    'config/smartfallmm/imu_8channel_dtw.yaml'
]

for config in configs_to_check:
    if os.path.exists(config):
        print(f"✓ Config exists: {config}")
    else:
        print(f"✗ Config missing: {config}")
        sys.exit(1)

# Test 7: Check comparison script exists
print("\n[Test 7] Verifying comparison script...")
if os.path.exists('run_imu_comparison.py') and os.access('run_imu_comparison.py', os.X_OK):
    print("✓ run_imu_comparison.py exists and is executable")
else:
    print("✗ run_imu_comparison.py missing or not executable")
    sys.exit(1)

print("\n" + "="*80)
print("ALL TESTS PASSED! ✓")
print("="*80)
print("\nYou can now run the comparison:")
print("  ./scripts/run_experiments.sh")
print("\nOr run the comparison script directly:")
print("  python run_imu_comparison.py --device 0 --num-epochs 80 --batch-size 64")
print("="*80 + "\n")
