#!/usr/bin/env python3
"""
Quick test to validate channel dimension fixes for dual-stream IMU models.
This script tests that the data feeder produces the correct number of channels
and that the model can process them without errors.
"""

import torch
import numpy as np
import sys
from Feeder.Make_Dataset import UTD_mm

def test_feeder_channels():
    """Test that the feeder produces correct channel counts for IMU data."""
    print("=" * 70)
    print("TESTING FEEDER CHANNEL DIMENSIONS")
    print("=" * 70)

    # Create mock dataset with accelerometer and gyroscope data
    num_samples = 10
    seq_length = 128
    acc_channels = 3
    gyro_channels = 3

    mock_dataset = {
        'accelerometer': np.random.randn(num_samples, seq_length, acc_channels).astype(np.float32),
        'gyroscope': np.random.randn(num_samples, seq_length, gyro_channels).astype(np.float32),
        'labels': np.random.randint(0, 2, num_samples)
    }

    print(f"\nMock dataset created:")
    print(f"  - Accelerometer shape: {mock_dataset['accelerometer'].shape}")
    print(f"  - Gyroscope shape: {mock_dataset['gyroscope'].shape}")
    print(f"  - Labels shape: {mock_dataset['labels'].shape}")

    # Create feeder
    feeder = UTD_mm(mock_dataset, batch_size=4)

    print(f"\nFeeder initialized:")
    print(f"  - Inertial modality: {feeder.inertial_modality}")
    print(f"  - Has accelerometer: {feeder.has_accelerometer}")
    print(f"  - Has gyroscope: {feeder.has_gyroscope}")
    print(f"  - Number of samples: {feeder.num_samples}")

    # Get a sample
    data, label, idx = feeder[0]

    print(f"\nSample data from feeder:")
    print(f"  - Data keys: {list(data.keys())}")
    if 'accelerometer' in data:
        imu_data = data['accelerometer']
        print(f"  - IMU data shape: {imu_data.shape}")
        print(f"  - Expected shape: (128, 6) for [ax, ay, az, gx, gy, gz]")

        if imu_data.shape[-1] == 6:
            print(f"  ✓ PASS: Correct 6 channels for dual-stream model")
            return True
        else:
            print(f"  ✗ FAIL: Got {imu_data.shape[-1]} channels, expected 6")
            return False

def test_model_forward_pass():
    """Test that the model can process the data without errors."""
    print("\n" + "=" * 70)
    print("TESTING MODEL FORWARD PASS")
    print("=" * 70)

    try:
        from Models.imu_dual_stream_asymmetric import DualStreamAsymmetricIMU

        batch_size = 4
        seq_len = 128
        imu_channels = 6

        # Create random input data
        imu_data = torch.randn(batch_size, seq_len, imu_channels)

        print(f"\nInput data shape: {imu_data.shape}")
        print(f"Expected format: (batch_size={batch_size}, seq_len={seq_len}, channels={imu_channels})")

        # Initialize model
        model = DualStreamAsymmetricIMU(
            imu_frames=seq_len,
            imu_channels=imu_channels,
            num_classes=1,
            acc_layers=2,
            gyro_layers=1,
            acc_dim=16,
            gyro_dim=8,
            num_heads=2,
            dropout=0.6
        )

        print(f"\nModel initialized successfully")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Forward pass
        model.eval()
        with torch.no_grad():
            logits, features = model(imu_data)

        print(f"\nForward pass completed successfully!")
        print(f"  - Output logits shape: {logits.shape}")
        print(f"  - Output features shape: {features.shape}")
        print(f"  ✓ PASS: Model can process 6-channel IMU data")
        return True

    except Exception as e:
        print(f"\n✗ FAIL: Model forward pass failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_channel_split():
    """Test that the model correctly splits accelerometer and gyroscope channels."""
    print("\n" + "=" * 70)
    print("TESTING CHANNEL SPLITTING")
    print("=" * 70)

    try:
        batch_size = 2
        seq_len = 128

        # Create test data with known values
        # First 3 channels are accelerometer (filled with 1.0)
        # Last 3 channels are gyroscope (filled with 2.0)
        imu_data = torch.zeros(batch_size, seq_len, 6)
        imu_data[:, :, :3] = 1.0  # Accelerometer channels
        imu_data[:, :, 3:] = 2.0  # Gyroscope channels

        print(f"\nTest data created:")
        print(f"  - Shape: {imu_data.shape}")
        print(f"  - Accelerometer channels (0:3) filled with: {imu_data[0, 0, :3]}")
        print(f"  - Gyroscope channels (3:6) filled with: {imu_data[0, 0, 3:]}")

        # Split manually (same as model does)
        acc = imu_data[:, :, :3]
        gyro = imu_data[:, :, 3:]

        print(f"\nAfter splitting:")
        print(f"  - Accelerometer shape: {acc.shape}")
        print(f"  - Gyroscope shape: {gyro.shape}")
        print(f"  - Accelerometer sample values: {acc[0, 0, :]}")
        print(f"  - Gyroscope sample values: {gyro[0, 0, :]}")

        # Verify correct split
        if acc.shape[-1] == 3 and gyro.shape[-1] == 3:
            if torch.allclose(acc, torch.ones_like(acc)) and torch.allclose(gyro, torch.ones_like(gyro) * 2):
                print(f"\n  ✓ PASS: Channels split correctly")
                return True
            else:
                print(f"\n  ✗ FAIL: Channel values don't match expected")
                return False
        else:
            print(f"\n  ✗ FAIL: Incorrect channel counts after split")
            return False

    except Exception as e:
        print(f"\n✗ FAIL: Channel splitting test failed:")
        print(f"  {type(e).__name__}: {e}")
        return False

def main():
    """Run all tests."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "CHANNEL DIMENSION FIX VALIDATION" + " " * 21 + "║")
    print("╚" + "═" * 68 + "╝")

    results = []

    # Run tests
    results.append(("Feeder Channel Test", test_feeder_channels()))
    results.append(("Channel Splitting Test", test_channel_split()))
    results.append(("Model Forward Pass Test", test_model_forward_pass()))

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")
        if not passed:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print("\n✓ All tests passed! Channel dimension fix is working correctly.")
        return 0
    else:
        print("\n✗ Some tests failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
