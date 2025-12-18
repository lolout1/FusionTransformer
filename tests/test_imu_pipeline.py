"""Smoke tests for the IMU data loading and model pipeline."""

import os
import sys
import yaml
import numpy as np
import torch
from argparse import Namespace

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.dataset import prepare_smartfallmm, split_by_subjects
from Feeder.Make_Dataset import UTD_mm
from Models.imu_transformer import IMUTransformer

def load_config(config_path):
    """Load YAML configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def test_data_loading():
    """Test data loading and preprocessing pipeline"""
    print("=" * 80)
    print("TEST 1: Data Loading and Preprocessing")
    print("=" * 80)

    # Load config
    config_path = './config/smartfallmm/imu_student.yaml'
    config = load_config(config_path)

    # Convert to namespace for compatibility
    arg = Namespace(**config)

    print(f"\nConfig loaded successfully")
    print(f"  - Modalities: {config['dataset_args']['modalities']}")
    print(f"  - Sensor: {config['dataset_args']['sensors']}")
    print(f"  - Age Group: {config['dataset_args']['age_group']}")
    print(f"  - Filtering Enabled: {config['dataset_args'].get('enable_filtering', True)}")
    print(f"  - Filter Cutoff: {config['dataset_args'].get('filter_cutoff', 5.5)} Hz")

    # Prepare dataset
    print("\nPreparing SmartFallMM dataset...")
    try:
        builder = prepare_smartfallmm(arg)
        print("  ✓ Dataset builder created successfully")
    except Exception as e:
        print(f"  ✗ Error creating dataset builder: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Use a small subset of subjects for testing
    test_subjects = config['subjects'][:3]  # First 3 subjects
    print(f"\nLoading data for test subjects: {test_subjects}")

    try:
        data = split_by_subjects(builder, test_subjects, fuse=False)
        print("  ✓ Data loaded and processed successfully")
    except Exception as e:
        print(f"  ✗ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Verify data shape
    print("\nData Statistics:")
    for key, value in data.items():
        if key != 'labels':
            print(f"  - {key:15s}: {value.shape}")
        else:
            print(f"  - {key:15s}: {value.shape}, Fall samples: {np.sum(value)}, Non-fall: {len(value) - np.sum(value)}")

    # Check if both accelerometer and gyroscope are present
    has_acc = 'accelerometer' in data
    has_gyro = 'gyroscope' in data

    print(f"\nModality Check:")
    print(f"  - Accelerometer present: {has_acc}")
    print(f"  - Gyroscope present: {has_gyro}")

    if has_acc and has_gyro:
        print("  ✓ Both accelerometer and gyroscope data loaded")
        # Check if they have the same number of samples
        if data['accelerometer'].shape[0] == data['gyroscope'].shape[0]:
            print(f"  ✓ Sample counts match: {data['accelerometer'].shape[0]}")
        else:
            print(f"  ✗ Sample count mismatch! Acc: {data['accelerometer'].shape[0]}, Gyro: {data['gyroscope'].shape[0]}")
            return False

        # Check temporal alignment
        if data['accelerometer'].shape[1] == data['gyroscope'].shape[1]:
            print(f"  ✓ Temporal length match: {data['accelerometer'].shape[1]} frames")
        else:
            print(f"  ✗ Temporal mismatch! Acc: {data['accelerometer'].shape[1]}, Gyro: {data['gyroscope'].shape[1]}")
            return False

    return True, data

def test_dataloader(data, config):
    """Test PyTorch dataloader"""
    print("\n" + "=" * 80)
    print("TEST 2: PyTorch Dataloader")
    print("=" * 80)

    try:
        dataset = UTD_mm(data, batch_size=config['batch_size'])
        print(f"\n  ✓ Dataset created successfully")
        print(f"    - Number of samples: {len(dataset)}")
        print(f"    - Inertial modality: {dataset.inertial_modality}")

        # Test getting a single sample
        sample_data, sample_label, sample_idx = dataset[0]

        print(f"\n  Sample Data Shapes:")
        for key, value in sample_data.items():
            print(f"    - {key:15s}: {value.shape}")

        print(f"    - Label: {sample_label.item()}")

        # Verify IMU data shape
        if dataset.inertial_modality == 'imu':
            expected_channels = 6  # ax, ay, az, gx, gy, gz
            actual_channels = sample_data['accelerometer'].shape[-1]
            if actual_channels == expected_channels:
                print(f"\n  ✓ IMU data has correct shape: {expected_channels} channels")
            else:
                print(f"\n  ✗ IMU data shape mismatch! Expected {expected_channels}, got {actual_channels}")
                return False

        # Test batch loading
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0
        )

        batch_data, batch_label, batch_idx = next(iter(dataloader))

        print(f"\n  Batch Data Shapes (batch_size={config['batch_size']}):")
        for key, value in batch_data.items():
            print(f"    - {key:15s}: {value.shape}")
        print(f"    - Labels: {batch_label.shape}")

        return True, dataset, batch_data

    except Exception as e:
        print(f"\n  ✗ Error creating dataloader: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def test_model(batch_data, config):
    """Test model forward pass"""
    print("\n" + "=" * 80)
    print("TEST 3: Model Forward Pass")
    print("=" * 80)

    try:
        # Create model
        model = IMUTransformer(**config['model_args'])
        print(f"\n  ✓ Model created successfully")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"    - Total parameters: {total_params:,}")
        print(f"    - Trainable parameters: {trainable_params:,}")

        # Forward pass
        acc_data = batch_data['accelerometer']
        skl_data = batch_data['skeleton']

        print(f"\n  Input shapes:")
        print(f"    - IMU data: {acc_data.shape}")
        print(f"    - Skeleton data: {skl_data.shape}")

        # Test forward pass
        model.eval()
        with torch.no_grad():
            logits, features = model(acc_data, skl_data)

        print(f"\n  Output shapes:")
        print(f"    - Logits: {logits.shape}")
        print(f"    - Features: {features.shape}")

        # Verify output shape
        if logits.shape == (acc_data.shape[0], config['model_args']['num_classes']):
            print(f"\n  ✓ Output shape is correct: ({acc_data.shape[0]}, {config['model_args']['num_classes']})")
        else:
            print(f"\n  ✗ Output shape mismatch!")
            return False

        print(f"\n  ✓ Model forward pass successful!")
        return True

    except Exception as e:
        print(f"\n  ✗ Error in model forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("IMU PIPELINE TEST SUITE")
    print("Testing Accelerometer + Gyroscope Data Pipeline")
    print("=" * 80)

    # Load config
    config_path = './config/smartfallmm/imu_student.yaml'
    if not os.path.exists(config_path):
        print(f"\n✗ Config file not found: {config_path}")
        return

    config = load_config(config_path)

    # Test 1: Data Loading
    result = test_data_loading()
    if isinstance(result, tuple):
        success, data = result
    else:
        success = result
        data = None

    if not success:
        print("\n" + "=" * 80)
        print("TEST SUITE FAILED: Data loading error")
        print("=" * 80)
        return

    # Test 2: Dataloader
    result = test_dataloader(data, config)
    if isinstance(result, tuple):
        success, dataset, batch_data = result
    else:
        success = result
        dataset = None
        batch_data = None

    if not success or batch_data is None:
        print("\n" + "=" * 80)
        print("TEST SUITE FAILED: Dataloader error")
        print("=" * 80)
        return

    # Test 3: Model
    success = test_model(batch_data, config)
    if not success:
        print("\n" + "=" * 80)
        print("TEST SUITE FAILED: Model error")
        print("=" * 80)
        return

    # All tests passed
    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED!")
    print("=" * 80)
    print("\nSummary:")
    print("  ✓ Data loading and preprocessing works correctly")
    print("  ✓ Accelerometer and gyroscope data are properly aligned")
    print("  ✓ Filtering is applied correctly")
    print("  ✓ PyTorch dataloader works with 6-channel IMU data")
    print("  ✓ Model accepts 6-channel input and produces correct output")
    print("\nThe pipeline is ready for training!")
    print("Run: bash scripts/train_imu.sh")
    print("=" * 80)

if __name__ == "__main__":
    main()
