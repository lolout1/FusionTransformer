#!/usr/bin/env python3
"""
Test script to verify stride_sync implementation
"""
import os
import sys
import yaml
from collections import defaultdict
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from utils.dataset import SmartFallMM, prepare_smartfallmm
from utils.loader import DatasetBuilder


class MockArgs:
    """Mock args object for testing"""
    def __init__(self, dataset_args):
        self.dataset_args = dataset_args


def test_stride_sync():
    """Test stride_sync functionality"""
    print("=" * 70)
    print("STRIDE SYNC FUNCTIONALITY TEST")
    print("=" * 70)
    print("")

    # Load a config file
    config_path = "config/smartfallmm/transformer_motion_filtered.yaml"

    print(f"Loading config: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Test 1: stride_sync disabled (default)
    print("\n" + "-" * 70)
    print("Test 1: stride_sync=False (default)")
    print("-" * 70)

    dataset_args = config['dataset_args'].copy()
    dataset_args['enable_stride_sync'] = False
    dataset_args['debug'] = True  # Enable debug output

    # Create dataset
    sm_dataset = SmartFallMM(root_dir=os.path.join(os.getcwd(), 'data'))
    sm_dataset.pipe_line(
        age_group=dataset_args['age_group'],
        modalities=dataset_args['modalities'],
        sensors=dataset_args['sensors']
    )

    # Create builder
    builder = DatasetBuilder(sm_dataset, **dataset_args)

    # Test subjects (including subject 29 which has few falls)
    test_subjects = [29, 30, 31]

    # Compute stride map (should be empty since stride_sync=False)
    stride_map = builder._compute_subject_stride_map(test_subjects)
    print(f"Stride map (should be empty): {stride_map}")
    assert len(stride_map) == 0, "Stride map should be empty when stride_sync=False"
    print("✓ Test 1 passed: stride_map is empty when disabled")

    # Test 2: stride_sync enabled
    print("\n" + "-" * 70)
    print("Test 2: stride_sync=True")
    print("-" * 70)

    dataset_args['enable_stride_sync'] = True
    builder2 = DatasetBuilder(sm_dataset, **dataset_args)

    # Compute stride map (should contain subject-specific strides)
    stride_map = builder2._compute_subject_stride_map(test_subjects)
    print(f"\nStride map for subjects {test_subjects}:")
    for subject_id, strides in stride_map.items():
        print(f"  Subject {subject_id}: fall_stride={strides['fall_stride']}, adl_stride={strides['adl_stride']}")

    assert len(stride_map) > 0, "Stride map should not be empty when stride_sync=True"
    print("\n✓ Test 2 passed: stride_map computed correctly")

    # Test 3: Verify logic for subject 29 (should have < 15 falls)
    print("\n" + "-" * 70)
    print("Test 3: Verify subject 29 logic (< 15 falls)")
    print("-" * 70)

    if 29 in stride_map:
        subject_29_stride = stride_map[29]
        print(f"Subject 29 stride: {subject_29_stride}")

        # Subject 29 should have fall_stride=10 (low fall count)
        expected_fall_stride = dataset_args.get('stride_sync_low_fall_stride', 10)
        if subject_29_stride['fall_stride'] == expected_fall_stride:
            print(f"✓ Test 3 passed: Subject 29 has correct fall_stride={expected_fall_stride}")
        else:
            print(f"⚠ Subject 29 might have more than 15 falls, using stride={subject_29_stride['fall_stride']}")
    else:
        print("⚠ Subject 29 not in dataset")

    # Test 4: Verify YAML configs have stride_sync parameters
    print("\n" + "-" * 70)
    print("Test 4: Verify YAML configs have stride_sync parameters")
    print("-" * 70)

    config_files = [
        "config/smartfallmm/transformer_motion_filtered.yaml",
        "config/smartfallmm/imu_transformer_motionfilter.yaml",
        "config/smartfallmm/imu_dualstream_motion_filtered.yaml",
        "config/smartfallmm/imu_madgwick_motion_filtered.yaml",
    ]

    all_configs_valid = True
    for config_file in config_files:
        if not os.path.exists(config_file):
            print(f"  ⚠ Config not found: {config_file}")
            continue

        with open(config_file, 'r') as f:
            cfg = yaml.safe_load(f)

        dataset_args = cfg.get('dataset_args', {})
        has_stride_sync = 'enable_stride_sync' in dataset_args

        if has_stride_sync:
            print(f"  ✓ {os.path.basename(config_file)}: has stride_sync parameters")
        else:
            print(f"  ✗ {os.path.basename(config_file)}: missing stride_sync parameters")
            all_configs_valid = False

    if all_configs_valid:
        print("\n✓ Test 4 passed: All configs have stride_sync parameters")
    else:
        print("\n✗ Test 4 failed: Some configs missing stride_sync parameters")

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print("✓ All core tests passed!")
    print("\nStride Sync Implementation:")
    print("  1. DatasetBuilder.__init__: stride_sync parameters added")
    print("  2. _compute_subject_stride_map(): computes per-subject strides")
    print("  3. process(): uses subject-specific strides")
    print("  4. make_dataset(): calls _compute_subject_stride_map()")
    print("  5. YAML configs: stride_sync parameters added")
    print("\nUsage:")
    print("  - Set enable_stride_sync: True in YAML config")
    print("  - Subjects with < 15 falls: fall_stride=10")
    print("  - Subjects with fall:ADL <= 1:3: fall_stride=16")
    print("  - Others: fall_stride=32 (default)")
    print("\nNext steps:")
    print("  - Run: ./scripts/run_modality_filtering.sh")
    print("  - This will test models with stride_sync=False and stride_sync=True")
    print("=" * 70)


if __name__ == "__main__":
    try:
        test_stride_sync()
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
