#!/usr/bin/env python3
"""
Quick validation test for sensor comparison configurations.
Tests that all 4 configs have correct settings and filtering disabled.
"""

import sys
import yaml
from pathlib import Path

def test_config(config_path, config_name, expected_sensor, expected_modalities):
    """Test a single config to ensure settings are correct"""
    print(f"\n{'='*70}")
    print(f"Testing: {config_name}")
    print(f"Config: {config_path}")
    print(f"{'='*70}\n")

    # Check if config exists
    if not Path(config_path).exists():
        print(f"✗ FAILED: Config file not found at {config_path}")
        return False

    # Load config
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"✗ FAILED: Error loading YAML: {e}")
        return False

    # Extract settings
    dataset_args = config.get('dataset_args', {})
    sensors = dataset_args.get('sensors', [])
    modalities = dataset_args.get('modalities', [])
    val_subjects = config.get('validation_subjects', [])

    # Critical filtering flags
    motion_filtering = dataset_args.get('enable_motion_filtering', False)
    freq_filtering = dataset_args.get('enable_filtering', False)
    normalization = dataset_args.get('enable_normalization', False)
    class_aware_stride = dataset_args.get('enable_class_aware_stride', False)
    stride_sync = dataset_args.get('enable_stride_sync', False)

    # Print current settings
    print(f"Sensor: {sensors}")
    print(f"Modalities: {modalities}")
    print(f"Validation subjects: {val_subjects}")
    print()
    print("Filtering/Preprocessing Status:")
    print(f"  Motion filtering: {motion_filtering}")
    print(f"  Frequency filtering: {freq_filtering}")
    print(f"  Normalization: {normalization}")
    print(f"  Class-aware stride: {class_aware_stride}")
    print(f"  Stride sync: {stride_sync}")
    print()

    # Validate settings
    issues = []

    # Check sensor
    if sensors != expected_sensor:
        issues.append(f"Expected sensor {expected_sensor}, got {sensors}")
    else:
        print(f"✓ Sensor correct: {sensors}")

    # Check modalities
    if modalities != expected_modalities:
        issues.append(f"Expected modalities {expected_modalities}, got {modalities}")
    else:
        print(f"✓ Modalities correct: {modalities}")

    # Check validation subjects
    if val_subjects != [48, 57]:
        issues.append(f"Expected validation_subjects [48, 57], got {val_subjects}")
    else:
        print(f"✓ Validation subjects correct: {val_subjects}")

    # Check that ALL filtering is disabled
    if motion_filtering:
        issues.append("Motion filtering should be False (DISABLED)")
    else:
        print(f"✓ Motion filtering disabled")

    if freq_filtering:
        issues.append("Frequency filtering should be False (DISABLED)")
    else:
        print(f"✓ Frequency filtering disabled")

    if normalization:
        issues.append("Normalization should be False (DISABLED)")
    else:
        print(f"✓ Normalization disabled")

    if class_aware_stride:
        issues.append("Class-aware stride should be False (DISABLED)")
    else:
        print(f"✓ Class-aware stride disabled")

    if stride_sync:
        issues.append("Stride sync should be False (DISABLED)")
    else:
        print(f"✓ Stride sync disabled")

    # Report results
    print()
    if issues:
        print(f"✗ FAILED with {len(issues)} issue(s):")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print(f"✓ ALL CHECKS PASSED")
        return True

def main():
    """Test all 4 sensor comparison configs"""

    # Define expected configurations
    configs = [
        ('config/smartfallmm/transformer_motion_filtered.yaml',
         'TransModel + Watch (acc only)',
         ['watch'],
         ['accelerometer']),
        ('config/smartfallmm/transformer_motion_filtered_meta.yaml',
         'TransModel + Meta_Wrist (acc only)',
         ['meta_wrist'],
         ['accelerometer']),
        ('config/smartfallmm/imu_transformer_motionfilter.yaml',
         'IMUTransformer + Watch (acc+gyro)',
         ['watch'],
         ['accelerometer', 'gyroscope']),
        ('config/smartfallmm/imu_transformer_motionfilter_meta.yaml',
         'IMUTransformer + Meta_Wrist (acc+gyro)',
         ['meta_wrist'],
         ['accelerometer', 'gyroscope']),
    ]

    print("="*70)
    print("SENSOR COMPARISON CONFIG VALIDATION TEST")
    print("="*70)
    print()
    print("Validating all 4 configurations...")
    print("Requirements:")
    print("  ✓ Correct sensor and modalities")
    print("  ✓ Validation subjects: [48, 57]")
    print("  ✓ ALL filtering DISABLED (motion, frequency, normalization, stride)")
    print()

    results = {}
    for config_path, config_name, expected_sensor, expected_modalities in configs:
        success = test_config(config_path, config_name, expected_sensor, expected_modalities)
        results[config_name] = success

    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print()

    all_passed = True
    for config_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {config_name}")
        if not success:
            all_passed = False

    print()
    if all_passed:
        print("="*70)
        print("✓ ALL CONFIGURATIONS VALIDATED SUCCESSFULLY!")
        print("="*70)
        print()
        print("Next steps:")
        print("  1. Verify data files exist:")
        print("     ls data/young/accelerometer/meta_wrist/S48A/")
        print("     ls data/young/gyroscope/meta_wrist/S48A/")
        print()
        print("  2. Run sensor comparison:")
        print("     bash run_sensor_comparison.sh")
        print()
        return 0
    else:
        print("="*70)
        print("✗ SOME CONFIGURATIONS FAILED VALIDATION")
        print("="*70)
        print()
        print("Please fix the issues above before running experiments")
        return 1

if __name__ == '__main__':
    sys.exit(main())
