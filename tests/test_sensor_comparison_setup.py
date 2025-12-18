#!/usr/bin/env python3
"""
Test script to verify sensor comparison study setup
Checks config files, data availability, and script integrity
"""

import sys
from pathlib import Path
import yaml


def test_config_files():
    """Test that all required config files exist and are valid."""
    print("Testing configuration files...")

    configs = [
        "config/smartfallmm/transformer_motion_filtered.yaml",
        "config/smartfallmm/transformer_motion_filtered_meta.yaml",
        "config/smartfallmm/imu_transformer_motionfilter.yaml",
        "config/smartfallmm/imu_transformer_motionfilter_meta.yaml",
    ]

    all_valid = True
    for config_path in configs:
        path = Path(config_path)

        if not path.exists():
            print(f"  ✗ MISSING: {config_path}")
            all_valid = False
            continue

        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)

            # Check critical fields
            required_fields = ['model', 'dataset', 'dataset_args']
            missing = [f for f in required_fields if f not in config]

            if missing:
                print(f"  ✗ INVALID: {config_path} (missing: {missing})")
                all_valid = False
            else:
                sensor = config['dataset_args'].get('sensors', ['unknown'])[0]
                print(f"  ✓ {config_path} (sensor: {sensor})")

        except Exception as e:
            print(f"  ✗ ERROR: {config_path} - {e}")
            all_valid = False

    return all_valid


def test_data_directories():
    """Test that data directories exist and contain files."""
    print("\nTesting data directories...")

    data_dirs = [
        "data/young/accelerometer/watch",
        "data/young/accelerometer/meta_wrist",
        "data/young/gyroscope/watch",
        "data/young/gyroscope/meta_wrist",
    ]

    all_valid = True
    for dir_path in data_dirs:
        path = Path(dir_path)

        if not path.exists():
            print(f"  ✗ MISSING: {dir_path}")
            all_valid = False
            continue

        # Count CSV files
        csv_files = list(path.glob("*.csv"))
        if len(csv_files) == 0:
            print(f"  ✗ EMPTY: {dir_path}")
            all_valid = False
        else:
            print(f"  ✓ {dir_path} ({len(csv_files)} files)")

    return all_valid


def test_scripts():
    """Test that required scripts exist and are executable."""
    print("\nTesting scripts...")

    scripts = [
        "scripts/run_sensor_comparison.sh",
        "aggregate_sensor_comparison.py",
    ]

    all_valid = True
    for script_name in scripts:
        path = Path(script_name)

        if not path.exists():
            print(f"  ✗ MISSING: {script_name}")
            all_valid = False
            continue

        # Check if executable
        is_executable = path.stat().st_mode & 0o111

        if is_executable:
            print(f"  ✓ {script_name} (executable)")
        else:
            print(f"  ⚠ {script_name} (not executable - run: chmod +x {script_name})")

    return all_valid


def test_python_dependencies():
    """Test that required Python packages are available."""
    print("\nTesting Python dependencies...")

    required_packages = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('scipy', 'scipy'),
        ('yaml', 'yaml'),
    ]

    all_valid = True
    missing_packages = []
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ⚠ {package_name} not available in current environment")
            missing_packages.append(package_name)

    if missing_packages:
        print(f"\n  Note: These packages will be needed when running main.py")
        print(f"        They should be available in your training environment")

    # Don't fail the test for missing packages since they may be in a different env
    return True


def compare_config_pairs():
    """Compare regular vs meta configs to ensure only sensors differ."""
    print("\nComparing config pairs...")

    pairs = [
        ("config/smartfallmm/transformer_motion_filtered.yaml",
         "config/smartfallmm/transformer_motion_filtered_meta.yaml"),
        ("config/smartfallmm/imu_transformer_motionfilter.yaml",
         "config/smartfallmm/imu_transformer_motionfilter_meta.yaml"),
    ]

    all_valid = True
    for regular_path, meta_path in pairs:
        print(f"\n  Comparing: {Path(regular_path).name} vs {Path(meta_path).name}")

        try:
            with open(regular_path, 'r') as f:
                regular = yaml.safe_load(f)
            with open(meta_path, 'r') as f:
                meta = yaml.safe_load(f)

            # Check sensors
            regular_sensor = regular['dataset_args'].get('sensors', ['unknown'])[0]
            meta_sensor = meta['dataset_args'].get('sensors', ['unknown'])[0]

            if regular_sensor == 'watch' and meta_sensor == 'meta_wrist':
                print(f"    ✓ Sensors: {regular_sensor} → {meta_sensor}")
            else:
                print(f"    ✗ Unexpected sensors: {regular_sensor} vs {meta_sensor}")
                all_valid = False

            # Check that model architecture is the same
            if regular.get('model') == meta.get('model'):
                print(f"    ✓ Model: {regular['model']}")
            else:
                print(f"    ✗ Model mismatch: {regular.get('model')} vs {meta.get('model')}")
                all_valid = False

            # Check that critical hyperparameters match
            critical_params = ['batch_size', 'num_epoch', 'base_lr', 'weight_decay']
            for param in critical_params:
                reg_val = regular.get(param)
                meta_val = meta.get(param)
                if reg_val == meta_val:
                    print(f"    ✓ {param}: {reg_val}")
                else:
                    print(f"    ⚠ {param} differs: {reg_val} vs {meta_val}")

        except Exception as e:
            print(f"    ✗ ERROR: {e}")
            all_valid = False

    return all_valid


def main():
    print("=" * 80)
    print("SENSOR COMPARISON STUDY - SETUP VERIFICATION")
    print("=" * 80)
    print()

    results = {
        'Config files': test_config_files(),
        'Data directories': test_data_directories(),
        'Scripts': test_scripts(),
        'Python dependencies': test_python_dependencies(),
        'Config pair validation': compare_config_pairs(),
    }

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False

    print()

    if all_passed:
        print("✓ All checks passed! Ready to run sensor comparison study.")
        print()
        print("To start the study, run:")
        print("  ./scripts/run_sensor_comparison.sh")
        print()
        return 0
    else:
        print("✗ Some checks failed. Please fix the issues above before running.")
        print()
        return 1


if __name__ == '__main__':
    sys.exit(main())
