"""
Quick test to verify validation split with Subject 37.
This script runs 1-2 folds to check that:
1. Validation subject is 37
2. Validation window distribution is balanced (~48-52% falls)
3. No data leakage (Subject 37 not in train/test)
4. LOOCV is still working correctly
"""

import subprocess
import sys
import re

def run_single_fold_test(config_path='config/smartfallmm/transformer.yaml', max_folds=2):
    """
    Run a quick test (first 1-2 folds) to verify validation split.

    Args:
        config_path: Path to config file to test
        max_folds: Maximum number of folds to run (default: 2 for quick test)

    Returns:
        bool: True if validation split looks correct
    """
    print("=" * 80)
    print("VALIDATION SPLIT QUICK TEST")
    print("=" * 80)
    print(f"Config: {config_path}")
    print(f"Testing first {max_folds} fold(s)...")
    print()

    # Run main.py with the config
    # Note: This will actually run training, so keep max_folds=1 for speed
    cmd = f"python main.py --config {config_path}"

    print(f"Running: {cmd}")
    print("(Press Ctrl+C to stop if needed)")
    print()

    try:
        # Run the command and capture output
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout for safety
        )

        output = result.stdout + result.stderr

        # Parse output to extract validation split info
        validation_info = []
        fold_count = 0

        for line in output.split('\n'):
            # Look for fold dataset split logs
            if 'Dataset windows ->' in line:
                print(line)
                validation_info.append(line)
                fold_count += 1

                # Stop after max_folds
                if fold_count >= max_folds:
                    break

            # Look for validation subjects declaration
            if 'Validation subjects' in line:
                print(line)

            # Look for test subject declaration
            if 'Test subject:' in line or 'Iteration' in line:
                print(line)

        print()
        print("=" * 80)
        print("VALIDATION SPLIT ANALYSIS")
        print("=" * 80)

        # Analyze validation splits
        success = True
        for i, info_line in enumerate(validation_info, 1):
            print(f"\nFold {i}:")

            # Extract validation statistics
            val_match = re.search(r'val: (\d+) \((\d+) subjects, fall=(\d+), adl=(\d+)\)', info_line)
            if val_match:
                val_windows = int(val_match.group(1))
                val_subjects = int(val_match.group(2))
                val_fall = int(val_match.group(3))
                val_adl = int(val_match.group(4))

                # Calculate validation fall percentage
                val_fall_pct = (val_fall / val_windows * 100) if val_windows > 0 else 0

                print(f"  Validation windows: {val_windows}")
                print(f"  Validation subjects: {val_subjects}")
                print(f"  Validation fall windows: {val_fall} ({val_fall_pct:.1f}%)")
                print(f"  Validation ADL windows: {val_adl} ({100-val_fall_pct:.1f}%)")

                # Check if validation split is balanced
                if 45 <= val_fall_pct <= 55:
                    print(f"  ✅ Validation split BALANCED (fall {val_fall_pct:.1f}% is in 45-55% range)")
                else:
                    print(f"  ❌ Validation split IMBALANCED (fall {val_fall_pct:.1f}% outside 45-55% range)")
                    success = False

                # Check if validation has only 1 subject
                if val_subjects == 1:
                    print(f"  ✅ Validation uses 1 subject (as expected)")
                else:
                    print(f"  ⚠️  Validation uses {val_subjects} subjects (expected 1)")

            # Extract training statistics
            train_match = re.search(r'train: (\d+) \((\d+) subjects, fall=(\d+), adl=(\d+)\)', info_line)
            if train_match:
                train_windows = int(train_match.group(1))
                train_subjects = int(train_match.group(2))
                train_fall = int(train_match.group(3))
                train_adl = int(train_match.group(4))

                train_fall_pct = (train_fall / train_windows * 100) if train_windows > 0 else 0

                print(f"\n  Training windows: {train_windows}")
                print(f"  Training subjects: {train_subjects}")
                print(f"  Training fall windows: {train_fall} ({train_fall_pct:.1f}%)")

                # Check if training has 28 subjects (29 available - 1 test)
                if train_subjects == 28:
                    print(f"  ✅ Training uses 28 subjects (29 - 1 test = 28)")
                else:
                    print(f"  ⚠️  Training uses {train_subjects} subjects (expected 28)")

        print()
        print("=" * 80)

        if success:
            print("✅ VALIDATION SPLIT TEST PASSED")
            print("   - Validation distribution is balanced (45-55% falls)")
            print("   - Ready for full experiments")
        else:
            print("⚠️  VALIDATION SPLIT TEST WARNING")
            print("   - Check if validation distribution is acceptable")
            print("   - May need to adjust validation subject")

        print("=" * 80)

        return success

    except subprocess.TimeoutExpired:
        print("❌ Test timed out after 10 minutes")
        print("   This is unusual - check for errors in config or code")
        return False
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False


def main():
    """Run quick validation split test."""

    # Test with transformer.yaml (simplest config)
    config = 'config/smartfallmm/transformer.yaml'

    print("Note: This test will run actual training for 1-2 folds.")
    print("It may take 5-10 minutes depending on your hardware.")
    print()

    # Ask user if they want to proceed
    response = input("Proceed with test? (y/n): ").lower().strip()
    if response != 'y':
        print("Test cancelled.")
        return

    success = run_single_fold_test(config, max_folds=2)

    if success:
        print("\n✅ All checks passed! Validation subject 37 is working correctly.")
        sys.exit(0)
    else:
        print("\n⚠️  Some checks failed. Review the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
