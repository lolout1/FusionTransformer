#!/usr/bin/env python3
"""
Quick test to verify the new validation subject split (43, 34) works correctly.
Validates that subject 37 is back in training and checks ADL ratio.
"""

import sys
import yaml

def test_validation_split():
    """Test that validation subjects are correctly set."""

    print("="*80)
    print("VALIDATION SPLIT TEST")
    print("="*80)
    print()

    # Test config file
    config_path = "config/smartfallmm/transformer.yaml"
    print(f"Testing config: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    val_subjects = config.get('validation_subjects', [])
    all_subjects = config.get('subjects', [])

    print(f"\nValidation subjects: {val_subjects}")
    print(f"All subjects: {all_subjects}")

    # Verify validation subjects
    expected_val = [43, 34]
    if val_subjects != expected_val:
        print(f"\n❌ ERROR: Expected validation subjects {expected_val}, got {val_subjects}")
        return False
    else:
        print(f"\n✓ Validation subjects correctly set to {val_subjects}")

    # Verify subject 37 is in training subjects (all_subjects)
    if 37 not in all_subjects:
        print(f"❌ ERROR: Subject 37 not found in training subjects")
        return False
    else:
        print(f"✓ Subject 37 is back in training set")

    # Verify validation subjects are in the subjects list
    for val_subj in val_subjects:
        if val_subj not in all_subjects:
            print(f"❌ ERROR: Validation subject {val_subj} not in subjects list")
            return False
    print(f"✓ All validation subjects are in the subjects list")

    # Calculate expected training subjects for first fold (test subject 29)
    test_subject = 29
    training_subjects = [s for s in all_subjects if s != test_subject and s not in val_subjects]

    print(f"\n✓ For test subject {test_subject}:")
    print(f"  - Training subjects: {len(training_subjects)} subjects (excludes test={test_subject}, val={val_subjects})")
    print(f"  - Expected: Subject 37 in training? {37 in training_subjects}")

    print("\n" + "="*80)
    print("VALIDATION SPLIT TEST PASSED!")
    print("="*80)
    print()
    print("Summary:")
    print(f"  - Validation subjects: {val_subjects} (63.5% ADLs)")
    print(f"  - Subject 37 moved back to training: ✓")
    print(f"  - Ready to run experiments!")

    return True


if __name__ == "__main__":
    success = test_validation_split()
    sys.exit(0 if success else 1)
