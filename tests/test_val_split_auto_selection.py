#!/usr/bin/env python3
"""
Test validation split auto-selection logic.
Verifies that the correct validation subjects are selected based on configuration.
"""

import sys
from utils.val_split_selector import get_optimal_validation_subjects, get_validation_split_info, VALIDATION_SPLITS


def test_auto_selection():
    """Test various configurations."""

    print("="*80)
    print("VALIDATION SPLIT AUTO-SELECTION TEST")
    print("="*80)

    test_cases = [
        {
            'name': 'Acc-only, no motion filtering',
            'config': {
                'modalities': ['accelerometer'],
                'enable_motion_filtering': False,
            },
            'expected': [38, 44],
        },
        {
            'name': 'Acc-only, WITH motion filtering',
            'config': {
                'modalities': ['accelerometer'],
                'enable_motion_filtering': True,
            },
            'expected': [48, 57],
        },
        {
            'name': 'Acc+Gyro, no motion filtering',
            'config': {
                'modalities': ['accelerometer', 'gyroscope'],
                'enable_motion_filtering': False,
            },
            'expected': [38, 44],
        },
        {
            'name': 'Acc+Gyro, WITH motion filtering',
            'config': {
                'modalities': ['accelerometer', 'gyroscope'],
                'enable_motion_filtering': True,
            },
            'expected': [48, 57],
        },
        {
            'name': 'Skeleton-based',
            'config': {
                'modalities': ['skeleton', 'accelerometer'],
                'enable_motion_filtering': False,
            },
            'expected': [38, 46],
        },
    ]

    all_passed = True

    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['name']}")
        print("-"*80)

        result = get_optimal_validation_subjects(test['config'])
        expected = test['expected']

        if result == expected:
            print(f"  ✓ PASS")
            print(f"    Selected: {result}")
            print(f"    Info: {get_validation_split_info(result)}")
        else:
            print(f"  ✗ FAIL")
            print(f"    Expected: {expected}")
            print(f"    Got:      {result}")
            all_passed = False

    print("\n" + "="*80)
    print("VALIDATION SPLIT METADATA")
    print("="*80)

    for split_name, split_info in VALIDATION_SPLITS.items():
        print(f"\n{split_name.upper()}:")
        print(f"  Subjects: {split_info['subjects']}")
        print(f"  Description: {split_info['description']}")
        print(f"  Use cases:")
        for use_case in split_info['use_cases']:
            print(f"    - {use_case}")
        print(f"  Performance:")
        for config, perf in split_info['performance'].items():
            print(f"    {config}: {perf['adl_ratio']:.1%} ADLs ({perf['windows']} windows)")

    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*80)

    return all_passed


if __name__ == "__main__":
    success = test_auto_selection()
    sys.exit(0 if success else 1)
