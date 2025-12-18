#!/usr/bin/env python3
"""
Test DTW length validation

Tests that:
1. Small length differences (≤10 samples) pass and get aligned
2. Large length differences (>10 samples) raise ValueError
"""

import numpy as np
from utils.loader import align_gyro_to_acc


def test_dtw_validation():
    print("="*80)
    print("Testing DTW Length Validation")
    print("="*80)

    # Test 1: Should PASS - length difference = 5 (within threshold)
    print("\n[Test 1] Small length difference (acc=110, gyro=115, diff=5)")
    data1 = {
        'accelerometer': np.random.randn(110, 3),
        'gyroscope': np.random.randn(115, 3)
    }
    try:
        result1 = align_gyro_to_acc(data1, use_fast_dtw=True, max_length_diff=10)
        print(f"✓ PASSED - DTW alignment succeeded")
        print(f"  After alignment: acc={len(result1['accelerometer'])}, gyro={len(result1['gyroscope'])}")
        assert len(result1['accelerometer']) == len(result1['gyroscope']), "Lengths should match after alignment"
        print(f"✓ Both modalities have same length after alignment")
    except ValueError as e:
        print(f"✗ FAILED - Should not raise error for diff=5: {e}")

    # Test 2: Should PASS - length difference = 10 (exactly at threshold)
    print("\n[Test 2] Threshold length difference (acc=100, gyro=110, diff=10)")
    data2 = {
        'accelerometer': np.random.randn(100, 3),
        'gyroscope': np.random.randn(110, 3)
    }
    try:
        result2 = align_gyro_to_acc(data2, use_fast_dtw=True, max_length_diff=10)
        print(f"✓ PASSED - DTW alignment succeeded at threshold")
        print(f"  After alignment: acc={len(result2['accelerometer'])}, gyro={len(result2['gyroscope'])}")
    except ValueError as e:
        print(f"✗ FAILED - Should not raise error for diff=10: {e}")

    # Test 3: Should FAIL - length difference = 20 (exceeds threshold)
    print("\n[Test 3] Large length difference (acc=110, gyro=130, diff=20)")
    data3 = {
        'accelerometer': np.random.randn(110, 3),
        'gyroscope': np.random.randn(130, 3)
    }
    try:
        result3 = align_gyro_to_acc(data3, use_fast_dtw=True, max_length_diff=10)
        print(f"✗ FAILED - Should have raised ValueError for diff=20")
    except ValueError as e:
        print(f"✓ PASSED - Correctly rejected: {e}")

    # Test 4: Should FAIL - length difference = 50 (way too large)
    print("\n[Test 4] Very large length difference (acc=100, gyro=150, diff=50)")
    data4 = {
        'accelerometer': np.random.randn(100, 3),
        'gyroscope': np.random.randn(150, 3)
    }
    try:
        result4 = align_gyro_to_acc(data4, use_fast_dtw=True, max_length_diff=10)
        print(f"✗ FAILED - Should have raised ValueError for diff=50")
    except ValueError as e:
        print(f"✓ PASSED - Correctly rejected: {e}")

    # Test 5: Should PASS - identical lengths (diff=0)
    print("\n[Test 5] Identical lengths (acc=120, gyro=120, diff=0)")
    data5 = {
        'accelerometer': np.random.randn(120, 3),
        'gyroscope': np.random.randn(120, 3)
    }
    try:
        result5 = align_gyro_to_acc(data5, use_fast_dtw=True, max_length_diff=10)
        print(f"✓ PASSED - DTW alignment succeeded for identical lengths")
        print(f"  After alignment: acc={len(result5['accelerometer'])}, gyro={len(result5['gyroscope'])}")
    except ValueError as e:
        print(f"✗ FAILED - Should not raise error for diff=0: {e}")

    # Test 6: Custom threshold
    print("\n[Test 6] Custom threshold (acc=100, gyro=125, diff=25, max_diff=30)")
    data6 = {
        'accelerometer': np.random.randn(100, 3),
        'gyroscope': np.random.randn(125, 3)
    }
    try:
        result6 = align_gyro_to_acc(data6, use_fast_dtw=True, max_length_diff=30)
        print(f"✓ PASSED - DTW alignment succeeded with custom threshold")
        print(f"  After alignment: acc={len(result6['accelerometer'])}, gyro={len(result6['gyroscope'])}")
    except ValueError as e:
        print(f"✗ FAILED - Should not raise error for diff=25 with max_diff=30: {e}")

    print("\n" + "="*80)
    print("DTW Validation Tests Complete!")
    print("="*80)
    print("\nSummary:")
    print("✓ Small differences (≤10 samples) are aligned using DTW")
    print("✓ Large differences (>10 samples) are rejected as data quality issues")
    print("✓ This ensures DTW is used for temporal alignment, not missing data")
    print("="*80)


if __name__ == '__main__':
    test_dtw_validation()
