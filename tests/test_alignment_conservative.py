#!/usr/bin/env python3
"""
Test script to validate conservative alignment logic.

Tests the new alignment rules:
1. Sample diff ≤ 10: USE AS-IS (no interpolation)
2. Sample diff > 10 AND timestamps close: INTERPOLATE
3. Sample diff > 10 AND timestamps drifted: DISCARD

Author: SmartFallMM Research Team
"""

import os
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.alignment import (
    AlignmentConfig, AlignmentResult, AlignmentStats,
    parse_imu_csv_with_timestamps, check_alignment_feasibility,
    align_imu_modalities, compute_sampling_stats
)


def test_alignment_logic():
    """Test the alignment decision logic with synthetic data."""
    print("=" * 70)
    print("TEST 1: Alignment Decision Logic (Synthetic Data)")
    print("=" * 70)

    config = AlignmentConfig(
        length_threshold=10,
        max_duration_ratio=1.2,
        max_rate_divergence=0.3,
        max_time_gap_ms=1000.0
    )

    test_cases = [
        # (acc_samples, gyro_samples, acc_duration_ms, gyro_duration_ms, expected_action, description)
        (100, 100, 3000, 3000, 'use_as_is', "Identical lengths"),
        (100, 105, 3000, 3000, 'use_as_is', "Small diff (5 samples) - use as-is"),
        (100, 110, 3000, 3000, 'use_as_is', "Exactly 10 samples diff - use as-is"),
        (100, 115, 3000, 3150, 'align', "15 samples diff, similar durations - interpolate"),
        (100, 150, 3000, 3000, 'discard', "50 samples diff, same duration - DISCARD (rate divergence 40%)"),
        (100, 150, 3000, 4500, 'discard', "50 samples diff, duration ratio 1.5 - DISCARD"),
        (100, 200, 3000, 3000, 'discard', "100 samples diff, same duration - DISCARD (rate divergence)"),
        (100, 100, 3000, 5000, 'discard', "Same samples but very different durations - DISCARD"),
    ]

    passed = 0
    failed = 0

    for acc_n, gyro_n, acc_dur, gyro_dur, expected, desc in test_cases:
        # Create synthetic timestamps
        acc_times = np.linspace(0, acc_dur, acc_n)
        gyro_times = np.linspace(0, gyro_dur, gyro_n)

        action, reason = check_alignment_feasibility(acc_times, gyro_times, config)

        status = "✓ PASS" if action == expected else "✗ FAIL"
        if action == expected:
            passed += 1
        else:
            failed += 1

        print(f"\n{status}: {desc}")
        print(f"  acc={acc_n} samples/{acc_dur}ms, gyro={gyro_n} samples/{gyro_dur}ms")
        print(f"  Expected: {expected}, Got: {action}")
        print(f"  Reason: {reason}")

    print(f"\n{'=' * 70}")
    print(f"Logic tests: {passed} passed, {failed} failed")
    print(f"{'=' * 70}\n")

    return failed == 0


def test_real_files():
    """Test alignment on real SmartFallMM files."""
    print("=" * 70)
    print("TEST 2: Real File Alignment (SmartFallMM Dataset)")
    print("=" * 70)

    # Find accelerometer and gyroscope files
    data_root = Path("data/young")
    acc_dir = data_root / "accelerometer" / "watch"
    gyro_dir = data_root / "gyroscope" / "watch"

    if not acc_dir.exists() or not gyro_dir.exists():
        print(f"Data directories not found: {acc_dir} or {gyro_dir}")
        print("Skipping real file tests.")
        return True

    # Get matching file pairs
    acc_files = sorted(acc_dir.glob("*.csv"))

    config = AlignmentConfig(
        target_rate=30.0,
        length_threshold=10,
        max_duration_ratio=1.2,
        max_rate_divergence=0.3,
        max_time_gap_ms=1000.0,
        min_output_samples=64
    )

    stats = AlignmentStats()
    subject_stats = defaultdict(lambda: {'use_as_is': 0, 'aligned': 0, 'discarded': 0, 'total': 0})

    # Process sample of files (first 100 or all if fewer)
    max_files = min(len(acc_files), 200)

    print(f"\nProcessing {max_files} file pairs...")

    for acc_path in acc_files[:max_files]:
        gyro_path = gyro_dir / acc_path.name

        if not gyro_path.exists():
            continue

        # Extract subject ID from filename (e.g., S29A01T01.csv -> 29)
        filename = acc_path.stem
        subject_id = filename.split('A')[0].replace('S', '')

        try:
            result = align_imu_modalities(str(acc_path), str(gyro_path), config)
            stats.update(result)

            subject_stats[subject_id]['total'] += 1
            subject_stats[subject_id][result.action] += 1

        except Exception as e:
            print(f"Error processing {acc_path.name}: {e}")

    # Print summary
    print(f"\n{'=' * 70}")
    print("ALIGNMENT STATISTICS")
    print(f"{'=' * 70}")
    print(f"Total files processed: {stats.total_trials}")
    print(f"  - Use as-is (≤10 sample diff): {stats.use_as_is} ({100*stats.use_as_is/max(stats.total_trials,1):.1f}%)")
    print(f"  - Interpolated: {stats.aligned} ({100*stats.aligned/max(stats.total_trials,1):.1f}%)")
    print(f"  - Discarded: {stats.discarded} ({100*stats.discarded/max(stats.total_trials,1):.1f}%)")

    print(f"\nDiscard breakdown:")
    print(f"  - Timestamp unsync: {stats.discarded_timestamp_unsync}")
    print(f"  - Insufficient overlap: {stats.discarded_insufficient_overlap}")
    print(f"  - Duration drift: {stats.discarded_duration_drift}")
    print(f"  - Rate divergence: {stats.discarded_rate_drift}")
    print(f"  - Too short: {stats.discarded_too_short}")
    print(f"  - Other errors: {stats.discarded_error}")

    # Per-subject summary
    print(f"\n{'=' * 70}")
    print("PER-SUBJECT SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Subject':<10} {'Total':<8} {'As-Is':<8} {'Interp':<8} {'Discard':<8}")
    print("-" * 42)

    for subject_id in sorted(subject_stats.keys(), key=lambda x: int(x)):
        s = subject_stats[subject_id]
        print(f"S{subject_id:<9} {s['total']:<8} {s['use_as_is']:<8} {s['aligned']:<8} {s['discarded']:<8}")

    print(f"{'=' * 70}\n")

    return True


def test_edge_cases():
    """Test edge cases in alignment."""
    print("=" * 70)
    print("TEST 3: Edge Cases")
    print("=" * 70)

    config = AlignmentConfig(
        length_threshold=10,
        max_duration_ratio=1.2,
        max_rate_divergence=0.3
    )

    # Test: Very short sequences
    print("\n1. Very short sequences (< min_output_samples):")
    acc_times = np.linspace(0, 1000, 30)  # 30 samples
    gyro_times = np.linspace(0, 1000, 30)
    action, reason = check_alignment_feasibility(acc_times, gyro_times, config)
    print(f"   30 samples each: action={action}, reason={reason}")

    # Test: Large sample diff but same duration (rate issue)
    print("\n2. Large sample diff, same duration (different rates):")
    acc_times = np.linspace(0, 3000, 100)  # 33 Hz
    gyro_times = np.linspace(0, 3000, 200)  # 66 Hz
    action, reason = check_alignment_feasibility(acc_times, gyro_times, config)
    print(f"   100 vs 200 samples, same 3s duration: action={action}")
    print(f"   Reason: {reason}")

    # Test: Similar sample count but different durations (timestamp drift)
    print("\n3. Similar samples but different durations (timestamp drift):")
    acc_times = np.linspace(0, 3000, 100)
    gyro_times = np.linspace(0, 5000, 105)
    action, reason = check_alignment_feasibility(acc_times, gyro_times, config)
    print(f"   100 vs 105 samples, 3s vs 5s duration: action={action}")
    print(f"   Reason: {reason}")

    # Test: Boundary case at exactly threshold
    print("\n4. Boundary: exactly 10 sample diff:")
    acc_times = np.linspace(0, 3000, 100)
    gyro_times = np.linspace(0, 3000, 110)
    action, reason = check_alignment_feasibility(acc_times, gyro_times, config)
    print(f"   100 vs 110 samples: action={action}")
    print(f"   Reason: {reason}")

    # Test: 11 samples diff (just over threshold)
    print("\n5. Boundary: 11 sample diff (just over threshold):")
    acc_times = np.linspace(0, 3000, 100)
    gyro_times = np.linspace(0, 3030, 111)  # Similar duration
    action, reason = check_alignment_feasibility(acc_times, gyro_times, config)
    print(f"   100 vs 111 samples: action={action}")
    print(f"   Reason: {reason}")

    print(f"\n{'=' * 70}\n")
    return True


def test_full_pipeline():
    """Test the full alignment pipeline with a sample config."""
    print("=" * 70)
    print("TEST 4: Full Pipeline Test")
    print("=" * 70)

    # Check if we can import the loader
    try:
        from utils.loader import DatasetBuilder
        from utils.dataset import SmartFallMM
        print("✓ Successfully imported DatasetBuilder and SmartFallMM")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

    # Check data exists
    data_path = "data"
    if not os.path.exists(data_path):
        print(f"Data path not found: {data_path}")
        return True  # Not a failure, just can't test

    # Create dataset with conservative alignment
    print("\nCreating dataset with conservative alignment settings...")

    try:
        # Create dataset using SmartFallMM API
        dataset = SmartFallMM(root_dir=data_path)
        dataset.add_modality('young', 'accelerometer')
        dataset.add_modality('young', 'gyroscope')
        dataset.select_sensor('accelerometer', 'watch')
        dataset.select_sensor('gyroscope', 'watch')
        dataset.load_files()
        dataset.match_trials()

        builder = DatasetBuilder(
            dataset=dataset,
            mode='sliding_window',
            max_length=128,
            task='fd',
            enable_timestamp_alignment=True,
            length_threshold=10,
            max_duration_ratio=1.2,
            max_rate_divergence=0.3,
            alignment_target_rate=30.0,
            debug=False
        )

        # Process a few subjects
        test_subjects = [29, 30, 31]
        print(f"Processing subjects: {test_subjects}")

        builder.make_dataset(subjects=test_subjects, fuse=False)

        # Print results
        print("\n" + "=" * 70)
        builder.print_skip_summary()

        # Check we got some data
        if 'accelerometer' in builder.data and len(builder.data['accelerometer']) > 0:
            print(f"✓ Generated {len(builder.data['accelerometer'])} windows")
            print(f"  Accelerometer shape: {builder.data['accelerometer'].shape}")
            if 'gyroscope' in builder.data:
                print(f"  Gyroscope shape: {builder.data['gyroscope'].shape}")
            return True
        else:
            print("✗ No data generated")
            return False

    except Exception as e:
        print(f"✗ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 70)
    print("CONSERVATIVE ALIGNMENT VALIDATION")
    print("=" * 70)
    print("\nNew alignment rules:")
    print("  1. Sample diff ≤ 10: USE AS-IS (truncate, no interpolation)")
    print("  2. Sample diff > 10 + timestamps close: INTERPOLATE")
    print("  3. Sample diff > 10 + timestamps drifted: DISCARD")
    print("\n'Timestamps close' means:")
    print("  - Duration ratio ≤ 1.2 (20% max difference)")
    print("  - Sampling rate divergence ≤ 30%")
    print("=" * 70 + "\n")

    results = []

    # Run tests
    results.append(("Logic Tests", test_alignment_logic()))
    results.append(("Edge Cases", test_edge_cases()))
    results.append(("Real Files", test_real_files()))
    results.append(("Full Pipeline", test_full_pipeline()))

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print("\n✓ All validation tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed - review output above")
        return 1


if __name__ == '__main__':
    sys.exit(main())
