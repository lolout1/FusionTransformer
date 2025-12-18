#!/usr/bin/env python3
"""
Test script for the dynamic test fold grouper.

This script validates that the TestFoldGrouper correctly groups subjects
to achieve consistent fall:ADL ratios across test folds.
"""

import sys
sys.path.insert(0, '.')

from utils.test_fold_grouper import (
    SubjectStats,
    TestFoldGroup,
    TestFoldGrouper,
    create_test_fold_groups,
    TestFoldGroupingResult
)


def test_subject_stats():
    """Test SubjectStats dataclass."""
    print("\n=== Testing SubjectStats ===")

    # Normal case
    stats = SubjectStats(subject_id=1, fall_windows=30, adl_windows=70)
    assert stats.total_windows == 100
    assert abs(stats.fall_ratio - 0.3) < 0.001
    assert abs(stats.adl_ratio - 0.7) < 0.001
    print(f"  SubjectStats(1): fall_ratio={stats.fall_ratio:.3f} ✓")

    # Edge case: no windows
    stats_empty = SubjectStats(subject_id=2, fall_windows=0, adl_windows=0)
    assert stats_empty.total_windows == 0
    assert stats_empty.fall_ratio == 0.0
    print(f"  SubjectStats(2, empty): fall_ratio={stats_empty.fall_ratio:.3f} ✓")

    # Edge case: all falls
    stats_all_fall = SubjectStats(subject_id=3, fall_windows=100, adl_windows=0)
    assert abs(stats_all_fall.fall_ratio - 1.0) < 0.001
    print(f"  SubjectStats(3, all falls): fall_ratio={stats_all_fall.fall_ratio:.3f} ✓")

    print("  All SubjectStats tests passed! ✓")


def test_test_fold_group():
    """Test TestFoldGroup dataclass."""
    print("\n=== Testing TestFoldGroup ===")

    group = TestFoldGroup(
        subjects=[1, 2, 3],
        combined_fall_windows=60,
        combined_adl_windows=140
    )
    assert group.combined_total == 200
    assert abs(group.fall_ratio - 0.3) < 0.001
    assert abs(group.deviation_from_target(0.4) - 0.1) < 0.001
    print(f"  TestFoldGroup: fall_ratio={group.fall_ratio:.3f}, dev_from_0.4={group.deviation_from_target(0.4):.3f} ✓")

    print("  All TestFoldGroup tests passed! ✓")


def test_grouper_basic():
    """Test basic grouper functionality."""
    print("\n=== Testing TestFoldGrouper (basic) ===")

    # Create mock subject stats with varying ratios
    subject_stats = {
        # High fall ratio subjects
        43: SubjectStats(43, fall_windows=80, adl_windows=20),  # 0.8 fall
        44: SubjectStats(44, fall_windows=70, adl_windows=30),  # 0.7 fall
        # Low fall ratio subjects
        45: SubjectStats(45, fall_windows=20, adl_windows=80),  # 0.2 fall
        46: SubjectStats(46, fall_windows=30, adl_windows=70),  # 0.3 fall
        # Medium fall ratio subjects
        47: SubjectStats(47, fall_windows=40, adl_windows=60),  # 0.4 fall
        48: SubjectStats(48, fall_windows=45, adl_windows=55),  # 0.45 fall
    }

    target_ratio = 0.4  # Target: 40% falls

    grouper = TestFoldGrouper(
        subject_stats=subject_stats,
        target_fall_ratio=target_ratio,
        min_group_size=2,
        max_group_size=3,
        ratio_tolerance=0.15
    )

    result = grouper.get_result()

    print(f"\n  Target fall ratio: {target_ratio:.3f}")
    print(f"  Number of groups: {len(result.test_folds)}")
    print(f"  Mean deviation: {result.mean_deviation:.4f}")
    print(f"  Max deviation: {result.max_deviation:.4f}")
    print(f"  Extreme subjects: {result.extreme_subjects}")

    # Verify all subjects are assigned exactly once
    all_assigned = []
    for fold in result.test_folds:
        all_assigned.extend(fold)

    all_subjects = set(subject_stats.keys()) - set(result.extreme_subjects)
    assert set(all_assigned) == all_subjects, "Not all subjects assigned!"
    print(f"\n  All {len(all_subjects)} subjects assigned to {len(result.test_folds)} folds ✓")

    # Verify groups have reasonable ratios
    for i, fold in enumerate(result.fold_details):
        print(f"  Fold {i+1}: subjects={fold.subjects}, ratio={fold.fall_ratio:.3f}")

    print("\n  Basic grouper tests passed! ✓")


def test_extreme_subject_handling():
    """Test that extreme subjects are moved to train_only."""
    print("\n=== Testing Extreme Subject Handling ===")

    subject_stats = {
        # Normal subjects
        50: SubjectStats(50, fall_windows=40, adl_windows=60),  # 0.4 fall
        51: SubjectStats(51, fall_windows=35, adl_windows=65),  # 0.35 fall
        52: SubjectStats(52, fall_windows=45, adl_windows=55),  # 0.45 fall
        53: SubjectStats(53, fall_windows=50, adl_windows=50),  # 0.5 fall
        # Extreme subjects (should be excluded)
        54: SubjectStats(54, fall_windows=2, adl_windows=98),   # 0.02 fall (< 0.05)
        55: SubjectStats(55, fall_windows=98, adl_windows=2),   # 0.98 fall (> 0.95)
        56: SubjectStats(56, fall_windows=0, adl_windows=0),    # No data
    }

    grouper = TestFoldGrouper(
        subject_stats=subject_stats,
        target_fall_ratio=0.4,
        min_group_size=2,
        max_group_size=2,
        extreme_ratio_threshold=0.05
    )

    result = grouper.get_result()

    print(f"  Extreme subjects identified: {result.extreme_subjects}")

    # Verify extreme subjects are excluded
    assert 54 in result.extreme_subjects, "Subject 54 should be extreme (0.02 fall)"
    assert 55 in result.extreme_subjects, "Subject 55 should be extreme (0.98 fall)"
    assert 56 in result.extreme_subjects, "Subject 56 should be extreme (no data)"

    # Verify extreme subjects are not in any fold
    for fold in result.test_folds:
        for extreme in result.extreme_subjects:
            assert extreme not in fold, f"Extreme subject {extreme} found in fold!"

    print(f"  {len(result.extreme_subjects)} extreme subjects correctly excluded ✓")
    print("  Extreme subject handling tests passed! ✓")


def test_edge_cases():
    """Test edge cases."""
    print("\n=== Testing Edge Cases ===")

    # Case 1: Too few subjects
    print("  Case 1: Too few subjects...")
    subject_stats_few = {
        60: SubjectStats(60, fall_windows=40, adl_windows=60),
    }
    grouper_few = TestFoldGrouper(
        subject_stats=subject_stats_few,
        target_fall_ratio=0.4,
        min_group_size=2,
        max_group_size=3
    )
    result_few = grouper_few.get_result()
    assert len(result_few.test_folds) == 1, "Should create single group"
    print(f"    Created {len(result_few.test_folds)} fold with {len(result_few.test_folds[0])} subjects ✓")

    # Case 2: Odd number of subjects (5)
    print("  Case 2: Odd number of subjects...")
    subject_stats_odd = {
        70: SubjectStats(70, fall_windows=50, adl_windows=50),  # 0.5
        71: SubjectStats(71, fall_windows=30, adl_windows=70),  # 0.3
        72: SubjectStats(72, fall_windows=60, adl_windows=40),  # 0.6
        73: SubjectStats(73, fall_windows=40, adl_windows=60),  # 0.4
        74: SubjectStats(74, fall_windows=35, adl_windows=65),  # 0.35
    }
    grouper_odd = TestFoldGrouper(
        subject_stats=subject_stats_odd,
        target_fall_ratio=0.4,
        min_group_size=2,
        max_group_size=3
    )
    result_odd = grouper_odd.get_result()
    total_subjects = sum(len(f) for f in result_odd.test_folds)
    assert total_subjects == 5, "All 5 subjects should be assigned"
    print(f"    All {total_subjects} subjects assigned to {len(result_odd.test_folds)} folds ✓")

    print("  Edge case tests passed! ✓")


def test_determinism():
    """Test that groupings are deterministic."""
    print("\n=== Testing Determinism ===")

    subject_stats = {
        80: SubjectStats(80, fall_windows=70, adl_windows=30),
        81: SubjectStats(81, fall_windows=25, adl_windows=75),
        82: SubjectStats(82, fall_windows=50, adl_windows=50),
        83: SubjectStats(83, fall_windows=40, adl_windows=60),
        84: SubjectStats(84, fall_windows=60, adl_windows=40),
        85: SubjectStats(85, fall_windows=35, adl_windows=65),
    }

    # Run grouper multiple times
    results = []
    for i in range(3):
        grouper = TestFoldGrouper(
            subject_stats=subject_stats,
            target_fall_ratio=0.45,
            min_group_size=2,
            max_group_size=3
        )
        result = grouper.get_result()
        results.append(result.test_folds)

    # All runs should produce identical results
    for i in range(1, len(results)):
        assert results[i] == results[0], f"Run {i+1} differs from run 1!"

    print(f"  {len(results)} runs produced identical groupings ✓")
    print("  Determinism tests passed! ✓")


def main():
    """Run all tests."""
    print("=" * 60)
    print("TEST FOLD GROUPER UNIT TESTS")
    print("=" * 60)

    try:
        test_subject_stats()
        test_test_fold_group()
        test_grouper_basic()
        test_extreme_subject_handling()
        test_edge_cases()
        test_determinism()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"\n\nTEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n\nUNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
