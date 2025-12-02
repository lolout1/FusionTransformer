"""
Dynamic test subject grouping for consistent fall:ADL ratios in LOSO CV.

This module groups test subjects in sets of 2-3 to ensure each test fold
maintains a fall:ADL ratio consistent with the validation set, preventing
evaluation bias due to subject-level class imbalance.

Key Features:
- Modality-aware: Works correctly for both acc-only and acc+gyro
- Extreme subject handling: Subjects with 0% or 100% falls moved to train_only
- Uses exact same preprocessing as training for accurate statistics
- Backwards compatible: Can be disabled to use single-subject LOSO

Usage:
    from utils.test_fold_grouper import create_test_fold_groups

    result = create_test_fold_groups(
        arg=self.arg,
        builder=stats_builder,
        test_candidates=test_candidates,
        validation_subjects=self.arg.validation_subjects,
    )

    test_folds = result.test_folds
    # Add extreme subjects to train_only
    train_only_subjects += result.extreme_subjects
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from itertools import combinations
from collections import defaultdict
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class SubjectStats:
    """Per-subject window statistics for fall detection."""
    subject_id: int
    fall_windows: int
    adl_windows: int

    @property
    def total_windows(self) -> int:
        """Total number of windows for this subject."""
        return self.fall_windows + self.adl_windows

    @property
    def fall_ratio(self) -> float:
        """Proportion of windows that are falls (0.0 to 1.0)."""
        if self.total_windows == 0:
            return 0.0
        return self.fall_windows / self.total_windows

    @property
    def adl_ratio(self) -> float:
        """Proportion of windows that are ADLs (0.0 to 1.0)."""
        return 1.0 - self.fall_ratio

    def __repr__(self) -> str:
        return (f"SubjectStats(id={self.subject_id}, "
                f"fall={self.fall_windows}, adl={self.adl_windows}, "
                f"ratio={self.fall_ratio:.3f})")


@dataclass
class TestFoldGroup:
    """A group of subjects forming one test fold."""
    subjects: List[int]
    combined_fall_windows: int
    combined_adl_windows: int

    @property
    def combined_total(self) -> int:
        """Total windows across all subjects in this group."""
        return self.combined_fall_windows + self.combined_adl_windows

    @property
    def fall_ratio(self) -> float:
        """Combined fall ratio for this group."""
        if self.combined_total == 0:
            return 0.0
        return self.combined_fall_windows / self.combined_total

    def deviation_from_target(self, target_fall_ratio: float) -> float:
        """Absolute deviation from target ratio."""
        return abs(self.fall_ratio - target_fall_ratio)

    def __repr__(self) -> str:
        return (f"TestFoldGroup(subjects={self.subjects}, "
                f"fall={self.combined_fall_windows}, adl={self.combined_adl_windows}, "
                f"ratio={self.fall_ratio:.3f})")


@dataclass
class TestFoldGroupingResult:
    """Result of test fold grouping computation."""
    test_folds: List[List[int]]         # Groups of subjects for testing
    extreme_subjects: List[int]          # Subjects moved to train_only
    target_fall_ratio: float             # Validation set fall ratio used as target
    mean_deviation: float                # Average deviation from target across folds
    max_deviation: float                 # Maximum deviation from target
    fold_details: List[TestFoldGroup]    # Detailed info for each fold

    def __repr__(self) -> str:
        return (f"TestFoldGroupingResult(\n"
                f"  folds={len(self.test_folds)}, "
                f"extreme={self.extreme_subjects},\n"
                f"  target_ratio={self.target_fall_ratio:.3f}, "
                f"mean_dev={self.mean_deviation:.3f}, "
                f"max_dev={self.max_deviation:.3f}\n"
                f")")


class TestFoldGrouper:
    """
    Groups test subjects to achieve consistent fall:ADL ratios across folds.

    Uses greedy bin-packing to pair high-fall-ratio subjects with
    low-fall-ratio subjects, targeting the validation set ratio.

    Subjects with extreme ratios (0% or 100% falls) are automatically
    excluded from test folds and should be moved to train_only_subjects.
    """

    def __init__(
        self,
        subject_stats: Dict[int, SubjectStats],
        target_fall_ratio: float,
        min_group_size: int = 2,
        max_group_size: int = 3,
        ratio_tolerance: float = 0.10,
        min_windows_per_group: int = 50,
        extreme_ratio_threshold: float = 0.05
    ):
        """
        Initialize the test fold grouper.

        Args:
            subject_stats: Dictionary mapping subject_id to SubjectStats
            target_fall_ratio: Target fall ratio (from validation set)
            min_group_size: Minimum subjects per group (default: 2)
            max_group_size: Maximum subjects per group (default: 3)
            ratio_tolerance: Acceptable deviation from target (default: 0.10)
            min_windows_per_group: Minimum windows per group (default: 50)
            extreme_ratio_threshold: Subjects with fall ratio < threshold or
                                      > (1-threshold) are moved to train_only
        """
        self.subject_stats = subject_stats
        self.target_fall_ratio = target_fall_ratio
        self.min_group_size = min_group_size
        self.max_group_size = max_group_size
        self.ratio_tolerance = ratio_tolerance
        self.min_windows_per_group = min_windows_per_group
        self.extreme_ratio_threshold = extreme_ratio_threshold

        self._groups: List[TestFoldGroup] = []
        self._extreme_subjects: List[int] = []
        self._is_computed = False

    def identify_extreme_subjects(self) -> List[int]:
        """
        Identify subjects with extreme fall ratios that should be moved to train set.

        Returns:
            List of subject IDs with 0%, 100%, or near-extreme fall ratios
        """
        extreme = []
        for sid, stats in self.subject_stats.items():
            if stats.total_windows == 0:
                extreme.append(sid)
                logger.warning(
                    f"Subject {sid}: No windows available - moving to train_only"
                )
            elif stats.fall_ratio < self.extreme_ratio_threshold:
                extreme.append(sid)
                logger.warning(
                    f"Subject {sid}: fall_ratio={stats.fall_ratio:.3f} "
                    f"(< {self.extreme_ratio_threshold}) - moving to train_only"
                )
            elif stats.fall_ratio > (1.0 - self.extreme_ratio_threshold):
                extreme.append(sid)
                logger.warning(
                    f"Subject {sid}: fall_ratio={stats.fall_ratio:.3f} "
                    f"(> {1.0 - self.extreme_ratio_threshold}) - moving to train_only"
                )

        self._extreme_subjects = extreme
        return extreme

    def compute_optimal_groupings(self) -> List[TestFoldGroup]:
        """
        Compute optimal subject groupings using greedy bin-packing.

        Algorithm:
        1. Identify and exclude extreme subjects (move to train_only)
        2. Sort remaining by deviation from target (largest first)
        3. Greedily form groups that minimize deviation from target
        4. Handle remainders by adding to existing groups or forming final group

        Returns:
            List of TestFoldGroup objects
        """
        if self._is_computed:
            return self._groups

        # Step 1: Identify extreme subjects to exclude
        extreme_subjects = self.identify_extreme_subjects()

        # Filter to only non-extreme subjects with data
        subjects = [s for s in self.subject_stats.values()
                    if s.subject_id not in extreme_subjects
                    and s.total_windows > 0]

        # Handle edge case: too few subjects
        if len(subjects) < self.min_group_size:
            if len(subjects) > 0:
                logger.warning(
                    f"Too few subjects ({len(subjects)}) for grouping. "
                    f"Creating single group with all remaining subjects."
                )
                self._groups = [self._create_group([s.subject_id for s in subjects])]
            else:
                logger.error("No subjects available for grouping after filtering extremes.")
                self._groups = []
            self._is_computed = True
            return self._groups

        # Sort by deviation from target (largest deviation first)
        # This ensures we handle the hardest-to-place subjects first
        subjects_sorted = sorted(
            subjects,
            key=lambda s: abs(s.fall_ratio - self.target_fall_ratio),
            reverse=True
        )

        unassigned: Set[int] = set(s.subject_id for s in subjects_sorted)
        groups: List[TestFoldGroup] = []

        while len(unassigned) >= self.min_group_size:
            best_group: Optional[TestFoldGroup] = None
            best_score = float('inf')

            # Try all combinations of min_size to max_size
            for size in range(self.min_group_size, self.max_group_size + 1):
                if len(unassigned) < size:
                    continue

                for combo in combinations(unassigned, size):
                    group = self._create_group(list(combo))
                    score = self._compute_group_score(group)

                    if score < best_score:
                        best_score = score
                        best_group = group

            if best_group is None:
                break

            groups.append(best_group)
            for sid in best_group.subjects:
                unassigned.discard(sid)

        # Handle remainder subjects
        if unassigned:
            remainder = list(unassigned)
            if len(remainder) == 1 and groups:
                # Add single remainder to best-matching existing group
                best_idx = self._find_best_group_for_subject(groups, remainder[0])
                old_subjects = groups[best_idx].subjects
                new_subjects = old_subjects + remainder
                groups[best_idx] = self._create_group(new_subjects)
                logger.info(
                    f"Added remainder subject {remainder[0]} to fold {best_idx + 1} "
                    f"(now {len(new_subjects)} subjects)"
                )
            else:
                # Form final group with remainder
                groups.append(self._create_group(remainder))
                logger.info(
                    f"Created final group with {len(remainder)} remainder subjects: {remainder}"
                )

        self._groups = groups
        self._is_computed = True

        self._log_grouping_summary()

        return self._groups

    def _compute_group_score(self, group: TestFoldGroup) -> float:
        """
        Compute a score for a potential group (lower is better).

        The score combines:
        - Deviation from target ratio (primary)
        - Penalty for insufficient windows (secondary)
        """
        score = group.deviation_from_target(self.target_fall_ratio)

        # Penalty for too few windows
        if group.combined_total < self.min_windows_per_group:
            score += 1.0

        return score

    def _create_group(self, subject_ids: List[int]) -> TestFoldGroup:
        """Create a TestFoldGroup from subject IDs."""
        fall_sum = sum(
            self.subject_stats[sid].fall_windows
            for sid in subject_ids
            if sid in self.subject_stats
        )
        adl_sum = sum(
            self.subject_stats[sid].adl_windows
            for sid in subject_ids
            if sid in self.subject_stats
        )
        return TestFoldGroup(
            subjects=list(subject_ids),
            combined_fall_windows=fall_sum,
            combined_adl_windows=adl_sum
        )

    def _find_best_group_for_subject(
        self,
        groups: List[TestFoldGroup],
        subject_id: int
    ) -> int:
        """
        Find index of group that would benefit most from adding this subject.

        Returns the index of the group where adding this subject
        would result in the smallest deviation from target.
        """
        subject = self.subject_stats[subject_id]
        best_idx = 0
        best_new_deviation = float('inf')

        for i, group in enumerate(groups):
            # Simulate adding subject
            new_fall = group.combined_fall_windows + subject.fall_windows
            new_adl = group.combined_adl_windows + subject.adl_windows
            new_total = new_fall + new_adl
            new_ratio = new_fall / new_total if new_total > 0 else 0
            new_dev = abs(new_ratio - self.target_fall_ratio)

            if new_dev < best_new_deviation:
                best_new_deviation = new_dev
                best_idx = i

        return best_idx

    def _log_grouping_summary(self) -> None:
        """Log summary of computed groupings."""
        if not self._groups:
            logger.warning("No groups were created.")
            return

        deviations = [g.deviation_from_target(self.target_fall_ratio) for g in self._groups]

        print(f"\n{'='*70}")
        print("TEST FOLD GROUPING SUMMARY")
        print(f"{'='*70}")
        print(f"Target fall ratio: {self.target_fall_ratio:.3f}")
        print(f"Number of test folds: {len(self._groups)}")
        if self._extreme_subjects:
            print(f"Extreme subjects (→ train_only): {self._extreme_subjects}")
        print(f"\nFold Details:")
        print(f"{'Fold':<6} {'Subjects':<20} {'Falls':<8} {'ADLs':<8} {'Ratio':<8} {'Dev':<8} {'Status'}")
        print("-" * 70)

        for i, group in enumerate(self._groups):
            dev = group.deviation_from_target(self.target_fall_ratio)
            status = "OK" if dev <= self.ratio_tolerance else "WARNING"
            subjects_str = str(group.subjects)
            if len(subjects_str) > 18:
                subjects_str = subjects_str[:15] + "..."
            print(
                f"{i+1:<6} {subjects_str:<20} {group.combined_fall_windows:<8} "
                f"{group.combined_adl_windows:<8} {group.fall_ratio:<8.3f} "
                f"{dev:<8.3f} {status}"
            )

        print("-" * 70)
        print(f"Mean deviation: {np.mean(deviations):.4f}")
        print(f"Max deviation:  {np.max(deviations):.4f}")
        print(f"{'='*70}\n")

        # Also log for file output
        logger.info(f"Test Fold Grouping: {len(self._groups)} folds, "
                   f"target_ratio={self.target_fall_ratio:.3f}, "
                   f"mean_dev={np.mean(deviations):.4f}, "
                   f"max_dev={np.max(deviations):.4f}")

    def get_fold_assignments(self) -> List[List[int]]:
        """Return list of subject lists for each test fold."""
        if not self._is_computed:
            self.compute_optimal_groupings()
        return [group.subjects for group in self._groups]

    def get_extreme_subjects(self) -> List[int]:
        """Return list of subjects with extreme ratios (to be moved to train_only)."""
        if not self._is_computed:
            self.compute_optimal_groupings()
        return self._extreme_subjects

    def get_result(self) -> TestFoldGroupingResult:
        """Get comprehensive grouping result."""
        if not self._is_computed:
            self.compute_optimal_groupings()

        if not self._groups:
            return TestFoldGroupingResult(
                test_folds=[],
                extreme_subjects=self._extreme_subjects,
                target_fall_ratio=self.target_fall_ratio,
                mean_deviation=0.0,
                max_deviation=0.0,
                fold_details=[]
            )

        deviations = [g.deviation_from_target(self.target_fall_ratio) for g in self._groups]

        return TestFoldGroupingResult(
            test_folds=[g.subjects for g in self._groups],
            extreme_subjects=self._extreme_subjects,
            target_fall_ratio=self.target_fall_ratio,
            mean_deviation=float(np.mean(deviations)),
            max_deviation=float(np.max(deviations)),
            fold_details=self._groups
        )


def collect_subject_stats_from_builder(
    builder,
    subjects: List[int]
) -> Dict[int, SubjectStats]:
    """
    Extract per-subject statistics from DatasetBuilder.

    Args:
        builder: DatasetBuilder instance after make_dataset() called
        subjects: List of subject IDs to collect stats for

    Returns:
        Dictionary mapping subject_id to SubjectStats
    """
    stats = {}
    for sid in subjects:
        if sid in builder.subject_modality_stats:
            s = builder.subject_modality_stats[sid]
            stats[sid] = SubjectStats(
                subject_id=sid,
                fall_windows=s.get('fall_windows', 0),
                adl_windows=s.get('adl_windows', 0)
            )
        else:
            logger.warning(f"Subject {sid} not found in builder stats, skipping")

    return stats


def create_test_fold_groups(
    arg,
    builder,
    test_candidates: List[int],
    validation_subjects: List[int],
    min_group_size: int = 2,
    max_group_size: int = 3,
    ratio_tolerance: float = 0.10,
    min_windows_per_group: int = 50,
    extreme_ratio_threshold: float = 0.05
) -> TestFoldGroupingResult:
    """
    High-level API for creating test fold groups.

    This function:
    1. Collects subject statistics from the builder
    2. Computes validation set target fall ratio
    3. Identifies extreme subjects to move to train_only
    4. Creates optimal groupings for remaining test candidates

    Args:
        arg: Argument namespace with dataset configuration
        builder: DatasetBuilder instance after make_dataset()
        test_candidates: List of subject IDs available for testing
        validation_subjects: List of validation subject IDs (for target ratio)
        min_group_size: Minimum subjects per group (default: 2)
        max_group_size: Maximum subjects per group (default: 3)
        ratio_tolerance: Acceptable deviation from target (default: 0.10)
        min_windows_per_group: Minimum windows per group (default: 50)
        extreme_ratio_threshold: Subjects with fall ratio < threshold or
                                  > (1-threshold) are moved to train_only

    Returns:
        TestFoldGroupingResult with test_folds, extreme_subjects, and statistics
    """
    # Collect stats for all relevant subjects
    all_subjects = list(set(test_candidates + validation_subjects))
    subject_stats = collect_subject_stats_from_builder(builder, all_subjects)

    if not subject_stats:
        logger.error("No subject statistics available. Cannot create fold groups.")
        return TestFoldGroupingResult(
            test_folds=[[s] for s in test_candidates],  # Fallback to single-subject
            extreme_subjects=[],
            target_fall_ratio=0.4,
            mean_deviation=0.0,
            max_deviation=0.0,
            fold_details=[]
        )

    # Calculate target ratio from validation subjects
    val_fall = sum(
        subject_stats[s].fall_windows
        for s in validation_subjects
        if s in subject_stats
    )
    val_adl = sum(
        subject_stats[s].adl_windows
        for s in validation_subjects
        if s in subject_stats
    )
    val_total = val_fall + val_adl

    if val_total == 0:
        logger.warning(
            "Validation set has no windows. Using default 0.4 fall ratio."
        )
        target_fall_ratio = 0.4
    else:
        target_fall_ratio = val_fall / val_total

    print(f"\nValidation set statistics:")
    print(f"  Falls: {val_fall}, ADLs: {val_adl}")
    print(f"  Target fall ratio: {target_fall_ratio:.3f}")

    # Filter to only test candidates with stats
    test_stats = {
        sid: subject_stats[sid]
        for sid in test_candidates
        if sid in subject_stats
    }

    if not test_stats:
        logger.error(
            "No test candidates have statistics. Returning single-subject folds."
        )
        return TestFoldGroupingResult(
            test_folds=[[s] for s in test_candidates],
            extreme_subjects=[],
            target_fall_ratio=target_fall_ratio,
            mean_deviation=0.0,
            max_deviation=0.0,
            fold_details=[]
        )

    # Log per-subject statistics before grouping
    print(f"\nTest candidate statistics ({len(test_stats)} subjects):")
    print(f"{'Subject':<10} {'Falls':<8} {'ADLs':<8} {'Total':<8} {'Ratio':<8}")
    print("-" * 42)
    for sid in sorted(test_stats.keys()):
        s = test_stats[sid]
        print(f"{sid:<10} {s.fall_windows:<8} {s.adl_windows:<8} "
              f"{s.total_windows:<8} {s.fall_ratio:<8.3f}")

    # Create grouper and compute
    grouper = TestFoldGrouper(
        subject_stats=test_stats,
        target_fall_ratio=target_fall_ratio,
        min_group_size=min_group_size,
        max_group_size=max_group_size,
        ratio_tolerance=ratio_tolerance,
        min_windows_per_group=min_windows_per_group,
        extreme_ratio_threshold=extreme_ratio_threshold
    )

    return grouper.get_result()
