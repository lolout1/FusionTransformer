"""
Validation Split Selector - Automatically selects optimal validation subjects
based on preprocessing configuration (motion filtering, modalities, etc.)

This module provides intelligent validation split selection to maintain
proper ADL ratios across different experimental configurations.

IMPORTANT: Some subjects have poor gyroscope data quality and should NOT be
used for validation/testing when using IMU (acc+gyro) models with timestamp alignment.
"""


# Subjects with corrupt gyroscope timestamps - DO NOT use for validation/test with IMU models
# These subjects have all their fall trials discarded after timestamp alignment
# They should be permanently fixed in training (train_only_subjects) for IMU experiments
POOR_GYRO_SUBJECTS = [29, 32, 35, 39]


def get_train_only_subjects(dataset_args, force_consistency=True):
    """
    Get subjects that should be permanently fixed in training (never used for testing).

    These subjects have data quality issues that make them unsuitable for evaluation,
    but they can still contribute valuable training data.

    Args:
        dataset_args: Dictionary of dataset arguments from config
        force_consistency: If True, always return POOR_GYRO_SUBJECTS for consistency
                          across all experiments (recommended for academic research)

    Returns:
        list: Subject IDs that should only be used for training

    Rationale:
        - Subjects [29, 32, 35, 39] have corrupt gyroscope timestamps that cause
          ALL their fall trials to be discarded after timestamp alignment.
        - For CONSISTENCY across all experiments (acc-only and acc+gyro), we exclude
          these subjects from testing in ALL configs. This ensures:
          1. Fair comparison between acc-only and acc+gyro models
          2. Same test subjects across all experimental conditions
          3. Reproducible and comparable results for academic publication
    """
    # For consistency across all experiments, always exclude poor gyro subjects
    # This ensures fair comparisons between acc-only and acc+gyro models
    if force_consistency:
        return POOR_GYRO_SUBJECTS.copy()

    # Legacy behavior: only exclude for IMU models with timestamp alignment
    modalities = dataset_args.get('modalities', ['accelerometer'])
    uses_gyro = 'gyroscope' in modalities
    enable_timestamp_alignment = dataset_args.get('enable_timestamp_alignment', False)

    if uses_gyro and enable_timestamp_alignment:
        return POOR_GYRO_SUBJECTS.copy()

    return []


def validate_imu_validation_subjects(validation_subjects, dataset_args, print_warning=True):
    """
    Check if validation subjects are appropriate for IMU models.

    Args:
        validation_subjects: List of validation subject IDs
        dataset_args: Dictionary of dataset arguments from config
        print_warning: Whether to print a warning if issues found

    Returns:
        tuple: (is_valid, problematic_subjects)
    """
    modalities = dataset_args.get('modalities', ['accelerometer'])
    uses_gyro = 'gyroscope' in modalities
    enable_timestamp_alignment = dataset_args.get('enable_timestamp_alignment', False)

    if not uses_gyro:
        return True, []

    problematic = [s for s in validation_subjects if s in POOR_GYRO_SUBJECTS]

    if problematic and print_warning:
        print(f"\n{'='*70}")
        print("WARNING: POOR GYROSCOPE DATA IN VALIDATION SUBJECTS")
        print(f"{'='*70}")
        print(f"Validation subjects {problematic} have corrupt gyroscope timestamps!")
        print(f"These subjects will have ALL fall trials discarded after alignment.")
        print(f"\nRecommendation: Use different validation subjects for IMU models.")
        print(f"Current validation: {validation_subjects}")
        print(f"Problematic subjects (train-only): {POOR_GYRO_SUBJECTS}")
        print(f"{'='*70}\n")

    return len(problematic) == 0, problematic


def get_optimal_validation_subjects(dataset_args):
    """
    Select optimal validation subjects based on dataset configuration.

    Args:
        dataset_args: Dictionary of dataset arguments from config

    Returns:
        list: Optimal validation subject IDs for this configuration

    Strategy:
        - Non-motion-filtering: Use subjects optimized for ~60% ADLs
        - Motion-filtering: Use subjects optimized for ~45-50% ADLs (adjusted for filtering)
    """

    enable_motion_filtering = dataset_args.get('enable_motion_filtering', False)
    modalities = dataset_args.get('modalities', ['accelerometer'])

    # Determine if using skeleton (different validation strategy)
    use_skeleton = 'skeleton' in modalities

    if use_skeleton:
        # For skeleton-based experiments (not motion filtered)
        # Use original validation split
        return [38, 46]

    elif enable_motion_filtering:
        # Motion filtering enabled - use split optimized for motion-filtered data
        # These subjects maintain better ADL ratios after aggressive motion filtering
        #
        # Analysis showed that with motion filtering:
        # - Many ADL trials get filtered out (low motion)
        # - Need subjects with more robust ADL data
        #
        # Subjects [48, 57] provide:
        # - Acc-only with motion filter: ~48-50% ADLs combined
        # - Acc+gyro with motion filter: ~46-48% ADLs combined
        # - Sufficient validation samples even after filtering
        return [48, 57]

    else:
        # Standard non-motion-filtering experiments
        # Use split optimized for ~60% ADLs
        #
        # Subjects [38, 44] provide:
        # - Acc-only: 60.2% ADLs (349 windows)
        # - Acc+gyro: 59.8% ADLs (246 windows)
        # - Validated to work across both modality configurations
        return [38, 44]


def get_validation_split_info(validation_subjects):
    """
    Get human-readable information about a validation split.

    Args:
        validation_subjects: List of validation subject IDs

    Returns:
        str: Description of the validation split
    """

    split_descriptions = {
        str([38, 44]): "Standard split (60% ADLs) - optimized for non-filtered experiments",
        str([48, 57]): "Motion-filter split (45-50% ADLs) - optimized for motion-filtered experiments",
        str([38, 46]): "Skeleton split - optimized for skeleton-based experiments",
    }

    return split_descriptions.get(str(validation_subjects), f"Custom split: {validation_subjects}")


# Validation split metadata for documentation
VALIDATION_SPLITS = {
    'standard': {
        'subjects': [38, 44],
        'use_cases': ['acc-only without motion filtering', 'acc+gyro without motion filtering'],
        'performance': {
            'acc_only': {'adl_ratio': 0.602, 'windows': 349},
            'acc_gyro': {'adl_ratio': 0.598, 'windows': 246},
        },
        'description': 'Optimized for ~60% ADL ratio in standard experiments'
    },
    'motion_filtered': {
        'subjects': [48, 57],
        'use_cases': ['acc-only with motion filtering', 'acc+gyro with motion filtering'],
        'performance': {
            'acc_only_filtered': {'adl_ratio': 0.48, 'windows': 200},  # Estimated
            'acc_gyro_filtered': {'adl_ratio': 0.46, 'windows': 140},  # Estimated
        },
        'description': 'Optimized for ~45-50% ADL ratio in motion-filtered experiments'
    },
    'skeleton': {
        'subjects': [38, 46],
        'use_cases': ['skeleton-based experiments'],
        'performance': {
            'skeleton': {'adl_ratio': 0.55, 'windows': 300},  # Estimated
        },
        'description': 'Optimized for skeleton-based fall detection'
    },
}
