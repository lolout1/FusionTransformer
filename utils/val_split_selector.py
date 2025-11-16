"""
Validation Split Selector - Automatically selects optimal validation subjects
based on preprocessing configuration (motion filtering, modalities, etc.)

This module provides intelligent validation split selection to maintain
proper ADL ratios across different experimental configurations.
"""


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
