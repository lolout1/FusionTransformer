#!/usr/bin/env python3
"""
Calculate actual window counts for different stride configurations on SmartFallMM.

Usage:
    python tools/calculate_stride_stats.py
"""

import os
import sys
sys.path.insert(0, '.')

import yaml
import numpy as np
from collections import defaultdict
from utils.dataset import SmartFallMM
from utils.loader import DatasetBuilder


def calculate_stats(fall_stride: int, adl_stride: int, dataset: SmartFallMM,
                    base_args: dict, subjects: list) -> dict:
    """Calculate window counts for given stride configuration."""

    args = base_args.copy()
    # Always enable class-aware stride to ensure explicit control
    args['enable_class_aware_stride'] = True
    args['fall_stride'] = fall_stride
    args['adl_stride'] = adl_stride
    args['stride'] = adl_stride  # Default stride (used as fallback)

    builder = DatasetBuilder(dataset, **args)
    builder.make_dataset(subjects, fuse=True)

    labels = builder.data.get('labels', np.array([]))
    if len(labels) > 0:
        fall_windows = int(np.sum(labels == 1))
        adl_windows = int(np.sum(labels == 0))
    else:
        fall_windows = 0
        adl_windows = 0

    fall_trials = builder.skip_stats.get('fall_trials', 0)
    adl_trials = builder.skip_stats.get('adl_trials', 0)
    total = fall_windows + adl_windows

    return {
        'fall_windows': fall_windows,
        'adl_windows': adl_windows,
        'fall_trials': fall_trials,
        'adl_trials': adl_trials,
        'total_windows': total,
        'ratio': fall_windows / adl_windows if adl_windows > 0 else 0,
        'pct_fall': 100 * fall_windows / total if total > 0 else 0,
    }


def main():
    with open('config/best_config/smartfallmm/kalman.yaml') as f:
        cfg = yaml.safe_load(f)

    subjects = cfg['subjects']
    train_only = cfg.get('train_only_subjects', [])
    dataset_args = cfg['dataset_args']

    print("Loading SmartFallMM dataset...")

    data_path = os.path.join(os.getcwd(), 'data')
    dataset = SmartFallMM(root_dir=data_path)
    dataset.pipe_line(
        age_group=dataset_args['age_group'],
        modalities=dataset_args['modalities'],
        sensors=dataset_args['sensors']
    )

    print(f"\nData path: {data_path}")
    print(f"Total subjects: {len(subjects)}")
    print(f"Train-only subjects: {len(train_only)}")
    print(f"Test subjects: {len(subjects) - len(train_only)}")
    print(f"Matched trials: {len(dataset.matched_trials)}")
    print()

    fall_stride = 10
    adl_strides = [10, 20, 30, 40, 50]

    print("=" * 95)
    print(f"{'ADL Stride':<12} {'Fall Win':<12} {'ADL Win':<12} {'Ratio':<10} {'% Fall':<10} {'Fall Tr':<10} {'ADL Tr':<10}")
    print("=" * 95)

    for adl_stride in adl_strides:
        stats = calculate_stats(fall_stride, adl_stride, dataset, dataset_args, subjects)
        print(f"{adl_stride:<12} {stats['fall_windows']:<12} {stats['adl_windows']:<12} "
              f"{stats['ratio']:.4f}    {stats['pct_fall']:.1f}%      "
              f"{stats['fall_trials']:<10} {stats['adl_trials']:<10}")

    print("=" * 95)
    print(f"\nFall stride fixed at: {fall_stride}")


if __name__ == '__main__':
    main()
