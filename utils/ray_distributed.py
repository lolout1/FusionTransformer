"""
Ray Distributed LOSO Training for FusionTransformer.

This module provides Ray-based distributed training across multiple GPUs
with round-robin fold assignment, metrics aggregation, and W&B integration.

Features:
- Robust GPU resource management via Ray
- Graceful error handling with detailed reporting
- Comprehensive metrics aggregation and validation
- Progress tracking with ETA estimation
"""

import os
import sys
import time
import traceback
import shutil
import pickle
from pathlib import Path
from datetime import datetime
from argparse import Namespace
from typing import Dict, List, Tuple, Optional, Any
from copy import deepcopy

import numpy as np
import pandas as pd
import yaml
import ray

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def assign_folds_to_gpus(
    test_candidates: List[int],
    num_gpus: int
) -> Dict[int, List[Tuple[int, int]]]:
    """
    Assign folds to GPUs using round-robin distribution.

    Example with 21 folds, 3 GPUs:
    - GPU 0: folds 0, 3, 6, 9, 12, 15, 18 (7 folds)
    - GPU 1: folds 1, 4, 7, 10, 13, 16, 19 (7 folds)
    - GPU 2: folds 2, 5, 8, 11, 14, 17, 20 (7 folds)

    Args:
        test_candidates: List of test subject IDs
        num_gpus: Number of GPUs to distribute across

    Returns:
        Dict mapping gpu_id -> list of (fold_idx, subject_id) tuples
    """
    assignments = {gpu_id: [] for gpu_id in range(num_gpus)}
    for fold_idx, subject in enumerate(test_candidates):
        gpu_id = fold_idx % num_gpus
        assignments[gpu_id].append((fold_idx, subject))
    return assignments


def check_gpu_availability(required_gpus: int) -> Tuple[bool, int, str]:
    """
    Check if required GPUs are available.

    Args:
        required_gpus: Number of GPUs required

    Returns:
        Tuple of (is_available, actual_count, message)
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return False, 0, "CUDA is not available"

        actual_count = torch.cuda.device_count()
        if actual_count < required_gpus:
            return False, actual_count, f"Only {actual_count} GPUs available, but {required_gpus} requested"

        return True, actual_count, f"{actual_count} GPUs available"
    except ImportError:
        # torch not installed - defer to Ray's GPU detection
        return True, -1, "torch not imported, will use Ray GPU detection"
    except Exception as e:
        return False, 0, f"Error checking GPUs: {e}"


class ProgressTracker:
    """Track and display training progress across distributed folds."""

    def __init__(self, total_folds: int, num_gpus: int):
        self.total = total_folds
        self.num_gpus = num_gpus
        self.completed = 0
        self.successful = 0
        self.failed = 0
        self.start_time = time.time()
        self.results = []
        self.fold_times = []

    def update(self, result: Dict) -> None:
        """Update progress with completed fold result."""
        self.completed += 1
        self.results.append(result)

        elapsed = time.time() - self.start_time

        if result.get('status') == 'failed':
            self.failed += 1
            status_str = "FAILED"
        else:
            self.successful += 1
            test_f1 = result.get('test', {}).get('f1_score', 0.0)
            status_str = f"F1={test_f1:.1f}%"
            # Track fold time for ETA calculation
            if 'elapsed_time' in result:
                self.fold_times.append(result['elapsed_time'])

        # Calculate ETA based on average fold time
        if self.fold_times:
            avg_fold_time = np.mean(self.fold_times)
            remaining = self.total - self.completed
            eta = avg_fold_time * remaining / self.num_gpus  # Parallel execution
        elif self.completed > 0:
            avg_time = elapsed / self.completed
            remaining = self.total - self.completed
            eta = avg_time * remaining
        else:
            eta = 0

        print(f"[{self.completed}/{self.total}] Fold {result.get('fold_idx', '?')} "
              f"(Subject {result.get('test_subject', '?')}) {status_str} | "
              f"Elapsed: {elapsed/60:.1f}min | ETA: {eta/60:.1f}min | "
              f"Success: {self.successful}, Failed: {self.failed}")

    def get_summary(self) -> Dict:
        """Get summary statistics from completed folds."""
        successful_results = [r for r in self.results if r.get('status') != 'failed']

        if successful_results:
            test_f1s = [r['test']['f1_score'] for r in successful_results if r.get('test')]
            test_accs = [r['test']['accuracy'] for r in successful_results if r.get('test')]
            val_f1s = [r['val']['f1_score'] for r in successful_results if r.get('val')]
        else:
            test_f1s = []
            test_accs = []
            val_f1s = []

        return {
            'total_folds': self.total,
            'successful': self.successful,
            'failed': self.failed,
            'mean_test_f1': float(np.mean(test_f1s)) if test_f1s else 0.0,
            'std_test_f1': float(np.std(test_f1s)) if test_f1s else 0.0,
            'min_test_f1': float(np.min(test_f1s)) if test_f1s else 0.0,
            'max_test_f1': float(np.max(test_f1s)) if test_f1s else 0.0,
            'mean_test_acc': float(np.mean(test_accs)) if test_accs else 0.0,
            'std_test_acc': float(np.std(test_accs)) if test_accs else 0.0,
            'mean_val_f1': float(np.mean(val_f1s)) if val_f1s else 0.0,
            'total_time_min': (time.time() - self.start_time) / 60,
        }


def print_fold_results_table(results: List[Dict]) -> None:
    """
    Print a comprehensive per-fold results table with val/test metrics.

    Displays a formatted table showing:
    - Fold index and test subject
    - GPU assignment
    - Validation F1, Precision, Recall
    - Test F1, Precision, Recall, Accuracy
    - Training time
    - Best epoch
    """
    # Filter successful results and sort by fold index
    successful = [r for r in results if r.get('status') != 'failed']
    failed = [r for r in results if r.get('status') == 'failed']

    if not successful:
        print("\nNo successful folds to display.")
        return

    # Sort by fold index
    successful = sorted(successful, key=lambda x: x.get('fold_idx', 0))

    # Table header
    print("\n" + "=" * 150)
    print("PER-FOLD RESULTS TABLE")
    print("=" * 150)

    # Column headers
    header = (
        f"{'Fold':>4} | {'Subject':>7} | {'GPU':>3} | "
        f"{'Val F1':>7} {'Val P':>6} {'Val R':>6} | "
        f"{'Test F1':>7} {'Test P':>6} {'Test R':>6} {'Test Acc':>8} | "
        f"{'Epoch':>5} | {'Windows':>15} | {'Trials':>12} | {'Time':>6}"
    )
    print(header)
    print("-" * 150)

    # Collect metrics for statistics
    val_f1s, test_f1s, test_accs = [], [], []
    val_precs, val_recs = [], []
    test_precs, test_recs = [], []

    for r in successful:
        fold_idx = r.get('fold_idx', 0)
        test_subject = r.get('test_subject', '?')
        gpu_id = r.get('gpu_id', 0)
        best_epoch = r.get('best_epoch', 0)
        elapsed_min = r.get('elapsed_time', 0) / 60

        # Dataset stats
        fall_win = r.get('fall_windows', 0)
        adl_win = r.get('adl_windows', 0)
        fall_tri = r.get('fall_trials', 0)
        adl_tri = r.get('adl_trials', 0)

        # Validation metrics
        val = r.get('val', {})
        val_f1 = val.get('f1_score', 0.0)
        val_prec = val.get('precision', 0.0)
        val_rec = val.get('recall', 0.0)

        # Test metrics
        test = r.get('test', {})
        test_f1 = test.get('f1_score', 0.0)
        test_prec = test.get('precision', 0.0)
        test_rec = test.get('recall', 0.0)
        test_acc = test.get('accuracy', 0.0)

        # Collect for stats
        val_f1s.append(val_f1)
        val_precs.append(val_prec)
        val_recs.append(val_rec)
        test_f1s.append(test_f1)
        test_precs.append(test_prec)
        test_recs.append(test_rec)
        test_accs.append(test_acc)

        # Format row
        win_str = f"{fall_win}:{adl_win}"
        tri_str = f"{fall_tri}:{adl_tri}"
        row = (
            f"{fold_idx:>4} | {test_subject:>7} | {gpu_id:>3} | "
            f"{val_f1:>6.2f}% {val_prec:>5.2f}% {val_rec:>5.2f}% | "
            f"{test_f1:>6.2f}% {test_prec:>5.2f}% {test_rec:>5.2f}% {test_acc:>7.2f}% | "
            f"{best_epoch:>5} | {win_str:>15} | {tri_str:>12} | {elapsed_min:>5.1f}m"
        )
        print(row)

    # Statistics row
    print("-" * 150)

    # Mean row
    mean_row = (
        f"{'MEAN':>4} | {'':>7} | {'':>3} | "
        f"{np.mean(val_f1s):>6.2f}% {np.mean(val_precs):>5.2f}% {np.mean(val_recs):>5.2f}% | "
        f"{np.mean(test_f1s):>6.2f}% {np.mean(test_precs):>5.2f}% {np.mean(test_recs):>5.2f}% {np.mean(test_accs):>7.2f}% | "
        f"{'':>5} | {'':>15} | {'':>12} | {'':>6}"
    )
    print(mean_row)

    # Std row
    std_row = (
        f"{'STD':>4} | {'':>7} | {'':>3} | "
        f"{np.std(val_f1s):>6.2f}  {np.std(val_precs):>5.2f}  {np.std(val_recs):>5.2f}  | "
        f"{np.std(test_f1s):>6.2f}  {np.std(test_precs):>5.2f}  {np.std(test_recs):>5.2f}  {np.std(test_accs):>7.2f}  | "
        f"{'':>5} | {'':>15} | {'':>12} | {'':>6}"
    )
    print(std_row)

    # Min/Max row
    min_row = (
        f"{'MIN':>4} | {'':>7} | {'':>3} | "
        f"{np.min(val_f1s):>6.2f}% {'':>5} {'':>5} | "
        f"{np.min(test_f1s):>6.2f}% {'':>5} {'':>5} {np.min(test_accs):>7.2f}% | "
        f"{'':>5} | {'':>15} | {'':>12} | {'':>6}"
    )
    print(min_row)

    max_row = (
        f"{'MAX':>4} | {'':>7} | {'':>3} | "
        f"{np.max(val_f1s):>6.2f}% {'':>5} {'':>5} | "
        f"{np.max(test_f1s):>6.2f}% {'':>5} {'':>5} {np.max(test_accs):>7.2f}% | "
        f"{'':>5} | {'':>15} | {'':>12} | {'':>6}"
    )
    print(max_row)

    print("=" * 150)

    # Best and worst folds
    if test_f1s:
        best_idx = int(np.argmax(test_f1s))
        worst_idx = int(np.argmin(test_f1s))
        best_fold = successful[best_idx]
        worst_fold = successful[worst_idx]

        print(f"\nBest Fold:  #{best_fold.get('fold_idx', 0)} (Subject {best_fold.get('test_subject', '?')}) - "
              f"Test F1: {test_f1s[best_idx]:.2f}%")
        print(f"Worst Fold: #{worst_fold.get('fold_idx', 0)} (Subject {worst_fold.get('test_subject', '?')}) - "
              f"Test F1: {test_f1s[worst_idx]:.2f}%")

    # Threshold analysis summary
    threshold_data = [r.get('threshold_analysis') for r in successful if r.get('threshold_analysis')]
    if threshold_data:
        opt_thresholds = [t['optimal_f1_threshold']['threshold'] for t in threshold_data]
        opt_f1s = [t['optimal_f1_threshold']['f1'] * 100 for t in threshold_data]
        opt_precs = [t['optimal_f1_threshold']['precision'] * 100 for t in threshold_data]
        opt_recs = [t['optimal_f1_threshold']['recall'] * 100 for t in threshold_data]
        opt_accs = [t['optimal_f1_threshold']['accuracy'] * 100 for t in threshold_data]
        opt_specs = [t['optimal_f1_threshold']['specificity'] * 100 for t in threshold_data]

        default_f1s = [t['default_threshold']['f1'] * 100 for t in threshold_data]
        default_precs = [t['default_threshold']['precision'] * 100 for t in threshold_data]
        default_recs = [t['default_threshold']['recall'] * 100 for t in threshold_data]
        default_accs = [t['default_threshold']['accuracy'] * 100 for t in threshold_data]
        default_specs = [t['default_threshold']['specificity'] * 100 for t in threshold_data]

        print(f"\n{'='*90}")
        print(f"THRESHOLD ANALYSIS ({len(threshold_data)} folds)")
        print(f"{'='*90}")

        # Per-fold optimal analysis (upper bound - not realistic for deployment)
        print(f"\n--- Per-Fold Optimal (Upper Bound) ---")
        print(f"Mean Optimal Threshold: {np.mean(opt_thresholds):.3f} ± {np.std(opt_thresholds):.3f}")
        print(f"\n{'Metric':<12} | {'@ Per-Fold Opt τ':>18} | {'@ Default 0.5':>18} | {'Δ':>8}")
        print(f"{'-'*70}")
        print(f"{'F1':<12} | {np.mean(opt_f1s):>6.2f}% ± {np.std(opt_f1s):>5.2f}% | {np.mean(default_f1s):>6.2f}% ± {np.std(default_f1s):>5.2f}% | {np.mean(opt_f1s)-np.mean(default_f1s):>+6.2f}%")
        print(f"{'Precision':<12} | {np.mean(opt_precs):>6.2f}% ± {np.std(opt_precs):>5.2f}% | {np.mean(default_precs):>6.2f}% ± {np.std(default_precs):>5.2f}% | {np.mean(opt_precs)-np.mean(default_precs):>+6.2f}%")
        print(f"{'Recall':<12} | {np.mean(opt_recs):>6.2f}% ± {np.std(opt_recs):>5.2f}% | {np.mean(default_recs):>6.2f}% ± {np.std(default_recs):>5.2f}% | {np.mean(opt_recs)-np.mean(default_recs):>+6.2f}%")
        print(f"{'Accuracy':<12} | {np.mean(opt_accs):>6.2f}% ± {np.std(opt_accs):>5.2f}% | {np.mean(default_accs):>6.2f}% ± {np.std(default_accs):>5.2f}% | {np.mean(opt_accs)-np.mean(default_accs):>+6.2f}%")
        print(f"{'Specificity':<12} | {np.mean(opt_specs):>6.2f}% ± {np.std(opt_specs):>5.2f}% | {np.mean(default_specs):>6.2f}% ± {np.std(default_specs):>5.2f}% | {np.mean(opt_specs)-np.mean(default_specs):>+6.2f}%")

        # Global threshold analysis (deployment-realistic)
        try:
            from utils.threshold_analysis import compute_global_threshold_metrics, evaluate_fixed_thresholds
            global_results = compute_global_threshold_metrics(successful)

            if 'error' not in global_results:
                print(f"\n--- Global Threshold (Deployment-Realistic) ---")
                print(f"Global Threshold: τ = {global_results['global_threshold']:.3f} (mean of per-fold optimal)")
                print(f"\n{'Metric':<12} | {'@ Global τ':>18} | {'@ Default 0.5':>18} | {'Δ':>8}")
                print(f"{'-'*70}")
                print(f"{'F1':<12} | {global_results['mean_f1']*100:>6.2f}% ± {global_results['std_f1']*100:>5.2f}% | {np.mean(default_f1s):>6.2f}% ± {np.std(default_f1s):>5.2f}% | {global_results['mean_f1']*100-np.mean(default_f1s):>+6.2f}%")
                print(f"{'Precision':<12} | {global_results['mean_precision']*100:>6.2f}% ± {global_results['std_precision']*100:>5.2f}% | {np.mean(default_precs):>6.2f}% ± {np.std(default_precs):>5.2f}% | {global_results['mean_precision']*100-np.mean(default_precs):>+6.2f}%")
                print(f"{'Recall':<12} | {global_results['mean_recall']*100:>6.2f}% ± {global_results['std_recall']*100:>5.2f}% | {np.mean(default_recs):>6.2f}% ± {np.std(default_recs):>5.2f}% | {global_results['mean_recall']*100-np.mean(default_recs):>+6.2f}%")
                print(f"{'Accuracy':<12} | {global_results['mean_accuracy']*100:>6.2f}% ± {global_results['std_accuracy']*100:>5.2f}% | {np.mean(default_accs):>6.2f}% ± {np.std(default_accs):>5.2f}% | {global_results['mean_accuracy']*100-np.mean(default_accs):>+6.2f}%")
                print(f"{'Specificity':<12} | {global_results['mean_specificity']*100:>6.2f}% ± {global_results['std_specificity']*100:>5.2f}% | {np.mean(default_specs):>6.2f}% ± {np.std(default_specs):>5.2f}% | {global_results['mean_specificity']*100-np.mean(default_specs):>+6.2f}%")

            # Fixed threshold comparison
            fixed_thresholds = [0.5, 0.55, 0.6, 0.7, 0.9]
            fixed_results = evaluate_fixed_thresholds(successful, fixed_thresholds)
            if 'error' not in fixed_results:
                print(f"\n--- Fixed Threshold Comparison ---")
                print(f"{'Threshold':<10} | {'F1':>18} | {'Precision':>18} | {'Recall':>18} | {'Accuracy':>18}")
                print(f"{'-'*95}")
                for t in fixed_thresholds:
                    r = fixed_results[t]
                    print(f"τ = {t:<6} | {r['mean_f1']*100:>6.2f}% ± {r['std_f1']*100:>5.2f}% | "
                          f"{r['mean_precision']*100:>6.2f}% ± {r['std_precision']*100:>5.2f}% | "
                          f"{r['mean_recall']*100:>6.2f}% ± {r['std_recall']*100:>5.2f}% | "
                          f"{r['mean_accuracy']*100:>6.2f}% ± {r['std_accuracy']*100:>5.2f}%")

            if 'error' not in global_results:
                print(f"\n{'='*90}")
                print(f"DEPLOYMENT RECOMMENDATION")
                print(f"{'='*90}")
                print(f"Use threshold τ = {global_results['global_threshold']:.3f} for deployment")
                print(f"Expected F1: {global_results['mean_f1']*100:.2f}% ± {global_results['std_f1']*100:.2f}%")
                print(f"Note: Per-fold optimal ({np.mean(opt_f1s):.2f}%) is an upper bound; global ({global_results['mean_f1']*100:.2f}%) is realistic")
        except Exception as e:
            print(f"\nGlobal threshold analysis failed: {e}")

        print(f"{'='*90}")

    # Failed folds summary
    if failed:
        print(f"\nFailed Folds ({len(failed)}):")
        for f in failed:
            print(f"  - Fold {f.get('fold_idx', '?')} (Subject {f.get('test_subject', '?')})")

    print()


def train_single_fold(
    actor_id: int,
    base_config: dict,
    test_subject: int,
    fold_idx: int,
    test_candidates: List[int],
    train_only_subjects: List[int],
    validation_subjects: List[int],
    work_dir: str
) -> Dict:
    """
    Train a single LOSO fold. This is the core training function executed by Ray.

    Args:
        actor_id: Logical actor ID for logging
        base_config: Base configuration dict from YAML
        test_subject: Subject ID for testing
        fold_idx: Fold index (for logging)
        test_candidates: List of all test candidate subjects
        train_only_subjects: Subjects fixed in training set
        validation_subjects: Validation subjects
        work_dir: Directory for saving outputs

    Returns:
        Dictionary with fold metrics and status
    """
    import torch

    fold_start_time = time.time()

    # Get GPU info from Ray
    gpu_ids = ray.get_gpu_ids()
    physical_gpu = gpu_ids[0] if gpu_ids else 'cpu'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print(f"\n[Actor {actor_id} | GPU {physical_gpu}] Starting fold {fold_idx} (test subject {test_subject})")

    try:
        # Build args namespace
        cfg = base_config
        args = Namespace()

        # Model configuration
        args.model = cfg.get('model')
        args.model_args = cfg.get('model_args', {}).copy()

        # Dataset configuration
        args.dataset = cfg.get('dataset', 'smartfallmm')
        args.dataset_args = cfg.get('dataset_args', {}).copy()
        args.feeder = cfg.get('feeder', 'Feeder.Make_Dataset.UTD_mm')

        # Training parameters
        args.batch_size = cfg.get('batch_size', 64)
        args.test_batch_size = cfg.get('test_batch_size', args.batch_size)
        args.val_batch_size = cfg.get('val_batch_size', args.batch_size)
        args.num_epoch = cfg.get('num_epoch', 80)
        args.start_epoch = cfg.get('start_epoch', 0)

        # Optimizer
        args.optimizer = cfg.get('optimizer', 'adamw')
        args.base_lr = cfg.get('base_lr', 1e-3)
        args.weight_decay = cfg.get('weight_decay', 5e-4)

        # LR Scheduler
        args.lr_scheduler = cfg.get('lr_scheduler', 'none')
        args.warmup_epochs = cfg.get('warmup_epochs', 10)
        args.min_lr = cfg.get('min_lr', 1e-6)

        # Loss
        args.loss = cfg.get('loss', 'loss.BCE')
        args.loss_args = cfg.get('loss_args', {})
        args.loss_type = cfg.get('loss_type', 'focal')

        # Subjects
        args.subjects = cfg.get('subjects', [])
        args.validation_subjects = validation_subjects
        args.train_only_subjects = train_only_subjects

        # Single-fold mode
        args.single_fold = test_subject

        # Feeder args
        args.train_feeder_args = cfg.get('train_feeder_args', {'batch_size': args.batch_size})
        args.val_feeder_args = cfg.get('val_feeder_args', {'batch_size': args.batch_size})
        args.test_feeder_args = cfg.get('test_feeder_args', {'batch_size': args.batch_size})

        # Device - use cuda:0 since Ray makes only allocated GPU visible
        args.device = [0]

        # Work directory
        args.work_dir = work_dir
        args.model_saved_name = cfg.get('model_saved_name', 'model')

        # Misc
        args.config = None
        args.phase = 'train'
        args.print_log = True
        args.num_worker = cfg.get('num_worker', 0)
        args.include_val = cfg.get('include_val', True)
        args.seed = cfg.get('seed', 2)
        args.log_interval = cfg.get('log_interval', 10)
        args.weights = None
        args.result_file = None
        args.save_best_val_f1 = cfg.get('save_best_val_f1', False)

        # Test grouping (disabled for single-fold)
        args.enable_test_grouping = False
        args.test_group_min_size = 2
        args.test_group_max_size = 3
        args.test_group_ratio_tolerance = 0.10
        args.test_group_extreme_threshold = 0.05

        # Kalman preprocessing
        args.enable_kalman_preprocessing = cfg.get('enable_kalman_preprocessing', False)
        args.kalman_args = cfg.get('kalman_args', {})

        # W&B (disabled for workers - main process handles this)
        args.enable_wandb = False
        args.wandb_project = 'smartfall-mm'
        args.wandb_entity = 'abheek-texas-state-university'

        # Import training components
        from main import Trainer, init_seed

        # Initialize seed for reproducibility
        init_seed(args.seed + fold_idx)  # Vary seed slightly per fold

        # Create trainer
        trainer = Trainer(args)

        # Configure fold splits
        trainer.train_subjects = [s for s in test_candidates if s != test_subject] + train_only_subjects
        trainer.val_subject = validation_subjects
        trainer.test_subject = [test_subject]
        trainer._init_fold_tracking(fold_idx, test_subject)

        # Load model
        trainer.model = trainer.load_model(args.model, args.model_args)

        # Load data
        if not trainer.load_data():
            return {
                'test_subject': str(test_subject),
                'fold_idx': fold_idx,
                'gpu_id': actor_id,
                'physical_gpu': physical_gpu,
                'status': 'failed',
                'error': 'Data loading failed - empty or invalid data',
                'elapsed_time': time.time() - fold_start_time,
            }

        # Initialize optimizer and loss
        trainer.load_optimizer(trainer.model.parameters())
        trainer.load_loss()

        # Training loop
        for epoch in range(args.start_epoch, args.num_epoch):
            trainer.train(epoch)

            # Step scheduler if available (old Trainer may not have it)
            if hasattr(trainer, 'scheduler') and trainer.scheduler is not None:
                trainer.scheduler.step()

            # Early stopping
            if trainer.early_stop.early_stop:
                print(f"[Actor {actor_id}] Early stopping at epoch {epoch+1}")
                break

        # Load best weights and evaluate on test
        trainer.load_model(args.model, args.model_args)
        trainer.load_weights()
        trainer.model.eval()
        trainer.eval(epoch=0, loader_name='test')

        # Extract metrics with validation
        train_metrics = deepcopy(trainer.current_fold_metrics.get('train', {}))
        val_metrics = deepcopy(trainer.best_val_metrics) if trainer.best_val_metrics else deepcopy(trainer.current_fold_metrics.get('val', {}))
        test_metrics = deepcopy(trainer.current_fold_metrics.get('test', {}))

        # Validate metrics are present
        if not test_metrics or 'f1_score' not in test_metrics:
            return {
                'test_subject': str(test_subject),
                'fold_idx': fold_idx,
                'gpu_id': actor_id,
                'physical_gpu': physical_gpu,
                'status': 'failed',
                'error': 'Test metrics not computed properly',
                'elapsed_time': time.time() - fold_start_time,
            }

        elapsed_time = time.time() - fold_start_time

        # Extract threshold analysis if available
        threshold_analysis = trainer.current_fold_metrics.get('threshold_analysis')

        result = {
            'test_subject': str(test_subject),
            'fold_idx': fold_idx,
            'gpu_id': actor_id,
            'physical_gpu': physical_gpu,
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics,
            'best_epoch': getattr(trainer, 'best_epoch', 0),
            'fall_windows': getattr(trainer, 'fold_fall_windows', 0),
            'adl_windows': getattr(trainer, 'fold_adl_windows', 0),
            'fall_trials': getattr(trainer, 'fold_fall_trials', 0),
            'adl_trials': getattr(trainer, 'fold_adl_trials', 0),
            'status': 'success',
            'elapsed_time': elapsed_time,
        }

        if threshold_analysis:
            result['threshold_analysis'] = threshold_analysis

        # Log with optimal threshold info if available
        opt_info = ""
        if threshold_analysis and 'optimal_f1_threshold' in threshold_analysis:
            opt = threshold_analysis['optimal_f1_threshold']
            opt_info = f", OptThresh={opt['threshold']:.2f} (F1={opt['f1']*100:.1f}%)"

        print(f"[Actor {actor_id} | GPU {physical_gpu}] Fold {fold_idx} complete in {elapsed_time/60:.1f}min: "
              f"Test F1={test_metrics.get('f1_score', 0):.1f}%, Acc={test_metrics.get('accuracy', 0):.1f}%{opt_info}")

        return result

    except Exception as e:
        error_msg = str(e)
        tb = traceback.format_exc()
        print(f"[Actor {actor_id}] Fold {fold_idx} FAILED: {error_msg}")
        print(tb)

        return {
            'test_subject': str(test_subject),
            'fold_idx': fold_idx,
            'gpu_id': actor_id,
            'physical_gpu': physical_gpu if 'physical_gpu' in dir() else 'unknown',
            'status': 'failed',
            'error': error_msg,
            'traceback': tb,
            'elapsed_time': time.time() - fold_start_time,
        }


# Create Ray remote function for fold training
@ray.remote(num_gpus=1)
def ray_train_fold(
    actor_id: int,
    base_config: dict,
    test_subject: int,
    fold_idx: int,
    test_candidates: List[int],
    train_only_subjects: List[int],
    validation_subjects: List[int],
    work_dir: str
) -> Dict:
    """Ray remote wrapper for train_single_fold."""
    return train_single_fold(
        actor_id=actor_id,
        base_config=base_config,
        test_subject=test_subject,
        fold_idx=fold_idx,
        test_candidates=test_candidates,
        train_only_subjects=train_only_subjects,
        validation_subjects=validation_subjects,
        work_dir=work_dir,
    )


class RayWandbLogger:
    """W&B integration for Ray distributed training."""

    def __init__(
        self,
        config: dict,
        project: str,
        entity: str,
        run_name: Optional[str] = None
    ):
        """Initialize W&B run in main process."""
        try:
            import wandb
            self.wandb = wandb
            self.enabled = True

            self.run = wandb.init(
                project=project,
                entity=entity,
                config=config,
                name=run_name or f"ray_loso_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                job_type="distributed_train",
                reinit=True,
            )
            self.completed_count = 0
            print(f"[W&B] Initialized run: {self.run.name}")

        except ImportError:
            print("[W&B] wandb not installed, logging disabled")
            self.enabled = False
            self.wandb = None
            self.run = None
        except Exception as e:
            print(f"[W&B] Failed to initialize: {e}")
            self.enabled = False
            self.wandb = None
            self.run = None

    def log_fold_complete(self, result: Dict) -> None:
        """Log completed fold metrics."""
        if not self.enabled:
            return

        try:
            self.completed_count += 1
            fold_idx = result.get('fold_idx', 0)

            if result.get('status') == 'failed':
                self.wandb.log({
                    f"fold_{fold_idx}/status": "failed",
                    "completed_folds": self.completed_count,
                })
            else:
                test_metrics = result.get('test', {})
                val_metrics = result.get('val', {})

                log_dict = {
                    f"fold_{fold_idx}/test_f1": test_metrics.get('f1_score', 0),
                    f"fold_{fold_idx}/test_accuracy": test_metrics.get('accuracy', 0),
                    f"fold_{fold_idx}/val_f1": val_metrics.get('f1_score', 0),
                    f"fold_{fold_idx}/best_epoch": result.get('best_epoch', 0),
                    f"fold_{fold_idx}/gpu_id": result.get('gpu_id', 0),
                    f"fold_{fold_idx}/elapsed_min": result.get('elapsed_time', 0) / 60,
                    "completed_folds": self.completed_count,
                }
                self.wandb.log(log_dict)
        except Exception as e:
            print(f"[W&B] Error logging fold: {e}")

    def log_summary(self, results: List[Dict], summary: Dict) -> None:
        """Log final summary statistics."""
        if not self.enabled:
            return

        try:
            self.run.summary.update({
                'mean_test_f1': summary.get('mean_test_f1', 0),
                'std_test_f1': summary.get('std_test_f1', 0),
                'min_test_f1': summary.get('min_test_f1', 0),
                'max_test_f1': summary.get('max_test_f1', 0),
                'mean_test_accuracy': summary.get('mean_test_acc', 0),
                'mean_val_f1': summary.get('mean_val_f1', 0),
                'num_folds': summary.get('total_folds', 0),
                'successful_folds': summary.get('successful', 0),
                'failed_folds': summary.get('failed', 0),
                'total_time_min': summary.get('total_time_min', 0),
            })

            # Create summary table
            successful = [r for r in results if r.get('status') != 'failed']
            if successful:
                columns = ['fold', 'test_subject', 'gpu_id', 'test_f1', 'test_accuracy', 'val_f1', 'best_epoch', 'time_min']
                table = self.wandb.Table(columns=columns)

                for r in successful:
                    table.add_data(
                        r.get('fold_idx', 0),
                        r.get('test_subject', ''),
                        r.get('gpu_id', 0),
                        r['test'].get('f1_score', 0) if r.get('test') else 0,
                        r['test'].get('accuracy', 0) if r.get('test') else 0,
                        r['val'].get('f1_score', 0) if r.get('val') else 0,
                        r.get('best_epoch', 0),
                        r.get('elapsed_time', 0) / 60,
                    )

                self.wandb.log({"fold_summary": table})
        except Exception as e:
            print(f"[W&B] Error logging summary: {e}")

    def finish(self) -> None:
        """Finish W&B run."""
        if self.enabled and self.run:
            try:
                self.run.finish()
            except Exception as e:
                print(f"[W&B] Error finishing run: {e}")


class RayDistributedTrainer:
    """
    Orchestrates distributed LOSO training across multiple GPUs.

    Features:
    - Robust GPU resource management
    - Graceful error handling
    - Comprehensive metrics aggregation
    - Progress tracking with ETA
    """

    def __init__(
        self,
        config_path: str,
        num_gpus: int = 3,
        work_dir: Optional[str] = None,
        model_override: Optional[str] = None,
        model_args_override: Optional[str] = None,
        loss_type_override: Optional[str] = None,
        embed_dim_override: Optional[int] = None,
        adl_stride_override: Optional[int] = None,
        enable_wandb: bool = False,
        wandb_project: str = 'smartfall-mm',
        wandb_entity: str = 'abheek-texas-state-university',
        ray_address: Optional[str] = None,
        max_folds: Optional[int] = None,
        seed: int = 2,
        **kwargs,  # For extensibility: preprocessing/encoder overrides
    ):
        """
        Initialize the distributed trainer.

        Args:
            config_path: Path to YAML config file
            num_gpus: Number of GPUs to use
            work_dir: Output directory
            model_override: Override model class (e.g., 'Models.encoder_ablation.KalmanConv1dLinear')
            model_args_override: Override model args as dict string
            loss_type_override: Override loss function (bce, focal, cb_focal)
            embed_dim_override: Override embedding dimension
            adl_stride_override: Override ADL stride
            enable_wandb: Whether to enable W&B logging
            wandb_project: W&B project name
            wandb_entity: W&B entity
            ray_address: Ray cluster address (None for local)
            max_folds: Maximum number of folds to run (for testing)
            seed: Random seed
            **kwargs: Additional overrides for preprocessing/encoder:
                - remove_gravity_override (bool): Enable gravity removal
                - gravity_cutoff_override (float): Gravity filter cutoff Hz
                - include_smv_override (bool): Include SMV in features
                - fall_stride_override (int): Fall stride for class-aware windowing
                - acc_encoder_override (str): Accelerometer encoder type
                - ori_encoder_override (str): Orientation encoder type
                - acc_kernel_override (int): Accelerometer kernel size
                - ori_kernel_override (int): Orientation kernel size
        """
        self.config_path = config_path
        self.num_gpus = num_gpus
        self.enable_wandb = enable_wandb
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.ray_address = ray_address
        self.max_folds = max_folds
        self.seed = seed

        # Load config
        self.config = self._load_config()
        self.config['seed'] = seed

        # Apply model override (for ablation studies)
        if model_override:
            self.config['model'] = model_override

        if model_args_override:
            import ast
            try:
                override_args = ast.literal_eval(model_args_override)
                existing_args = self.config.get('model_args', {})
                existing_args.update(override_args)
                self.config['model_args'] = existing_args
            except (ValueError, SyntaxError) as e:
                print(f"WARNING: Failed to parse --model-args: {e}")

        # Apply loss type override
        if loss_type_override:
            self.config['loss_type'] = loss_type_override

        # Apply embed_dim override
        if embed_dim_override:
            model_args = self.config.get('model_args', {})
            model_args['embed_dim'] = embed_dim_override
            self.config['model_args'] = model_args

        # Apply adl_stride override
        if adl_stride_override:
            dataset_args = self.config.get('dataset_args', {})
            dataset_args['adl_stride'] = adl_stride_override
            self.config['dataset_args'] = dataset_args

        # Apply preprocessing overrides from kwargs (extensible design)
        self._apply_preprocessing_overrides(kwargs)

        # Compute test candidates
        self.test_candidates = self._compute_test_candidates()
        if max_folds and max_folds < len(self.test_candidates):
            self.test_candidates = self.test_candidates[:max_folds]

        # Setup work directory
        if work_dir:
            self.work_dir = work_dir
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = self.config.get('model', 'model').split('.')[-1]
            self.work_dir = f"results/ray_{model_name}_{timestamp}"

        os.makedirs(self.work_dir, exist_ok=True)

        # Copy config to work_dir
        shutil.copy(config_path, self.work_dir)

        # Extract model name for reports
        self.model_name = self.config.get('model', 'model').split('.')[-1]

        # Get fixed subjects
        self.validation_subjects = self.config.get('validation_subjects', [48, 57])
        self.train_only_subjects = self.config.get('train_only_subjects', [])

        self.wandb_logger = None

    def _load_config(self) -> dict:
        """Load YAML config file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def _compute_test_candidates(self) -> List[int]:
        """Compute test candidate subjects."""
        subjects = self.config.get('subjects', [])
        validation = self.config.get('validation_subjects', [48, 57])
        train_only = self.config.get('train_only_subjects', [])

        candidates = [s for s in subjects
                     if s not in validation and s not in train_only]
        return candidates

    def _apply_preprocessing_overrides(self, kwargs: Dict[str, Any]) -> None:
        """
        Apply preprocessing and encoder overrides from kwargs.

        Modular design: add new overrides by extending the mappings below.
        """
        # Dataset args overrides (preprocessing)
        dataset_overrides = {
            'remove_gravity_override': 'remove_gravity',
            'gravity_cutoff_override': 'gravity_filter_cutoff',
            'include_smv_override': 'kalman_include_smv',
            'fall_stride_override': 'fall_stride',
            'adl_stride_override': 'adl_stride',  # Also handled in __init__ for backwards compatibility
        }

        # Model args overrides (encoder architecture)
        model_overrides = {
            'acc_encoder_override': 'acc_encoder',
            'ori_encoder_override': 'ori_encoder',
            'acc_kernel_override': 'acc_kernel_size',
            'ori_kernel_override': 'ori_kernel_size',
        }

        # Apply dataset_args overrides
        for kwarg_key, config_key in dataset_overrides.items():
            if kwarg_key in kwargs and kwargs[kwarg_key] is not None:
                dataset_args = self.config.get('dataset_args', {})
                dataset_args[config_key] = kwargs[kwarg_key]
                self.config['dataset_args'] = dataset_args

        # Apply model_args overrides
        for kwarg_key, config_key in model_overrides.items():
            if kwarg_key in kwargs and kwargs[kwarg_key] is not None:
                model_args = self.config.get('model_args', {})
                model_args[config_key] = kwargs[kwarg_key]
                self.config['model_args'] = model_args

        # Handle SMV channel count adjustment
        # When SMV is disabled, reduce imu_channels from 7 to 6
        if kwargs.get('include_smv_override') is False:
            model_args = self.config.get('model_args', {})
            model_args['imu_channels'] = 6
            model_args['acc_coords'] = 6
            self.config['model_args'] = model_args

    def _aggregate_results(self, results: List[Dict]) -> pd.DataFrame:
        """
        Aggregate fold results into DataFrame with comprehensive validation.

        Ensures all metrics are properly computed and averaged.
        """
        rows = []

        # Define expected metrics
        expected_metrics = ['loss', 'accuracy', 'f1_score', 'precision', 'recall', 'auc']

        for result in results:
            if result.get('status') == 'failed':
                continue

            row = {'test_subject': result['test_subject']}

            for split_name in ['train', 'val', 'test']:
                metrics = result.get(split_name, {})
                if metrics:
                    for metric_name in expected_metrics:
                        col_name = f'{split_name}_{metric_name}'
                        value = metrics.get(metric_name)
                        if value is not None:
                            precision = 6 if 'loss' in metric_name else 2
                            row[col_name] = round(float(value), precision)

            # Add metadata
            row['best_epoch'] = result.get('best_epoch', 0)
            row['elapsed_min'] = round(result.get('elapsed_time', 0) / 60, 2)

            rows.append(row)

        if not rows:
            print("WARNING: No successful folds to aggregate!")
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # Sort by test_subject
        df['test_subject_int'] = pd.to_numeric(df['test_subject'], errors='coerce')
        df = df.sort_values('test_subject_int').drop('test_subject_int', axis=1)
        df = df.reset_index(drop=True)

        # Calculate and add average row
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        avg_row = {'test_subject': 'Average'}
        for col in numeric_cols:
            avg_row[col] = round(df[col].mean(), 6 if 'loss' in col else 2)

        # Also add std for key metrics
        std_row = {'test_subject': 'Std'}
        for col in numeric_cols:
            if any(m in col for m in ['f1_score', 'accuracy']):
                std_row[col] = round(df[col].std(), 2)
            else:
                std_row[col] = None

        df = pd.concat([df, pd.DataFrame([avg_row]), pd.DataFrame([std_row])], ignore_index=True)

        return df

    def run(self) -> pd.DataFrame:
        """
        Execute distributed LOSO training.

        Returns:
            DataFrame with aggregated fold results
        """
        print("=" * 70)
        print("RAY DISTRIBUTED LOSO TRAINING")
        print("=" * 70)
        print(f"Config: {self.config_path}")
        print(f"Model: {self.model_name}")
        print(f"Work directory: {self.work_dir}")
        print(f"Requested GPUs: {self.num_gpus}")
        print(f"Total folds: {len(self.test_candidates)}")
        print(f"Test candidates: {self.test_candidates}")
        print(f"Validation subjects: {self.validation_subjects}")
        print(f"Train-only subjects: {len(self.train_only_subjects)} subjects")
        print("=" * 70)

        # Check GPU availability before starting (preliminary - Ray will verify)
        gpu_ok, gpu_count, gpu_msg = check_gpu_availability(self.num_gpus)
        print(f"\n[GPU Check] {gpu_msg}")

        if not gpu_ok:
            print(f"ERROR: {gpu_msg}")
            print("Aborting. Please check GPU availability or reduce --num-gpus.")
            return pd.DataFrame(), []

        # Initialize Ray (local mode - no need to upload files)
        print("\n[Ray] Initializing...")
        try:
            if self.ray_address:
                ray.init(address=self.ray_address)
            else:
                ray.init(ignore_reinit_error=True)

            resources = ray.cluster_resources()
            print(f"[Ray] Cluster resources: {resources}")

            available_gpus = resources.get('GPU', 0)
            if available_gpus < self.num_gpus:
                print(f"WARNING: Ray sees only {available_gpus} GPUs, but {self.num_gpus} requested")
                print(f"Adjusting to use {int(available_gpus)} GPUs")
                self.num_gpus = int(available_gpus)

                if self.num_gpus == 0:
                    print("ERROR: No GPUs available to Ray. Aborting.")
                    ray.shutdown()
                    return pd.DataFrame(), []

        except Exception as e:
            print(f"ERROR: Failed to initialize Ray: {e}")
            return pd.DataFrame(), []

        # Initialize W&B
        if self.enable_wandb:
            self.wandb_logger = RayWandbLogger(
                config=self.config,
                project=self.wandb_project,
                entity=self.wandb_entity,
            )

        # Assign folds to GPUs (for logging purposes - Ray handles actual assignment)
        assignments = assign_folds_to_gpus(self.test_candidates, self.num_gpus)

        print(f"\nFold Distribution (round-robin across {self.num_gpus} GPUs):")
        for gpu_id, folds in assignments.items():
            subjects = [s for _, s in folds]
            print(f"  Worker {gpu_id}: {len(folds)} folds (subjects: {subjects})")
        print()

        # Submit all fold training tasks
        print("Submitting fold training tasks...")
        futures = []
        future_to_info = {}

        for gpu_id, fold_assignments in assignments.items():
            for fold_idx, test_subject in fold_assignments:
                future = ray_train_fold.remote(
                    actor_id=gpu_id,
                    base_config=self.config,
                    test_subject=test_subject,
                    fold_idx=fold_idx,
                    test_candidates=self.test_candidates,
                    train_only_subjects=self.train_only_subjects,
                    validation_subjects=self.validation_subjects,
                    work_dir=self.work_dir,
                )
                futures.append(future)
                future_to_info[future] = (fold_idx, test_subject, gpu_id)

        # Collect results with progress tracking
        print(f"\nTraining {len(futures)} folds across {self.num_gpus} GPUs...\n")
        print("-" * 70)

        progress = ProgressTracker(len(self.test_candidates), self.num_gpus)
        results = []
        failed_folds = []

        while futures:
            # Wait for any fold to complete
            try:
                ready, futures = ray.wait(futures, num_returns=1, timeout=None)
            except Exception as e:
                print(f"ERROR: Ray wait failed: {e}")
                break

            for future in ready:
                fold_idx, test_subject, gpu_id = future_to_info[future]

                try:
                    result = ray.get(future)
                    results.append(result)
                    progress.update(result)

                    if result.get('status') == 'failed':
                        failed_folds.append(result)

                    # Log to W&B
                    if self.wandb_logger:
                        self.wandb_logger.log_fold_complete(result)

                except ray.exceptions.RayTaskError as e:
                    error_result = {
                        'test_subject': str(test_subject),
                        'fold_idx': fold_idx,
                        'gpu_id': gpu_id,
                        'status': 'failed',
                        'error': f"RayTaskError: {str(e)}",
                        'elapsed_time': 0,
                    }
                    results.append(error_result)
                    failed_folds.append(error_result)
                    progress.update(error_result)

                except Exception as e:
                    error_result = {
                        'test_subject': str(test_subject),
                        'fold_idx': fold_idx,
                        'gpu_id': gpu_id,
                        'status': 'failed',
                        'error': str(e),
                        'elapsed_time': 0,
                    }
                    results.append(error_result)
                    failed_folds.append(error_result)
                    progress.update(error_result)

        # Get final summary
        summary = progress.get_summary()

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Total folds:        {summary['total_folds']}")
        print(f"Successful:         {summary['successful']}")
        print(f"Failed:             {summary['failed']}")
        print("-" * 40)
        print(f"Mean Test F1:       {summary['mean_test_f1']:.2f} +/- {summary['std_test_f1']:.2f}%")
        print(f"Min/Max Test F1:    {summary['min_test_f1']:.2f}% / {summary['max_test_f1']:.2f}%")
        print(f"Mean Test Accuracy: {summary['mean_test_acc']:.2f} +/- {summary['std_test_acc']:.2f}%")
        print(f"Mean Val F1:        {summary['mean_val_f1']:.2f}%")
        print("-" * 40)
        print(f"Total time:         {summary['total_time_min']:.1f} minutes")
        print("=" * 70)

        # Report failed folds with details
        if failed_folds:
            print(f"\nWARNING: {len(failed_folds)} folds failed:")
            for f in failed_folds:
                error = f.get('error', 'Unknown error')
                # Truncate long errors
                if len(error) > 100:
                    error = error[:100] + "..."
                print(f"  - Fold {f.get('fold_idx')} (Subject {f.get('test_subject')}): {error}")

        # Aggregate results
        results_df = self._aggregate_results(results)

        # Save results
        if not results_df.empty:
            scores_path = f'{self.work_dir}/scores.csv'
            results_df.to_csv(scores_path, index=False)
            print(f"\nResults saved to: {scores_path}")

            # Save fold results with probabilities for post-hoc threshold analysis
            try:
                fold_results_path = f'{self.work_dir}/fold_results.pkl'
                with open(fold_results_path, 'wb') as f:
                    pickle.dump(results, f)
                print(f"Fold results saved to: {fold_results_path}")
                print(f"  → Use: python scripts/analyze_thresholds.py --fold-results {fold_results_path}")
            except Exception as e:
                print(f"Warning: Could not save fold results: {e}")

            # Generate enhanced reports using existing utils
            try:
                from utils.metrics_report import save_enhanced_results

                fold_metrics = []
                for r in results:
                    if r.get('status') != 'failed' and r.get('test') and r.get('val') and r.get('train'):
                        fold_metrics.append({
                            'test_subject': r['test_subject'],
                            'train': r['train'],
                            'val': r['val'],
                            'test': r['test'],
                        })

                if fold_metrics:
                    save_enhanced_results(fold_metrics, self.work_dir, self.model_name)
                else:
                    print("WARNING: No valid fold metrics for enhanced reports")

            except Exception as e:
                print(f"Warning: Could not generate enhanced reports: {e}")
                traceback.print_exc()

        # Finalize W&B
        if self.wandb_logger:
            self.wandb_logger.log_summary(results, summary)
            self.wandb_logger.finish()

        # Shutdown Ray
        print("\n[Ray] Shutting down...")
        ray.shutdown()

        print(f"\nAll outputs saved to: {self.work_dir}")

        return results_df, results
