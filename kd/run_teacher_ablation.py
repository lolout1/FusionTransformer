#!/usr/bin/env python3
"""
Skeleton Teacher Architecture Ablation

Systematic comparison of teacher model configurations for skeleton-based fall detection.
Uses fixed train/val/test split with 2 validation subjects for consistent comparison.

Ablation factors:
  - embed_dim: [48, 64, 96]
  - num_layers: [1, 2, 3]
  - num_heads: [2, 4, 8]
  - dropout: [0.3, 0.5]
  - loss: [focal, bce]

Usage:
    python kd/run_teacher_ablation.py --num-gpus 4 --parallel 2
    python kd/run_teacher_ablation.py --quick --num-gpus 2
    python kd/run_teacher_ablation.py --results-only --work-dir exps/teacher_ablation_XXX
    python kd/run_teacher_ablation.py --dry-run
"""

import argparse
import copy
import json
import os
import pickle
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Fix module path BEFORE any local imports
_script_dir = Path(__file__).parent.parent.resolve()
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Young subjects with skeleton data
YOUNG_SUBJECTS = [29, 30, 31, 32, 34, 35, 36, 37, 38, 39,
                  43, 44, 45, 46, 48, 49, 50, 51, 52, 53,
                  54, 55, 56, 57, 58, 59, 60, 61, 62, 63]

# Fixed split: 2 subjects for validation, 2 for test, rest for training
VAL_SUBJECTS = [29, 30]
TEST_SUBJECTS = [31, 32]
TRAIN_SUBJECTS = [s for s in YOUNG_SUBJECTS if s not in VAL_SUBJECTS + TEST_SUBJECTS]

# Ablation configurations
# Skeleton input: 96 channels (32 joints × 3 xyz), so embed_dim should be >= 96
EMBED_DIMS = [96, 128, 192]
NUM_LAYERS = [2, 3, 4]
NUM_HEADS = [4, 8]
DROPOUTS = [0.3, 0.5]
USE_POS_ENC = [True, False]  # Test with/without positional encoding
LOSSES = ['bce']  # BCE only - focal can cause NaN with extreme values

# Training settings
DEFAULT_EPOCHS = 80
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 0.001
DEFAULT_WEIGHT_DECAY = 0.001
WINDOW_SIZE = 128
PATIENCE = 15


@dataclass
class TeacherConfig:
    """Configuration for a single teacher experiment."""
    name: str
    embed_dim: int
    num_layers: int
    num_heads: int
    dropout: float
    loss_type: str
    use_pos_enc: bool = True
    num_joints: int = 32
    coords_per_joint: int = 3
    se_reduction: int = 4


@dataclass
class ExperimentResult:
    """Result from a single experiment."""
    name: str
    embed_dim: int
    num_layers: int
    num_heads: int
    dropout: float
    loss_type: str
    # Metrics
    val_f1: float = 0.0
    val_accuracy: float = 0.0
    val_precision: float = 0.0
    val_recall: float = 0.0
    test_f1: float = 0.0
    test_f1_std: float = 0.0
    test_accuracy: float = 0.0
    test_precision: float = 0.0
    test_recall: float = 0.0
    # Training info
    best_epoch: int = 0
    total_epochs: int = 0
    elapsed_time: float = 0.0
    num_params: int = 0
    status: str = 'pending'
    error_message: str = ''
    # Per-run metrics for variance
    run_f1s: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def create_teacher_model(config: TeacherConfig, device: str = 'cuda'):
    """Create skeleton teacher model from config."""
    from kd.skeleton_encoder import SkeletonTransformer

    model = SkeletonTransformer(
        num_joints=config.num_joints,
        coords_per_joint=config.coords_per_joint,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
        se_reduction=config.se_reduction,
        use_pos_enc=config.use_pos_enc,
    )
    return model.to(device)


def create_loss_fn(loss_type: str):
    """Create loss function."""
    if loss_type == 'focal':
        # Binary focal loss for imbalanced data
        class BinaryFocalLoss(nn.Module):
            def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma

            def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
                probs = torch.sigmoid(logits)
                ce_loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, targets, reduction='none'
                )
                p_t = probs * targets + (1 - probs) * (1 - targets)
                focal_weight = (1 - p_t) ** self.gamma
                alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
                return (alpha_t * focal_weight * ce_loss).mean()

        return BinaryFocalLoss()
    else:
        return nn.BCEWithLogitsLoss()


class SkeletonOnlyDataset(Dataset):
    """Fast skeleton-only dataset - no IMU loading overhead."""

    def __init__(self, windows: List[Dict], window_size: int = 128):
        self.windows = windows
        self.window_size = window_size

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        w = self.windows[idx]
        skeleton = w['skeleton']  # Pre-loaded numpy array

        # Pad/truncate to window_size
        if len(skeleton) < self.window_size:
            pad = np.zeros((self.window_size - len(skeleton), 96), dtype=np.float32)
            skeleton = np.concatenate([skeleton, pad], axis=0)
        else:
            skeleton = skeleton[:self.window_size]

        return {
            'skeleton': torch.from_numpy(skeleton.astype(np.float32)),
            'label': torch.tensor(w['label'], dtype=torch.long),
        }


@dataclass
class DataLoadingStats:
    """Statistics from data loading process."""
    subjects_requested: int = 0
    subjects_with_data: int = 0
    trials_found: int = 0
    trials_skipped: int = 0
    fall_trials: int = 0
    adl_trials: int = 0
    fall_windows: int = 0
    adl_windows: int = 0
    files_loaded: int = 0
    files_failed: int = 0
    skeleton_empty: int = 0
    acc_empty: int = 0
    errors: List[str] = field(default_factory=list)

    def summary(self, name: str = 'Data') -> str:
        lines = [
            f'{name} Loading Statistics:',
            f'  Subjects: {self.subjects_with_data}/{self.subjects_requested} with data',
            f'  Trials: {self.trials_found} found, {self.trials_skipped} skipped',
            f'  Fall/ADL trials: {self.fall_trials}/{self.adl_trials}',
            f'  Fall/ADL windows: {self.fall_windows}/{self.adl_windows}',
            f'  Files: {self.files_loaded} loaded, {self.files_failed} failed',
        ]
        if self.skeleton_empty > 0:
            lines.append(f'  Empty skeleton files: {self.skeleton_empty}')
        if self.acc_empty > 0:
            lines.append(f'  Empty accelerometer files: {self.acc_empty}')
        if self.errors:
            lines.append(f'  Errors ({len(self.errors)}): {self.errors[:3]}')
        return '\n'.join(lines)


def create_dataloaders(
    data_root: str,
    train_subjects: List[int],
    val_subjects: List[int],
    test_subjects: List[int],
    batch_size: int = 64,
    window_size: int = 128,
    num_workers: int = 4,
    verbose: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, DataLoadingStats]]:
    """Create train/val/test dataloaders for skeleton data with logging.

    Uses fast skeleton-only loading (no IMU timestamp parsing).
    """
    from kd.data_loader import TrialMatcher, load_skeleton

    matcher = TrialMatcher(data_root, age_group='young', device='watch')
    all_stats = {}

    def load_split_with_stats(
        subjects: List[int],
        split_name: str,
        class_aware_stride: bool,
        fall_stride: int,
        adl_stride: int,
    ) -> Tuple[SkeletonOnlyDataset, DataLoadingStats]:
        stats = DataLoadingStats(subjects_requested=len(subjects))
        subjects_seen = set()
        windows = []

        # Find trials (require_skeleton but we don't need gyro for teacher)
        trials = matcher.find_matched_trials(subjects=subjects, require_skeleton=True, require_gyro=False)
        stats.trials_found = len(trials)

        # Load and window each trial
        for trial in trials:
            subject_id = trial['subject_id']
            subjects_seen.add(subject_id)

            try:
                skel_path = trial['files'].get('skeleton')
                if not skel_path:
                    stats.trials_skipped += 1
                    continue

                skeleton = load_skeleton(skel_path)
                if len(skeleton) == 0:
                    stats.skeleton_empty += 1
                    stats.trials_skipped += 1
                    continue

                stats.files_loaded += 1
                label = trial['label']

                # Count trial by class
                if label == 1:
                    stats.fall_trials += 1
                else:
                    stats.adl_trials += 1

                # Create windows
                stride = fall_stride if (class_aware_stride and label == 1) else adl_stride
                if not class_aware_stride:
                    stride = window_size // 2

                n_frames = len(skeleton)
                for start in range(0, max(1, n_frames - window_size // 2), stride):
                    end = min(start + window_size, n_frames)
                    if end - start < window_size // 2:
                        continue

                    windows.append({
                        'skeleton': skeleton[start:end],
                        'label': label,
                    })

                    if label == 1:
                        stats.fall_windows += 1
                    else:
                        stats.adl_windows += 1

            except Exception as e:
                stats.files_failed += 1
                stats.trials_skipped += 1
                if len(stats.errors) < 5:
                    stats.errors.append(f'{trial["trial_id"]}: {str(e)[:50]}')

        stats.subjects_with_data = len(subjects_seen)

        # Create fast skeleton-only dataset
        dataset = SkeletonOnlyDataset(windows, window_size=window_size)
        return dataset, stats

    # Load each split (stride 16:32 for fall:adl to balance classes better)
    train_dataset, train_stats = load_split_with_stats(
        train_subjects, 'train', class_aware_stride=True, fall_stride=16, adl_stride=32
    )
    val_dataset, val_stats = load_split_with_stats(
        val_subjects, 'val', class_aware_stride=False, fall_stride=64, adl_stride=64
    )
    test_dataset, test_stats = load_split_with_stats(
        test_subjects, 'test', class_aware_stride=False, fall_stride=64, adl_stride=64
    )

    all_stats = {'train': train_stats, 'val': val_stats, 'test': test_stats}

    if verbose:
        print('\n' + '=' * 60)
        print('DATA LOADING SUMMARY')
        print('=' * 60)
        for name, stats in all_stats.items():
            print(f'\n{stats.summary(name.upper())}')
        print('=' * 60 + '\n')

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, all_stats


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    valid_batches = 0

    for batch in loader:
        skeleton = batch.get('skeleton')
        labels = batch.get('label') if 'label' in batch else batch.get('labels')

        if skeleton is None or labels is None:
            continue

        skeleton = skeleton.to(device)
        labels = labels.to(device).float()

        # Skip batches with NaN/Inf
        if torch.isnan(skeleton).any() or torch.isinf(skeleton).any():
            continue

        # Normalize skeleton per-batch (zero mean, unit std)
        skeleton = (skeleton - skeleton.mean()) / (skeleton.std() + 1e-8)

        optimizer.zero_grad()
        logits, _ = model(skeleton)
        loss = criterion(logits.squeeze(-1), labels)

        # Skip if loss is NaN
        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        valid_batches += 1
        preds = (torch.sigmoid(logits.squeeze(-1)) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return {
        'loss': total_loss / max(valid_batches, 1),
        'accuracy': correct / max(total, 1),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    valid_batches = 0

    for batch in loader:
        skeleton = batch.get('skeleton')
        labels = batch.get('label') if 'label' in batch else batch.get('labels')

        if skeleton is None or labels is None:
            continue

        skeleton = skeleton.to(device)
        labels = labels.to(device).float()

        # Skip batches with NaN/Inf
        if torch.isnan(skeleton).any() or torch.isinf(skeleton).any():
            continue

        # Normalize skeleton per-batch
        skeleton = (skeleton - skeleton.mean()) / (skeleton.std() + 1e-8)

        logits, _ = model(skeleton)
        loss = criterion(logits.squeeze(-1), labels)

        if not (torch.isnan(loss) or torch.isinf(loss)):
            total_loss += loss.item()
            valid_batches += 1

        preds = (torch.sigmoid(logits.squeeze(-1)) > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()
    tn = ((all_preds == 0) & (all_labels == 0)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / max(len(all_labels), 1)

    return {
        'loss': total_loss / max(valid_batches, 1),
        'f1': float(f1) * 100,
        'accuracy': float(accuracy) * 100,
        'precision': float(precision) * 100,
        'recall': float(recall) * 100,
    }


def run_single_experiment(
    config: TeacherConfig,
    data_root: str,
    work_dir: Path,
    num_epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    patience: int = PATIENCE,
    device: str = 'cuda',
    seed: int = 42,
    verbose: bool = False,
) -> ExperimentResult:
    """Run a single teacher training experiment."""
    start_time = time.time()

    result = ExperimentResult(
        name=config.name,
        embed_dim=config.embed_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
        loss_type=config.loss_type,
    )

    try:
        # Set seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create model and count parameters
        model = create_teacher_model(config, device)
        result.num_params = sum(p.numel() for p in model.parameters())

        # Create loss and optimizer
        criterion = create_loss_fn(config.loss_type)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=DEFAULT_WEIGHT_DECAY)

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        # Create dataloaders (verbose=False for parallel runs to avoid spam)
        train_loader, val_loader, test_loader, _ = create_dataloaders(
            data_root, TRAIN_SUBJECTS, VAL_SUBJECTS, TEST_SUBJECTS,
            batch_size=batch_size, window_size=WINDOW_SIZE, num_workers=2,
            verbose=verbose,
        )

        # Training loop with early stopping
        best_val_f1 = 0.0
        best_epoch = 0
        epochs_without_improvement = 0
        best_model_state = None

        print(f'  > {config.name} [{device}] starting...', flush=True)

        for epoch in range(num_epochs):
            train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            # Log progress every 10 epochs or on improvement
            improved = val_metrics['f1'] > best_val_f1
            if improved or epoch % 10 == 0 or epoch == num_epochs - 1:
                status = '*' if improved else ' '
                print(f'    {config.name} [{device}] ep {epoch+1:2d}: '
                      f'loss={train_metrics["loss"]:.3f} '
                      f'val_f1={val_metrics["f1"]:.1f}%{status}', flush=True)

            if improved:
                best_val_f1 = val_metrics['f1']
                best_epoch = epoch
                epochs_without_improvement = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f'    {config.name} early stop at epoch {epoch+1} (best={best_epoch+1})', flush=True)
                break

        # Load best model and evaluate on test set
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        test_metrics = evaluate(model, test_loader, criterion, device)

        # Update result
        result.val_f1 = best_val_f1
        result.val_accuracy = val_metrics['accuracy']
        result.val_precision = val_metrics['precision']
        result.val_recall = val_metrics['recall']
        result.test_f1 = test_metrics['f1']
        result.test_accuracy = test_metrics['accuracy']
        result.test_precision = test_metrics['precision']
        result.test_recall = test_metrics['recall']
        result.best_epoch = best_epoch
        result.total_epochs = epoch + 1
        result.status = 'completed'

        # Save best model
        model_path = work_dir / f'{config.name}_best.pth'
        torch.save(best_model_state, model_path)

    except Exception as e:
        result.status = 'failed'
        result.error_message = str(e)[:500]

    result.elapsed_time = time.time() - start_time
    return result


def run_experiments_parallel(
    configs: List[TeacherConfig],
    data_root: str,
    work_dir: Path,
    num_gpus: int,
    parallel: int,
    num_epochs: int,
) -> List[ExperimentResult]:
    """Run experiments in parallel across GPUs."""
    results = []
    gpu_ids = list(range(num_gpus))

    # Create work directories
    (work_dir / 'models').mkdir(parents=True, exist_ok=True)

    def run_on_gpu(config: TeacherConfig, gpu_id: int) -> ExperimentResult:
        device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
        return run_single_experiment(
            config, data_root, work_dir / 'models',
            num_epochs=num_epochs, device=device,
        )

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {}
        for i, config in enumerate(configs):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            future = executor.submit(run_on_gpu, config, gpu_id)
            futures[future] = config.name

        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                results.append(result)
                status = '+' if result.status == 'completed' else 'x'
                f1_str = f'{result.test_f1:.2f}%' if result.test_f1 > 0 else 'N/A'
                print(f'  {status} {name}: test_f1={f1_str}, val_f1={result.val_f1:.2f}%', flush=True)
            except Exception as e:
                print(f'  x {name}: {e}', flush=True)
                results.append(ExperimentResult(
                    name=name, embed_dim=0, num_layers=0, num_heads=0,
                    dropout=0, loss_type='', status='error', error_message=str(e),
                ))

    return results


def generate_report(results: List[ExperimentResult], output_path: Path) -> str:
    """Generate markdown report with analysis."""
    lines = [
        '# Skeleton Teacher Architecture Ablation',
        f'\nGenerated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
        '',
        '## Configuration',
        '',
        f'- Train subjects: {len(TRAIN_SUBJECTS)} ({TRAIN_SUBJECTS[:3]}...)',
        f'- Val subjects: {VAL_SUBJECTS}',
        f'- Test subjects: {TEST_SUBJECTS}',
        f'- Window size: {WINDOW_SIZE}',
        f'- Epochs: {DEFAULT_EPOCHS} (early stopping patience={PATIENCE})',
        '',
        '## Full Results',
        '',
        '| Config | embed | layers | heads | drop | loss | Val F1 | Test F1 | Params |',
        '|--------|-------|--------|-------|------|------|--------|---------|--------|',
    ]

    # Sort by test F1
    results_sorted = sorted(results, key=lambda x: x.test_f1, reverse=True)
    best_f1 = results_sorted[0].test_f1 if results_sorted else 0

    for r in results_sorted:
        if r.status != 'completed':
            continue
        marker = ' **' if abs(r.test_f1 - best_f1) < 0.1 else ''
        end_marker = '**' if marker else ''
        lines.append(
            f'| {marker}{r.name}{end_marker} | {r.embed_dim} | {r.num_layers} | {r.num_heads} | '
            f'{r.dropout} | {r.loss_type} | {r.val_f1:.2f} | {r.test_f1:.2f} | {r.num_params:,} |'
        )

    # Factor analysis
    lines.extend(['', '## Factor Analysis', ''])

    # Embed dim effect
    lines.append('### Embedding Dimension Effect')
    lines.append('')
    lines.append('| embed_dim | Mean F1 | Std | Count |')
    lines.append('|-----------|---------|-----|-------|')
    for dim in EMBED_DIMS:
        subset = [r for r in results if r.embed_dim == dim and r.status == 'completed']
        if subset:
            mean_f1 = statistics.mean([r.test_f1 for r in subset])
            std_f1 = statistics.stdev([r.test_f1 for r in subset]) if len(subset) > 1 else 0
            lines.append(f'| {dim} | {mean_f1:.2f} | {std_f1:.2f} | {len(subset)} |')

    # Num layers effect
    lines.append('')
    lines.append('### Number of Layers Effect')
    lines.append('')
    lines.append('| num_layers | Mean F1 | Std | Count |')
    lines.append('|------------|---------|-----|-------|')
    for nl in NUM_LAYERS:
        subset = [r for r in results if r.num_layers == nl and r.status == 'completed']
        if subset:
            mean_f1 = statistics.mean([r.test_f1 for r in subset])
            std_f1 = statistics.stdev([r.test_f1 for r in subset]) if len(subset) > 1 else 0
            lines.append(f'| {nl} | {mean_f1:.2f} | {std_f1:.2f} | {len(subset)} |')

    # Num heads effect
    lines.append('')
    lines.append('### Number of Heads Effect')
    lines.append('')
    lines.append('| num_heads | Mean F1 | Std | Count |')
    lines.append('|-----------|---------|-----|-------|')
    for nh in NUM_HEADS:
        subset = [r for r in results if r.num_heads == nh and r.status == 'completed']
        if subset:
            mean_f1 = statistics.mean([r.test_f1 for r in subset])
            std_f1 = statistics.stdev([r.test_f1 for r in subset]) if len(subset) > 1 else 0
            lines.append(f'| {nh} | {mean_f1:.2f} | {std_f1:.2f} | {len(subset)} |')

    # Dropout effect
    lines.append('')
    lines.append('### Dropout Effect')
    lines.append('')
    lines.append('| dropout | Mean F1 | Std | Count |')
    lines.append('|---------|---------|-----|-------|')
    for drop in DROPOUTS:
        subset = [r for r in results if abs(r.dropout - drop) < 0.01 and r.status == 'completed']
        if subset:
            mean_f1 = statistics.mean([r.test_f1 for r in subset])
            std_f1 = statistics.stdev([r.test_f1 for r in subset]) if len(subset) > 1 else 0
            lines.append(f'| {drop} | {mean_f1:.2f} | {std_f1:.2f} | {len(subset)} |')

    # Loss type effect
    lines.append('')
    lines.append('### Loss Type Effect')
    lines.append('')
    lines.append('| loss | Mean F1 | Std | Count |')
    lines.append('|------|---------|-----|-------|')
    for loss in LOSSES:
        subset = [r for r in results if r.loss_type == loss and r.status == 'completed']
        if subset:
            mean_f1 = statistics.mean([r.test_f1 for r in subset])
            std_f1 = statistics.stdev([r.test_f1 for r in subset]) if len(subset) > 1 else 0
            lines.append(f'| {loss} | {mean_f1:.2f} | {std_f1:.2f} | {len(subset)} |')

    # Summary
    lines.extend(['', '## Summary', ''])
    if results_sorted and results_sorted[0].status == 'completed':
        best = results_sorted[0]
        lines.append(f'**Best configuration**: {best.name}')
        lines.append(f'- embed_dim={best.embed_dim}, num_layers={best.num_layers}, '
                    f'num_heads={best.num_heads}, dropout={best.dropout}, loss={best.loss_type}')
        lines.append(f'- Test F1: {best.test_f1:.2f}%')
        lines.append(f'- Val F1: {best.val_f1:.2f}%')
        lines.append(f'- Parameters: {best.num_params:,}')
        lines.append(f'- Training time: {best.elapsed_time:.1f}s')

    # Statistics
    completed = [r for r in results if r.status == 'completed']
    if completed:
        lines.append('')
        lines.append(f'**Statistics**: {len(completed)}/{len(results)} completed')
        lines.append(f'- Mean test F1: {statistics.mean([r.test_f1 for r in completed]):.2f}%')
        lines.append(f'- Std test F1: {statistics.stdev([r.test_f1 for r in completed]):.2f}%' if len(completed) > 1 else '')
        lines.append(f'- Range: {min(r.test_f1 for r in completed):.2f}% - {max(r.test_f1 for r in completed):.2f}%')

    report = '\n'.join(lines)
    with open(output_path, 'w') as f:
        f.write(report)
    return report


def main():
    parser = argparse.ArgumentParser(
        description='Skeleton teacher architecture ablation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--num-gpus', type=int, default=4)
    parser.add_argument('--parallel', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--quick', action='store_true', help='Reduced config grid (faster)')
    parser.add_argument('--work-dir', type=Path, default=None)
    parser.add_argument('--data-root', type=str, default='data')
    parser.add_argument('--results-only', action='store_true', help='Regenerate report')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    if args.work_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.work_dir = Path(f'exps/teacher_ablation_{timestamp}')

    # Results-only mode
    if args.results_only:
        results_path = args.work_dir / 'results.json'
        if results_path.exists():
            with open(results_path) as f:
                data = json.load(f)
            results = [ExperimentResult(**r) for r in data]
            report = generate_report(results, args.work_dir / 'teacher_ablation_report.md')
            print(report)
        else:
            print(f'Error: No results.json in {args.work_dir}')
            sys.exit(1)
        return

    # Generate experiment configs
    if args.quick:
        # Reduced grid for quick testing
        embed_dims = [96, 128]
        num_layers = [2]
        num_heads = [4]
        dropouts = [0.5]
        use_pos_enc = [False]
        losses = ['bce']
    else:
        embed_dims = EMBED_DIMS
        num_layers = NUM_LAYERS
        num_heads = NUM_HEADS
        use_pos_enc = USE_POS_ENC
        dropouts = DROPOUTS
        losses = LOSSES

    configs = []
    for ed, nl, nh, do, pos, loss in product(embed_dims, num_layers, num_heads, dropouts, use_pos_enc, losses):
        # Skip invalid combinations (heads must divide embed_dim)
        if ed % nh != 0:
            continue
        pos_str = 'pos' if pos else 'nopos'
        name = f'e{ed}_l{nl}_h{nh}_d{int(do*10)}_{pos_str}_{loss}'
        configs.append(TeacherConfig(
            name=name,
            embed_dim=ed,
            num_layers=nl,
            num_heads=nh,
            dropout=do,
            use_pos_enc=pos,
            loss_type=loss,
        ))

    print('Skeleton Teacher Architecture Ablation')
    print('=' * 50)
    print(f'Total experiments: {len(configs)}')
    print(f'Factors: embed_dim={embed_dims}, num_layers={num_layers}, '
          f'num_heads={num_heads}, dropout={dropouts}, loss={losses}')
    print(f'Train subjects: {len(TRAIN_SUBJECTS)}, Val: {VAL_SUBJECTS}, Test: {TEST_SUBJECTS}')
    print(f'GPUs: {args.num_gpus}, Parallel: {args.parallel}')
    print(f'Epochs: {args.epochs}')
    print(f'Output: {args.work_dir}')
    print()

    if args.dry_run:
        print('Dry run - would execute:')
        for c in configs[:10]:
            print(f'  - {c.name}')
        if len(configs) > 10:
            print(f'  ... and {len(configs) - 10} more')
        return

    # Create output directory
    args.work_dir.mkdir(parents=True, exist_ok=True)

    # Save spec
    spec = {
        'configs': [c.name for c in configs],
        'train_subjects': TRAIN_SUBJECTS,
        'val_subjects': VAL_SUBJECTS,
        'test_subjects': TEST_SUBJECTS,
        'num_gpus': args.num_gpus,
        'parallel': args.parallel,
        'epochs': args.epochs,
        'timestamp': datetime.now().isoformat(),
    }
    with open(args.work_dir / 'spec.json', 'w') as f:
        json.dump(spec, f, indent=2)

    # Validate data loading and show statistics (run once with verbose)
    print('Validating data loading...')
    _, _, _, data_stats = create_dataloaders(
        args.data_root, TRAIN_SUBJECTS, VAL_SUBJECTS, TEST_SUBJECTS,
        batch_size=64, window_size=WINDOW_SIZE, num_workers=0,
        verbose=True,
    )

    # Save data stats
    stats_summary = {
        split: {
            'subjects_requested': s.subjects_requested,
            'subjects_with_data': s.subjects_with_data,
            'trials_found': s.trials_found,
            'trials_skipped': s.trials_skipped,
            'fall_trials': s.fall_trials,
            'adl_trials': s.adl_trials,
            'fall_windows': s.fall_windows,
            'adl_windows': s.adl_windows,
            'files_loaded': s.files_loaded,
            'files_failed': s.files_failed,
        }
        for split, s in data_stats.items()
    }
    with open(args.work_dir / 'data_stats.json', 'w') as f:
        json.dump(stats_summary, f, indent=2)

    # Check for critical data issues
    total_train_windows = data_stats['train'].fall_windows + data_stats['train'].adl_windows
    if total_train_windows == 0:
        print('ERROR: No training windows found. Check data path.')
        sys.exit(1)

    # Run experiments
    print('Running experiments...', flush=True)
    results = run_experiments_parallel(
        configs, args.data_root, args.work_dir,
        args.num_gpus, args.parallel, args.epochs,
    )

    # Save results
    with open(args.work_dir / 'results.json', 'w') as f:
        json.dump([r.to_dict() for r in results], f, indent=2)

    # Generate report
    print()
    report = generate_report(results, args.work_dir / 'teacher_ablation_report.md')
    print(report)
    print(f'\nResults saved to: {args.work_dir}')


if __name__ == '__main__':
    main()
