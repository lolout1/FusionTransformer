#!/usr/bin/env python3
"""
Full LOSO ablation comparing EventTokenResampler students vs baseline conv1d transformers.

All models use raw IMU input (acc + gyro, no Kalman fusion).
"""

import argparse
import copy
import json
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add paths for imports
_script_dir = Path(__file__).parent.parent.resolve()
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

FUSION_TRANSFORMER_PATH = Path('/home/sww35/FusionTransformer')
if str(FUSION_TRANSFORMER_PATH) not in sys.path:
    sys.path.insert(0, str(FUSION_TRANSFORMER_PATH))

from kd.data_loader import TrialMatcher, WindowedKDDataset, collate_kd_batch
from kd.resampler import TimestampAwareStudent, DualStreamStudent

# Baseline models from main repo
from Models.encoder_ablation import KalmanConv1dConv1d
from Models.single_stream_transformer import SingleStreamTransformerSE

# Subject configuration (SmartFallMM young subjects)
YOUNG_SUBJECTS = [
    29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 43, 44, 45, 46,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63
]
VALIDATION_SUBJECTS = [48, 57]
TRAIN_ONLY_SUBJECTS = [29, 32, 35, 39]
TEST_CANDIDATES = [
    s for s in YOUNG_SUBJECTS
    if s not in VALIDATION_SUBJECTS and s not in TRAIN_ONLY_SUBJECTS
]


def get_model_registry(config: dict) -> Dict:
    """Build model registry with configurations."""
    return {
        'single_resampler': {
            'class': TimestampAwareStudent,
            'args': {
                'input_dim': 6,
                'embed_dim': config['embed_dim'],
                'num_tokens': config['num_tokens'],
                'num_heads': config['num_heads'],
                'num_layers': config['num_layers'],
                'dropout': config['dropout'],
                'time_mode': config['time_mode'],  # default: position (no timestamps)
            },
            'is_timestamp_aware': True,
            'input_channels': 6,
        },
        'dual_resampler': {
            'class': DualStreamStudent,
            'args': {
                'acc_dim': 3,
                'gyro_dim': 3,
                'embed_dim': config['embed_dim'],
                'num_tokens': config['num_tokens'],
                'num_heads': config['num_heads'],
                'num_layers': config['num_layers'],
                'dropout': config['dropout'],
                'acc_ratio': config['acc_ratio'],
                'time_mode': config['time_mode'],  # default: position (no timestamps)
            },
            'is_timestamp_aware': True,
            'input_channels': 6,
        },
        'conv1d_conv1d': {
            'class': KalmanConv1dConv1d,
            'args': {
                'imu_frames': config['window_size'],
                'imu_channels': 7,
                'acc_coords': 7,
                'embed_dim': config['embed_dim'],
                'num_heads': config['num_heads'],
                'num_layers': config['num_layers'],
                'dropout': config['dropout'],
                'acc_ratio': config['acc_ratio'],
            },
            'is_timestamp_aware': False,
            'input_channels': 7,
        },
        'single_stream_se': {
            'class': SingleStreamTransformerSE,
            'args': {
                'imu_frames': config['window_size'],
                'imu_channels': 7,
                'embed_dim': config['embed_dim'],
                'num_heads': config['num_heads'],
                'num_layers': config['num_layers'],
                'dropout': config['dropout'],
            },
            'is_timestamp_aware': False,
            'input_channels': 7,
        },
    }


class BinaryFocalLoss(nn.Module):
    """Focal loss for imbalanced binary classification."""

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
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * ce_loss).mean()


class UnifiedTrainer:
    """Trainer that handles both timestamp-aware and baseline models."""

    def __init__(
        self,
        model: nn.Module,
        is_timestamp_aware: bool,
        input_channels: int,
        device: str,
        lr: float = 1e-3,
        loss_type: str = 'focal',
    ):
        self.model = model.to(device)
        self.is_timestamp_aware = is_timestamp_aware
        self.input_channels = input_channels
        self.device = device

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = BinaryFocalLoss() if loss_type == 'focal' else nn.BCEWithLogitsLoss()

    def _prepare_input(self, batch: dict) -> Tuple:
        """Prepare batch for model input."""
        acc = batch['acc_values'].to(self.device)
        gyro = batch.get('gyro_values')
        if gyro is not None:
            gyro = gyro.to(self.device)
            imu = torch.cat([acc, gyro], dim=-1)
        else:
            imu = acc

        # Handle both 'label' and 'labels' keys (collate outputs 'labels')
        labels = batch.get('labels', batch.get('label')).float().to(self.device)

        if self.input_channels == 7:
            smv = torch.norm(acc, dim=-1, keepdim=True)
            imu = torch.cat([smv, imu], dim=-1)

        if self.is_timestamp_aware:
            timestamps = batch['acc_timestamps'].to(self.device)
            B, T, _ = imu.shape
            mask = torch.ones(B, T, dtype=torch.bool, device=self.device)
            return imu, timestamps, mask, labels
        else:
            return imu, None, None, labels

    def train_epoch(self, train_loader: DataLoader) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            imu, timestamps, mask, labels = self._prepare_input(batch)

            self.optimizer.zero_grad()

            if self.is_timestamp_aware:
                logits, _ = self.model(imu, timestamps, mask)
            else:
                logits, _ = self.model(imu)

            loss = self.criterion(logits.squeeze(-1), labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return {'loss': total_loss / max(n_batches, 1)}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        """Evaluate on given loader."""
        self.model.eval()
        all_preds, all_labels = [], []
        total_loss = 0.0

        for batch in loader:
            imu, timestamps, mask, labels = self._prepare_input(batch)

            if self.is_timestamp_aware:
                logits, _ = self.model(imu, timestamps, mask)
            else:
                logits, _ = self.model(imu)

            loss = self.criterion(logits.squeeze(-1), labels)
            total_loss += loss.item()

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
            'loss': total_loss / max(len(loader), 1),
            'f1': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            'accuracy': float(accuracy),
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        patience: int,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        verbose: bool = False,
        log_every: int = 10,
    ) -> dict:
        """Full training loop with early stopping."""
        best_f1 = 0.0
        best_epoch = 0
        best_weights = None
        best_val_metrics = {}
        no_improve = 0

        for epoch in range(num_epochs):
            train_m = self.train_epoch(train_loader)
            val_m = self.evaluate(val_loader)

            if val_m['f1'] > best_f1:
                best_f1 = val_m['f1']
                best_epoch = epoch
                best_weights = copy.deepcopy(self.model.state_dict())
                best_val_metrics = val_m.copy()
                no_improve = 0
            else:
                no_improve += 1

            if verbose and (epoch + 1) % log_every == 0:
                print(f"        Epoch {epoch+1}: loss={train_m['loss']:.4f} "
                      f"val_f1={val_m['f1']*100:.1f}% (best={best_f1*100:.1f}%)", flush=True)

            if scheduler:
                scheduler.step()

            if no_improve >= patience:
                if verbose:
                    print(f"        Early stop at epoch {epoch+1}", flush=True)
                break

        if best_weights:
            self.model.load_state_dict(best_weights)

        return {
            'best_f1': best_f1,
            'best_epoch': best_epoch,
            'best_val_metrics': best_val_metrics,
            'final_epoch': epoch + 1,
        }


def count_class_distribution(trials: List[dict], dataset: 'WindowedKDDataset') -> dict:
    """Count fall vs ADL trials and windows."""
    fall_trials = sum(1 for t in trials if t['label'] == 1)
    adl_trials = len(trials) - fall_trials

    fall_windows = sum(1 for w in dataset.windows if w['label'] == 1)
    adl_windows = len(dataset.windows) - fall_windows

    return {
        'fall_trials': fall_trials,
        'adl_trials': adl_trials,
        'fall_windows': fall_windows,
        'adl_windows': adl_windows,
    }


def create_fold_dataloaders(
    data_root: str,
    test_subject: int,
    config: dict,
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """Create train/val/test dataloaders for one LOSO fold."""
    train_subjects = [s for s in TEST_CANDIDATES if s != test_subject] + TRAIN_ONLY_SUBJECTS
    val_subjects = VALIDATION_SUBJECTS
    test_subjects = [test_subject]

    matcher = TrialMatcher(data_root, device='watch')

    train_trials = matcher.find_matched_trials(subjects=train_subjects, require_skeleton=False)
    val_trials = matcher.find_matched_trials(subjects=val_subjects, require_skeleton=False)
    test_trials = matcher.find_matched_trials(subjects=test_subjects, require_skeleton=False)

    train_ds = WindowedKDDataset(
        train_trials,
        window_size=config['window_size'],
        fall_stride=config['fall_stride'],
        adl_stride=config['adl_stride'],
        class_aware_stride=True,
    )
    val_ds = WindowedKDDataset(
        val_trials,
        window_size=config['window_size'],
        stride=config['window_size'] // 2,
        class_aware_stride=False,
    )
    test_ds = WindowedKDDataset(
        test_trials,
        window_size=config['window_size'],
        stride=config['window_size'] // 2,
        class_aware_stride=False,
    )

    # Collect statistics
    stats = {
        'train': count_class_distribution(train_trials, train_ds),
        'val': count_class_distribution(val_trials, val_ds),
        'test': count_class_distribution(test_trials, test_ds),
    }

    train_loader = DataLoader(
        train_ds, batch_size=config['batch_size'], shuffle=True,
        collate_fn=collate_kd_batch, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config['batch_size'], shuffle=False,
        collate_fn=collate_kd_batch, num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config['batch_size'], shuffle=False,
        collate_fn=collate_kd_batch, num_workers=2, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, stats


def run_fold(
    fold_idx: int,
    test_subject: int,
    model_name: str,
    model_registry: dict,
    data_root: str,
    config: dict,
    device: str,
    verbose: bool = False,
) -> dict:
    """Train and evaluate single fold for single model."""
    try:
        train_loader, val_loader, test_loader, data_stats = create_fold_dataloaders(
            data_root, test_subject, config
        )

        if verbose:
            ts = data_stats['train']
            print(f"      Data: train={ts['fall_windows']}F/{ts['adl_windows']}A windows "
                  f"({ts['fall_trials']}F/{ts['adl_trials']}A trials)", flush=True)

        model_info = model_registry[model_name]
        model = model_info['class'](**model_info['args'])

        trainer = UnifiedTrainer(
            model=model,
            is_timestamp_aware=model_info['is_timestamp_aware'],
            input_channels=model_info['input_channels'],
            device=device,
            lr=config['lr'],
            loss_type=config['loss'],
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer, T_max=config['num_epochs']
        )

        train_result = trainer.train(
            train_loader, val_loader,
            num_epochs=config['num_epochs'],
            patience=config['patience'],
            scheduler=scheduler,
            verbose=verbose,
            log_every=20,
        )

        test_metrics = trainer.evaluate(test_loader)

        return {
            'data_stats': data_stats,
            'fold_idx': fold_idx,
            'test_subject': test_subject,
            'model': model_name,
            'val_f1': train_result['best_f1'],
            'test_f1': test_metrics['f1'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'test_accuracy': test_metrics['accuracy'],
            'best_epoch': train_result['best_epoch'],
            'status': 'success',
        }

    except Exception as e:
        return {
            'fold_idx': fold_idx,
            'test_subject': test_subject,
            'model': model_name,
            'test_f1': 0.0,
            'status': 'failed',
            'error': str(e),
        }


def run_model_ablation(
    model_name: str,
    model_registry: dict,
    data_root: str,
    config: dict,
    num_gpus: int,
    parallel: int,
    max_folds: Optional[int] = None,
    verbose: bool = False,
) -> List[dict]:
    """Run all folds for one model with parallel execution."""
    test_subjects = TEST_CANDIDATES[:max_folds] if max_folds else TEST_CANDIDATES
    gpu_ids = list(range(num_gpus))

    fold_configs = [
        (fold_idx, subj, gpu_ids[fold_idx % len(gpu_ids)])
        for fold_idx, subj in enumerate(test_subjects)
    ]

    results = []

    def format_result(result: dict) -> str:
        """Format result string with metrics."""
        if result['status'] == 'success':
            s = f"F1={result['test_f1']*100:.1f}%"
            s += f" P={result.get('test_precision', 0)*100:.0f}%"
            s += f" R={result.get('test_recall', 0)*100:.0f}%"
            s += f" @ep{result.get('best_epoch', '?')}"
            if 'data_stats' in result:
                ts = result['data_stats']['test']
                s += f" [{ts['fall_windows']}F/{ts['adl_windows']}A]"
            return s
        return f"FAILED - {result.get('error', 'unknown')}"

    if parallel > 1 and len(fold_configs) > 1:
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {}
            for fold_idx, subj, gpu_id in fold_configs:
                device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
                future = executor.submit(
                    run_fold, fold_idx, subj, model_name,
                    model_registry, data_root, config, device, verbose
                )
                futures[future] = (fold_idx, subj)

            for future in as_completed(futures):
                fold_idx, subj = futures[future]
                result = future.result()
                results.append(result)
                print(f"    Fold {fold_idx} (S{subj}): {format_result(result)}", flush=True)
    else:
        for fold_idx, subj, gpu_id in fold_configs:
            device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
            print(f"    Fold {fold_idx} (S{subj}): starting...", flush=True)
            result = run_fold(
                fold_idx, subj, model_name,
                model_registry, data_root, config, device, verbose
            )
            results.append(result)
            print(f"    Fold {fold_idx} (S{subj}): {format_result(result)}", flush=True)

    return results


def aggregate_results(all_results: dict) -> dict:
    """Compute summary statistics per model."""
    summary = {}

    for model_name, fold_results in all_results.items():
        successful = [r for r in fold_results if r.get('status') == 'success']

        if successful:
            f1_scores = [r['test_f1'] for r in successful]
            precision_scores = [r['test_precision'] for r in successful]
            recall_scores = [r['test_recall'] for r in successful]
            acc_scores = [r['test_accuracy'] for r in successful]

            summary[model_name] = {
                'mean_f1': np.mean(f1_scores) * 100,
                'std_f1': np.std(f1_scores) * 100,
                'mean_precision': np.mean(precision_scores) * 100,
                'mean_recall': np.mean(recall_scores) * 100,
                'mean_accuracy': np.mean(acc_scores) * 100,
                'n_successful': len(successful),
                'n_failed': len(fold_results) - len(successful),
            }
        else:
            summary[model_name] = {
                'mean_f1': 0.0, 'std_f1': 0.0,
                'n_successful': 0, 'n_failed': len(fold_results),
            }

    return summary


def save_results(all_results: dict, summary: dict, output_dir: Path):
    """Save results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'full_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# Full LOSO Ablation Results",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Folds**: {len(TEST_CANDIDATES)}",
        "",
        "## Results",
        "",
        "| Model | F1 (%) | Std | Precision | Recall | Folds |",
        "|-------|--------|-----|-----------|--------|-------|",
    ]

    for model, stats in sorted(summary.items(), key=lambda x: -x[1]['mean_f1']):
        lines.append(
            f"| {model} | {stats['mean_f1']:.2f} | ±{stats['std_f1']:.2f} | "
            f"{stats.get('mean_precision', 0):.1f}% | {stats.get('mean_recall', 0):.1f}% | "
            f"{stats['n_successful']}/{stats['n_successful'] + stats['n_failed']} |"
        )

    with open(output_dir / 'RESULTS.md', 'w') as f:
        f.write('\n'.join(lines))


def run_full_ablation(args):
    """Run full ablation: all models, all folds."""
    print("=" * 60, flush=True)
    print("FULL LOSO ABLATION", flush=True)
    print(f"Models: {args.models or 'all'}", flush=True)
    print(f"Folds: {args.max_folds or len(TEST_CANDIDATES)}", flush=True)
    print(f"GPUs: {args.num_gpus}, Parallel: {args.parallel}", flush=True)
    print("=" * 60, flush=True)

    config = {
        'window_size': args.window_size,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'patience': args.patience,
        'lr': args.lr,
        'loss': args.loss,
        'fall_stride': args.fall_stride,
        'adl_stride': args.adl_stride,
        'embed_dim': args.embed_dim,
        'num_heads': args.num_heads,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'num_tokens': args.num_tokens,
        'time_mode': args.time_mode,
        'acc_ratio': args.acc_ratio,
    }

    model_registry = get_model_registry(config)
    models_to_run = args.models or list(model_registry.keys())

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f'full_ablation_{timestamp}'

    all_results = {}

    for i, model_name in enumerate(models_to_run, 1):
        if model_name not in model_registry:
            print(f"Unknown model: {model_name}, skipping", flush=True)
            continue

        print(f"\n[{i}/{len(models_to_run)}] {model_name}", flush=True)
        print("-" * 40, flush=True)

        results = run_model_ablation(
            model_name=model_name,
            model_registry=model_registry,
            data_root=args.data_root,
            config=config,
            num_gpus=args.num_gpus,
            parallel=args.parallel,
            max_folds=args.max_folds,
            verbose=args.verbose,
        )

        all_results[model_name] = results

        successful = [r for r in results if r.get('status') == 'success']
        if successful:
            f1_scores = [r['test_f1'] for r in successful]
            print(f"  Summary: {np.mean(f1_scores)*100:.2f}% ± {np.std(f1_scores)*100:.2f}%", flush=True)

    summary = aggregate_results(all_results)
    save_results(all_results, summary, output_dir)

    print("\n" + "=" * 60, flush=True)
    print("FINAL RESULTS", flush=True)
    print("-" * 60, flush=True)
    print(f"{'Model':<20} {'F1 (%)':<12} {'Precision':<12} {'Recall':<12}", flush=True)
    print("-" * 60, flush=True)

    for model, stats in sorted(summary.items(), key=lambda x: -x[1]['mean_f1']):
        print(
            f"{model:<20} {stats['mean_f1']:.2f} ± {stats['std_f1']:.2f}  "
            f"{stats.get('mean_precision', 0):.1f}%         {stats.get('mean_recall', 0):.1f}%",
            flush=True
        )

    print("=" * 60, flush=True)
    print(f"Results saved to: {output_dir}", flush=True)

    return all_results, summary


def main():
    parser = argparse.ArgumentParser(description='Full LOSO ablation study')
    parser.add_argument('--data-root', default='data', help='Path to SmartFallMM data')
    parser.add_argument('--output-dir', default='exps', help='Output directory')
    parser.add_argument('--num-gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--parallel', type=int, default=1, help='Parallel folds per model')
    parser.add_argument('--models', nargs='+', default=None,
                        choices=['single_resampler', 'dual_resampler', 'conv1d_conv1d', 'single_stream_se'],
                        help='Models to run (default: all)')
    parser.add_argument('--max-folds', type=int, default=None, help='Limit folds for testing')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose epoch logging')

    parser.add_argument('--window-size', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-epochs', type=int, default=80)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--loss', choices=['bce', 'focal'], default='focal')
    parser.add_argument('--fall-stride', type=int, default=16)
    parser.add_argument('--adl-stride', type=int, default=64)

    parser.add_argument('--embed-dim', type=int, default=48)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num-tokens', type=int, default=64)
    parser.add_argument('--time-mode', default='position', choices=['position', 'timestamps', 'cleaned'])
    parser.add_argument('--acc-ratio', type=float, default=0.65)

    args = parser.parse_args()
    run_full_ablation(args)


if __name__ == '__main__':
    main()
