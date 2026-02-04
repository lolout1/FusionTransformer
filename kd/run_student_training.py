#!/usr/bin/env python3
"""
Student model training with EventTokenResampler.

Supports single-stream and dual-stream variants with configurable time modes.
"""

import argparse
import json
import sys
from pathlib import Path

_script_dir = Path(__file__).parent.parent.resolve()
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from kd.data_loader import TrialMatcher, WindowedKDDataset, create_kd_dataloaders
from kd.resampler import TimestampAwareStudent, DualStreamStudent
from kd.skeleton_encoder import SkeletonTransformer
from kd.losses import CombinedKDLoss


class StudentTrainer:
    """Unified trainer for single-stream and dual-stream student models."""

    def __init__(self, config: dict, device: str = 'cuda'):
        self.config = config
        self.device = device
        self.model = self._build_model()
        self.teacher = None
        self.kd_loss = None
        self.criterion = nn.BCEWithLogitsLoss()

    def _build_model(self) -> nn.Module:
        cfg = self.config
        if cfg.get('dual_stream', False):
            model = DualStreamStudent(
                acc_dim=cfg.get('acc_dim', 3),
                gyro_dim=cfg.get('gyro_dim', 3),
                embed_dim=cfg.get('embed_dim', 48),
                num_tokens=cfg.get('num_tokens', 64),
                num_heads=cfg.get('num_heads', 4),
                num_layers=cfg.get('num_layers', 2),
                dropout=cfg.get('dropout', 0.3),
                acc_ratio=cfg.get('acc_ratio', 0.65),
                time_mode=cfg.get('time_mode', 'position'),
            )
        else:
            model = TimestampAwareStudent(
                input_dim=cfg.get('input_dim', 6),
                embed_dim=cfg.get('embed_dim', 48),
                num_tokens=cfg.get('num_tokens', 64),
                num_heads=cfg.get('num_heads', 4),
                num_layers=cfg.get('num_layers', 2),
                dropout=cfg.get('dropout', 0.3),
                time_mode=cfg.get('time_mode', 'position'),
            )
        return model.to(self.device)

    def load_teacher(self, weights_path: str, teacher_config: dict):
        """Load frozen teacher for KD."""
        self.teacher = SkeletonTransformer(
            embed_dim=teacher_config.get('embed_dim', 96),
            num_heads=teacher_config.get('num_heads', 4),
            num_layers=teacher_config.get('num_layers', 2),
            dropout=teacher_config.get('dropout', 0.3),
        ).to(self.device)

        self.teacher.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Attention KD config
        attn_enabled = self.config.get('attention_kd', False)
        self.attention_enabled = attn_enabled

        self.kd_loss = CombinedKDLoss(
            embed_dim=self.config.get('embed_dim', 48),
            teacher_embed_dim=teacher_config.get('embed_dim', 96),
            task_loss=self.criterion,
            task_weight=1.0,
            embedding_weight=self.config.get('kd_embedding_weight', 1.0),
            embedding_enabled=True,
            gram_weight=self.config.get('kd_gram_weight', 0.5),
            gram_enabled=True,
            comodo_weight=0.0,
            comodo_enabled=False,
            attention_enabled=attn_enabled,
            attention_weight=self.config.get('attention_weight', 0.5),
            attention_type=self.config.get('attention_type', 'pool'),
            attention_loss_type=self.config.get('attention_loss', 'kl'),
            attention_temperature=self.config.get('attention_temp', 1.0),
        ).to(self.device)

    def _unpack_batch(self, batch: dict):
        """Extract tensors from batch dict."""
        labels = batch.get('labels', batch.get('label')).to(self.device)

        acc = batch['acc_values'].to(self.device)
        gyro = batch.get('gyro_values')
        if gyro is not None:
            gyro = gyro.to(self.device)
            imu = torch.cat([acc, gyro], dim=-1)
        else:
            imu = acc

        # Compute SMV (speed of motion) if configured
        if self.config.get('include_smv', False):
            smv = torch.norm(acc, dim=-1, keepdim=True)
            imu = torch.cat([smv, imu], dim=-1)

        ts = batch.get('acc_timestamps')
        if ts is not None:
            ts = ts.to(self.device)
        else:
            B, T, _ = imu.shape
            ts = torch.linspace(0, T / 30.0, T, device=self.device).unsqueeze(0).expand(B, -1)

        skeleton = batch.get('skeleton')
        if skeleton is not None:
            skeleton = skeleton.to(self.device)
            skeleton = (skeleton - skeleton.mean()) / (skeleton.std() + 1e-8)

        mask = batch.get('mask')
        if mask is not None:
            mask = mask.to(self.device)

        return imu, ts, skeleton, labels, mask

    def train_epoch(self, loader: DataLoader) -> dict:
        self.model.train()
        total_loss, total_kd, correct, total = 0.0, 0.0, 0, 0
        attention_enabled = getattr(self, 'attention_enabled', False)

        for batch in loader:
            imu, ts, skeleton, labels, mask = self._unpack_batch(batch)

            # Forward with optional attention extraction
            s_attn = None
            if attention_enabled:
                logits, features, s_attn = self.model(imu, ts, mask, return_attention=True)
            else:
                logits, features = self.model(imu, ts, mask)

            if self.teacher is not None and skeleton is not None:
                with torch.no_grad():
                    if attention_enabled:
                        _, t_embed, t_attn = self.teacher(skeleton, return_attention=True)
                    else:
                        _, t_embed = self.teacher(skeleton)
                        t_attn = None
                s_tokens, _ = self.model.resampler(imu, ts, mask)
                loss, loss_dict = self.kd_loss(
                    student_logits=logits,
                    student_embed=features,
                    labels=labels,
                    teacher_embed=t_embed,
                    student_tokens=s_tokens,
                    student_attn=s_attn,
                    teacher_attn=t_attn,
                )
                total_kd += loss_dict.get('embedding', 0) + loss_dict.get('gram', 0)
                if 'attention' in loss_dict:
                    total_kd += loss_dict['attention']
            else:
                loss = self.criterion(logits.squeeze(-1), labels.float())

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            preds = (torch.sigmoid(logits.squeeze(-1)) > 0.5).float()
            correct += (preds == labels.float()).sum().item()
            total += labels.size(0)

        n = len(loader)
        return {'loss': total_loss / n, 'kd_loss': total_kd / n, 'acc': correct / max(total, 1)}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        self.model.eval()
        preds_all, labels_all, total_loss = [], [], 0.0

        for batch in loader:
            imu, ts, _, labels, mask = self._unpack_batch(batch)
            logits, _ = self.model(imu, ts, mask)
            loss = self.criterion(logits.squeeze(-1), labels.float())
            total_loss += loss.item()

            preds = (torch.sigmoid(logits.squeeze(-1)) > 0.5).float()
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

        preds_all = np.array(preds_all)
        labels_all = np.array(labels_all)

        tp = ((preds_all == 1) & (labels_all == 1)).sum()
        fp = ((preds_all == 1) & (labels_all == 0)).sum()
        fn = ((preds_all == 0) & (labels_all == 1)).sum()
        tn = ((preds_all == 0) & (labels_all == 0)).sum()

        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)

        return {
            'loss': total_loss / max(len(loader), 1),
            'acc': (tp + tn) / max(len(labels_all), 1),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 80,
        patience: int = 15,
        lr: float = 1e-3,
        save_dir: Path = None,
    ) -> dict:
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs)

        best_f1, best_epoch, no_improve = 0.0, 0, 0

        for epoch in range(num_epochs):
            train_m = self.train_epoch(train_loader)
            val_m = self.evaluate(val_loader)

            if val_m['f1'] > best_f1:
                best_f1 = val_m['f1']
                best_epoch = epoch
                no_improve = 0
                if save_dir:
                    save_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(self.model.state_dict(), save_dir / 'best_model.pth')
            else:
                no_improve += 1

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}: loss={train_m['loss']:.4f} val_f1={val_m['f1']*100:.1f}%", flush=True)

            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch+1}", flush=True)
                break

            self.scheduler.step()

        return {'best_f1': best_f1, 'best_epoch': best_epoch}


def get_all_subjects(data_root: Path) -> list:
    """Get all subject IDs from accelerometer directory."""
    matcher = TrialMatcher(data_root)
    trials = matcher.find_matched_trials(require_skeleton=True)
    return sorted(set(t['subject_id'] for t in trials))


def run_experiment(args):
    """Run single experiment with given config."""
    print("=" * 60, flush=True)
    mode_str = "DUAL-STREAM" if args.dual_stream else "SINGLE-STREAM"
    kd_str = "with KD" if args.teacher_weights else "baseline"
    print(f"STUDENT: {mode_str} ({args.time_mode}) {kd_str}", flush=True)
    print("=" * 60, flush=True)

    data_root = Path(args.data_root)

    # Get subjects
    all_subjects = get_all_subjects(data_root)
    print(f"Found {len(all_subjects)} subjects with skeleton data", flush=True)

    # Create data loaders
    train_subjects = [s for s in all_subjects if s not in args.val_subjects + args.test_subjects]
    print(f"Split: train={len(train_subjects)}, val={len(args.val_subjects)}, test={len(args.test_subjects)}", flush=True)

    train_loader, val_loader, test_loader = create_kd_dataloaders(
        data_root=str(data_root),
        train_subjects=train_subjects,
        val_subjects=args.val_subjects,
        test_subjects=args.test_subjects,
        window_size=args.window_size,
        batch_size=args.batch_size,
        fall_stride=args.fall_stride,
        adl_stride=args.adl_stride,
    )
    print(f"Windows: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}", flush=True)

    # Build config (add SMV channel if enabled)
    input_dim = 7 if args.include_smv else 6
    acc_dim = 4 if args.include_smv else 3  # smv + acc_xyz or just acc_xyz

    config = {
        'dual_stream': args.dual_stream,
        'input_dim': input_dim,
        'acc_dim': acc_dim,
        'gyro_dim': 3,
        'embed_dim': args.embed_dim,
        'num_tokens': args.num_tokens,
        'num_heads': args.num_heads,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'acc_ratio': args.acc_ratio,
        'time_mode': args.time_mode,
        'include_smv': args.include_smv,
        # Attention KD config
        'attention_kd': args.attention_kd,
        'attention_type': args.attention_type,
        'attention_weight': args.attention_weight,
        'attention_loss': args.attention_loss,
        'attention_temp': args.attention_temp,
    }

    trainer = StudentTrainer(config, device=args.device)
    print(f"Model params: {sum(p.numel() for p in trainer.model.parameters()):,}", flush=True)

    # Load teacher if KD
    if args.teacher_weights and Path(args.teacher_weights).exists():
        trainer.load_teacher(args.teacher_weights, {
            'embed_dim': args.teacher_embed_dim,
            'num_heads': 4,
            'num_layers': 2,
        })
        print(f"Loaded teacher from {args.teacher_weights}", flush=True)

    # Train
    save_name = f"{'dual' if args.dual_stream else 'single'}_{args.time_mode}"
    if args.teacher_weights:
        save_name += "_kd"
    save_dir = Path(args.output_dir) / save_name

    print("\nTraining...", flush=True)
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        save_dir=save_dir,
    )

    # Evaluate
    best_path = save_dir / 'best_model.pth'
    if best_path.exists():
        trainer.model.load_state_dict(torch.load(best_path))
    test_m = trainer.evaluate(test_loader)

    print("\n" + "=" * 60, flush=True)
    print(f"Val F1: {results['best_f1']*100:.2f}% (epoch {results['best_epoch']})", flush=True)
    print(f"Test F1: {test_m['f1']*100:.2f}%", flush=True)
    print(f"Test Precision: {test_m['precision']*100:.2f}%", flush=True)
    print(f"Test Recall: {test_m['recall']*100:.2f}%", flush=True)
    print("=" * 60, flush=True)

    # Save results
    with open(save_dir / 'results.json', 'w') as f:
        json.dump({
            'config': config,
            'val_f1': results['best_f1'],
            'test_f1': test_m['f1'],
            'test_precision': test_m['precision'],
            'test_recall': test_m['recall'],
        }, f, indent=2)

    return results, test_m


def run_single_config(config_tuple):
    """Run single config (for parallel execution)."""
    dual, time_mode, gpu_id, args_dict = config_tuple

    import copy
    args = argparse.Namespace(**args_dict)
    args.dual_stream = dual
    args.time_mode = time_mode
    args.device = f'cuda:{gpu_id}'

    try:
        _, test_m = run_experiment(args)
        return {
            'dual_stream': dual,
            'time_mode': time_mode,
            'test_f1': test_m['f1'],
            'gpu': gpu_id,
            'error': None,
        }
    except Exception as e:
        return {
            'dual_stream': dual,
            'time_mode': time_mode,
            'test_f1': 0.0,
            'gpu': gpu_id,
            'error': str(e),
        }


def run_ablation(args):
    """Run ablation over architectures with parallel GPU support."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print("=" * 60, flush=True)
    print("ABLATION: Single vs Dual-Stream", flush=True)
    print(f"GPUs: {args.num_gpus}, Parallel: {args.parallel}", flush=True)
    print("=" * 60, flush=True)

    # Build experiment configs
    configs = []
    args_dict = vars(args).copy()

    for dual in [False, True]:
        for time_mode in ['position']:
            configs.append((dual, time_mode))

    # Assign GPUs round-robin
    gpu_ids = list(range(args.num_gpus))
    config_tuples = [
        (dual, tm, gpu_ids[i % len(gpu_ids)], args_dict)
        for i, (dual, tm) in enumerate(configs)
    ]

    results = []

    if args.parallel > 1 and len(configs) > 1:
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {executor.submit(run_single_config, ct): ct for ct in config_tuples}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                arch = "Dual" if result['dual_stream'] else "Single"
                if result['error']:
                    print(f"[GPU {result['gpu']}] {arch}: ERROR - {result['error']}", flush=True)
                else:
                    print(f"[GPU {result['gpu']}] {arch}: {result['test_f1']*100:.2f}%", flush=True)
    else:
        for ct in config_tuples:
            result = run_single_config(ct)
            results.append(result)

    print("\n" + "=" * 60, flush=True)
    print("ABLATION RESULTS", flush=True)
    print("-" * 40, flush=True)
    for r in sorted(results, key=lambda x: x['dual_stream']):
        arch = "Dual-Stream" if r['dual_stream'] else "Single-Stream"
        if r['error']:
            print(f"{arch}: FAILED", flush=True)
        else:
            print(f"{arch}: {r['test_f1']*100:.2f}%", flush=True)
    print("=" * 60, flush=True)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['baseline', 'kd', 'ablation'], default='baseline')
    parser.add_argument('--dual-stream', action='store_true', help='Use dual-stream resampler')
    parser.add_argument('--time-mode', choices=['position', 'timestamps', 'cleaned'], default='position')
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--output-dir', default='exps/student')
    parser.add_argument('--teacher-weights', default=None)
    parser.add_argument('--teacher-embed-dim', type=int, default=96)

    # Model
    parser.add_argument('--embed-dim', type=int, default=48)
    parser.add_argument('--num-tokens', type=int, default=64)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--acc-ratio', type=float, default=0.65)
    parser.add_argument('--include-smv', action='store_true',
                        help='Add SMV (acceleration magnitude) as first channel (7ch total)')

    # Attention KD (2026-02-04)
    parser.add_argument('--attention-kd', action='store_true',
                        help='Enable attention distillation from teacher to student')
    parser.add_argument('--attention-type', default='pool', choices=['pool'],
                        help='Attention source: pool (TemporalAttentionPooling weights)')
    parser.add_argument('--attention-weight', type=float, default=0.5,
                        help='Weight for attention KD loss component')
    parser.add_argument('--attention-loss', default='kl', choices=['kl', 'mse', 'js'],
                        help='Attention loss function')
    parser.add_argument('--attention-temp', type=float, default=1.0,
                        help='Temperature for attention softmax (higher=softer)')

    # Training
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--window-size', type=int, default=128)
    parser.add_argument('--fall-stride', type=int, default=8)
    parser.add_argument('--adl-stride', type=int, default=64)

    # Data split (match main repo defaults)
    parser.add_argument('--val-subjects', type=int, nargs='+', default=[48, 57])
    parser.add_argument('--test-subjects', type=int, nargs='+', default=[30, 31])

    # Hardware
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-gpus', type=int, default=1, help='Number of GPUs for ablation')
    parser.add_argument('--parallel', type=int, default=1, help='Parallel experiments')

    args = parser.parse_args()

    if args.mode == 'ablation':
        run_ablation(args)
    elif args.mode == 'kd':
        if not args.teacher_weights:
            print("ERROR: --teacher-weights required for KD mode", flush=True)
            sys.exit(1)
        run_experiment(args)
    else:
        run_experiment(args)


if __name__ == '__main__':
    main()
