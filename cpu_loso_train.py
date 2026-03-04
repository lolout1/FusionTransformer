#!/usr/bin/env python3
"""
CPU-only LOSO Training for SmartFallMM.

Runs sequential Leave-One-Subject-Out cross-validation on CPU.
Saves fold_results.pkl compatible with significance testing scripts.
Optionally runs temporal queue evaluation after all folds complete.

Usage:
    # Default config (dual-stream Kalman baseline)
    python cpu_loso_train.py --config config/best_config/smartfallmm/kalman_baseline.yaml

    # Quick test (2 folds)
    python cpu_loso_train.py --config config/best_config/smartfallmm/kalman_baseline.yaml --max-folds 2

    # With temporal queue evaluation after LOSO
    python cpu_loso_train.py --config config/best_config/smartfallmm/kalman_baseline.yaml --queue-eval

    # Resume from fold 10
    python cpu_loso_train.py --config config/best_config/smartfallmm/kalman_baseline.yaml --resume-from 10

    # Custom output
    python cpu_loso_train.py --config config/best_config/smartfallmm/kalman_baseline.yaml -o exps/my_run
"""

import argparse
import os
import sys
import time
import json
import pickle
import random
import traceback
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
import logging
logging.disable(logging.INFO)
from copy import deepcopy
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
)
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from utils.dataset import prepare_smartfallmm, split_by_subjects
from utils.callbacks import EarlyStopping
from utils.loss import BinaryFocalLoss, ClassBalancedFocalLoss

THRESHOLD = 0.5


def parse_args():
    p = argparse.ArgumentParser(description='CPU LOSO Training for SmartFallMM')
    p.add_argument('--config', '-c', required=True, help='YAML config path')
    p.add_argument('-o', '--work-dir', default=None, help='Output directory')
    p.add_argument('--max-folds', type=int, default=None, help='Limit number of folds')
    p.add_argument('--resume-from', type=int, default=0, help='Resume from fold index')
    p.add_argument('--seed', type=int, default=2)
    p.add_argument('--num-workers', type=int, default=0, help='DataLoader workers')
    p.add_argument('--verbose', '-v', action='store_true')
    p.add_argument('--queue-eval', action='store_true',
                   help='Run temporal queue evaluation after all LOSO folds complete')
    p.add_argument('--save-models', action='store_true', default=None,
                   help='Save per-fold model weights (auto-enabled with --queue-eval)')
    return p.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def import_class(import_str):
    try:
        from fusionlib.registry import MODEL_REGISTRY
        return MODEL_REGISTRY.get(import_str)
    except (ImportError, KeyError):
        pass
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    return getattr(sys.modules[mod_str], class_str)


def load_config(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_namespace(cfg, args):
    """Convert YAML config + CLI args into an argparse.Namespace for the Trainer."""
    ns = argparse.Namespace()
    ns.config = args.config
    ns.dataset = cfg.get('dataset', 'smartfallmm')
    ns.model = cfg['model']
    ns.model_args = cfg.get('model_args', {})
    ns.model_saved_name = 'model'
    ns.dataset_args = cfg.get('dataset_args', {})
    ns.subjects = cfg.get('subjects', [])
    ns.validation_subjects = cfg.get('validation_subjects', [48, 57])
    ns.train_only_subjects = cfg.get('train_only_subjects', [])
    ns.batch_size = cfg.get('batch_size', 64)
    ns.test_batch_size = cfg.get('test_batch_size', 64)
    ns.val_batch_size = cfg.get('val_batch_size', 64)
    ns.num_epoch = cfg.get('num_epoch', 80)
    ns.start_epoch = 0
    ns.optimizer = cfg.get('optimizer', 'adamw')
    ns.base_lr = cfg.get('base_lr', 0.001)
    ns.weight_decay = cfg.get('weight_decay', 0.001)
    ns.loss_type = cfg.get('dataset_args', {}).get('loss_type', 'focal')
    ns.loss_args = cfg.get('loss_args', {})
    ns.seed = args.seed
    ns.device = ['cpu']
    ns.phase = 'train'
    ns.num_worker = args.num_workers
    ns.include_val = True
    ns.print_log = True
    ns.feeder = cfg.get('feeder', 'Feeder.Make_Dataset.UTD_mm')
    ns.train_feeder_args = cfg.get('train_feeder_args', {'batch_size': 64})
    ns.val_feeder_args = cfg.get('val_feeder_args', {'batch_size': 64})
    ns.test_feeder_args = cfg.get('test_feeder_args', {'batch_size': 64})
    ns.weights = None
    ns.work_dir = args.work_dir
    ns.single_fold = None
    ns.enable_test_grouping = False
    ns.enable_kalman_preprocessing = False
    ns.kalman_args = {}
    return ns


def create_model(model_str, model_args):
    Model = import_class(model_str)
    return Model(**model_args).cpu()


def create_loss(loss_type, loss_args, pos_weight):
    if loss_type == 'focal':
        alpha = loss_args.get('alpha', 0.75) if isinstance(loss_args, dict) else 0.75
        gamma = loss_args.get('gamma', 2.0) if isinstance(loss_args, dict) else 2.0
        return BinaryFocalLoss(alpha=alpha, gamma=gamma)
    elif loss_type == 'cb_focal':
        beta = loss_args.get('beta', 0.9999) if isinstance(loss_args, dict) else 0.9999
        gamma = loss_args.get('gamma', 2.0) if isinstance(loss_args, dict) else 2.0
        return ClassBalancedFocalLoss(beta=beta, gamma=gamma)
    else:
        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def create_optimizer(model, opt_name, lr, weight_decay):
    if opt_name.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif opt_name.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=lr)
    raise ValueError(f'Unknown optimizer: {opt_name}')


def make_loader(data_dict, batch_size, shuffle, num_workers, feeder_str, dataset_args=None):
    Feeder = import_class(feeder_str)
    include_smv = True
    if dataset_args:
        include_smv = dataset_args.get('include_smv', not dataset_args.get('enable_kalman_fusion', False))
    ds = Feeder(data_dict, batch_size=batch_size, include_smv=include_smv)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, drop_last=False
    )


def compute_metrics(labels, preds, probs):
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.0
    macro = f1_score(labels, preds, average='macro', zero_division=0)
    return {
        'accuracy': float(acc), 'f1_score': float(f1),
        'macro_f1': float(macro), 'precision': float(prec),
        'recall': float(rec), 'auc': float(auc)
    }


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []
    n_batches = 0

    ctx = torch.no_grad() if not train else torch.enable_grad()
    with ctx:
        for batch in loader:
            inputs, targets, _ = batch
            # Handle dict inputs (multi-modality feeder)
            if isinstance(inputs, dict):
                for k in inputs:
                    if isinstance(inputs[k], torch.Tensor):
                        key = k
                        break
                x = inputs[key].float().to(device)
            else:
                x = inputs.float().to(device)

            targets = targets.float().to(device)
            logits, _ = model(x)
            logits = logits.squeeze(-1)

            if logits.dim() == 0:
                logits = logits.unsqueeze(0)

            loss = criterion(logits, targets)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            preds = (probs >= THRESHOLD).astype(int)
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(targets.cpu().numpy().astype(int).tolist())

    avg_loss = total_loss / max(n_batches, 1)
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics['loss'] = avg_loss
    return metrics, all_labels, all_probs


def get_test_candidates(cfg, ns):
    """Determine LOSO test subjects, respecting validation and train-only exclusions."""
    from utils.val_split_selector import (
        get_optimal_validation_subjects, get_train_only_subjects
    )
    val_subjects = get_optimal_validation_subjects(ns.dataset_args)
    train_only = get_train_only_subjects(ns.dataset_args)

    # Use config-specified if non-default
    if ns.validation_subjects != [38, 44]:
        val_subjects = ns.validation_subjects
    else:
        ns.validation_subjects = val_subjects

    if not ns.train_only_subjects and train_only:
        ns.train_only_subjects = train_only

    candidates = [s for s in ns.subjects
                  if s not in val_subjects and s not in ns.train_only_subjects]
    return candidates


def train_single_fold(ns, test_subject, fold_idx, total_folds, verbose=False):
    """Train one LOSO fold on CPU. Returns (result_dict, best_state_dict)."""
    fold_start = time.time()
    device = torch.device('cpu')

    train_only = ns.train_only_subjects or []
    candidates = [s for s in ns.subjects
                  if s not in ns.validation_subjects and s not in train_only]
    train_subjects = [s for s in candidates if s != test_subject] + train_only

    # Build dataset
    builder = prepare_smartfallmm(ns)
    all_subjects = list(set(train_subjects + ns.validation_subjects + [test_subject]))
    builder.make_dataset(all_subjects, fuse=len(ns.dataset_args.get('modalities', [])) > 1)

    norm_train = split_by_subjects(builder, train_subjects, fuse=True)
    norm_val = split_by_subjects(builder, ns.validation_subjects, fuse=True)
    norm_test = split_by_subjects(builder, [test_subject], fuse=True)

    # Validate data
    for name, split in [('train', norm_train), ('val', norm_val), ('test', norm_test)]:
        if not split or not any(isinstance(v, np.ndarray) and len(v) > 0
                                for v in split.values() if isinstance(v, np.ndarray)):
            print(f'  [Fold {fold_idx+1}] WARNING: {name} split empty, skipping')
            return None, None

    # Class distribution
    train_labels = norm_train.get('labels', [])
    label_counts = Counter(train_labels)
    n_adl, n_fall = label_counts.get(0, 1), label_counts.get(1, 1)
    pos_weight = torch.tensor([n_adl / max(n_fall, 1)])

    if verbose:
        test_labels = norm_test.get('labels', [])
        test_counts = Counter(test_labels)
        print(f'  Train: {len(train_labels)} windows (fall={n_fall}, adl={n_adl})')
        print(f'  Test:  {len(test_labels)} windows (fall={test_counts.get(1,0)}, adl={test_counts.get(0,0)})')

    # Data loaders
    feeder_str = ns.feeder
    train_loader = make_loader(norm_train, ns.batch_size, True, ns.num_worker, feeder_str, ns.dataset_args)
    val_loader = make_loader(norm_val, ns.val_batch_size, False, ns.num_worker, feeder_str, ns.dataset_args)
    test_loader = make_loader(norm_test, ns.test_batch_size, False, ns.num_worker, feeder_str, ns.dataset_args)

    # Model, loss, optimizer
    model = create_model(ns.model, ns.model_args)
    criterion = create_loss(ns.loss_type, ns.loss_args, pos_weight)
    optimizer = create_optimizer(model, ns.optimizer, ns.base_lr, ns.weight_decay)

    # Training loop with early stopping
    early_stop = EarlyStopping(patience=15, min_delta=0.0001)
    best_val_loss = float('inf')
    best_state = None
    best_epoch = 0
    best_val_metrics = None

    for epoch in range(ns.num_epoch):
        train_metrics, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_metrics, _, _ = run_epoch(model, val_loader, criterion, None, device, train=False)

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            best_val_metrics = deepcopy(val_metrics)

        early_stop(val_metrics['loss'])
        if early_stop.early_stop:
            break

    # Evaluate best model on test set
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    test_metrics, test_labels, test_probs = run_epoch(
        model, test_loader, criterion, None, device, train=False
    )

    elapsed = time.time() - fold_start

    # Build result dict (compatible with ray_distributed format)
    result = {
        'test_subject': str(test_subject),
        'fold_idx': fold_idx,
        'gpu_id': -1,
        'physical_gpu': 'cpu',
        'train': best_val_metrics or {},
        'val': best_val_metrics or {},
        'test': test_metrics,
        'best_epoch': best_epoch,
        'fall_windows': int(Counter(norm_train['labels']).get(1, 0)),
        'adl_windows': int(Counter(norm_train['labels']).get(0, 0)),
        'fall_trials': 0,
        'adl_trials': 0,
        'status': 'success',
        'elapsed_time': elapsed,
    }

    return result, best_state


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)

    # Build namespace
    ns = build_namespace(cfg, args)

    # Output directory
    if args.work_dir is None:
        config_stem = Path(args.config).stem
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        work_dir = f'exps/cpu_loso_{config_stem}_{timestamp}'
    else:
        work_dir = args.work_dir
    os.makedirs(work_dir, exist_ok=True)
    ns.work_dir = work_dir

    # Copy config
    import shutil
    shutil.copy2(args.config, work_dir)

    # Auto-enable model saving when queue eval is requested
    save_models = args.save_models if args.save_models is not None else args.queue_eval

    # Determine test candidates
    candidates = get_test_candidates(cfg, ns)
    if args.max_folds:
        candidates = candidates[:args.max_folds]

    total_folds = len(candidates)
    start_fold = args.resume_from

    print('=' * 70)
    print(f'CPU LOSO Training: {Path(args.config).name}')
    print(f'Model: {ns.model}')
    print(f'Folds: {total_folds} (starting from {start_fold})')
    print(f'Output: {work_dir}')
    print(f'Device: CPU')
    if save_models:
        print(f'Saving per-fold model weights: model_{{subject}}.pth')
    if args.queue_eval:
        print(f'Queue evaluation: enabled (runs after all folds)')
    print('=' * 70)

    # Load existing results if resuming
    pkl_path = os.path.join(work_dir, 'fold_results.pkl')
    if start_fold > 0 and os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            results = pickle.load(f)
        print(f'Resumed: loaded {len(results)} existing fold results')
    else:
        results = []

    completed_subjects = {r['test_subject'] for r in results}
    run_start = time.time()
    successful = 0
    failed = 0

    for fold_idx in range(start_fold, total_folds):
        subject = candidates[fold_idx]
        if str(subject) in completed_subjects:
            print(f'\n[Fold {fold_idx+1}/{total_folds}] Subject {subject} already done, skipping')
            continue

        elapsed_so_far = time.time() - run_start
        folds_done = fold_idx - start_fold
        if folds_done > 0:
            eta = (elapsed_so_far / folds_done) * (total_folds - fold_idx)
            eta_str = str(timedelta(seconds=int(eta)))
        else:
            eta_str = '...'

        print(f'\n[Fold {fold_idx+1}/{total_folds}] Subject {subject}  (ETA: {eta_str})')

        try:
            set_seed(args.seed)
            result, best_state = train_single_fold(ns, subject, fold_idx, total_folds, verbose=args.verbose)
            if result is not None:
                results.append(result)
                successful += 1
                test_f1 = result['test']['f1_score'] * 100
                test_acc = result['test']['accuracy'] * 100
                epoch = result['best_epoch']
                t = result['elapsed_time']
                print(f'  -> F1={test_f1:.1f}%  Acc={test_acc:.1f}%  epoch={epoch}  ({t:.0f}s)')

                # Save per-fold model weights for queue eval
                if save_models and best_state is not None:
                    model_path = os.path.join(work_dir, f'model_{subject}.pth')
                    torch.save(best_state, model_path)
            else:
                failed += 1
                print(f'  -> SKIPPED (empty data)')
        except Exception as e:
            failed += 1
            print(f'  -> FAILED: {e}')
            if args.verbose:
                traceback.print_exc()

        # Checkpoint after every fold (ensure directory exists)
        os.makedirs(work_dir, exist_ok=True)
        with open(pkl_path, 'wb') as f:
            pickle.dump(results, f)

    # Final summary
    total_time = time.time() - run_start
    print('\n' + '=' * 70)
    print('RESULTS')
    print('=' * 70)

    if results:
        f1s = [r['test']['f1_score'] * 100 for r in results]
        accs = [r['test']['accuracy'] * 100 for r in results]
        precs = [r['test']['precision'] * 100 for r in results]
        recs = [r['test']['recall'] * 100 for r in results]

        print(f'F1:        {np.mean(f1s):.2f} ± {np.std(f1s):.2f}')
        print(f'Accuracy:  {np.mean(accs):.2f} ± {np.std(accs):.2f}')
        print(f'Precision: {np.mean(precs):.2f} ± {np.std(precs):.2f}')
        print(f'Recall:    {np.mean(recs):.2f} ± {np.std(recs):.2f}')
        print(f'\nFolds: {successful} successful, {failed} failed')
        print(f'Time:  {timedelta(seconds=int(total_time))}')

        # Save summary JSON
        summary = {
            'config': args.config,
            'model': ns.model,
            'n_folds': len(results),
            'f1_mean': round(np.mean(f1s), 2),
            'f1_std': round(np.std(f1s), 2),
            'accuracy_mean': round(np.mean(accs), 2),
            'precision_mean': round(np.mean(precs), 2),
            'recall_mean': round(np.mean(recs), 2),
            'total_time_seconds': round(total_time, 1),
            'device': 'cpu',
            'timestamp': datetime.now().isoformat(),
        }
        with open(os.path.join(work_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        # Save per-fold CSV
        rows = []
        for r in results:
            row = {'test_subject': r['test_subject'], 'best_epoch': r['best_epoch']}
            for k, v in r['test'].items():
                row[f'test_{k}'] = round(v * 100 if k != 'loss' else v, 4)
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(work_dir, 'scores.csv'), index=False)

    print(f'\nOutput: {work_dir}')
    print(f'  fold_results.pkl  (for significance testing)')
    print(f'  summary.json')
    print(f'  scores.csv')
    if save_models:
        print(f'  model_{{subject}}.pth  (per-fold weights)')
    print('=' * 70)

    # Temporal queue evaluation
    if args.queue_eval and results:
        n_models = len([f for f in os.listdir(work_dir) if f.startswith('model_') and f.endswith('.pth')])
        if n_models == 0:
            print('\nWARNING: No model weights found, skipping queue evaluation')
        else:
            print(f'\nStarting temporal queue evaluation ({n_models} fold models)...')
            try:
                from utils.temporal_queue_eval import run_temporal_queue_evaluation
                run_temporal_queue_evaluation(
                    work_dir=work_dir,
                    config=cfg,
                    device='cpu',
                    print_results=True,
                )
            except Exception as e:
                print(f'Queue evaluation failed: {e}')
                if args.verbose:
                    traceback.print_exc()


if __name__ == '__main__':
    main()
