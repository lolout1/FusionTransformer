#!/usr/bin/env python3
"""Baseline ablation: No-Kalman vs Kalman, Single vs Dual stream."""

import os
import sys
import argparse
import yaml
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

CONFIGS = [
    'acc_only_single',      # 4ch single stream (no gyro)
    'acc_gyro_single',      # 7ch single stream (raw gyro)
    'acc_gyro_dual',        # 7ch dual stream (raw gyro)
    'kalman_single',        # 7ch single stream (kalman euler)
]

CONFIG_DIR = Path(__file__).parent.parent.parent / 'config/smartfallmm/baseline_ablation'


def load_config(name: str) -> dict:
    base = yaml.safe_load(open(CONFIG_DIR / '_base.yaml'))
    variant = yaml.safe_load(open(CONFIG_DIR / f'{name}.yaml'))
    for k, v in variant.items():
        if k in ['dataset_args', 'model_args'] and k in base:
            base[k].update(v)
        else:
            base[k] = v
    return base


def config_to_args(config: dict):
    from argparse import Namespace
    args = Namespace()
    for k, v in config.items():
        setattr(args, k.replace('-', '_'), v)
    args.phase = 'train'
    args.print_log = True
    args.num_worker = 0
    args.device = [0]
    args.single_fold = None
    args.enable_kalman_preprocessing = False
    args.kalman_args = {}
    args.include_val = True
    args.enable_test_grouping = False
    args.model_saved_name = 'model'
    args.weights = None
    args.result_file = None
    args.start_epoch = 0
    args.lr_scheduler = 'none'
    args.warmup_epochs = 10
    args.min_lr = 1e-6
    args.config = None
    args.enable_wandb = False
    args.test_batch_size = config.get('batch_size', 32)
    args.val_batch_size = config.get('batch_size', 32)
    return args


def run_single(name: str, wandb: bool = False, work_dir: str = None):
    import numpy as np
    import torch

    config = load_config(name)
    if work_dir is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        work_dir = f'results/baseline_ablation/{name}_{ts}'
    config['work_dir'] = work_dir
    os.makedirs(work_dir, exist_ok=True)

    with open(f'{work_dir}/config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    args = config_to_args(config)
    args.enable_wandb = wandb

    seed = config.get('seed', 2)
    torch.manual_seed(seed)
    np.random.seed(seed)

    from main import Trainer
    trainer = Trainer(args)
    trainer.start()

    return {
        'name': name,
        'test_f1': getattr(trainer, 'test_f1', None),
        'val_f1': getattr(trainer, 'best_val_f1', None),
    }


def run_sequential(wandb: bool = True):
    results = []
    for name in CONFIGS:
        print(f'\n{"="*60}\nRunning: {name}\n{"="*60}')
        try:
            r = run_single(name, wandb=wandb)
            results.append(r)
            print(f'Result: Test F1 = {r["test_f1"]:.4f}')
        except Exception as e:
            print(f'Error: {e}')
            results.append({'name': name, 'test_f1': None, 'error': str(e)})

    print(f'\n{"="*60}\nBASELINE ABLATION RESULTS\n{"="*60}')
    for r in sorted([x for x in results if x.get('test_f1')], key=lambda x: x['test_f1'], reverse=True):
        print(f"{r['name']:25s}  Test F1: {r['test_f1']*100:.2f}%")

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('results/baseline_ablation', exist_ok=True)
    with open(f'results/baseline_ablation/summary_{ts}.yaml', 'w') as f:
        yaml.dump(results, f)
    return results


def main():
    parser = argparse.ArgumentParser(description='Baseline Ablation Study')
    parser.add_argument('--mode', choices=['sequential', 'single'], default='sequential')
    parser.add_argument('--config', type=str, help='Config name for single mode')
    parser.add_argument('--wandb', action='store_true', help='Enable W&B')
    parser.add_argument('--work-dir', type=str, help='Work dir for single mode')
    args = parser.parse_args()

    if args.mode == 'single':
        if not args.config:
            print('Available configs:', CONFIGS)
            return
        run_single(args.config, wandb=args.wandb, work_dir=args.work_dir)
    else:
        run_sequential(wandb=args.wandb)


if __name__ == '__main__':
    main()
