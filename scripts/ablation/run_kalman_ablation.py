#!/usr/bin/env python3
"""
Kalman Filter Ablation Study Runner

Runs systematic comparison of:
- Models: KalmanBalancedFlexible, KalmanGatedHierarchical
- Filters: Linear (Euler), EKF (Euler), UKF (Euler), Linear (Gravity)

Uses Ray Tune for parallel execution and W&B for tracking.

Usage:
    python scripts/ablation/run_kalman_ablation.py --mode ray
    python scripts/ablation/run_kalman_ablation.py --mode sequential
    python scripts/ablation/run_kalman_ablation.py --mode single --config balanced_lkf_euler
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Ablation grid
ABLATION_GRID = {
    'model': [
        ('balanced', 'Models.kalman_transformer_variants.KalmanBalancedFlexible'),
        ('kghf', 'Models.kalman_gated_hierarchical.KalmanGatedHierarchical'),
    ],
    'filter': [
        ('lkf_euler', {'kalman_filter_type': 'linear', 'kalman_output_format': 'euler'}),
        ('ekf_euler', {'kalman_filter_type': 'ekf', 'kalman_output_format': 'euler'}),
        ('ukf_euler', {'kalman_filter_type': 'ukf', 'kalman_output_format': 'euler'}),
        ('lkf_gravity', {'kalman_filter_type': 'linear', 'kalman_output_format': 'gravity_vector'}),
        ('ekf_quat', {'kalman_filter_type': 'ekf', 'kalman_output_format': 'quaternion'}),
        ('ukf_quat', {'kalman_filter_type': 'ukf', 'kalman_output_format': 'quaternion'}),
    ],
}

CONFIG_DIR = Path(__file__).parent.parent.parent / 'config/smartfallmm/kalman_ablation'


def load_merged_config(variant_name: str) -> dict:
    """Load base config and merge with variant."""
    base_path = CONFIG_DIR / '_base.yaml'
    variant_path = CONFIG_DIR / f'{variant_name}.yaml'

    with open(base_path) as f:
        config = yaml.safe_load(f)

    if variant_path.exists():
        with open(variant_path) as f:
            variant = yaml.safe_load(f)

        # Deep merge
        for key, value in variant.items():
            if key == 'dataset_args' and 'dataset_args' in config:
                config['dataset_args'].update(value)
            elif key == 'model_args' and 'model_args' in config:
                config['model_args'].update(value)
            else:
                config[key] = value

    return config


def config_to_args(config: dict):
    """Convert config dict to argparse.Namespace."""
    from argparse import Namespace

    args = Namespace()
    for key, value in config.items():
        setattr(args, key.replace('-', '_'), value)

    # Required defaults
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


def run_single_experiment(variant_name: str, enable_wandb: bool = False, work_dir: str = None):
    """Run a single ablation experiment."""
    import numpy as np
    import torch

    config = load_merged_config(variant_name)

    # Set work dir
    if work_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        work_dir = f'results/kalman_ablation/{variant_name}_{timestamp}'

    config['work_dir'] = work_dir
    os.makedirs(work_dir, exist_ok=True)

    # Save config
    with open(f'{work_dir}/config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    args = config_to_args(config)
    args.enable_wandb = enable_wandb

    # Seed
    seed = config.get('seed', 2)
    torch.manual_seed(seed)
    np.random.seed(seed)

    from main import Trainer
    trainer = Trainer(args)
    trainer.start()

    return {
        'variant': variant_name,
        'test_f1': trainer.test_f1 if hasattr(trainer, 'test_f1') else None,
        'val_f1': trainer.best_val_f1 if hasattr(trainer, 'best_val_f1') else None,
    }


def run_ray_ablation(num_workers: int = 8, enable_wandb: bool = True):
    """Run ablation study with Ray Tune (1 GPU per trial)."""
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler

    try:
        from ray.tune.integration.wandb import WandbLoggerCallback
        WANDB_AVAILABLE = True
    except ImportError:
        WANDB_AVAILABLE = False

    ray.init(num_gpus=num_workers)  # Tell Ray about available GPUs

    # Generate all variants
    variants = []
    for model_name, _ in ABLATION_GRID['model']:
        for filter_name, _ in ABLATION_GRID['filter']:
            variants.append(f'{model_name}_{filter_name}')

    def train_variant(config):
        """Ray Tune trainable (runs on assigned GPU)."""
        import numpy as np
        import torch

        variant_name = config['variant']
        full_config = load_merged_config(variant_name)

        seed = full_config.get('seed', 2)
        torch.manual_seed(seed)
        np.random.seed(seed)

        args = config_to_args(full_config)
        args.work_dir = f'ray_results/kalman_ablation/{variant_name}'
        args.device = [0]  # Ray sets CUDA_VISIBLE_DEVICES to assigned GPU
        args.enable_wandb = False  # W&B handled by Ray callback

        from main import Trainer
        trainer = Trainer(args)

        # Run subset of folds for faster iteration
        max_folds = config.get('max_folds', 5)
        test_candidates = [s for s in args.subjects
                          if s not in args.validation_subjects
                          and s not in args.train_only_subjects][:max_folds]

        fold_f1s = []
        for fold_idx, test_subject in enumerate(test_candidates):
            trainer.train_subjects = [s for s in test_candidates if s != test_subject]
            trainer.test_subject = [test_subject]
            trainer.val_subject = args.validation_subjects

            trainer._init_fold_tracking(fold_idx, test_subject)
            trainer.model = trainer.load_model(args.model, args.model_args)

            if not trainer.load_data():
                continue

            trainer.load_optimizer(trainer.model.parameters())
            trainer.load_scheduler(args.num_epoch)
            trainer.load_loss()

            for epoch in range(args.num_epoch):
                trainer.train(epoch)
                val_f1 = trainer.current_fold_metrics.get('val', {}).get('f1_score', 0)

                tune.report(
                    val_f1=val_f1,
                    fold=fold_idx,
                    epoch=epoch,
                    variant=variant_name,
                )

                if trainer.early_stop.early_stop:
                    break

            trainer.load_weights()
            trainer.eval(epoch=0, loader_name='test')
            fold_f1s.append(trainer.test_f1)

        mean_f1 = np.mean(fold_f1s) if fold_f1s else 0
        tune.report(mean_test_f1=mean_f1, variant=variant_name)

    # Search space
    search_space = {
        'variant': tune.grid_search(variants),
        'max_folds': 5,
    }

    # Callbacks
    callbacks = []
    if enable_wandb and WANDB_AVAILABLE:
        callbacks.append(WandbLoggerCallback(
            project='smartfall-mm',
            entity='abheek-texas-state-university',
            group='kalman-filter-ablation',
            log_config=True,
        ))

    # Run (1 GPU + 4 CPUs per trial)
    analysis = tune.run(
        train_variant,
        config=search_space,
        num_samples=1,
        resources_per_trial={'cpu': 4, 'gpu': 1},
        storage_path='ray_results/',
        callbacks=callbacks,
        max_concurrent_trials=num_workers,
        verbose=1,
    )

    # Results summary
    print('\n' + '='*60)
    print('KALMAN FILTER ABLATION RESULTS')
    print('='*60)

    results = []
    for trial in analysis.trials:
        result = trial.last_result
        results.append({
            'variant': result.get('variant', 'unknown'),
            'mean_test_f1': result.get('mean_test_f1', 0),
            'val_f1': result.get('val_f1', 0),
        })

    results.sort(key=lambda x: x['mean_test_f1'], reverse=True)

    for r in results:
        print(f"{r['variant']:30s}  Test F1: {r['mean_test_f1']*100:.2f}%  Val F1: {r['val_f1']*100:.2f}%")

    return analysis


def run_parallel_ablation(num_gpus: int = 8, enable_wandb: bool = True):
    """Run ablation with simple multiprocessing (1 GPU per variant)."""
    import subprocess
    import concurrent.futures
    from datetime import datetime

    # Generate all variants
    variants = []
    for model_name, _ in ABLATION_GRID['model']:
        for filter_name, _ in ABLATION_GRID['filter']:
            variants.append(f'{model_name}_{filter_name}')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    def run_on_gpu(gpu_id: int, variant: str):
        """Run single variant on specific GPU."""
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        work_dir = f'results/kalman_ablation/{variant}_{timestamp}'
        cmd = [
            sys.executable, 'scripts/ablation/run_kalman_ablation.py',
            '--mode', 'single',
            '--config', variant,
            '--work-dir', work_dir,
        ]
        if enable_wandb:
            cmd.append('--wandb')

        print(f'[GPU {gpu_id}] Starting {variant}')
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)

        if result.returncode != 0:
            print(f'[GPU {gpu_id}] {variant} FAILED:\n{result.stderr[-500:]}')
            return {'variant': variant, 'gpu': gpu_id, 'success': False}

        print(f'[GPU {gpu_id}] {variant} DONE')
        return {'variant': variant, 'gpu': gpu_id, 'success': True}

    # Run all variants in parallel (up to num_gpus at a time)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = {}
        for i, variant in enumerate(variants):
            gpu_id = i % num_gpus
            futures[executor.submit(run_on_gpu, gpu_id, variant)] = variant

        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    # Summary
    print('\n' + '='*60)
    print('PARALLEL ABLATION COMPLETE')
    print('='*60)
    success = sum(1 for r in results if r['success'])
    print(f'{success}/{len(results)} variants completed successfully')

    return results


def run_sequential_ablation(enable_wandb: bool = True):
    """Run ablation study sequentially (no Ray)."""
    results = []

    for model_name, _ in ABLATION_GRID['model']:
        for filter_name, _ in ABLATION_GRID['filter']:
            variant = f'{model_name}_{filter_name}'
            print(f'\n{"="*60}')
            print(f'Running: {variant}')
            print('='*60)

            try:
                result = run_single_experiment(variant, enable_wandb=enable_wandb)
                results.append(result)
                print(f'Result: Test F1 = {result["test_f1"]:.4f}')
            except Exception as e:
                print(f'Error: {e}')
                results.append({'variant': variant, 'test_f1': None, 'error': str(e)})

    # Summary
    print('\n' + '='*60)
    print('ABLATION STUDY RESULTS')
    print('='*60)

    valid_results = [r for r in results if r.get('test_f1') is not None]
    valid_results.sort(key=lambda x: x['test_f1'], reverse=True)

    for r in valid_results:
        print(f"{r['variant']:30s}  Test F1: {r['test_f1']*100:.2f}%")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = f'results/kalman_ablation/summary_{timestamp}.yaml'
    os.makedirs('results/kalman_ablation', exist_ok=True)

    with open(results_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)

    print(f'\nResults saved to: {results_path}')
    return results


def main():
    parser = argparse.ArgumentParser(description='Kalman Filter Ablation Study')
    parser.add_argument('--mode', choices=['ray', 'parallel', 'sequential', 'single'], default='parallel',
                        help='Execution mode: ray, parallel (multi-GPU), sequential, or single')
    parser.add_argument('--config', type=str, default=None,
                        help='Config name for single mode (e.g., balanced_lkf_euler)')
    parser.add_argument('--gpus', type=int, default=8,
                        help='Number of GPUs for parallel mode')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable W&B logging')
    parser.add_argument('--work-dir', type=str, default=None,
                        help='Work directory for single mode')
    args = parser.parse_args()

    if args.mode == 'ray':
        run_ray_ablation(num_workers=args.gpus, enable_wandb=args.wandb)
    elif args.mode == 'parallel':
        run_parallel_ablation(num_gpus=args.gpus, enable_wandb=args.wandb)
    elif args.mode == 'sequential':
        run_sequential_ablation(enable_wandb=args.wandb)
    elif args.mode == 'single':
        if args.config is None:
            print('Error: --config required for single mode')
            print('Available configs:')
            for f in sorted(CONFIG_DIR.glob('*.yaml')):
                if not f.name.startswith('_'):
                    print(f'  {f.stem}')
            sys.exit(1)
        result = run_single_experiment(args.config, enable_wandb=args.wandb, work_dir=args.work_dir)
        print(f'\nResult: {result}')


if __name__ == '__main__':
    main()
