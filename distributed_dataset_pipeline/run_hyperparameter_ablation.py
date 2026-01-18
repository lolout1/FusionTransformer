#!/usr/bin/env python3
"""Hyperparameter ablation study orchestrator for UP-FALL and WEDA-FALL datasets."""

import argparse
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import yaml
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from distributed_dataset_pipeline.ablation_config_generator import (
    AblationConfigGenerator,
    AblationConfig,
    DATASET_CONFIG,
    STRIDE_CONFIGS,
    MODEL_VARIANTS,
    DEFAULT_MODEL_VARIANTS,
    EMBED_DIMS,
)


class AblationCheckpoint:
    """Manages checkpoint state for resume capability."""

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.completed: Dict[str, Dict] = {}  # exp_name -> result
        self.failed: Dict[str, str] = {}  # exp_name -> error
        self.in_progress: List[str] = []

        if checkpoint_path.exists():
            self._load()

    def _load(self):
        """Load checkpoint from file."""
        with open(self.checkpoint_path, 'r') as f:
            data = json.load(f)
            self.completed = data.get('completed', {})
            self.failed = data.get('failed', {})
            self.in_progress = data.get('in_progress', [])

    def save(self):
        """Save checkpoint to file."""
        data = {
            'completed': self.completed,
            'failed': self.failed,
            'in_progress': self.in_progress,
            'last_updated': datetime.now().isoformat(),
        }
        with open(self.checkpoint_path, 'w') as f:
            json.dump(data, f, indent=2)

    def is_completed(self, exp_name: str) -> bool:
        """Check if experiment is completed."""
        return exp_name in self.completed

    def mark_started(self, exp_name: str):
        """Mark experiment as started."""
        if exp_name not in self.in_progress:
            self.in_progress.append(exp_name)
        self.save()

    def mark_completed(self, exp_name: str, result: Dict):
        """Mark experiment as completed."""
        self.completed[exp_name] = result
        if exp_name in self.in_progress:
            self.in_progress.remove(exp_name)
        self.save()

    def mark_failed(self, exp_name: str, error: str):
        """Mark experiment as failed."""
        self.failed[exp_name] = error
        if exp_name in self.in_progress:
            self.in_progress.remove(exp_name)
        self.save()

    def get_summary(self) -> Dict:
        """Get checkpoint summary."""
        return {
            'completed': len(self.completed),
            'failed': len(self.failed),
            'in_progress': len(self.in_progress),
        }


def parse_summary_report(summary_path: Path) -> Dict:
    """Parse summary_report.txt for metrics."""
    results = {}
    if not summary_path.exists():
        return {'error': 'Summary file not found'}

    content = summary_path.read_text()

    # Extract F1
    match = re.search(r'Test F1:\s+([\d.]+)\s*[+-±]\s*([\d.]+)%', content)
    if match:
        results['test_f1'] = float(match.group(1))
        results['test_f1_std'] = float(match.group(2))

    # Extract Macro-F1
    match = re.search(r'Test Macro-F1:\s+([\d.]+)\s*[+-±]\s*([\d.]+)%', content)
    if match:
        results['test_macro_f1'] = float(match.group(1))
        results['test_macro_f1_std'] = float(match.group(2))

    # Extract accuracy
    match = re.search(r'Test Accuracy:\s+([\d.]+)\s*[+-±]\s*([\d.]+)%', content)
    if match:
        results['test_accuracy'] = float(match.group(1))
        results['test_accuracy_std'] = float(match.group(2))

    # Extract precision
    match = re.search(r'Test Precision:\s+([\d.]+)\s*[+-±]\s*([\d.]+)%', content)
    if match:
        results['test_precision'] = float(match.group(1))
        results['test_precision_std'] = float(match.group(2))

    # Extract recall
    match = re.search(r'Test Recall:\s+([\d.]+)\s*[+-±]\s*([\d.]+)%', content)
    if match:
        results['test_recall'] = float(match.group(1))
        results['test_recall_std'] = float(match.group(2))

    # Extract AUC
    match = re.search(r'Test AUC:\s+([\d.]+)\s*[+-±]\s*([\d.]+)%', content)
    if match:
        results['test_auc'] = float(match.group(1))
        results['test_auc_std'] = float(match.group(2))

    # Extract validation metrics
    match = re.search(r'Val F1:\s+([\d.]+)%', content)
    if match:
        results['val_f1'] = float(match.group(1))

    match = re.search(r'Val Macro-F1:\s+([\d.]+)%', content)
    if match:
        results['val_macro_f1'] = float(match.group(1))

    return results


def load_fold_results(fold_results_path: Path) -> Dict:
    """Load and summarize fold results from pickle file."""
    if not fold_results_path.exists():
        return {'error': 'Fold results not found'}

    try:
        with open(fold_results_path, 'rb') as f:
            results = pickle.load(f)

        successful = [r for r in results if r.get('status') != 'failed']

        if not successful:
            return {'error': 'No successful folds'}

        # Extract metrics
        test_f1s = [r['test']['f1_score'] for r in successful if r.get('test')]
        test_macro_f1s = [r['test'].get('macro_f1', r['test']['f1_score']) for r in successful if r.get('test')]
        test_accs = [r['test']['accuracy'] for r in successful if r.get('test')]
        test_precs = [r['test']['precision'] for r in successful if r.get('test')]
        test_recs = [r['test']['recall'] for r in successful if r.get('test')]
        test_aucs = [r['test'].get('auc', 0) for r in successful if r.get('test')]

        return {
            'n_folds': len(successful),
            'test_f1': float(np.mean(test_f1s)),
            'test_f1_std': float(np.std(test_f1s)),
            'test_macro_f1': float(np.mean(test_macro_f1s)),
            'test_macro_f1_std': float(np.std(test_macro_f1s)),
            'test_accuracy': float(np.mean(test_accs)),
            'test_accuracy_std': float(np.std(test_accs)),
            'test_precision': float(np.mean(test_precs)),
            'test_precision_std': float(np.std(test_precs)),
            'test_recall': float(np.mean(test_recs)),
            'test_recall_std': float(np.std(test_recs)),
            'test_auc': float(np.mean(test_aucs)),
            'test_auc_std': float(np.std(test_aucs)),
        }
    except Exception as e:
        return {'error': str(e)}


def run_single_experiment(
    config_path: str,
    work_dir: str,
    gpus: List[int],
    exp_name: str,
) -> Dict:
    """
    Run a single experiment using ray_train.py.

    Args:
        config_path: Path to YAML config
        work_dir: Output directory for this experiment
        gpus: List of GPU IDs to use
        exp_name: Experiment name for logging

    Returns:
        Dictionary with experiment results
    """
    # Set CUDA_VISIBLE_DEVICES for this experiment
    gpu_str = ','.join(str(g) for g in gpus)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpu_str

    # Build command
    cmd = [
        sys.executable, str(PROJECT_ROOT / 'ray_train.py'),
        '--config', config_path,
        '--num-gpus', str(len(gpus)),
        '--work-dir', work_dir,
    ]

    print(f"\n[{exp_name}] Starting on GPUs {gpus}")
    print(f"[{exp_name}] Command: {' '.join(cmd)}")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout per experiment
        )

        elapsed = time.time() - start_time

        if result.returncode != 0:
            print(f"[{exp_name}] FAILED after {elapsed/60:.1f}min")
            print(f"[{exp_name}] stderr: {result.stderr[:500]}")
            return {
                'status': 'failed',
                'error': result.stderr[:1000],
                'elapsed_minutes': elapsed / 60,
            }

        # Parse results
        work_path = Path(work_dir)
        summary_path = work_path / 'summary_report.txt'
        fold_results_path = work_path / 'fold_results.pkl'

        metrics = parse_summary_report(summary_path)
        if 'error' in metrics:
            # Try fold_results.pkl as backup
            metrics = load_fold_results(fold_results_path)

        print(f"[{exp_name}] COMPLETED in {elapsed/60:.1f}min - F1: {metrics.get('test_f1', 0):.2f}%")

        return {
            'status': 'success',
            'elapsed_minutes': elapsed / 60,
            **metrics,
        }

    except subprocess.TimeoutExpired:
        print(f"[{exp_name}] TIMEOUT after 2 hours")
        return {
            'status': 'failed',
            'error': 'Timeout after 2 hours',
            'elapsed_minutes': 120,
        }
    except Exception as e:
        print(f"[{exp_name}] ERROR: {e}")
        return {
            'status': 'failed',
            'error': str(e),
            'elapsed_minutes': (time.time() - start_time) / 60,
        }


class HyperparameterAblationRunner:
    """Orchestrates the full hyperparameter ablation study."""

    def __init__(
        self,
        output_dir: Path,
        num_gpus: int = 6,
        gpus_per_exp: int = 3,
        datasets: Optional[List[str]] = None,
        window_sizes: Optional[Dict[str, List[int]]] = None,
        stride_names: Optional[List[str]] = None,
        embed_dims: Optional[List[int]] = None,
        model_names: Optional[List[str]] = None,
        resume: bool = False,
    ):
        """
        Initialize the ablation runner.

        Args:
            output_dir: Base output directory
            num_gpus: Total GPUs available
            gpus_per_exp: GPUs per experiment
            datasets: Datasets to run ('upfall', 'wedafall')
            window_sizes: Window sizes per dataset
            stride_names: Stride configurations to test
            embed_dims: Embedding dimensions to test
            model_names: Model variants to test
            resume: Whether to resume from checkpoint
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.num_gpus = num_gpus
        self.gpus_per_exp = gpus_per_exp
        self.max_parallel = num_gpus // gpus_per_exp

        self.datasets = datasets or ['upfall', 'wedafall']
        self.window_sizes = window_sizes or {}
        self.stride_names = stride_names or list(STRIDE_CONFIGS.keys())
        self.embed_dims = embed_dims or EMBED_DIMS
        self.model_names = model_names or DEFAULT_MODEL_VARIANTS

        self.generator = AblationConfigGenerator(PROJECT_ROOT)

        # Setup checkpoint
        self.checkpoint = AblationCheckpoint(self.output_dir / 'checkpoint.json')

        if resume:
            summary = self.checkpoint.get_summary()
            print(f"Resuming from checkpoint: {summary['completed']} completed, "
                  f"{summary['failed']} failed, {summary['in_progress']} in progress")

        # Generate all configs
        self.configs: List[Tuple[AblationConfig, Dict, Path]] = []
        self._generate_all_configs()

    def _generate_all_configs(self):
        """Generate all experiment configurations."""
        temp_config_dir = self.output_dir / 'configs'
        temp_config_dir.mkdir(exist_ok=True)

        for dataset in self.datasets:
            ws = self.window_sizes.get(dataset, DATASET_CONFIG[dataset]['window_sizes'])

            configs = self.generator.generate_all_configs(
                dataset=dataset,
                window_sizes=ws,
                stride_names=self.stride_names,
                embed_dims=self.embed_dims,
                model_names=self.model_names,
            )

            # Save configs and track them
            for ablation_cfg, config in configs:
                config_path = temp_config_dir / dataset / f"{ablation_cfg.name}.yaml"
                config_path.parent.mkdir(parents=True, exist_ok=True)

                # Add header comment
                header = f"""# Ablation: {ablation_cfg.name}
# Dataset: {dataset}, Window: {ablation_cfg.window_size}, Stride: {ablation_cfg.stride_name}
# Embed: {ablation_cfg.embed_dim}, Model: {ablation_cfg.model_name}

"""
                with open(config_path, 'w') as f:
                    f.write(header)
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

                self.configs.append((ablation_cfg, config, config_path))

        print(f"Generated {len(self.configs)} configurations")

    def _get_pending_experiments(self) -> List[Tuple[AblationConfig, Dict, Path]]:
        """Get experiments that haven't been completed yet."""
        pending = []
        for ablation_cfg, config, config_path in self.configs:
            exp_name = f"{ablation_cfg.dataset}/{ablation_cfg.name}"
            if not self.checkpoint.is_completed(exp_name):
                pending.append((ablation_cfg, config, config_path))
        return pending

    def run(self) -> Dict:
        """
        Run the full ablation study.

        Returns:
            Dictionary with comprehensive results
        """
        print("=" * 70)
        print("HYPERPARAMETER ABLATION STUDY")
        print("=" * 70)
        print(f"Output: {self.output_dir}")
        print(f"GPUs: {self.num_gpus} total, {self.gpus_per_exp} per experiment")
        print(f"Max parallel: {self.max_parallel} experiments")
        print(f"Datasets: {self.datasets}")
        print(f"Total configs: {len(self.configs)}")
        print("=" * 70)

        pending = self._get_pending_experiments()
        print(f"\nPending experiments: {len(pending)}")

        if not pending:
            print("All experiments completed!")
            return self._collect_results()

        start_time = time.time()

        # Partition GPUs for parallel experiments
        gpu_partitions = []
        for i in range(self.max_parallel):
            start_gpu = i * self.gpus_per_exp
            end_gpu = start_gpu + self.gpus_per_exp
            gpu_partitions.append(list(range(start_gpu, end_gpu)))

        print(f"GPU partitions: {gpu_partitions}")

        # Run experiments with ThreadPoolExecutor
        completed = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            futures = {}

            # Submit initial batch
            for i, (ablation_cfg, config, config_path) in enumerate(pending[:self.max_parallel]):
                gpus = gpu_partitions[i % len(gpu_partitions)]
                exp_name = f"{ablation_cfg.dataset}/{ablation_cfg.name}"
                work_dir = str(self.output_dir / ablation_cfg.dataset / ablation_cfg.name)

                self.checkpoint.mark_started(exp_name)

                future = executor.submit(
                    run_single_experiment,
                    str(config_path),
                    work_dir,
                    gpus,
                    exp_name,
                )
                futures[future] = (ablation_cfg, exp_name, i)

            # Process as they complete and submit new ones
            remaining = pending[self.max_parallel:]

            while futures:
                for future in as_completed(list(futures.keys())):
                    ablation_cfg, exp_name, partition_idx = futures.pop(future)

                    try:
                        result = future.result()
                    except Exception as e:
                        result = {'status': 'failed', 'error': str(e)}

                    # Record result
                    result['config'] = ablation_cfg.to_dict()

                    if result.get('status') == 'success':
                        self.checkpoint.mark_completed(exp_name, result)
                        completed += 1
                    else:
                        self.checkpoint.mark_failed(exp_name, result.get('error', 'Unknown'))
                        failed += 1

                    # Progress update
                    elapsed = time.time() - start_time
                    total = len(pending)
                    done = completed + failed
                    remaining_count = total - done

                    if done > 0:
                        avg_time = elapsed / done
                        eta = avg_time * remaining_count / self.max_parallel
                    else:
                        eta = 0

                    print(f"\n[Progress] {done}/{total} complete ({completed} success, {failed} failed) | "
                          f"Elapsed: {elapsed/60:.1f}min | ETA: {eta/60:.1f}min")

                    # Submit next experiment if available
                    if remaining:
                        ablation_cfg, config, config_path = remaining.pop(0)
                        gpus = gpu_partitions[partition_idx]
                        exp_name = f"{ablation_cfg.dataset}/{ablation_cfg.name}"
                        work_dir = str(self.output_dir / ablation_cfg.dataset / ablation_cfg.name)

                        self.checkpoint.mark_started(exp_name)

                        new_future = executor.submit(
                            run_single_experiment,
                            str(config_path),
                            work_dir,
                            gpus,
                            exp_name,
                        )
                        futures[new_future] = (ablation_cfg, exp_name, partition_idx)

                    break  # Process one at a time to maintain submission queue

        # Final summary
        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("ABLATION STUDY COMPLETE")
        print("=" * 70)
        print(f"Total experiments: {len(pending)}")
        print(f"Successful: {completed}")
        print(f"Failed: {failed}")
        print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
        print("=" * 70)

        return self._collect_results()

    def _collect_results(self) -> Dict:
        """Collect all results into comprehensive format."""
        results = {
            'meta': {
                'timestamp': datetime.now().isoformat(),
                'datasets': self.datasets,
                'stride_configs': self.stride_names,
                'embed_dims': self.embed_dims,
                'model_variants': self.model_names,
                'total_experiments': len(self.configs),
            },
            'experiments': {},
            'summary': {},
        }

        # Collect from checkpoint
        for exp_name, result in self.checkpoint.completed.items():
            results['experiments'][exp_name] = result

        # Generate CSV
        self._save_comprehensive_csv(results)

        # Save JSON
        json_path = self.output_dir / 'comprehensive_results.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {self.output_dir}")

        return results

    def _save_comprehensive_csv(self, results: Dict):
        """Save results as comprehensive CSV."""
        import csv

        csv_path = self.output_dir / 'comprehensive_results.csv'

        fieldnames = [
            'dataset', 'window_size', 'stride_name', 'fall_stride', 'adl_stride',
            'embed_dim', 'model', 'status',
            'test_f1', 'test_f1_std', 'test_macro_f1', 'test_macro_f1_std',
            'test_accuracy', 'test_accuracy_std',
            'test_precision', 'test_precision_std',
            'test_recall', 'test_recall_std',
            'test_auc', 'test_auc_std',
            'n_folds', 'elapsed_minutes',
        ]

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for exp_name, result in results['experiments'].items():
                config = result.get('config', {})
                row = {
                    'dataset': config.get('dataset', ''),
                    'window_size': config.get('window_size', ''),
                    'stride_name': config.get('stride_name', ''),
                    'fall_stride': config.get('fall_stride', ''),
                    'adl_stride': config.get('adl_stride', ''),
                    'embed_dim': config.get('embed_dim', ''),
                    'model': config.get('model_name', ''),
                    'status': result.get('status', 'unknown'),
                    'test_f1': result.get('test_f1', ''),
                    'test_f1_std': result.get('test_f1_std', ''),
                    'test_macro_f1': result.get('test_macro_f1', ''),
                    'test_macro_f1_std': result.get('test_macro_f1_std', ''),
                    'test_accuracy': result.get('test_accuracy', ''),
                    'test_accuracy_std': result.get('test_accuracy_std', ''),
                    'test_precision': result.get('test_precision', ''),
                    'test_precision_std': result.get('test_precision_std', ''),
                    'test_recall': result.get('test_recall', ''),
                    'test_recall_std': result.get('test_recall_std', ''),
                    'test_auc': result.get('test_auc', ''),
                    'test_auc_std': result.get('test_auc_std', ''),
                    'n_folds': result.get('n_folds', ''),
                    'elapsed_minutes': result.get('elapsed_minutes', ''),
                }
                writer.writerow(row)

        print(f"CSV saved: {csv_path}")


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Hyperparameter Ablation Study for Fall Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (generate configs, show summary)
  python run_hyperparameter_ablation.py --dry-run

  # Full run with 6 GPUs
  python run_hyperparameter_ablation.py --num-gpus 6

  # Resume interrupted run
  python run_hyperparameter_ablation.py --resume --output-dir results/hyperparameter_ablation_20260117

  # Subset of experiments
  python run_hyperparameter_ablation.py --datasets upfall --window-sizes 128 --models kalman_conv1d_linear

  # Analysis only
  python run_hyperparameter_ablation.py --analyze-only --results-dir results/hyperparameter_ablation_20260117
        """
    )

    # Output configuration
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Output directory (default: results/hyperparameter_ablation_<timestamp>)'
    )

    # GPU configuration
    parser.add_argument(
        '--num-gpus', '-g',
        type=int,
        default=6,
        help='Total number of GPUs available (default: 6)'
    )
    parser.add_argument(
        '--gpus-per-exp',
        type=int,
        default=3,
        help='GPUs per experiment (default: 3)'
    )

    # Experiment selection
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['upfall', 'wedafall'],
        choices=['upfall', 'wedafall'],
        help='Datasets to run (default: both)'
    )
    parser.add_argument(
        '--window-sizes',
        nargs='+',
        type=int,
        default=None,
        help='Window sizes to test (default: dataset-specific)'
    )
    parser.add_argument(
        '--stride-configs',
        nargs='+',
        default=None,
        choices=['aggressive', 'standard', 'moderate', 'equal'],
        help='Stride configurations (default: all)'
    )
    parser.add_argument(
        '--embed-dims',
        nargs='+',
        type=int,
        default=None,
        help='Embedding dimensions (default: 48, 64)'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=None,
        choices=['kalman_conv1d_linear', 'kalman_conv1d_conv1d'],
        help='Model variants (default: kalman_conv1d_linear, kalman_conv1d_conv1d)'
    )

    # Execution modes
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show experiment matrix without running'
    )
    parser.add_argument(
        '--generate-configs',
        action='store_true',
        help='Only generate config files'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint'
    )
    parser.add_argument(
        '--analyze-only',
        action='store_true',
        help='Only run analysis on existing results'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default=None,
        help='Results directory for analysis-only mode'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = get_args()

    # Handle window sizes per dataset
    window_sizes = {}
    if args.window_sizes:
        for dataset in args.datasets:
            window_sizes[dataset] = args.window_sizes

    # Generate output directory if not specified
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.results_dir:
        output_dir = Path(args.results_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = PROJECT_ROOT / f'results/hyperparameter_ablation_{timestamp}'

    # Dry run mode
    if args.dry_run:
        generator = AblationConfigGenerator(PROJECT_ROOT)
        generator.print_experiment_summary(
            datasets=args.datasets,
            window_sizes=window_sizes if window_sizes else None,
            stride_names=args.stride_configs,
            embed_dims=args.embed_dims,
            model_names=args.models,
        )
        return

    # Generate configs only
    if args.generate_configs:
        generator = AblationConfigGenerator(PROJECT_ROOT)
        output_dir.mkdir(parents=True, exist_ok=True)

        for dataset in args.datasets:
            ws = window_sizes.get(dataset, DATASET_CONFIG[dataset]['window_sizes'])
            configs = generator.generate_all_configs(
                dataset=dataset,
                window_sizes=ws,
                stride_names=args.stride_configs,
                embed_dims=args.embed_dims,
                model_names=args.models,
            )
            paths = generator.save_configs(configs, output_dir / 'configs')
            print(f"Generated {len(paths)} configs for {dataset}")

        print(f"\nConfigs saved to: {output_dir / 'configs'}")
        return

    # Analysis only mode
    if args.analyze_only:
        if not output_dir.exists():
            print(f"ERROR: Results directory not found: {output_dir}")
            sys.exit(1)

        from distributed_dataset_pipeline.ablation_analysis import AblationAnalyzer
        analyzer = AblationAnalyzer(output_dir)
        analyzer.run_full_analysis()
        return

    # Full run
    runner = HyperparameterAblationRunner(
        output_dir=output_dir,
        num_gpus=args.num_gpus,
        gpus_per_exp=args.gpus_per_exp,
        datasets=args.datasets,
        window_sizes=window_sizes if window_sizes else None,
        stride_names=args.stride_configs,
        embed_dims=args.embed_dims,
        model_names=args.models,
        resume=args.resume,
    )

    results = runner.run()

    # Run analysis
    try:
        from distributed_dataset_pipeline.ablation_analysis import AblationAnalyzer
        analyzer = AblationAnalyzer(output_dir)
        analyzer.run_full_analysis()
    except Exception as e:
        print(f"Warning: Analysis failed: {e}")

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()
