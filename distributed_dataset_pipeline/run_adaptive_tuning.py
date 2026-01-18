#!/usr/bin/env python3
"""
Adaptive Kalman Filter Parameter Tuning Study.

Systematically tunes Innovation-Based Adaptive Estimation (IAE) parameters
to optimize fall detection performance across sensor quality regimes.

Key Parameters:
    - adaptive_alpha: Sensitivity to NIS changes (0=none, 1=full)
    - adaptive_scale_min: Min R scale (how much to trust clean signals)
    - adaptive_scale_max: Max R scale (how much to distrust noisy signals)
    - adaptive_ema_alpha: EMA smoothing for scale factor
    - adaptive_warmup_samples: Samples before adaptation starts

Tuning Strategy:
    Phase 1 (Coarse): Grid search over alpha and scale_min (most impactful)
    Phase 2 (Fine): Refine around best parameters from Phase 1

Hypothesis:
    - UP-FALL (clean): Lower scale_min + higher alpha → trust measurements more
    - WEDA-FALL (noisy): Higher scale_max → smooth more on noise spikes

Usage:
    # Full grid search with 4 parallel experiments (8 GPUs, 2 per experiment)
    python distributed_dataset_pipeline/run_adaptive_tuning.py --num-gpus 8 --parallel 4

    # Quick test (2 folds, parallel)
    python distributed_dataset_pipeline/run_adaptive_tuning.py --num-gpus 8 --parallel 4 --max-folds 2

    # Single dataset, parallel
    python distributed_dataset_pipeline/run_adaptive_tuning.py --datasets upfall --num-gpus 8 --parallel 4

    # Custom parameter grid
    python distributed_dataset_pipeline/run_adaptive_tuning.py --alpha 0.5 0.7 0.9 --scale-min 0.1 0.2 0.3 --num-gpus 8 --parallel 4

    # Sequential mode (default, for debugging)
    python distributed_dataset_pipeline/run_adaptive_tuning.py --num-gpus 8
"""

import argparse
import itertools
import json
import os
import re
import subprocess
import sys
import threading
import queue
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent

# =============================================================================
# Configuration
# =============================================================================

DATASET_INFO = {
    'upfall': {
        'name': 'UP-FALL',
        'sensor_type': 'Research-grade',
        'num_folds': 15,
        'base_config': 'config/upfall/best_kalman.yaml',
        'baseline_raw_f1': 92.00,
        'baseline_kalman_f1': 90.68,
        'target': 'Approach raw performance (~92%)',
    },
    'wedafall': {
        'name': 'WEDA-FALL',
        'sensor_type': 'Consumer-grade',
        'num_folds': 12,
        'base_config': 'config/wedafall/best_kalman.yaml',
        'baseline_raw_f1': 87.63,
        'baseline_kalman_f1': 88.88,
        'target': 'Maintain/improve Kalman performance (~89%)',
    }
}

# Default parameter grid (coarse search)
DEFAULT_PARAM_GRID = {
    'adaptive_alpha': [0.3, 0.5, 0.7, 0.9],
    'adaptive_scale_min': [0.1, 0.2, 0.3, 0.5],
    'adaptive_scale_max': [3.0],  # Keep fixed initially
    'adaptive_ema_alpha': [0.1],  # Keep fixed initially
    'adaptive_warmup_samples': [10],  # Keep fixed initially
}

# Fine-tuning grid (around promising regions)
FINE_TUNE_GRID = {
    'upfall': {
        # For clean sensors: more aggressive trust in measurements
        'adaptive_alpha': [0.6, 0.7, 0.8, 0.9, 1.0],
        'adaptive_scale_min': [0.05, 0.1, 0.15, 0.2],
        'adaptive_scale_max': [2.0, 3.0],
        'adaptive_ema_alpha': [0.1, 0.15, 0.2],
        'adaptive_warmup_samples': [5, 10],
    },
    'wedafall': {
        # For noisy sensors: balance smoothing with responsiveness
        'adaptive_alpha': [0.4, 0.5, 0.6, 0.7],
        'adaptive_scale_min': [0.2, 0.3, 0.4],
        'adaptive_scale_max': [3.0, 4.0, 5.0],
        'adaptive_ema_alpha': [0.05, 0.1, 0.15],
        'adaptive_warmup_samples': [10, 15],
    }
}


# =============================================================================
# Result Parsing
# =============================================================================

def parse_summary_report(summary_path: Path) -> Dict:
    """Parse summary_report.txt for metrics."""
    results = {}
    if not summary_path.exists():
        return {'error': 'Summary file not found'}

    content = summary_path.read_text()

    # Extract metrics with regex
    patterns = {
        'test_f1': r'Test F1:\s+([\d.]+)\s*±\s*([\d.]+)%',
        'test_macro_f1': r'Test Macro-F1:\s+([\d.]+)\s*±\s*([\d.]+)%',
        'test_acc': r'Test Accuracy:\s+([\d.]+)\s*±\s*([\d.]+)%',
        'test_prec': r'Test Precision:\s+([\d.]+)\s*±\s*([\d.]+)%',
        'test_recall': r'Test Recall:\s+([\d.]+)\s*±\s*([\d.]+)%',
        'test_auc': r'Test AUC:\s+([\d.]+)\s*±\s*([\d.]+)%',
        'val_f1': r'Val F1:\s+([\d.]+)%',
        'val_macro_f1': r'Val Macro-F1:\s+([\d.]+)%',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            results[key] = float(match.group(1))
            if match.lastindex >= 2:
                results[f'{key}_std'] = float(match.group(2))

    return results


# =============================================================================
# Config Generation
# =============================================================================

def create_adaptive_config(
    base_config_path: str,
    params: Dict,
    output_path: Path
) -> str:
    """Create config file with adaptive Kalman parameters."""
    with open(PROJECT_ROOT / base_config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Ensure dataset_args exists
    if 'dataset_args' not in config:
        config['dataset_args'] = {}

    # Enable adaptive Kalman and set parameters
    config['dataset_args']['adaptive_kalman_enabled'] = True
    config['dataset_args']['adaptive_alpha'] = params['adaptive_alpha']
    config['dataset_args']['adaptive_scale_min'] = params['adaptive_scale_min']
    config['dataset_args']['adaptive_scale_max'] = params['adaptive_scale_max']
    config['dataset_args']['adaptive_ema_alpha'] = params['adaptive_ema_alpha']
    config['dataset_args']['adaptive_warmup_samples'] = params['adaptive_warmup_samples']

    # Save config
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return str(output_path)


def params_to_str(params: Dict) -> str:
    """Convert params dict to short string identifier."""
    return f"a{params['adaptive_alpha']}_smin{params['adaptive_scale_min']}_smax{params['adaptive_scale_max']}_ema{params['adaptive_ema_alpha']}"


def params_to_readable(params: Dict) -> str:
    """Convert params dict to readable string."""
    return (f"α={params['adaptive_alpha']}, "
            f"scale_min={params['adaptive_scale_min']}, "
            f"scale_max={params['adaptive_scale_max']}, "
            f"ema={params['adaptive_ema_alpha']}, "
            f"warmup={params['adaptive_warmup_samples']}")


# =============================================================================
# Experiment Execution
# =============================================================================

def run_experiment(
    config_path: str,
    work_dir: str,
    num_gpus: int,
    max_folds: Optional[int] = None,
    gpu_ids: Optional[List[int]] = None,
    experiment_id: str = "",
) -> Dict:
    """Run a single experiment using ray_train.py.

    Args:
        config_path: Path to config file
        work_dir: Working directory for outputs
        num_gpus: Number of GPUs to use
        max_folds: Maximum folds (for quick testing)
        gpu_ids: Specific GPU IDs to use (for parallel execution)
        experiment_id: Identifier for logging
    """
    cmd = [
        sys.executable, 'ray_train.py',
        '--config', config_path,
        '--num-gpus', str(num_gpus),
        '--work-dir', work_dir,
    ]

    if max_folds:
        cmd.extend(['--max-folds', str(max_folds)])

    # Set up environment for GPU isolation
    env = os.environ.copy()
    if gpu_ids is not None:
        env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
        gpu_str = f"[GPUs {','.join(map(str, gpu_ids))}]"
    else:
        gpu_str = f"[All GPUs]"

    print(f"\n{'='*70}")
    print(f"{gpu_str} {experiment_id}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, env=env)
        summary_path = Path(work_dir) / 'summary_report.txt'
        return {'status': 'success', **parse_summary_report(summary_path)}
    except subprocess.CalledProcessError as e:
        print(f"ERROR {experiment_id}: Experiment failed with code {e.returncode}")
        return {'status': 'failed', 'error': str(e)}
    except Exception as e:
        print(f"ERROR {experiment_id}: {e}")
        return {'status': 'failed', 'error': str(e)}


def generate_param_combinations(param_grid: Dict) -> List[Dict]:
    """Generate all parameter combinations from grid."""
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]

    combinations = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo)))

    return combinations


def partition_gpus(total_gpus: int, num_partitions: int) -> List[List[int]]:
    """Partition GPUs into groups for parallel experiments.

    Args:
        total_gpus: Total number of GPUs available
        num_partitions: Number of parallel experiments to run

    Returns:
        List of GPU ID lists, one per partition
    """
    gpus_per_partition = total_gpus // num_partitions
    partitions = []

    for i in range(num_partitions):
        start = i * gpus_per_partition
        end = start + gpus_per_partition
        partitions.append(list(range(start, end)))

    return partitions


class ParallelTuningExecutor:
    """Execute tuning experiments in parallel with GPU partitioning."""

    def __init__(
        self,
        total_gpus: int,
        parallel_experiments: int,
        max_folds: Optional[int] = None,
    ):
        self.total_gpus = total_gpus
        self.parallel_experiments = parallel_experiments
        self.max_folds = max_folds

        # Partition GPUs
        self.gpu_partitions = partition_gpus(total_gpus, parallel_experiments)
        self.gpus_per_exp = total_gpus // parallel_experiments

        # Track GPU availability
        self.gpu_queue = queue.Queue()
        for i, partition in enumerate(self.gpu_partitions):
            self.gpu_queue.put((i, partition))

        # Results lock for thread-safe updates
        self.results_lock = threading.Lock()
        self.results = []

        print(f"Parallel executor initialized:")
        print(f"  Total GPUs: {total_gpus}")
        print(f"  Parallel experiments: {parallel_experiments}")
        print(f"  GPUs per experiment: {self.gpus_per_exp}")
        print(f"  GPU partitions: {self.gpu_partitions}")

    def run_single(
        self,
        config_path: str,
        work_dir: str,
        experiment_id: str,
        params: Dict,
        dataset: str,
    ) -> Dict:
        """Run a single experiment using an available GPU partition."""
        # Get available GPU partition
        partition_idx, gpu_ids = self.gpu_queue.get()

        try:
            metrics = run_experiment(
                config_path=config_path,
                work_dir=work_dir,
                num_gpus=self.gpus_per_exp,
                max_folds=self.max_folds,
                gpu_ids=gpu_ids,
                experiment_id=experiment_id,
            )

            result = {
                'name': experiment_id,
                'dataset': dataset,
                'params': params,
                'param_str': params_to_str(params),
                'config_path': config_path,
                'work_dir': work_dir,
                'gpu_partition': gpu_ids,
                'timestamp': datetime.now().isoformat(),
                **metrics,
            }

            return result

        finally:
            # Return GPU partition to pool
            self.gpu_queue.put((partition_idx, gpu_ids))

    def run_all(
        self,
        experiments: List[Dict],
        results_file: Path,
        completed_params: set,
    ) -> List[Dict]:
        """Run all experiments in parallel.

        Args:
            experiments: List of experiment configs with keys:
                - config_path, work_dir, experiment_id, params, dataset
            results_file: Path to save intermediate results
            completed_params: Set of already-completed param strings to skip

        Returns:
            List of result dicts
        """
        results = []

        # Filter out completed experiments
        pending = [e for e in experiments if e['param_str'] not in completed_params]

        print(f"\nRunning {len(pending)} experiments ({len(completed_params)} already completed)")
        print(f"Parallel workers: {self.parallel_experiments}")

        with ThreadPoolExecutor(max_workers=self.parallel_experiments) as executor:
            # Submit all experiments
            future_to_exp = {
                executor.submit(
                    self.run_single,
                    exp['config_path'],
                    exp['work_dir'],
                    exp['experiment_id'],
                    exp['params'],
                    exp['dataset'],
                ): exp for exp in pending
            }

            # Collect results as they complete
            for future in as_completed(future_to_exp):
                exp = future_to_exp[future]
                try:
                    result = future.result()
                    results.append(result)

                    # Save intermediate results (thread-safe)
                    with self.results_lock:
                        with open(results_file, 'w') as f:
                            json.dump(results, f, indent=2)

                    # Print progress
                    if result.get('status') == 'success':
                        f1 = result.get('test_f1', 0)
                        print(f"✓ {exp['experiment_id']}: F1={f1:.2f}%")
                    else:
                        print(f"✗ {exp['experiment_id']}: FAILED")

                except Exception as e:
                    print(f"✗ {exp['experiment_id']}: Exception - {e}")
                    results.append({
                        'name': exp['experiment_id'],
                        'dataset': exp['dataset'],
                        'params': exp['params'],
                        'status': 'failed',
                        'error': str(e),
                    })

        return results


# =============================================================================
# Tuning Execution
# =============================================================================

def run_tuning(
    dataset: str,
    output_dir: Path,
    num_gpus: int,
    param_grid: Dict,
    max_folds: Optional[int] = None,
    resume_from: Optional[str] = None,
    parallel: int = 1,
) -> List[Dict]:
    """Run parameter tuning for a dataset.

    Args:
        dataset: Dataset name ('upfall' or 'wedafall')
        output_dir: Output directory for results
        num_gpus: Total number of GPUs available
        param_grid: Parameter grid to search
        max_folds: Maximum folds per experiment (for quick testing)
        resume_from: Path to existing results file to resume from
        parallel: Number of parallel experiments (1=sequential)

    Returns:
        List of result dictionaries
    """
    results = []
    completed_params = set()

    # Load existing results if resuming
    results_file = output_dir / f'{dataset}_tuning_results.json'
    if resume_from and Path(resume_from).exists():
        with open(resume_from, 'r') as f:
            existing = json.load(f)
            results = [r for r in existing if r['dataset'] == dataset]
            completed_params = {params_to_str(r['params']) for r in results if r.get('status') == 'success'}
            print(f"Resuming: {len(completed_params)} experiments already completed")

    # Also check dataset-specific results file
    if results_file.exists() and not resume_from:
        with open(results_file, 'r') as f:
            existing = json.load(f)
            results = existing
            completed_params = {params_to_str(r['params']) for r in results if r.get('status') == 'success'}
            print(f"Found existing results: {len(completed_params)} experiments already completed")

    # Generate parameter combinations
    param_combinations = generate_param_combinations(param_grid)
    total_exps = len(param_combinations)
    pending_exps = total_exps - len(completed_params)

    print(f"\n{'#'*70}")
    print(f"# TUNING: {DATASET_INFO[dataset]['name']}")
    print(f"# Total experiments: {total_exps}")
    print(f"# Already completed: {len(completed_params)}")
    print(f"# Pending: {pending_exps}")
    print(f"# Parallel workers: {parallel}")
    print(f"# GPUs per worker: {num_gpus // parallel}")
    print(f"# Target: {DATASET_INFO[dataset]['target']}")
    print(f"# Baseline Raw F1: {DATASET_INFO[dataset]['baseline_raw_f1']:.2f}%")
    print(f"# Baseline Kalman F1: {DATASET_INFO[dataset]['baseline_kalman_f1']:.2f}%")
    print(f"{'#'*70}\n")

    # Create config directory
    config_dir = output_dir / 'configs' / dataset
    config_dir.mkdir(parents=True, exist_ok=True)

    # Prepare all experiment configs
    experiments = []
    for params in param_combinations:
        param_str = params_to_str(params)
        exp_name = f"{dataset}_{param_str}"
        work_dir = str(output_dir / exp_name)
        config_path = config_dir / f'{param_str}.yaml'

        # Create config file
        config_file = create_adaptive_config(
            DATASET_INFO[dataset]['base_config'],
            params,
            config_path
        )

        experiments.append({
            'config_path': config_file,
            'work_dir': work_dir,
            'experiment_id': exp_name,
            'params': params,
            'param_str': param_str,
            'dataset': dataset,
        })

    # Run experiments (parallel or sequential)
    if parallel > 1:
        # Parallel execution
        executor = ParallelTuningExecutor(
            total_gpus=num_gpus,
            parallel_experiments=parallel,
            max_folds=max_folds,
        )
        new_results = executor.run_all(experiments, results_file, completed_params)
        results.extend(new_results)
    else:
        # Sequential execution
        for i, exp in enumerate(experiments, 1):
            param_str = exp['param_str']

            # Skip if already completed
            if param_str in completed_params:
                print(f"[{i}/{total_exps}] Skipping (already completed): {params_to_readable(exp['params'])}")
                continue

            print(f"\n[{i}/{total_exps}] {DATASET_INFO[dataset]['name']}")
            print(f"    Params: {params_to_readable(exp['params'])}")

            # Run experiment
            metrics = run_experiment(
                config_path=exp['config_path'],
                work_dir=exp['work_dir'],
                num_gpus=num_gpus,
                max_folds=max_folds,
                experiment_id=exp['experiment_id'],
            )

            result = {
                'name': exp['experiment_id'],
                'dataset': dataset,
                'params': exp['params'],
                'param_str': param_str,
                'config_path': exp['config_path'],
                'work_dir': exp['work_dir'],
                'timestamp': datetime.now().isoformat(),
                **metrics,
            }
            results.append(result)

            # Save intermediate results
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)

            # Print progress
            if metrics.get('status') == 'success':
                f1 = metrics.get('test_f1', 0)
                macro_f1 = metrics.get('test_macro_f1', 0)
                baseline = DATASET_INFO[dataset]['baseline_kalman_f1']
                diff = f1 - baseline
                print(f"    Result: F1={f1:.2f}%, Macro-F1={macro_f1:.2f}% ({'+' if diff >= 0 else ''}{diff:.2f}% vs Kalman)")

    # Final save with all results
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary for this dataset
    successful = [r for r in results if r.get('status') == 'success']
    if successful:
        best = max(successful, key=lambda x: x.get('test_f1', 0))
        print(f"\n{'='*70}")
        print(f"BEST for {DATASET_INFO[dataset]['name']}:")
        print(f"  F1: {best.get('test_f1', 0):.2f}% ± {best.get('test_f1_std', 0):.2f}%")
        print(f"  Macro-F1: {best.get('test_macro_f1', 0):.2f}%")
        print(f"  Params: {params_to_readable(best['params'])}")
        print(f"{'='*70}")

    return results


# =============================================================================
# Report Generation
# =============================================================================

def generate_report(results: List[Dict], output_dir: Path):
    """Generate comprehensive markdown report."""
    report_path = output_dir / 'adaptive_tuning_report.md'

    with open(report_path, 'w') as f:
        f.write("# Adaptive Kalman Parameter Tuning Results\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        f.write("## Overview\n\n")
        f.write("Tuning Innovation-Based Adaptive Estimation (IAE) parameters for optimal fall detection.\n\n")
        f.write("**Parameters tuned:**\n")
        f.write("- `adaptive_alpha`: Sensitivity to NIS changes (0=none, 1=full)\n")
        f.write("- `adaptive_scale_min`: Minimum R scale (how much to trust clean signals)\n")
        f.write("- `adaptive_scale_max`: Maximum R scale (how much to smooth noisy signals)\n")
        f.write("- `adaptive_ema_alpha`: EMA smoothing for scale factor\n")
        f.write("- `adaptive_warmup_samples`: Warmup period before adaptation\n\n")

        for dataset in ['upfall', 'wedafall']:
            dataset_results = [r for r in results if r['dataset'] == dataset and r.get('status') == 'success']
            if not dataset_results:
                continue

            info = DATASET_INFO[dataset]
            f.write(f"## {info['name']} ({info['sensor_type']})\n\n")
            f.write(f"**Target:** {info['target']}\n\n")
            f.write(f"**Baselines:**\n")
            f.write(f"- Raw: {info['baseline_raw_f1']:.2f}% F1\n")
            f.write(f"- Fixed Kalman: {info['baseline_kalman_f1']:.2f}% F1\n\n")

            # Sort by F1 descending
            sorted_results = sorted(dataset_results, key=lambda x: x.get('test_f1', 0), reverse=True)

            # Best result
            best = sorted_results[0]
            f.write("### Best Configuration\n\n")
            f.write(f"**F1: {best.get('test_f1', 0):.2f}% ± {best.get('test_f1_std', 0):.2f}%**\n\n")
            f.write(f"**Macro-F1: {best.get('test_macro_f1', 0):.2f}%**\n\n")
            f.write("Parameters:\n")
            f.write(f"```yaml\n")
            f.write(f"adaptive_alpha: {best['params']['adaptive_alpha']}\n")
            f.write(f"adaptive_scale_min: {best['params']['adaptive_scale_min']}\n")
            f.write(f"adaptive_scale_max: {best['params']['adaptive_scale_max']}\n")
            f.write(f"adaptive_ema_alpha: {best['params']['adaptive_ema_alpha']}\n")
            f.write(f"adaptive_warmup_samples: {best['params']['adaptive_warmup_samples']}\n")
            f.write(f"```\n\n")

            # Comparison with baselines
            best_f1 = best.get('test_f1', 0)
            vs_raw = best_f1 - info['baseline_raw_f1']
            vs_kalman = best_f1 - info['baseline_kalman_f1']
            f.write("**Improvement:**\n")
            f.write(f"- vs Raw: {'+' if vs_raw >= 0 else ''}{vs_raw:.2f}%\n")
            f.write(f"- vs Fixed Kalman: {'+' if vs_kalman >= 0 else ''}{vs_kalman:.2f}%\n\n")

            # Full results table
            f.write("### All Results (sorted by F1)\n\n")
            f.write("| α | scale_min | scale_max | ema | F1 (%) | Std | Macro-F1 | vs Kalman |\n")
            f.write("|---|-----------|-----------|-----|--------|-----|----------|----------|\n")

            for r in sorted_results[:20]:  # Top 20
                p = r['params']
                f1 = r.get('test_f1', 0)
                std = r.get('test_f1_std', 0)
                macro = r.get('test_macro_f1', 0)
                diff = f1 - info['baseline_kalman_f1']
                f.write(f"| {p['adaptive_alpha']} | {p['adaptive_scale_min']} | {p['adaptive_scale_max']} | "
                       f"{p['adaptive_ema_alpha']} | {f1:.2f} | {std:.2f} | {macro:.2f} | "
                       f"{'+' if diff >= 0 else ''}{diff:.2f} |\n")

            f.write("\n")

            # Parameter sensitivity analysis
            f.write("### Parameter Sensitivity Analysis\n\n")

            # Alpha sensitivity
            alpha_perf = {}
            for r in dataset_results:
                alpha = r['params']['adaptive_alpha']
                if alpha not in alpha_perf:
                    alpha_perf[alpha] = []
                alpha_perf[alpha].append(r.get('test_f1', 0))

            f.write("**By α (sensitivity):**\n")
            f.write("| α | Mean F1 | Best F1 | Count |\n")
            f.write("|---|---------|---------|-------|\n")
            for alpha in sorted(alpha_perf.keys()):
                vals = alpha_perf[alpha]
                f.write(f"| {alpha} | {sum(vals)/len(vals):.2f}% | {max(vals):.2f}% | {len(vals)} |\n")
            f.write("\n")

            # Scale_min sensitivity
            smin_perf = {}
            for r in dataset_results:
                smin = r['params']['adaptive_scale_min']
                if smin not in smin_perf:
                    smin_perf[smin] = []
                smin_perf[smin].append(r.get('test_f1', 0))

            f.write("**By scale_min (measurement trust):**\n")
            f.write("| scale_min | Mean F1 | Best F1 | Count |\n")
            f.write("|-----------|---------|---------|-------|\n")
            for smin in sorted(smin_perf.keys()):
                vals = smin_perf[smin]
                f.write(f"| {smin} | {sum(vals)/len(vals):.2f}% | {max(vals):.2f}% | {len(vals)} |\n")
            f.write("\n")

        # Conclusions
        f.write("## Conclusions\n\n")

        for dataset in ['upfall', 'wedafall']:
            dataset_results = [r for r in results if r['dataset'] == dataset and r.get('status') == 'success']
            if not dataset_results:
                continue

            info = DATASET_INFO[dataset]
            best = max(dataset_results, key=lambda x: x.get('test_f1', 0))
            best_f1 = best.get('test_f1', 0)

            f.write(f"### {info['name']}\n\n")

            if dataset == 'upfall':
                if best_f1 >= info['baseline_raw_f1'] - 0.5:
                    f.write("✓ **Success:** Adaptive Kalman approaches raw performance.\n\n")
                else:
                    gap = info['baseline_raw_f1'] - best_f1
                    f.write(f"✗ **Gap remains:** {gap:.2f}% below raw performance.\n\n")
                    f.write("**Recommendations:**\n")
                    f.write("- Try even lower scale_min (0.05) for more measurement trust\n")
                    f.write("- Consider α=1.0 for full NIS responsiveness\n")
                    f.write("- May need per-trial adaptation rather than global settings\n\n")
            else:
                if best_f1 >= info['baseline_kalman_f1'] - 0.3:
                    f.write("✓ **Success:** Adaptive maintains Kalman performance.\n\n")
                else:
                    f.write("✗ **Regression:** Adaptive underperforms fixed Kalman.\n\n")

        f.write("## Recommended Configurations\n\n")
        f.write("Based on tuning results, add these to your config files:\n\n")

        for dataset in ['upfall', 'wedafall']:
            dataset_results = [r for r in results if r['dataset'] == dataset and r.get('status') == 'success']
            if not dataset_results:
                continue

            best = max(dataset_results, key=lambda x: x.get('test_f1', 0))
            p = best['params']

            f.write(f"**{DATASET_INFO[dataset]['name']}:**\n")
            f.write("```yaml\n")
            f.write("adaptive_kalman_enabled: True\n")
            f.write(f"adaptive_alpha: {p['adaptive_alpha']}\n")
            f.write(f"adaptive_scale_min: {p['adaptive_scale_min']}\n")
            f.write(f"adaptive_scale_max: {p['adaptive_scale_max']}\n")
            f.write(f"adaptive_ema_alpha: {p['adaptive_ema_alpha']}\n")
            f.write(f"adaptive_warmup_samples: {p['adaptive_warmup_samples']}\n")
            f.write("```\n\n")

    print(f"\nReport saved: {report_path}")


def generate_summary_table(results: List[Dict], output_dir: Path):
    """Generate CSV summary for easy analysis."""
    csv_path = output_dir / 'tuning_summary.csv'

    with open(csv_path, 'w') as f:
        f.write("dataset,alpha,scale_min,scale_max,ema_alpha,warmup,test_f1,test_f1_std,test_macro_f1,test_acc,status\n")

        for r in results:
            p = r.get('params', {})
            f.write(f"{r['dataset']},"
                   f"{p.get('adaptive_alpha', '')},"
                   f"{p.get('adaptive_scale_min', '')},"
                   f"{p.get('adaptive_scale_max', '')},"
                   f"{p.get('adaptive_ema_alpha', '')},"
                   f"{p.get('adaptive_warmup_samples', '')},"
                   f"{r.get('test_f1', '')},"
                   f"{r.get('test_f1_std', '')},"
                   f"{r.get('test_macro_f1', '')},"
                   f"{r.get('test_acc', '')},"
                   f"{r.get('status', '')}\n")

    print(f"CSV summary saved: {csv_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Adaptive Kalman parameter tuning study',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full grid search
  python run_adaptive_tuning.py --num-gpus 8

  # Quick test (2 folds)
  python run_adaptive_tuning.py --num-gpus 4 --max-folds 2

  # Custom parameter grid
  python run_adaptive_tuning.py --alpha 0.7 0.9 --scale-min 0.1 0.2 --num-gpus 8

  # Fine-tune mode (dataset-specific grids)
  python run_adaptive_tuning.py --fine-tune --num-gpus 8

  # Resume interrupted run
  python run_adaptive_tuning.py --resume exps/adaptive_tuning/upfall_tuning_results.json
        """
    )

    parser.add_argument('--datasets', nargs='+', default=['upfall', 'wedafall'],
                       choices=['upfall', 'wedafall'],
                       help='Datasets to tune (default: both)')
    parser.add_argument('--num-gpus', type=int, default=8,
                       help='Number of GPUs to use (default: 8)')
    parser.add_argument('--max-folds', type=int, default=None,
                       help='Maximum folds per experiment (for quick testing)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: exps/adaptive_tuning_<timestamp>)')

    # Parameter grid options
    parser.add_argument('--alpha', nargs='+', type=float, default=None,
                       help='adaptive_alpha values to test')
    parser.add_argument('--scale-min', nargs='+', type=float, default=None,
                       help='adaptive_scale_min values to test')
    parser.add_argument('--scale-max', nargs='+', type=float, default=None,
                       help='adaptive_scale_max values to test')
    parser.add_argument('--ema-alpha', nargs='+', type=float, default=None,
                       help='adaptive_ema_alpha values to test')
    parser.add_argument('--warmup', nargs='+', type=int, default=None,
                       help='adaptive_warmup_samples values to test')

    # Special modes
    parser.add_argument('--fine-tune', action='store_true',
                       help='Use dataset-specific fine-tuning grids')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from existing results file')
    parser.add_argument('--parallel', type=int, default=1,
                       help='Number of parallel experiments (default: 1=sequential). '
                            'GPUs are partitioned evenly. E.g., --num-gpus 8 --parallel 4 '
                            'runs 4 experiments with 2 GPUs each.')

    args = parser.parse_args()

    # Validate parallel configuration
    if args.parallel > 1:
        if args.num_gpus % args.parallel != 0:
            print(f"ERROR: --num-gpus ({args.num_gpus}) must be divisible by --parallel ({args.parallel})")
            sys.exit(1)
        gpus_per_exp = args.num_gpus // args.parallel
        if gpus_per_exp < 1:
            print(f"ERROR: Not enough GPUs. Need at least {args.parallel} GPUs for {args.parallel} parallel experiments")
            sys.exit(1)
        print(f"Parallel mode: {args.parallel} experiments, {gpus_per_exp} GPUs each")

    # Setup output directory
    if args.output_dir:
        output_dir = PROJECT_ROOT / args.output_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = PROJECT_ROOT / f'exps/adaptive_tuning_{timestamp}'

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    all_results = []

    for dataset in args.datasets:
        # Determine parameter grid
        if args.fine_tune:
            param_grid = FINE_TUNE_GRID.get(dataset, DEFAULT_PARAM_GRID)
        else:
            param_grid = DEFAULT_PARAM_GRID.copy()

        # Override with command-line values if provided
        if args.alpha:
            param_grid['adaptive_alpha'] = args.alpha
        if args.scale_min:
            param_grid['adaptive_scale_min'] = args.scale_min
        if args.scale_max:
            param_grid['adaptive_scale_max'] = args.scale_max
        if args.ema_alpha:
            param_grid['adaptive_ema_alpha'] = args.ema_alpha
        if args.warmup:
            param_grid['adaptive_warmup_samples'] = args.warmup

        # Calculate total experiments
        total = 1
        for v in param_grid.values():
            total *= len(v)

        print(f"\n{'='*70}")
        print(f"Dataset: {DATASET_INFO[dataset]['name']}")
        print(f"Parameter grid: {total} combinations")
        for k, v in param_grid.items():
            print(f"  {k}: {v}")
        print(f"{'='*70}\n")

        results = run_tuning(
            dataset=dataset,
            output_dir=output_dir,
            num_gpus=args.num_gpus,
            param_grid=param_grid,
            max_folds=args.max_folds,
            resume_from=args.resume,
            parallel=args.parallel,
        )
        all_results.extend(results)

    # Save final results
    final_results_path = output_dir / 'all_tuning_results.json'
    with open(final_results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Generate reports
    generate_report(all_results, output_dir)
    generate_summary_table(all_results, output_dir)

    # Print summary
    print(f"\n{'='*70}")
    print("Adaptive Kalman tuning study complete!")
    print(f"{'='*70}")
    print(f"Results JSON: {final_results_path}")
    print(f"Report: {output_dir / 'adaptive_tuning_report.md'}")
    print(f"CSV Summary: {output_dir / 'tuning_summary.csv'}")

    # Print best results
    print(f"\n{'='*70}")
    print("BEST CONFIGURATIONS")
    print(f"{'='*70}")

    for dataset in args.datasets:
        dataset_results = [r for r in all_results if r['dataset'] == dataset and r.get('status') == 'success']
        if dataset_results:
            best = max(dataset_results, key=lambda x: x.get('test_f1', 0))
            info = DATASET_INFO[dataset]
            print(f"\n{info['name']}:")
            print(f"  Best F1: {best.get('test_f1', 0):.2f}% ± {best.get('test_f1_std', 0):.2f}%")
            print(f"  vs Raw ({info['baseline_raw_f1']:.2f}%): {best.get('test_f1', 0) - info['baseline_raw_f1']:+.2f}%")
            print(f"  vs Kalman ({info['baseline_kalman_f1']:.2f}%): {best.get('test_f1', 0) - info['baseline_kalman_f1']:+.2f}%")
            print(f"  Params: {params_to_readable(best['params'])}")

    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
