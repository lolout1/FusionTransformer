#!/usr/bin/env python3
"""Kalman filter ablation study for fall detection."""

import os
import sys
import argparse
import yaml
import json
import pickle
import time
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy import stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class ExperimentConfig:
    """Configuration for a single ablation experiment."""
    name: str
    config_path: str
    phase: str  # 'feature', 'adaptation', 'mistuned', 'sensitivity'
    dataset: str  # 'upfall' or 'wedafall'
    description: str = ""
    priority: int = 1  # Lower = higher priority


@dataclass
class FoldMetrics:
    """Metrics for a single fold."""
    fold_id: int
    test_subject: int
    train_subjects: List[int]
    val_subjects: List[int]

    # Classification metrics
    test_f1: float = 0.0
    test_macro_f1: float = 0.0
    test_accuracy: float = 0.0
    test_precision: float = 0.0
    test_recall: float = 0.0
    test_auc: float = 0.0
    test_specificity: float = 0.0

    # Per-class metrics
    fall_precision: float = 0.0
    fall_recall: float = 0.0
    adl_precision: float = 0.0
    adl_recall: float = 0.0

    # Adaptive Kalman diagnostics
    adaptive_scale_mean: float = 1.0
    adaptive_scale_std: float = 0.0
    adaptive_scale_min_observed: float = 1.0
    adaptive_scale_max_observed: float = 1.0
    pct_scale_below_1: float = 0.0
    pct_scale_above_1_5: float = 0.0

    # Training info
    epochs_trained: int = 0
    best_epoch: int = 0
    final_val_loss: float = 0.0
    training_time_seconds: float = 0.0


@dataclass
class ExperimentResult:
    """Results for a complete experiment (all folds)."""
    config: ExperimentConfig
    fold_results: List[FoldMetrics] = field(default_factory=list)

    # Aggregated metrics (mean ± std)
    mean_f1: float = 0.0
    std_f1: float = 0.0
    mean_accuracy: float = 0.0
    std_accuracy: float = 0.0
    mean_precision: float = 0.0
    std_precision: float = 0.0
    mean_recall: float = 0.0
    std_recall: float = 0.0
    mean_auc: float = 0.0
    std_auc: float = 0.0

    # 95% confidence intervals
    ci_f1_lower: float = 0.0
    ci_f1_upper: float = 0.0

    # Timing
    total_time_seconds: float = 0.0

    def compute_aggregates(self):
        """Compute mean, std, and confidence intervals from fold results."""
        if not self.fold_results:
            return

        f1_scores = [f.test_f1 for f in self.fold_results]
        acc_scores = [f.test_accuracy for f in self.fold_results]
        prec_scores = [f.test_precision for f in self.fold_results]
        rec_scores = [f.test_recall for f in self.fold_results]
        auc_scores = [f.test_auc for f in self.fold_results]

        self.mean_f1 = np.mean(f1_scores)
        self.std_f1 = np.std(f1_scores)
        self.mean_accuracy = np.mean(acc_scores)
        self.std_accuracy = np.std(acc_scores)
        self.mean_precision = np.mean(prec_scores)
        self.std_precision = np.std(prec_scores)
        self.mean_recall = np.mean(rec_scores)
        self.std_recall = np.std(rec_scores)
        self.mean_auc = np.mean(auc_scores)
        self.std_auc = np.std(auc_scores)

        # 95% CI (t-distribution for small samples)
        n = len(f1_scores)
        if n > 1:
            se = self.std_f1 / np.sqrt(n)
            t_critical = stats.t.ppf(0.975, n - 1)
            self.ci_f1_lower = self.mean_f1 - t_critical * se
            self.ci_f1_upper = self.mean_f1 + t_critical * se

        self.total_time_seconds = sum(f.training_time_seconds for f in self.fold_results)


def discover_ablation_configs(config_dir: str) -> List[ExperimentConfig]:
    """
    Discover all ablation config files and categorize them.

    Args:
        config_dir: Base config directory (e.g., 'config')

    Returns:
        List of ExperimentConfig objects
    """
    experiments = []

    for dataset in ['upfall', 'wedafall']:
        ablation_dir = Path(config_dir) / dataset / 'ablation'
        if not ablation_dir.exists():
            print(f"Warning: {ablation_dir} does not exist")
            continue

        for config_file in sorted(ablation_dir.glob('*.yaml')):
            name = config_file.stem
            config_path = str(config_file)

            # Determine phase based on filename
            if name.startswith('kalman_') or name.startswith('hybrid_'):
                phase = 'feature'
                priority = 1
            elif name.startswith('adapt_'):
                phase = 'adaptation'
                priority = 2
            elif name.startswith('mistuned_'):
                phase = 'mistuned'
                priority = 3
            elif name.startswith('signal_a'):
                phase = 'sensitivity'
                priority = 4
            else:
                phase = 'other'
                priority = 5

            # Generate description
            description = f"{dataset.upper()} - {phase} - {name}"

            experiments.append(ExperimentConfig(
                name=name,
                config_path=config_path,
                phase=phase,
                dataset=dataset,
                description=description,
                priority=priority
            ))

    # Sort by priority, then dataset, then name
    experiments.sort(key=lambda x: (x.priority, x.dataset, x.name))

    return experiments


def run_single_experiment(
    config: ExperimentConfig,
    gpu_ids: List[int],
    output_dir: str,
    ray_train_path: str = 'ray_train.py'
) -> Optional[ExperimentResult]:
    """
    Run a single ablation experiment using ray_train.py.

    Args:
        config: Experiment configuration
        gpu_ids: List of GPU IDs to use
        output_dir: Directory to save results
        ray_train_path: Path to ray_train.py

    Returns:
        ExperimentResult or None if failed
    """
    start_time = time.time()

    # Create experiment output directory
    exp_output_dir = Path(output_dir) / config.dataset / config.phase / config.name
    exp_output_dir.mkdir(parents=True, exist_ok=True)

    # Copy config for reproducibility
    shutil.copy(config.config_path, exp_output_dir / 'config_used.yaml')

    # Build command
    gpu_str = ','.join(map(str, gpu_ids))
    cmd = [
        'python', ray_train_path,
        '--config', config.config_path,
        '--num-gpus', str(len(gpu_ids)),
        '--output-dir', str(exp_output_dir),
    ]

    # Set CUDA_VISIBLE_DEVICES
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpu_str

    print(f"\n{'='*60}")
    print(f"Running: {config.description}")
    print(f"GPUs: {gpu_str}")
    print(f"Output: {exp_output_dir}")
    print(f"{'='*60}")

    try:
        # Run experiment
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout per experiment
        )

        if result.returncode != 0:
            print(f"ERROR: Experiment failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr[:1000]}")
            # Save error log
            with open(exp_output_dir / 'error.log', 'w') as f:
                f.write(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")
            return None

        # Save stdout/stderr
        with open(exp_output_dir / 'stdout.log', 'w') as f:
            f.write(result.stdout)

        # Load results
        exp_result = load_experiment_results(exp_output_dir, config)
        if exp_result:
            exp_result.total_time_seconds = time.time() - start_time

        return exp_result

    except subprocess.TimeoutExpired:
        print(f"ERROR: Experiment timed out after 2 hours")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def load_experiment_results(
    output_dir: Path,
    config: ExperimentConfig
) -> Optional[ExperimentResult]:
    """
    Load results from an experiment output directory.

    Args:
        output_dir: Experiment output directory
        config: Experiment configuration

    Returns:
        ExperimentResult or None if not found
    """
    result = ExperimentResult(config=config)

    # Try to load fold_results.pkl
    pkl_path = output_dir / 'fold_results.pkl'
    if pkl_path.exists():
        with open(pkl_path, 'rb') as f:
            fold_data = pickle.load(f)

        for fold_id, fold_info in enumerate(fold_data.get('fold_results', [])):
            metrics = FoldMetrics(
                fold_id=fold_id,
                test_subject=fold_info.get('test_subject', 0),
                train_subjects=fold_info.get('train_subjects', []),
                val_subjects=fold_info.get('val_subjects', []),
                test_f1=fold_info.get('test_f1', 0.0),
                test_accuracy=fold_info.get('test_accuracy', 0.0),
                test_precision=fold_info.get('test_precision', 0.0),
                test_recall=fold_info.get('test_recall', 0.0),
                test_auc=fold_info.get('test_auc', 0.0),
                epochs_trained=fold_info.get('epochs_trained', 0),
                best_epoch=fold_info.get('best_epoch', 0),
                training_time_seconds=fold_info.get('training_time', 0.0),
            )
            result.fold_results.append(metrics)

    # Alternatively, try summary.json
    summary_path = output_dir / 'summary.json'
    if summary_path.exists() and not result.fold_results:
        with open(summary_path, 'r') as f:
            summary = json.load(f)

        result.mean_f1 = summary.get('mean_f1', 0.0)
        result.std_f1 = summary.get('std_f1', 0.0)
        result.mean_accuracy = summary.get('mean_accuracy', 0.0)
        result.std_accuracy = summary.get('std_accuracy', 0.0)

    if result.fold_results:
        result.compute_aggregates()

    return result


def run_ablation_study(
    experiments: List[ExperimentConfig],
    num_gpus: int,
    gpus_per_exp: int,
    parallel: int,
    output_dir: str,
    resume: bool = False
) -> Dict[str, ExperimentResult]:
    """
    Run the full ablation study.

    Args:
        experiments: List of experiments to run
        num_gpus: Total number of available GPUs
        gpus_per_exp: GPUs per experiment
        parallel: Number of parallel experiments
        output_dir: Output directory
        resume: Skip completed experiments

    Returns:
        Dictionary mapping experiment name to results
    """
    results = {}

    # Validate GPU allocation
    if gpus_per_exp * parallel > num_gpus:
        print(f"Warning: Requested {gpus_per_exp * parallel} GPUs but only {num_gpus} available")
        parallel = num_gpus // gpus_per_exp
        print(f"Reducing parallel to {parallel}")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save experiment manifest
    manifest = {
        'experiments': [asdict(e) for e in experiments],
        'num_gpus': num_gpus,
        'gpus_per_exp': gpus_per_exp,
        'parallel': parallel,
        'start_time': datetime.now().isoformat(),
    }
    with open(Path(output_dir) / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    # Filter completed experiments if resuming
    if resume:
        remaining = []
        for exp in experiments:
            exp_dir = Path(output_dir) / exp.dataset / exp.phase / exp.name
            if (exp_dir / 'summary.json').exists() or (exp_dir / 'fold_results.pkl').exists():
                print(f"Skipping completed: {exp.name}")
                # Load existing results
                result = load_experiment_results(exp_dir, exp)
                if result:
                    results[f"{exp.dataset}/{exp.name}"] = result
            else:
                remaining.append(exp)
        experiments = remaining
        print(f"\nResuming: {len(experiments)} experiments remaining")

    # Run experiments
    total = len(experiments)
    completed = 0

    if parallel == 1:
        # Sequential execution
        all_gpus = list(range(num_gpus))
        for exp in experiments:
            gpu_ids = all_gpus[:gpus_per_exp]
            result = run_single_experiment(exp, gpu_ids, output_dir)
            if result:
                results[f"{exp.dataset}/{exp.name}"] = result
            completed += 1
            print(f"\nProgress: {completed}/{total} experiments")
    else:
        # Parallel execution using process pool
        # Partition GPUs for parallel experiments
        gpu_partitions = []
        for i in range(parallel):
            start = i * gpus_per_exp
            end = start + gpus_per_exp
            gpu_partitions.append(list(range(start, end)))

        with ProcessPoolExecutor(max_workers=parallel) as executor:
            futures = {}
            exp_queue = list(experiments)
            active_partitions = set()

            while exp_queue or futures:
                # Submit new jobs for available partitions
                while exp_queue and len(futures) < parallel:
                    for i, partition in enumerate(gpu_partitions):
                        if i not in active_partitions and exp_queue:
                            exp = exp_queue.pop(0)
                            future = executor.submit(
                                run_single_experiment,
                                exp, partition, output_dir
                            )
                            futures[future] = (exp, i)
                            active_partitions.add(i)
                            break

                # Wait for any completion
                if futures:
                    done, _ = as_completed(futures.keys(), timeout=60).__iter__(), None
                    for future in list(futures.keys()):
                        if future.done():
                            exp, partition_idx = futures.pop(future)
                            active_partitions.discard(partition_idx)
                            try:
                                result = future.result()
                                if result:
                                    results[f"{exp.dataset}/{exp.name}"] = result
                            except Exception as e:
                                print(f"Error in {exp.name}: {e}")
                            completed += 1
                            print(f"\nProgress: {completed}/{total} experiments")
                            break

    return results


def generate_report(
    results: Dict[str, ExperimentResult],
    output_dir: str
):
    """
    Generate comprehensive report from ablation results.

    Args:
        results: Dictionary of experiment results
        output_dir: Output directory
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("COMPREHENSIVE KALMAN FILTER ABLATION STUDY RESULTS")
    report_lines.append("=" * 80)
    report_lines.append(f"\nGenerated: {datetime.now().isoformat()}")
    report_lines.append(f"Total experiments: {len(results)}")

    # Group by phase and dataset
    by_phase_dataset = {}
    for key, result in results.items():
        phase = result.config.phase
        dataset = result.config.dataset
        group_key = (phase, dataset)
        if group_key not in by_phase_dataset:
            by_phase_dataset[group_key] = []
        by_phase_dataset[group_key].append(result)

    # Report each phase
    for (phase, dataset), phase_results in sorted(by_phase_dataset.items()):
        report_lines.append(f"\n\n{'='*60}")
        report_lines.append(f"Phase: {phase.upper()} | Dataset: {dataset.upper()}")
        report_lines.append("=" * 60)

        # Sort by F1 score
        phase_results.sort(key=lambda x: x.mean_f1, reverse=True)

        report_lines.append(f"\n{'Config':<35} {'F1':>12} {'Acc':>12} {'AUC':>12}")
        report_lines.append("-" * 75)

        for result in phase_results:
            f1_str = f"{result.mean_f1*100:.2f} ± {result.std_f1*100:.2f}"
            acc_str = f"{result.mean_accuracy*100:.2f} ± {result.std_accuracy*100:.2f}"
            auc_str = f"{result.mean_auc*100:.2f} ± {result.std_auc*100:.2f}"
            report_lines.append(f"{result.config.name:<35} {f1_str:>12} {acc_str:>12} {auc_str:>12}")

    # Best results summary
    report_lines.append(f"\n\n{'='*80}")
    report_lines.append("BEST RESULTS BY DATASET")
    report_lines.append("=" * 80)

    for dataset in ['upfall', 'wedafall']:
        dataset_results = [r for k, r in results.items() if r.config.dataset == dataset]
        if dataset_results:
            best = max(dataset_results, key=lambda x: x.mean_f1)
            report_lines.append(f"\n{dataset.upper()}:")
            report_lines.append(f"  Best Config: {best.config.name}")
            report_lines.append(f"  F1: {best.mean_f1*100:.2f}% ± {best.std_f1*100:.2f}%")
            report_lines.append(f"  95% CI: [{best.ci_f1_lower*100:.2f}%, {best.ci_f1_upper*100:.2f}%]")

    # Save report
    report_text = "\n".join(report_lines)
    print(report_text)

    with open(Path(output_dir) / 'summary_report.txt', 'w') as f:
        f.write(report_text)

    # Save CSV for statistical analysis
    csv_lines = ["dataset,phase,config,f1_mean,f1_std,acc_mean,acc_std,auc_mean,auc_std,ci_lower,ci_upper"]
    for key, result in results.items():
        csv_lines.append(",".join([
            result.config.dataset,
            result.config.phase,
            result.config.name,
            f"{result.mean_f1:.6f}",
            f"{result.std_f1:.6f}",
            f"{result.mean_accuracy:.6f}",
            f"{result.std_accuracy:.6f}",
            f"{result.mean_auc:.6f}",
            f"{result.std_auc:.6f}",
            f"{result.ci_f1_lower:.6f}",
            f"{result.ci_f1_upper:.6f}",
        ]))

    with open(Path(output_dir) / 'metrics.csv', 'w') as f:
        f.write("\n".join(csv_lines))

    # Save full results as pickle
    with open(Path(output_dir) / 'all_results.pkl', 'wb') as f:
        pickle.dump(results, f)


def run_statistical_tests(
    results: Dict[str, ExperimentResult],
    output_dir: str
):
    """
    Run statistical significance tests between conditions.

    Args:
        results: Dictionary of experiment results
        output_dir: Output directory
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("STATISTICAL SIGNIFICANCE TESTS")
    report_lines.append("=" * 80)

    for dataset in ['upfall', 'wedafall']:
        report_lines.append(f"\n\n{'='*60}")
        report_lines.append(f"Dataset: {dataset.upper()}")
        report_lines.append("=" * 60)

        dataset_results = {k: v for k, v in results.items() if v.config.dataset == dataset}

        # Compare adaptation modes against baseline
        baseline_key = f"{dataset}/adapt_none"
        if baseline_key in dataset_results:
            baseline = dataset_results[baseline_key]
            baseline_f1s = [f.test_f1 for f in baseline.fold_results]

            report_lines.append(f"\nBaseline (adapt_none): F1 = {baseline.mean_f1*100:.2f}%")
            report_lines.append("\nPaired t-tests vs baseline:")
            report_lines.append("-" * 50)

            for key, result in dataset_results.items():
                if key == baseline_key:
                    continue

                other_f1s = [f.test_f1 for f in result.fold_results]

                # Only compare if same number of folds
                if len(baseline_f1s) == len(other_f1s) and len(baseline_f1s) > 1:
                    t_stat, p_value = stats.ttest_rel(baseline_f1s, other_f1s)

                    # Cohen's d effect size
                    diff = np.array(baseline_f1s) - np.array(other_f1s)
                    cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0

                    sig_marker = "*" if p_value < 0.05 else ""
                    if p_value < 0.01:
                        sig_marker = "**"
                    if p_value < 0.001:
                        sig_marker = "***"

                    report_lines.append(
                        f"  {result.config.name:<30} "
                        f"Δ={result.mean_f1*100 - baseline.mean_f1*100:+.2f}% "
                        f"p={p_value:.4f}{sig_marker} "
                        f"d={cohens_d:.3f}"
                    )

    report_text = "\n".join(report_lines)
    print(report_text)

    with open(Path(output_dir) / 'statistical_tests.txt', 'w') as f:
        f.write(report_text)


def main():
    parser = argparse.ArgumentParser(
        description='Run comprehensive Kalman filter ablation study'
    )
    parser.add_argument(
        '--num-gpus', type=int, default=6,
        help='Total number of GPUs available'
    )
    parser.add_argument(
        '--gpus-per-exp', type=int, default=3,
        help='GPUs per experiment'
    )
    parser.add_argument(
        '--parallel', type=int, default=2,
        help='Number of parallel experiments'
    )
    parser.add_argument(
        '--output-dir', type=str, default='results/ablation',
        help='Output directory for results'
    )
    parser.add_argument(
        '--config-dir', type=str, default='config',
        help='Base config directory'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Skip completed experiments'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Only show what would be run'
    )
    parser.add_argument(
        '--phase', type=str, default=None,
        choices=['feature', 'adaptation', 'mistuned', 'sensitivity'],
        help='Only run specific phase'
    )
    parser.add_argument(
        '--dataset', type=str, default=None,
        choices=['upfall', 'wedafall'],
        help='Only run specific dataset'
    )

    args = parser.parse_args()

    # Discover experiments
    experiments = discover_ablation_configs(args.config_dir)

    # Filter by phase/dataset if specified
    if args.phase:
        experiments = [e for e in experiments if e.phase == args.phase]
    if args.dataset:
        experiments = [e for e in experiments if e.dataset == args.dataset]

    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE KALMAN FILTER ABLATION STUDY")
    print(f"{'='*60}")
    print(f"Total experiments: {len(experiments)}")
    print(f"GPUs: {args.num_gpus} ({args.gpus_per_exp} per exp, {args.parallel} parallel)")
    print(f"Output: {args.output_dir}")

    # Group by phase
    by_phase = {}
    for exp in experiments:
        if exp.phase not in by_phase:
            by_phase[exp.phase] = []
        by_phase[exp.phase].append(exp)

    print(f"\nExperiments by phase:")
    for phase, phase_exps in sorted(by_phase.items()):
        print(f"  {phase}: {len(phase_exps)}")

    if args.dry_run:
        print(f"\nDRY RUN - Would run:")
        for exp in experiments:
            print(f"  {exp.config_path}")
        return

    # Run ablation study
    results = run_ablation_study(
        experiments=experiments,
        num_gpus=args.num_gpus,
        gpus_per_exp=args.gpus_per_exp,
        parallel=args.parallel,
        output_dir=args.output_dir,
        resume=args.resume
    )

    # Generate reports
    generate_report(results, args.output_dir)
    run_statistical_tests(results, args.output_dir)

    print(f"\n{'='*60}")
    print(f"ABLATION STUDY COMPLETE")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
