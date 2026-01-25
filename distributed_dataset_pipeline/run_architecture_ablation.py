#!/usr/bin/env python3
"""
Comprehensive Architecture Ablation Study.

Tests multiple architectures (LSTM, Transformer, DeepCNN, Mamba) with and without
Kalman fusion across all datasets (SmartFallMM, UP-FALL, WEDA-FALL) at multiple
window sizes.

Features:
- Distributed execution via Ray
- Multiple window sizes (2s, 3s, 4s, default)
- ADL:Fall ratio tracking and reporting
- Automatic model input/output adaptation
- Comprehensive logging and result aggregation
- Well-formatted markdown report generation

Usage:
    python distributed_dataset_pipeline/run_architecture_ablation.py \
        --datasets all --num-gpus 4 --parallel 2

    # Quick test (2 folds per dataset)
    python distributed_dataset_pipeline/run_architecture_ablation.py \
        --datasets wedafall --quick --num-gpus 2

    # Specific window sizes
    python distributed_dataset_pipeline/run_architecture_ablation.py \
        --datasets upfall --window-sizes 2s 4s --num-gpus 2
"""

import os
import sys
import json
import yaml
import pickle
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import copy

# =============================================================================
# CONFIGURATION
# =============================================================================

# Architecture definitions with model paths and compatibility info
ARCHITECTURES = {
    'lstm': {
        'model': 'Models.dual_stream_cnn_lstm.DualStreamLSTM',
        'description': 'Bidirectional LSTM with dual-stream fusion',
        'model_args_override': {
            'num_lstm_layers': 2,
        },
        # Lower LR for LSTM - prevents early overfitting on short windows
        'training_override': {
            'base_lr': 0.0005,
        },
    },
    'baseline_transformer': {
        'model': 'Models.encoder_ablation.KalmanConv1dLinear',
        'description': 'Transformer with Conv1D(acc) + Linear(ori) encoders',
        'model_args_override': {},
        'training_override': {},
    },
    'deep_cnn_transformer': {
        'model': 'Models.short_window_variants.DeepCNNTransformer',
        'description': 'Multi-stage CNN + Transformer encoder',
        'model_args_override': {
            'cnn_stages': 3,
            'kernel_sizes': [8, 5, 3],
        },
        'training_override': {},
    },
    'mamba': {
        'model': 'Models.dual_stream_mamba.DualStreamMamba',
        'description': 'State-space model with Conv1d + GRU blocks',
        'model_args_override': {
            'd_state': 16,
            'd_conv': 4,
            'expand': 2,
            # Mamba splits input: acc (4ch) + gyro (3ch) = 7ch total
            'acc_coords': 4,
            'gyro_coords': 3,
        },
        # Lower LR for Mamba - similar to LSTM, recurrent-style
        'training_override': {
            'base_lr': 0.0005,
        },
    },
}

# Input type definitions
INPUT_TYPES = {
    'kalman': {
        'enable_kalman_fusion': True,
        'imu_channels': 7,
        'description': 'Kalman-fused (smv, acc_xyz, roll, pitch, yaw)',
    },
    'raw': {
        'enable_kalman_fusion': False,
        'imu_channels': 7,
        'description': 'Raw IMU (smv, acc_xyz, gyro_xyz)',
    },
}

# Dataset configurations with window sizes
# NOTE: SmartFallMM excluded - only UP-FALL and WEDA-FALL included
DATASET_CONFIGS = {
    'upfall': {
        'base_config': 'best_config/upfall/kalman.yaml',
        'sampling_rate': 18,
        'feeder': 'Feeder.external_datasets.ExternalFallDataset',
        'description': 'UP-FALL (18Hz, 17 subjects, wrist IMU)',
        'window_sizes': {
            '2s': 36,    # 2.0s at 18Hz
            '3s': 54,    # 3.0s at 18Hz
            '4s': 72,    # 4.0s at 18Hz
            'default': 160,  # 8.9s (original optimal)
        },
        'default_window': 'default',
        'num_test_subjects': 15,
    },
    'wedafall': {
        'base_config': 'best_config/wedafall/kalman.yaml',
        'sampling_rate': 50,
        'feeder': 'Feeder.external_datasets.ExternalFallDataset',
        'description': 'WEDA-FALL (50Hz, young+elderly, wrist)',
        'window_sizes': {
            '2s': 100,   # 2.0s at 50Hz
            '3s': 150,   # 3.0s at 50Hz
            '4s': 200,   # 4.0s at 50Hz
            'default': 250,  # 5.0s (original optimal)
        },
        'default_window': 'default',
        'num_test_subjects': 12,
        # Include elderly subjects for training (ADL only)
        'include_elderly': True,
        'all_subjects': list(range(1, 15)) + list(range(21, 32)),  # Young (1-14) + Elderly (21-31)
    },
}


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    dataset: str
    architecture: str
    input_type: str
    window_name: str
    window_samples: int
    config_path: str
    work_dir: str

    # Derived fields
    model: str = ''
    description: str = ''
    window_duration: float = 0.0

    def __post_init__(self):
        arch_info = ARCHITECTURES[self.architecture]
        input_info = INPUT_TYPES[self.input_type]
        dataset_info = DATASET_CONFIGS[self.dataset]
        self.model = arch_info['model']
        self.description = f"{arch_info['description']} with {input_info['description']}"
        self.window_duration = self.window_samples / dataset_info['sampling_rate']


@dataclass
class ExperimentResult:
    """Results from a single experiment."""
    name: str
    dataset: str
    architecture: str
    input_type: str
    window_name: str
    window_samples: int
    window_duration: float

    # Metrics (mean ± std across folds)
    test_f1: float = 0.0
    test_f1_std: float = 0.0
    test_accuracy: float = 0.0
    test_accuracy_std: float = 0.0
    test_precision: float = 0.0
    test_recall: float = 0.0
    test_auc: float = 0.0

    # Training info
    num_folds: int = 0
    avg_best_epoch: float = 0.0
    status: str = 'pending'
    error_message: str = ''

    # Data statistics
    total_train_samples: int = 0
    train_falls: int = 0
    train_adls: int = 0
    fall_ratio: float = 0.0
    adl_fall_ratio: str = ''  # e.g., "1:2.5"

    # Per-fold details
    fold_results: List[Dict] = field(default_factory=list)


# =============================================================================
# CONFIG GENERATION
# =============================================================================

def generate_experiment_config(
    exp: ExperimentConfig,
    base_config: Dict,
    dataset_info: Dict,
    arch_info: Dict,
    input_info: Dict,
) -> Dict:
    """Generate YAML config for an experiment."""
    config = copy.deepcopy(base_config)

    # Update model
    config['model'] = exp.model

    # Update model args
    model_args = config.get('model_args', {})
    model_args['imu_channels'] = input_info['imu_channels']
    model_args['acc_coords'] = input_info['imu_channels']
    model_args['imu_frames'] = exp.window_samples
    model_args['acc_frames'] = exp.window_samples

    # Apply architecture-specific overrides
    for key, value in arch_info.get('model_args_override', {}).items():
        model_args[key] = value

    config['model_args'] = model_args

    # Update dataset args
    dataset_args = config.get('dataset_args', {})
    dataset_args['enable_kalman_fusion'] = input_info['enable_kalman_fusion']
    dataset_args['max_length'] = exp.window_samples  # CRITICAL: max_length controls actual window

    # WEDA-FALL: Include elderly subjects
    if exp.dataset == 'wedafall':
        dataset_args['age_group'] = ['young', 'old']  # Include elderly
        # Update subjects list to include all
        if 'all_subjects' in dataset_info:
            config['subjects'] = dataset_info['all_subjects']

    config['dataset_args'] = dataset_args

    # Apply architecture-specific training overrides (e.g., learning rate)
    training_override = arch_info.get('training_override', {})
    for key, value in training_override.items():
        config[key] = value

    return config


def create_experiment_configs(
    output_dir: Path,
    datasets: List[str],
    architectures: List[str],
    input_types: List[str],
    window_sizes: List[str],
) -> List[ExperimentConfig]:
    """Create all experiment configurations."""
    experiments = []
    configs_dir = output_dir / 'configs'

    for dataset in datasets:
        dataset_info = DATASET_CONFIGS[dataset]
        base_config_path = dataset_info['base_config']

        # Load base config
        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)

        dataset_configs_dir = configs_dir / dataset
        dataset_configs_dir.mkdir(parents=True, exist_ok=True)

        # Determine window sizes to test
        if 'all' in window_sizes:
            windows_to_test = list(dataset_info['window_sizes'].keys())
        elif 'default_only' in window_sizes:
            windows_to_test = [dataset_info['default_window']]
        else:
            windows_to_test = [w for w in window_sizes if w in dataset_info['window_sizes']]

        for window_name in windows_to_test:
            window_samples = dataset_info['window_sizes'][window_name]

            for arch in architectures:
                arch_info = ARCHITECTURES[arch]

                for input_type in input_types:
                    input_info = INPUT_TYPES[input_type]

                    # Generate experiment name
                    exp_name = f"{dataset}_{arch}_{input_type}_{window_name}"
                    config_path = dataset_configs_dir / f"{exp_name}.yaml"
                    work_dir = output_dir / 'runs' / exp_name

                    # Create experiment config object
                    exp = ExperimentConfig(
                        name=exp_name,
                        dataset=dataset,
                        architecture=arch,
                        input_type=input_type,
                        window_name=window_name,
                        window_samples=window_samples,
                        config_path=str(config_path),
                        work_dir=str(work_dir),
                    )

                    # Generate YAML config
                    config = generate_experiment_config(
                        exp, base_config, dataset_info, arch_info, input_info
                    )

                    # Write config
                    with open(config_path, 'w') as f:
                        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

                    experiments.append(exp)

    return experiments


# =============================================================================
# EXPERIMENT EXECUTION
# =============================================================================

def run_experiment(
    exp: ExperimentConfig,
    num_gpus: int = 2,
    max_folds: Optional[int] = None,
    quick_mode: bool = False,
) -> ExperimentResult:
    """Run a single experiment using ray_train.py."""
    dataset_info = DATASET_CONFIGS[exp.dataset]

    result = ExperimentResult(
        name=exp.name,
        dataset=exp.dataset,
        architecture=exp.architecture,
        input_type=exp.input_type,
        window_name=exp.window_name,
        window_samples=exp.window_samples,
        window_duration=exp.window_duration,
    )

    # Build command
    cmd = [
        'python', 'ray_train.py',
        '--config', exp.config_path,
        '--work-dir', exp.work_dir,
        '--num-gpus', str(num_gpus),
    ]

    if max_folds:
        cmd.extend(['--max-folds', str(max_folds)])
    elif quick_mode:
        cmd.extend(['--max-folds', '2'])

    print(f"\n{'='*70}")
    print(f"Running: {exp.name}")
    print(f"  Dataset: {exp.dataset} | Arch: {exp.architecture} | Input: {exp.input_type}")
    print(f"  Window: {exp.window_name} ({exp.window_samples} samples, {exp.window_duration:.1f}s)")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*70}")

    try:
        # Run training with live output
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Stream output in real-time
        output_lines = []
        for line in proc.stdout:
            print(line, end='')
            output_lines.append(line)

        proc.wait(timeout=7200)  # 2 hour timeout

        if proc.returncode != 0:
            result.status = 'failed'
            result.error_message = ''.join(output_lines[-20:]) if output_lines else 'Unknown error'
            print(f"FAILED: {exp.name}")
            return result

        # Load results
        result = load_experiment_results(exp, result)

    except subprocess.TimeoutExpired:
        proc.kill()
        result.status = 'timeout'
        result.error_message = 'Experiment timed out after 2 hours'
        print(f"TIMEOUT: {exp.name}")
    except KeyboardInterrupt:
        proc.kill()
        result.status = 'interrupted'
        result.error_message = 'User interrupted'
        print(f"\nINTERRUPTED: {exp.name}")
        raise  # Re-raise to stop the ablation
    except Exception as e:
        result.status = 'error'
        result.error_message = str(e)
        print(f"ERROR: {exp.name}: {e}")

    return result


def load_experiment_results(exp: ExperimentConfig, result: ExperimentResult) -> ExperimentResult:
    """Load results from completed experiment. Robust to different pickle formats."""
    import numpy as np

    work_dir = Path(exp.work_dir)

    # Try loading fold_results.pkl
    pkl_path = work_dir / 'fold_results.pkl'
    json_path = work_dir / 'summary.json'

    try:
        if pkl_path.exists():
            with open(pkl_path, 'rb') as f:
                fold_data = pickle.load(f)

            # Extract metrics from folds
            f1_scores = []
            acc_scores = []
            precision_scores = []
            recall_scores = []
            auc_scores = []
            best_epochs = []
            total_train = 0
            total_falls = 0
            total_adls = 0

            # Handle dict, list, or other formats robustly
            fold_items = []
            if isinstance(fold_data, dict):
                fold_items = list(fold_data.values())
            elif isinstance(fold_data, list):
                fold_items = fold_data
            else:
                # Try to iterate if possible
                try:
                    fold_items = list(fold_data)
                except:
                    fold_items = []

            for fold_info in fold_items:
                # Skip non-dict items
                if not isinstance(fold_info, dict):
                    continue

                # Handle both 'test' and 'test_metrics' keys
                test_metrics = fold_info.get('test', fold_info.get('test_metrics', {}))
                if not isinstance(test_metrics, dict):
                    continue

                # Handle both 'f1' and 'f1_score' keys (different naming conventions)
                f1_val = test_metrics.get('f1', test_metrics.get('f1_score'))
                if f1_val is not None:
                    f1_scores.append(f1_val)
                if 'accuracy' in test_metrics:
                    acc_scores.append(test_metrics['accuracy'])
                if 'precision' in test_metrics:
                    precision_scores.append(test_metrics['precision'])
                if 'recall' in test_metrics:
                    recall_scores.append(test_metrics['recall'])
                if 'auc' in test_metrics:
                    auc_scores.append(test_metrics['auc'])
                if 'best_epoch' in fold_info:
                    best_epochs.append(fold_info['best_epoch'])

                # Data statistics (from first fold as representative)
                if total_train == 0:
                    data_stats = fold_info.get('data_stats', {})
                    if isinstance(data_stats, dict):
                        total_train = data_stats.get('train_samples', 0)
                        total_falls = data_stats.get('train_falls', 0)
                        total_adls = data_stats.get('train_adls', 0)

            if f1_scores:
                result.test_f1 = float(np.mean(f1_scores)) * 100
                result.test_f1_std = float(np.std(f1_scores)) * 100
                result.test_accuracy = float(np.mean(acc_scores)) * 100 if acc_scores else 0
                result.test_accuracy_std = float(np.std(acc_scores)) * 100 if acc_scores else 0
                result.test_precision = float(np.mean(precision_scores)) * 100 if precision_scores else 0
                result.test_recall = float(np.mean(recall_scores)) * 100 if recall_scores else 0
                result.test_auc = float(np.mean(auc_scores)) * 100 if auc_scores else 0
                result.avg_best_epoch = float(np.mean(best_epochs)) if best_epochs else 0
                result.num_folds = len(f1_scores)
                result.status = 'completed'
                result.fold_results = fold_items

                # Data statistics
                result.total_train_samples = total_train
                result.train_falls = total_falls
                result.train_adls = total_adls
                if total_train > 0:
                    result.fall_ratio = total_falls / total_train * 100
                if total_falls > 0:
                    result.adl_fall_ratio = f"1:{total_adls/total_falls:.2f}"

        elif json_path.exists():
            with open(json_path, 'r') as f:
                summary = json.load(f)

            result.test_f1 = summary.get('test_f1_mean', 0) * 100
            result.test_f1_std = summary.get('test_f1_std', 0) * 100
            result.test_accuracy = summary.get('test_accuracy_mean', 0) * 100
            result.num_folds = summary.get('num_folds', 0)
            result.status = 'completed'

            # Data stats from summary
            result.total_train_samples = summary.get('train_samples', 0)
            result.train_falls = summary.get('train_falls', 0)
            result.train_adls = summary.get('train_adls', 0)
            if result.total_train_samples > 0:
                result.fall_ratio = result.train_falls / result.total_train_samples * 100
            if result.train_falls > 0:
                result.adl_fall_ratio = f"1:{result.train_adls/result.train_falls:.2f}"

        else:
            result.status = 'no_results'
            result.error_message = 'No results files found'

    except Exception as e:
        result.status = 'error'
        result.error_message = f'Failed to load results: {str(e)}'

    return result


def run_experiments_parallel(
    experiments: List[ExperimentConfig],
    num_gpus: int = 4,
    parallel: int = 2,
    max_folds: Optional[int] = None,
    quick_mode: bool = False,
) -> List[ExperimentResult]:
    """Run experiments with parallelism."""
    results = []
    gpus_per_exp = max(1, num_gpus // parallel)

    print(f"\n{'#'*70}")
    print(f"# Running {len(experiments)} experiments")
    print(f"# Parallel workers: {parallel}, GPUs per experiment: {gpus_per_exp}")
    print(f"{'#'*70}\n")

    # Run sequentially with interrupt handling
    try:
        for i, exp in enumerate(experiments):
            print(f"\n[{i+1}/{len(experiments)}] {exp.name}")
            result = run_experiment(exp, gpus_per_exp, max_folds, quick_mode)
            results.append(result)

            # Print intermediate result
            if result.status == 'completed':
                print(f"  -> F1: {result.test_f1:.2f}% ± {result.test_f1_std:.2f}%")
                if result.adl_fall_ratio:
                    print(f"  -> Data: {result.total_train_samples:,} samples (Falls: {result.train_falls:,}, ADLs: {result.train_adls:,}, Ratio: {result.adl_fall_ratio})")

    except KeyboardInterrupt:
        print(f"\n\n{'='*70}")
        print("ABLATION INTERRUPTED - Saving partial results...")
        print(f"Completed {len(results)}/{len(experiments)} experiments")
        print(f"{'='*70}\n")

    return results


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(
    results: List[ExperimentResult],
    output_dir: Path,
    experiments: List[ExperimentConfig],
) -> str:
    """Generate comprehensive markdown report."""
    report_lines = []

    # Header
    report_lines.append("# Architecture Ablation Study Results")
    report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Summary statistics
    completed = [r for r in results if r.status == 'completed']
    failed = [r for r in results if r.status != 'completed']

    report_lines.append("## Summary")
    report_lines.append(f"- Total experiments: {len(results)}")
    report_lines.append(f"- Completed: {len(completed)}")
    report_lines.append(f"- Failed: {len(failed)}")
    report_lines.append("")

    # Results by dataset
    for dataset in DATASET_CONFIGS.keys():
        dataset_results = [r for r in completed if r.dataset == dataset]
        if not dataset_results:
            continue

        dataset_info = DATASET_CONFIGS[dataset]
        report_lines.append(f"\n## {dataset.upper()}")
        report_lines.append(f"*{dataset_info['description']}*\n")

        # Group by window size
        window_groups = defaultdict(list)
        for r in dataset_results:
            window_groups[r.window_name].append(r)

        for window_name in sorted(window_groups.keys()):
            window_results = window_groups[window_name]
            if not window_results:
                continue

            sample_result = window_results[0]
            report_lines.append(f"\n### Window: {window_name} ({sample_result.window_samples} samples, {sample_result.window_duration:.1f}s)")

            # Data statistics
            if sample_result.total_train_samples > 0:
                report_lines.append(f"\n**Data Statistics (representative fold):**")
                report_lines.append(f"- Total train samples: {sample_result.total_train_samples:,}")
                report_lines.append(f"- Falls: {sample_result.train_falls:,} ({sample_result.fall_ratio:.1f}%)")
                report_lines.append(f"- ADLs: {sample_result.train_adls:,} ({100-sample_result.fall_ratio:.1f}%)")
                report_lines.append(f"- ADL:Fall ratio: {sample_result.adl_fall_ratio}")
                report_lines.append("")

            # Create comparison table
            report_lines.append("| Architecture | Input | F1 (%) | Acc (%) | Prec (%) | Recall (%) | AUC (%) | Folds |")
            report_lines.append("|--------------|-------|--------|---------|----------|------------|---------|-------|")

            # Sort by F1 descending
            window_results.sort(key=lambda x: x.test_f1, reverse=True)

            for r in window_results:
                report_lines.append(
                    f"| {r.architecture} | {r.input_type} | "
                    f"{r.test_f1:.2f}±{r.test_f1_std:.2f} | "
                    f"{r.test_accuracy:.2f} | "
                    f"{r.test_precision:.2f} | "
                    f"{r.test_recall:.2f} | "
                    f"{r.test_auc:.2f} | "
                    f"{r.num_folds} |"
                )

        # Kalman vs Raw comparison for this dataset
        report_lines.append(f"\n### Kalman vs Raw Comparison ({dataset})")
        report_lines.append("| Architecture | Window | Kalman F1 | Raw F1 | Δ F1 | Winner |")
        report_lines.append("|--------------|--------|-----------|--------|------|--------|")

        for window_name in sorted(window_groups.keys()):
            for arch in ARCHITECTURES.keys():
                kalman_r = next((r for r in dataset_results
                                if r.architecture == arch and r.input_type == 'kalman'
                                and r.window_name == window_name), None)
                raw_r = next((r for r in dataset_results
                             if r.architecture == arch and r.input_type == 'raw'
                             and r.window_name == window_name), None)

                if kalman_r and raw_r:
                    delta = kalman_r.test_f1 - raw_r.test_f1
                    winner = "Kalman" if delta > 0 else "Raw" if delta < 0 else "Tie"
                    report_lines.append(
                        f"| {arch} | {window_name} | {kalman_r.test_f1:.2f}% | {raw_r.test_f1:.2f}% | "
                        f"{delta:+.2f}% | **{winner}** |"
                    )

    # Window size impact analysis
    report_lines.append("\n## Window Size Impact Analysis")

    for dataset in DATASET_CONFIGS.keys():
        dataset_results = [r for r in completed if r.dataset == dataset]
        if not dataset_results:
            continue

        report_lines.append(f"\n### {dataset.upper()}")
        report_lines.append("| Architecture | Input | 2s F1 | 3s F1 | 4s F1 | Default F1 | Best Window |")
        report_lines.append("|--------------|-------|-------|-------|-------|------------|-------------|")

        for arch in ARCHITECTURES.keys():
            for input_type in INPUT_TYPES.keys():
                results_by_window = {}
                for r in dataset_results:
                    if r.architecture == arch and r.input_type == input_type:
                        results_by_window[r.window_name] = r.test_f1

                if results_by_window:
                    f1_2s = results_by_window.get('2s', 0)
                    f1_3s = results_by_window.get('3s', 0)
                    f1_4s = results_by_window.get('4s', 0)
                    f1_def = results_by_window.get('default', 0)

                    best_window = max(results_by_window.items(), key=lambda x: x[1])

                    report_lines.append(
                        f"| {arch} | {input_type} | "
                        f"{f1_2s:.1f}% | {f1_3s:.1f}% | {f1_4s:.1f}% | {f1_def:.1f}% | "
                        f"**{best_window[0]}** ({best_window[1]:.1f}%) |"
                    )

    # Cross-dataset comparison (best per architecture)
    report_lines.append("\n## Cross-Dataset Best Results")
    report_lines.append("\n### Best Configuration per Dataset")
    report_lines.append("| Dataset | Architecture | Input | Window | F1 (%) | Train Samples | Fall:ADL |")
    report_lines.append("|---------|--------------|-------|--------|--------|---------------|----------|")

    for dataset in DATASET_CONFIGS.keys():
        dataset_results = [r for r in completed if r.dataset == dataset]
        if dataset_results:
            best = max(dataset_results, key=lambda x: x.test_f1)
            report_lines.append(
                f"| {dataset} | {best.architecture} | {best.input_type} | {best.window_name} | "
                f"**{best.test_f1:.2f}%** ± {best.test_f1_std:.2f} | "
                f"{best.total_train_samples:,} | {best.adl_fall_ratio} |"
            )

    # Architecture ranking across datasets
    report_lines.append("\n### Architecture Ranking (Average F1 Across Datasets)")
    report_lines.append("| Rank | Architecture | Avg F1 (%) | Best Dataset |")
    report_lines.append("|------|--------------|------------|--------------|")

    arch_avg = defaultdict(list)
    for r in completed:
        arch_avg[r.architecture].append(r.test_f1)

    arch_ranking = [(arch, sum(scores)/len(scores)) for arch, scores in arch_avg.items()]
    arch_ranking.sort(key=lambda x: x[1], reverse=True)

    for rank, (arch, avg_f1) in enumerate(arch_ranking, 1):
        best_dataset = max(
            [r for r in completed if r.architecture == arch],
            key=lambda x: x.test_f1
        ).dataset
        report_lines.append(f"| {rank} | {arch} | {avg_f1:.2f}% | {best_dataset} |")

    # Kalman vs Raw overall
    report_lines.append("\n### Kalman vs Raw Overall")

    kalman_wins = 0
    raw_wins = 0
    ties = 0

    for dataset in DATASET_CONFIGS.keys():
        for arch in ARCHITECTURES.keys():
            for window in DATASET_CONFIGS[dataset]['window_sizes'].keys():
                kalman_r = next((r for r in completed
                                if r.dataset == dataset and r.architecture == arch
                                and r.input_type == 'kalman' and r.window_name == window), None)
                raw_r = next((r for r in completed
                             if r.dataset == dataset and r.architecture == arch
                             and r.input_type == 'raw' and r.window_name == window), None)

                if kalman_r and raw_r:
                    if kalman_r.test_f1 > raw_r.test_f1:
                        kalman_wins += 1
                    elif raw_r.test_f1 > kalman_r.test_f1:
                        raw_wins += 1
                    else:
                        ties += 1

    total_comparisons = kalman_wins + raw_wins + ties
    report_lines.append(f"- Total comparisons: {total_comparisons}")
    report_lines.append(f"- Kalman wins: {kalman_wins} ({kalman_wins/total_comparisons*100:.1f}%)" if total_comparisons > 0 else "- Kalman wins: 0")
    report_lines.append(f"- Raw wins: {raw_wins} ({raw_wins/total_comparisons*100:.1f}%)" if total_comparisons > 0 else "- Raw wins: 0")
    report_lines.append(f"- Ties: {ties}")

    # Data statistics summary
    report_lines.append("\n## Data Statistics Summary")
    report_lines.append("| Dataset | Window | Samples | Falls | ADLs | Fall % | ADL:Fall |")
    report_lines.append("|---------|--------|---------|-------|------|--------|----------|")

    for dataset in DATASET_CONFIGS.keys():
        for r in completed:
            if r.dataset == dataset and r.input_type == 'kalman' and r.total_train_samples > 0:
                # Only show once per dataset/window combo
                key = (dataset, r.window_name)
                report_lines.append(
                    f"| {dataset} | {r.window_name} ({r.window_duration:.1f}s) | "
                    f"{r.total_train_samples:,} | {r.train_falls:,} | {r.train_adls:,} | "
                    f"{r.fall_ratio:.1f}% | {r.adl_fall_ratio} |"
                )
                break  # Only one entry per dataset for this summary

    # Failed experiments
    if failed:
        report_lines.append("\n## Failed Experiments")
        for r in failed:
            report_lines.append(f"- **{r.name}**: {r.status} - {r.error_message[:100]}")

    # Configuration details
    report_lines.append("\n## Configuration Details")
    report_lines.append("\n### Architectures")
    for arch, info in ARCHITECTURES.items():
        report_lines.append(f"- **{arch}**: {info['description']}")
        report_lines.append(f"  - Model: `{info['model']}`")

    report_lines.append("\n### Datasets")
    for dataset, info in DATASET_CONFIGS.items():
        report_lines.append(f"- **{dataset}**: {info['description']}")
        windows_str = ', '.join([f"{k}={v}" for k, v in info['window_sizes'].items()])
        report_lines.append(f"  - Windows: {windows_str}")

    report = '\n'.join(report_lines)

    # Save report
    report_path = output_dir / 'architecture_ablation_report.md'
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\nReport saved to: {report_path}")

    return report


def save_results(
    results: List[ExperimentResult],
    experiments: List[ExperimentConfig],
    output_dir: Path,
):
    """Save all results to JSON."""
    # Convert to serializable format
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'experiments': [asdict(e) for e in experiments],
        'results': [asdict(r) for r in results],
        'summary': {
            'total': len(results),
            'completed': sum(1 for r in results if r.status == 'completed'),
            'failed': sum(1 for r in results if r.status != 'completed'),
        }
    }

    # Remove fold_results for compact JSON (too large)
    for r in results_data['results']:
        r['fold_results'] = f"[{len(r.get('fold_results', []))} folds]"

    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)

    print(f"Results saved to: {results_path}")


def print_summary_table(results: List[ExperimentResult]):
    """Print formatted summary table to console."""
    print("\n" + "="*120)
    print("ARCHITECTURE ABLATION RESULTS SUMMARY")
    print("="*120)

    completed = [r for r in results if r.status == 'completed']

    if not completed:
        print("No completed experiments.")
        return

    # Group by dataset
    for dataset in DATASET_CONFIGS.keys():
        dataset_results = [r for r in completed if r.dataset == dataset]
        if not dataset_results:
            continue

        print(f"\n{dataset.upper()}")
        print("-"*110)
        print(f"{'Architecture':<20} {'Input':<8} {'Window':<8} {'F1 (%)':<15} {'Acc (%)':<10} {'Samples':<12} {'Falls':<8} {'ADLs':<10}")
        print("-"*110)

        # Sort by F1
        dataset_results.sort(key=lambda x: (x.window_name, -x.test_f1))

        for r in dataset_results:
            f1_str = f"{r.test_f1:.2f} ± {r.test_f1_std:.2f}"
            print(f"{r.architecture:<20} {r.input_type:<8} {r.window_name:<8} {f1_str:<15} {r.test_accuracy:<10.2f} {r.total_train_samples:<12,} {r.train_falls:<8,} {r.train_adls:<10,}")

    print("\n" + "="*120)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive architecture ablation study',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=['all', 'upfall', 'wedafall'],
        default=['all'],
        help='Datasets to test (default: all = upfall + wedafall)'
    )
    parser.add_argument(
        '--architectures',
        nargs='+',
        choices=['all', 'lstm', 'baseline_transformer', 'deep_cnn_transformer', 'mamba'],
        default=['all'],
        help='Architectures to test (default: all)'
    )
    parser.add_argument(
        '--input-types',
        nargs='+',
        choices=['all', 'kalman', 'raw'],
        default=['all'],
        help='Input types to test (default: all)'
    )
    parser.add_argument(
        '--window-sizes',
        nargs='+',
        choices=['all', '2s', '3s', '4s', 'default', 'default_only'],
        default=['all'],
        help='Window sizes to test (default: all)'
    )
    parser.add_argument(
        '--num-gpus', type=int, default=4,
        help='Total GPUs available (default: 4)'
    )
    parser.add_argument(
        '--parallel', type=int, default=1,
        help='Number of parallel experiments (default: 1)'
    )
    parser.add_argument(
        '--max-folds', type=int, default=None,
        help='Maximum folds per experiment (default: all)'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick mode: 2 folds per experiment'
    )
    parser.add_argument(
        '--work-dir', type=str, default=None,
        help='Output directory (default: exps/architecture_ablation_{timestamp})'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Resume from existing work directory'
    )

    args = parser.parse_args()

    # Resolve 'all' selections
    datasets = list(DATASET_CONFIGS.keys()) if 'all' in args.datasets else args.datasets
    architectures = list(ARCHITECTURES.keys()) if 'all' in args.architectures else args.architectures
    input_types = list(INPUT_TYPES.keys()) if 'all' in args.input_types else args.input_types
    window_sizes = args.window_sizes  # Handled in create_experiment_configs

    # Setup output directory
    if args.resume:
        output_dir = Path(args.resume)
    elif args.work_dir:
        output_dir = Path(args.work_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(f'exps/architecture_ablation_{timestamp}')

    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate total experiments
    total_windows = 0
    for ds in datasets:
        if 'all' in window_sizes:
            total_windows += len(DATASET_CONFIGS[ds]['window_sizes'])
        elif 'default_only' in window_sizes:
            total_windows += 1
        else:
            total_windows += len([w for w in window_sizes if w in DATASET_CONFIGS[ds]['window_sizes']])

    total_exp = len(architectures) * len(input_types) * total_windows

    print(f"\n{'#'*70}")
    print(f"# Architecture Ablation Study")
    print(f"# Output: {output_dir}")
    print(f"# Datasets: {datasets}")
    print(f"# Architectures: {architectures}")
    print(f"# Input types: {input_types}")
    print(f"# Window sizes: {window_sizes}")
    print(f"# Total experiments: ~{total_exp}")
    print(f"{'#'*70}\n")

    # Create experiment configurations
    experiments = create_experiment_configs(output_dir, datasets, architectures, input_types, window_sizes)

    print(f"Created {len(experiments)} experiment configurations")

    # Run experiments
    results = run_experiments_parallel(
        experiments,
        num_gpus=args.num_gpus,
        parallel=args.parallel,
        max_folds=args.max_folds,
        quick_mode=args.quick,
    )

    # Save results
    save_results(results, experiments, output_dir)

    # Generate report
    report = generate_report(results, output_dir, experiments)

    # Print summary
    print_summary_table(results)

    # Print report to console
    print("\n" + "="*70)
    print("FULL REPORT")
    print("="*70)
    print(report)


if __name__ == '__main__':
    main()
