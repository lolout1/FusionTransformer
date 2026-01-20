#!/usr/bin/env python3
"""
Single-Stream vs Dual-Stream Architecture Comparison.

Compares different architectures (Transformer, CNN, LSTM, Mamba) with Kalman vs Raw inputs
across multiple fall detection datasets (SmartFallMM, UP-FALL, WEDA-FALL).

Uses validated best_config/*.yaml as base configurations to ensure proper preprocessing.

Usage:
    python distributed_dataset_pipeline/run_stream_architecture_comparison.py --num-gpus 8 --parallel
    python distributed_dataset_pipeline/run_stream_architecture_comparison.py --num-gpus 4 --datasets smartfallmm
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False


# =============================================================================
# Logging
# =============================================================================

def setup_logging(output_dir: Path, verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger('stream_comparison')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(output_dir / 'comparison.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)

    return logger


# =============================================================================
# Dataset Configuration - References to validated best_config files
# =============================================================================

DATASET_CONFIG = {
    'smartfallmm': {
        'name': 'SmartFallMM',
        'base_config_kalman': 'best_config/smartfallmm/kalman.yaml',
        'base_config_raw': 'best_config/smartfallmm/raw.yaml',
        'sampling_rate': 32,
        'num_folds': 22,
    },
    'upfall': {
        'name': 'UP-FALL',
        'base_config_kalman': 'best_config/upfall/kalman.yaml',
        'base_config_raw': 'best_config/upfall/raw.yaml',
        'sampling_rate': 18,
        'num_folds': 15,
    },
    'wedafall': {
        'name': 'WEDA-FALL',
        'base_config_kalman': 'best_config/wedafall/kalman.yaml',
        'base_config_raw': 'best_config/wedafall/raw.yaml',
        'sampling_rate': 50,
        'num_folds': 12,
    },
}


# =============================================================================
# Architecture Configurations
# =============================================================================

ARCHITECTURE_CONFIGS = {
    # Transformer Single-Stream
    'trans_single_kalman': {
        'name': 'Trans-Single-Kalman',
        'model': 'Models.kalman_transformer_variants.KalmanSingleStream',
        'category': 'transformer',
        'stream': 'single',
        'kalman': True,
    },
    'trans_single_raw': {
        'name': 'Trans-Single-Raw',
        'model': 'Models.kalman_transformer_variants.KalmanSingleStream',
        'category': 'transformer',
        'stream': 'single',
        'kalman': False,
    },
    # Transformer Dual-Stream (best model)
    'trans_dual_kalman': {
        'name': 'Trans-Dual-Kalman',
        'model': 'Models.encoder_ablation.KalmanConv1dLinear',
        'model_override': {
            'upfall': 'Models.encoder_ablation.KalmanConv1dConv1d',
            'wedafall': 'Models.encoder_ablation.KalmanConv1dConv1d',
        },
        'category': 'transformer',
        'stream': 'dual',
        'kalman': True,
    },
    'trans_dual_raw': {
        'name': 'Trans-Dual-Raw',
        'model': 'Models.dual_stream_baseline.DualStreamBaseline',
        'category': 'transformer',
        'stream': 'dual',
        'kalman': False,
    },
    # CNN
    'cnn_kalman': {
        'name': 'CNN-Kalman',
        'model': 'Models.dual_stream_cnn_lstm.DualStreamCNNKalman',
        'category': 'cnn',
        'stream': 'dual',
        'kalman': True,
    },
    'cnn_raw': {
        'name': 'CNN-Raw',
        'model': 'Models.dual_stream_cnn_lstm.DualStreamCNNRaw',
        'category': 'cnn',
        'stream': 'dual',
        'kalman': False,
    },
    # LSTM
    'lstm_kalman': {
        'name': 'LSTM-Kalman',
        'model': 'Models.dual_stream_cnn_lstm.DualStreamLSTMKalman',
        'category': 'lstm',
        'stream': 'dual',
        'kalman': True,
    },
    'lstm_raw': {
        'name': 'LSTM-Raw',
        'model': 'Models.dual_stream_cnn_lstm.DualStreamLSTMRaw',
        'category': 'lstm',
        'stream': 'dual',
        'kalman': False,
    },
    # Mamba
    'mamba_kalman': {
        'name': 'Mamba-Kalman',
        'model': 'Models.dual_stream_mamba.DualStreamMamba',
        'category': 'mamba',
        'stream': 'dual',
        'kalman': True,
    },
    'mamba_raw': {
        'name': 'Mamba-Raw',
        'model': 'Models.dual_stream_mamba.DualStreamMamba',
        'category': 'mamba',
        'stream': 'dual',
        'kalman': False,
    },
}


# =============================================================================
# Config Generation - Uses best_config as base
# =============================================================================

def load_base_config(dataset: str, use_kalman: bool) -> Dict:
    """Load base config from best_config/ directory."""
    ds_cfg = DATASET_CONFIG[dataset]

    if use_kalman:
        config_path = PROJECT_ROOT / ds_cfg['base_config_kalman']
    else:
        if ds_cfg['base_config_raw']:
            config_path = PROJECT_ROOT / ds_cfg['base_config_raw']
        else:
            # Derive raw config from kalman config
            config_path = PROJECT_ROOT / ds_cfg['base_config_kalman']

    if not config_path.exists():
        raise FileNotFoundError(f"Base config not found: {config_path}")

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_config(
    dataset: str,
    arch_key: str,
    embed_dim: int = 48,
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate experiment config by modifying base config."""
    arch_info = ARCHITECTURE_CONFIGS[arch_key]
    use_kalman = arch_info.get('kalman', True)

    # Load validated base config
    config = deepcopy(load_base_config(dataset, use_kalman))

    # Get model (check for dataset-specific override)
    model = arch_info.get('model_override', {}).get(dataset, arch_info['model'])
    config['model'] = model

    # Update model_args for this architecture
    # Kalman output: 7ch [smv, ax, ay, az, roll, pitch, yaw] -> acc_coords=4, gyro_coords=3
    # Raw output: 6ch [ax, ay, az, gx, gy, gz] -> acc_coords=3, gyro_coords=3
    if use_kalman:
        imu_channels = 7
        acc_coords = 4  # smv + ax + ay + az
        gyro_coords = 3  # roll + pitch + yaw (or gx, gy, gz)
    else:
        imu_channels = 6
        acc_coords = 3  # ax + ay + az
        gyro_coords = 3  # gx + gy + gz

    config['model_args']['imu_channels'] = imu_channels
    config['model_args']['acc_coords'] = acc_coords
    config['model_args']['gyro_coords'] = gyro_coords
    config['model_args']['embed_dim'] = embed_dim

    # Update dataset_args for kalman vs raw
    if not use_kalman:
        config['dataset_args']['enable_kalman_fusion'] = False
        config['dataset_args']['normalize_modalities'] = 'all'
        # For raw, we still need acc+gyro modalities
        if 'kalman_filter_type' in config['dataset_args']:
            del config['dataset_args']['kalman_filter_type']

    # Add cache config if specified
    if cache_dir:
        config['dataset_args']['cache_dir'] = cache_dir
        config['dataset_args']['use_cache'] = True

    return config


def generate_all_experiments(
    datasets: List[str],
    architectures: List[str],
    embed_dims: List[int],
    cache_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Generate all experiment configurations."""
    experiments = []

    for dataset in datasets:
        for arch_key in architectures:
            for embed_dim in embed_dims:
                try:
                    config = generate_config(dataset, arch_key, embed_dim, cache_dir)
                    arch_info = ARCHITECTURE_CONFIGS[arch_key]
                    exp_name = f"{dataset}_{arch_key}_ed{embed_dim}"

                    experiments.append({
                        'name': exp_name,
                        'dataset': dataset,
                        'architecture': arch_key,
                        'architecture_name': arch_info['name'],
                        'embed_dim': embed_dim,
                        'config': config,
                    })
                except Exception as e:
                    print(f"Warning: Failed to generate config for {dataset}/{arch_key}: {e}")

    return experiments


# =============================================================================
# Result Data Class
# =============================================================================

@dataclass
class ExperimentResult:
    name: str
    dataset: str
    architecture: str
    embed_dim: int
    # Test metrics
    test_f1: float = 0.0
    test_f1_std: float = 0.0
    test_acc: float = 0.0
    test_acc_std: float = 0.0
    test_precision: float = 0.0
    test_precision_std: float = 0.0
    test_recall: float = 0.0
    test_recall_std: float = 0.0
    test_auc: float = 0.0
    test_auc_std: float = 0.0
    test_loss: float = 0.0
    test_loss_std: float = 0.0
    test_macro_f1: float = 0.0
    test_macro_f1_std: float = 0.0
    # Validation metrics
    val_f1: float = 0.0
    val_acc: float = 0.0
    # Per-fold data
    fold_f1s: List[float] = field(default_factory=list)
    fold_accs: List[float] = field(default_factory=list)
    fold_precisions: List[float] = field(default_factory=list)
    fold_recalls: List[float] = field(default_factory=list)
    fold_aucs: List[float] = field(default_factory=list)
    # Status
    status: str = 'pending'
    error: str = ''
    time_sec: float = 0.0
    num_folds: int = 0


# =============================================================================
# Experiment Execution
# =============================================================================

def save_config(config: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def parse_summary_report(summary_path: Path) -> Dict:
    """Parse summary report text file for metrics."""
    results = {}
    if not summary_path.exists():
        return {'error': 'Summary file not found'}

    content = summary_path.read_text()

    patterns = {
        'test_f1': r'Test F1:\s+([\d.]+)\s*±\s*([\d.]+)%',
        'test_macro_f1': r'Test Macro-F1:\s+([\d.]+)\s*±\s*([\d.]+)%',
        'test_acc': r'Test Accuracy:\s+([\d.]+)\s*±\s*([\d.]+)%',
        'val_f1': r'Val F1:\s+([\d.]+)%',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            results[key] = float(match.group(1))
            if len(match.groups()) > 1:
                results[f'{key}_std'] = float(match.group(2))

    return results


def parse_fold_results(work_dir: Path) -> Dict:
    """Parse fold_results.pkl for comprehensive metrics."""
    import pickle
    import numpy as np

    pkl_path = work_dir / 'fold_results.pkl'
    if not pkl_path.exists():
        return {}

    try:
        with open(pkl_path, 'rb') as f:
            fold_data = pickle.load(f)

        if not fold_data or not isinstance(fold_data, list):
            return {}

        # Extract metrics from each fold
        metrics = {
            'f1': [], 'acc': [], 'precision': [], 'recall': [],
            'auc': [], 'loss': [], 'macro_f1': []
        }
        val_metrics = {'f1': [], 'acc': []}

        for fold in fold_data:
            if fold.get('status') != 'success':
                continue

            test = fold.get('test', {})
            val = fold.get('val', {})

            if 'f1_score' in test:
                metrics['f1'].append(test['f1_score'])
            if 'accuracy' in test:
                metrics['acc'].append(test['accuracy'])
            if 'precision' in test:
                metrics['precision'].append(test['precision'])
            if 'recall' in test:
                metrics['recall'].append(test['recall'])
            if 'auc' in test:
                metrics['auc'].append(test['auc'])
            if 'loss' in test:
                metrics['loss'].append(test['loss'])
            if 'macro_f1' in test:
                metrics['macro_f1'].append(test['macro_f1'])

            if 'f1_score' in val:
                val_metrics['f1'].append(val['f1_score'])
            if 'accuracy' in val:
                val_metrics['acc'].append(val['accuracy'])

        results = {'num_folds': len(fold_data)}

        # Calculate mean and std for each metric
        for key, values in metrics.items():
            if values:
                results[f'test_{key}'] = np.mean(values)
                results[f'test_{key}_std'] = np.std(values)
                results[f'fold_{key}s'] = values

        for key, values in val_metrics.items():
            if values:
                results[f'val_{key}'] = np.mean(values)

        return results

    except Exception as e:
        return {'parse_error': str(e)}


def run_experiment(
    exp_info: Dict,
    output_dir: Path,
    num_gpus: int,
    gpu_offset: int = 0,
    max_folds: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> ExperimentResult:
    exp_name = exp_info['name']
    work_dir = output_dir / exp_name
    config_path = output_dir / 'configs' / f'{exp_name}.yaml'

    save_config(exp_info['config'], config_path)

    env = os.environ.copy()
    if gpu_offset > 0:
        visible_gpus = ','.join(str(i) for i in range(gpu_offset, gpu_offset + num_gpus))
        env['CUDA_VISIBLE_DEVICES'] = visible_gpus

    cmd = [
        sys.executable, 'ray_train.py',
        '--config', str(config_path),
        '--num-gpus', str(num_gpus),
        '--work-dir', str(work_dir),
    ]

    if max_folds:
        cmd.extend(['--max-folds', str(max_folds)])

    start_time = time.time()

    if logger:
        logger.info(f"Starting: {exp_name} | GPUs: {gpu_offset}-{gpu_offset+num_gpus-1}")

    try:
        log_file = work_dir / 'train.log'
        work_dir.mkdir(parents=True, exist_ok=True)

        with open(log_file, 'w') as f:
            proc = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
            )

        elapsed = time.time() - start_time

        if proc.returncode != 0:
            return ExperimentResult(
                name=exp_name,
                dataset=exp_info['dataset'],
                architecture=exp_info['architecture'],
                embed_dim=exp_info['embed_dim'],
                status='failed',
                error=f'Exit code {proc.returncode}',
                time_sec=elapsed,
            )

        # Parse comprehensive metrics from fold_results.pkl
        metrics = parse_fold_results(work_dir)

        # Fallback to summary_report.txt if pkl not found
        if not metrics or 'test_f1' not in metrics:
            summary_path = work_dir / 'summary_report.txt'
            metrics = parse_summary_report(summary_path)

        if logger:
            f1 = metrics.get('test_f1', 0)
            acc = metrics.get('test_acc', 0)
            auc = metrics.get('test_auc', 0)
            logger.info(f"Completed: {exp_name} | F1: {f1:.2f}% | Acc: {acc:.2f}% | AUC: {auc:.2f}% | Time: {elapsed/60:.1f}min")

        return ExperimentResult(
            name=exp_name,
            dataset=exp_info['dataset'],
            architecture=exp_info['architecture'],
            embed_dim=exp_info['embed_dim'],
            # Test metrics
            test_f1=metrics.get('test_f1', 0),
            test_f1_std=metrics.get('test_f1_std', 0),
            test_acc=metrics.get('test_acc', 0),
            test_acc_std=metrics.get('test_acc_std', 0),
            test_precision=metrics.get('test_precision', 0),
            test_precision_std=metrics.get('test_precision_std', 0),
            test_recall=metrics.get('test_recall', 0),
            test_recall_std=metrics.get('test_recall_std', 0),
            test_auc=metrics.get('test_auc', 0),
            test_auc_std=metrics.get('test_auc_std', 0),
            test_loss=metrics.get('test_loss', 0),
            test_loss_std=metrics.get('test_loss_std', 0),
            test_macro_f1=metrics.get('test_macro_f1', 0),
            test_macro_f1_std=metrics.get('test_macro_f1_std', 0),
            # Validation metrics
            val_f1=metrics.get('val_f1', 0),
            val_acc=metrics.get('val_acc', 0),
            # Per-fold data
            fold_f1s=metrics.get('fold_f1s', []),
            fold_accs=metrics.get('fold_accs', []),
            fold_precisions=metrics.get('fold_precisions', []),
            fold_recalls=metrics.get('fold_recalls', []),
            fold_aucs=metrics.get('fold_aucs', []),
            # Status
            status='success',
            time_sec=elapsed,
            num_folds=metrics.get('num_folds', 0),
        )

    except Exception as e:
        elapsed = time.time() - start_time
        if logger:
            logger.error(f"Failed: {exp_name} | Error: {e}")
        return ExperimentResult(
            name=exp_name,
            dataset=exp_info['dataset'],
            architecture=exp_info['architecture'],
            embed_dim=exp_info['embed_dim'],
            status='failed',
            error=str(e),
            time_sec=elapsed,
        )


# =============================================================================
# GPU Allocation
# =============================================================================

@dataclass
class GPUSlot:
    slot_id: int
    gpu_start: int
    gpu_count: int
    in_use: bool = False


class GPUAllocator:
    """Flexible GPU allocation supporting asymmetric configurations."""

    def __init__(self, total_gpus: int, allocation: Optional[List[int]] = None):
        self.total_gpus = total_gpus
        self._lock = threading.Lock()

        if allocation:
            assert sum(allocation) <= total_gpus, f"Allocation {allocation} exceeds {total_gpus} GPUs"
            self.slots = []
            gpu_offset = 0
            for i, count in enumerate(allocation):
                self.slots.append(GPUSlot(slot_id=i, gpu_start=gpu_offset, gpu_count=count))
                gpu_offset += count
        else:
            gpus_per_slot = max(2, total_gpus // 4)
            num_slots = total_gpus // gpus_per_slot
            self.slots = [
                GPUSlot(slot_id=i, gpu_start=i * gpus_per_slot, gpu_count=gpus_per_slot)
                for i in range(num_slots)
            ]

    def acquire(self) -> Optional[GPUSlot]:
        with self._lock:
            for slot in self.slots:
                if not slot.in_use:
                    slot.in_use = True
                    return slot
            return None

    def release(self, slot: GPUSlot):
        with self._lock:
            slot.in_use = False

    @property
    def max_parallel(self) -> int:
        return len(self.slots)

    def __str__(self) -> str:
        parts = [f"{s.gpu_count}GPU" for s in self.slots]
        return f"GPUAllocator({'+'.join(parts)})"


def run_experiments_parallel(
    experiments: List[Dict],
    output_dir: Path,
    total_gpus: int,
    max_folds: Optional[int],
    logger: logging.Logger,
    gpu_allocation: Optional[List[int]] = None,
) -> List[ExperimentResult]:
    allocator = GPUAllocator(total_gpus, gpu_allocation)
    results = []

    logger.info(f"Running {len(experiments)} experiments | {allocator}")

    with ThreadPoolExecutor(max_workers=allocator.max_parallel) as executor:
        futures: Dict[Any, GPUSlot] = {}
        exp_queue = list(experiments)

        while exp_queue or futures:
            while exp_queue:
                slot = allocator.acquire()
                if slot is None:
                    break

                exp = exp_queue.pop(0)

                future = executor.submit(
                    run_experiment,
                    exp, output_dir, slot.gpu_count, slot.gpu_start, max_folds, logger
                )
                futures[future] = slot

            if futures:
                done_futures = [f for f in futures if f.done()]
                if not done_futures:
                    time.sleep(1)
                    continue

                for future in done_futures:
                    slot = futures.pop(future)
                    allocator.release(slot)

                    try:
                        result = future.result()
                        results.append(result)

                        with open(output_dir / 'results_partial.json', 'w') as f:
                            json.dump([r.__dict__ for r in results], f, indent=2)
                    except Exception as e:
                        logger.error(f"Slot {slot.slot_id} error: {e}")

    return results


def run_experiments_sequential(
    experiments: List[Dict],
    output_dir: Path,
    num_gpus: int,
    max_folds: Optional[int],
    logger: logging.Logger,
) -> List[ExperimentResult]:
    results = []

    for i, exp in enumerate(experiments):
        logger.info(f"[{i+1}/{len(experiments)}] {exp['name']}")
        result = run_experiment(exp, output_dir, num_gpus, 0, max_folds, logger)
        results.append(result)

        with open(output_dir / 'results_partial.json', 'w') as f:
            json.dump([r.__dict__ for r in results], f, indent=2)

    return results


# =============================================================================
# Visualization
# =============================================================================

def generate_visualizations(results: List[ExperimentResult], output_dir: Path, logger: logging.Logger):
    if not HAS_VIZ:
        logger.warning("matplotlib not available, skipping visualizations")
        return []

    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)
    paths = []

    datasets = sorted(set(r.dataset for r in results))

    # Bar chart per dataset
    for ds in datasets:
        ds_results = [r for r in results if r.dataset == ds and r.status == 'success']
        if not ds_results:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))

        arch_f1s = {}
        for r in ds_results:
            key = r.architecture
            if key not in arch_f1s:
                arch_f1s[key] = []
            arch_f1s[key].append((r.test_f1, r.test_f1_std))

        x = np.arange(len(arch_f1s))
        means = [np.mean([v[0] for v in arch_f1s[a]]) for a in arch_f1s]
        stds = [np.mean([v[1] for v in arch_f1s[a]]) for a in arch_f1s]
        labels = [ARCHITECTURE_CONFIGS[a]['name'] for a in arch_f1s]

        colors = plt.cm.Set2(np.linspace(0, 1, len(arch_f1s)))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor='black')

        ax.set_ylabel('Test F1 (%)', fontsize=12)
        ax.set_title(f'{DATASET_CONFIG[ds]["name"]} - Architecture Comparison', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)

        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{mean:.1f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        path = fig_dir / f'{ds}_bar_chart.png'
        plt.savefig(path, dpi=150)
        plt.close()
        paths.append(path)
        logger.info(f"Saved: {path}")

    # Cross-dataset comparison
    if len(datasets) > 1:
        fig, axes = plt.subplots(1, len(datasets), figsize=(6*len(datasets), 6), sharey=True)
        if len(datasets) == 1:
            axes = [axes]

        for ax, ds in zip(axes, datasets):
            ds_results = [r for r in results if r.dataset == ds and r.status == 'success']
            if not ds_results:
                continue

            archs = sorted(set(r.architecture for r in ds_results))
            f1s = []
            labels = []
            for a in archs:
                r = next((r for r in ds_results if r.architecture == a), None)
                if r:
                    f1s.append(r.test_f1)
                    labels.append(ARCHITECTURE_CONFIGS[a]['name'])

            y = np.arange(len(f1s))
            colors = ['#2ecc71' if 'Kalman' in l else '#e74c3c' for l in labels]
            ax.barh(y, f1s, color=colors, edgecolor='black')
            ax.set_yticks(y)
            ax.set_yticklabels(labels)
            ax.set_xlabel('Test F1 (%)')
            ax.set_title(DATASET_CONFIG[ds]['name'])
            ax.set_xlim(0, 100)

        plt.tight_layout()
        path = fig_dir / 'cross_dataset_comparison.png'
        plt.savefig(path, dpi=150)
        plt.close()
        paths.append(path)
        logger.info(f"Saved: {path}")

    return paths


def generate_report(results: List[ExperimentResult], output_dir: Path, logger: logging.Logger):
    """Generate comprehensive markdown report with all metrics."""
    report_path = output_dir / 'architecture_comparison_report.md'

    lines = [
        "# Architecture Comparison Results",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
    ]

    # Summary table with all metrics
    lines.append("## Summary\n")
    lines.append("| Dataset | Architecture | F1 | Acc | Precision | Recall | AUC | Status |")
    lines.append("|---------|--------------|-----|-----|-----------|--------|-----|--------|")

    for r in sorted(results, key=lambda x: (x.dataset, -x.test_f1)):
        arch_name = ARCHITECTURE_CONFIGS.get(r.architecture, {}).get('name', r.architecture)
        status = '✓' if r.status == 'success' else '✗'
        lines.append(
            f"| {r.dataset} | {arch_name} | "
            f"{r.test_f1:.1f}±{r.test_f1_std:.1f} | "
            f"{r.test_acc:.1f}±{r.test_acc_std:.1f} | "
            f"{r.test_precision:.1f}±{r.test_precision_std:.1f} | "
            f"{r.test_recall:.1f}±{r.test_recall_std:.1f} | "
            f"{r.test_auc:.1f}±{r.test_auc_std:.1f} | {status} |"
        )

    # Best per dataset with detailed metrics
    lines.append("\n## Best Results per Dataset\n")
    for ds in sorted(set(r.dataset for r in results)):
        ds_results = [r for r in results if r.dataset == ds and r.status == 'success']
        if ds_results:
            best = max(ds_results, key=lambda x: x.test_f1)
            arch_name = ARCHITECTURE_CONFIGS.get(best.architecture, {}).get('name', best.architecture)
            lines.append(f"### {DATASET_CONFIG[ds]['name']}: {arch_name}\n")
            lines.append(f"| Metric | Value |")
            lines.append(f"|--------|-------|")
            lines.append(f"| F1 Score | {best.test_f1:.2f}% ± {best.test_f1_std:.2f}% |")
            lines.append(f"| Accuracy | {best.test_acc:.2f}% ± {best.test_acc_std:.2f}% |")
            lines.append(f"| Precision | {best.test_precision:.2f}% ± {best.test_precision_std:.2f}% |")
            lines.append(f"| Recall | {best.test_recall:.2f}% ± {best.test_recall_std:.2f}% |")
            lines.append(f"| AUC | {best.test_auc:.2f}% ± {best.test_auc_std:.2f}% |")
            lines.append(f"| Folds | {best.num_folds} |")
            lines.append("")

    # Kalman vs Raw comparison
    lines.append("\n## Kalman vs Raw Comparison\n")
    lines.append("| Dataset | Architecture | Kalman F1 | Raw F1 | Δ F1 |")
    lines.append("|---------|--------------|-----------|--------|------|")

    for ds in sorted(set(r.dataset for r in results)):
        ds_results = [r for r in results if r.dataset == ds and r.status == 'success']
        arch_pairs = {}
        for r in ds_results:
            base_arch = r.architecture.replace('_kalman', '').replace('_raw', '')
            if base_arch not in arch_pairs:
                arch_pairs[base_arch] = {}
            if '_kalman' in r.architecture:
                arch_pairs[base_arch]['kalman'] = r
            elif '_raw' in r.architecture:
                arch_pairs[base_arch]['raw'] = r

        for base_arch, pair in sorted(arch_pairs.items()):
            if 'kalman' in pair and 'raw' in pair:
                k, r = pair['kalman'], pair['raw']
                delta = k.test_f1 - r.test_f1
                sign = '+' if delta > 0 else ''
                lines.append(f"| {ds} | {base_arch} | {k.test_f1:.1f}% | {r.test_f1:.1f}% | {sign}{delta:.1f}% |")

    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))

    logger.info(f"Report saved: {report_path}")

    # Also save comprehensive CSV
    csv_path = output_dir / 'all_metrics.csv'
    csv_lines = [
        "dataset,architecture,embed_dim,test_f1,test_f1_std,test_acc,test_acc_std,"
        "test_precision,test_precision_std,test_recall,test_recall_std,"
        "test_auc,test_auc_std,test_loss,test_loss_std,val_f1,val_acc,num_folds,status,time_sec"
    ]
    for r in sorted(results, key=lambda x: (x.dataset, x.architecture)):
        csv_lines.append(
            f"{r.dataset},{r.architecture},{r.embed_dim},"
            f"{r.test_f1:.4f},{r.test_f1_std:.4f},{r.test_acc:.4f},{r.test_acc_std:.4f},"
            f"{r.test_precision:.4f},{r.test_precision_std:.4f},{r.test_recall:.4f},{r.test_recall_std:.4f},"
            f"{r.test_auc:.4f},{r.test_auc_std:.4f},{r.test_loss:.6f},{r.test_loss_std:.6f},"
            f"{r.val_f1:.4f},{r.val_acc:.4f},{r.num_folds},{r.status},{r.time_sec:.1f}"
        )
    with open(csv_path, 'w') as f:
        f.write('\n'.join(csv_lines))
    logger.info(f"CSV saved: {csv_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Architecture Comparison Study')
    parser.add_argument('--num-gpus', type=int, default=8, help='Total GPUs available')
    parser.add_argument('--gpu-allocation', type=str, default=None,
                       help='GPU allocation per slot (e.g., "2,3,3" for 3 slots)')
    parser.add_argument('--datasets', type=str, default='smartfallmm',
                       help='Comma-separated datasets (smartfallmm,upfall,wedafall)')
    parser.add_argument('--architectures', type=str, default=None,
                       help='Comma-separated architectures (default: all)')
    parser.add_argument('--embed-dims', type=str, default='48',
                       help='Comma-separated embed dims (default: 48)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    parser.add_argument('--max-folds', type=int, default=None, help='Max folds per experiment')
    parser.add_argument('--parallel', action='store_true', help='Run experiments in parallel')
    parser.add_argument('--cache-dir', type=str, default=None, help='Preprocessing cache directory')
    parser.add_argument('--quick', action='store_true', help='Quick test (2 folds, 1 embed dim)')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    args = parser.parse_args()

    # Parse arguments
    datasets = [d.strip() for d in args.datasets.split(',')]
    embed_dims = [int(e.strip()) for e in args.embed_dims.split(',')]

    if args.architectures:
        architectures = [a.strip() for a in args.architectures.split(',')]
    else:
        architectures = list(ARCHITECTURE_CONFIGS.keys())

    if args.quick:
        args.max_folds = 2
        embed_dims = [48]

    gpu_allocation = None
    if args.gpu_allocation:
        gpu_allocation = [int(x) for x in args.gpu_allocation.split(',')]

    # Setup output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / 'exps' / f'stream_comparison_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir, args.verbose)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Datasets: {datasets}")
    logger.info(f"Architectures: {architectures}")
    logger.info(f"Embed dims: {embed_dims}")

    # Generate experiments
    experiments = generate_all_experiments(
        datasets=datasets,
        architectures=architectures,
        embed_dims=embed_dims,
        cache_dir=args.cache_dir,
    )

    logger.info(f"Generated {len(experiments)} experiments")

    # Save experiment plan
    with open(output_dir / 'experiment_plan.json', 'w') as f:
        json.dump({
            'datasets': datasets,
            'architectures': architectures,
            'embed_dims': embed_dims,
            'num_experiments': len(experiments),
        }, f, indent=2)

    # Run experiments
    if args.parallel and args.num_gpus >= 4:
        results = run_experiments_parallel(
            experiments, output_dir, args.num_gpus, args.max_folds, logger, gpu_allocation
        )
    else:
        results = run_experiments_sequential(
            experiments, output_dir, args.num_gpus, args.max_folds, logger
        )

    # Save results
    with open(output_dir / 'results.json', 'w') as f:
        json.dump([r.__dict__ for r in results], f, indent=2)

    # Generate visualizations and report
    generate_visualizations(results, output_dir, logger)
    generate_report(results, output_dir, logger)

    # Print summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for ds in datasets:
        ds_results = [r for r in results if r.dataset == ds and r.status == 'success']
        if ds_results:
            best = max(ds_results, key=lambda x: x.test_f1)
            arch_name = ARCHITECTURE_CONFIGS.get(best.architecture, {}).get('name', best.architecture)
            logger.info(f"{DATASET_CONFIG[ds]['name']}: Best = {arch_name} ({best.test_f1:.2f}%)")

    successful = sum(1 for r in results if r.status == 'success')
    failed = sum(1 for r in results if r.status == 'failed')
    logger.info(f"Completed: {successful}/{len(results)} experiments ({failed} failed)")


if __name__ == '__main__':
    main()
