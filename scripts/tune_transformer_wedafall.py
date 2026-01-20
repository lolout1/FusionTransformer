#!/usr/bin/env python3
"""
Transformer Tuning for WEDA-FALL.

Goal: Make transformer competitive with LSTM/CNN on WEDA-FALL.

Key hypotheses:
1. Small dataset (14 young subjects) limits transformer learning
2. Kalman params tuned for SmartFallMM (32Hz), not WEDA-FALL (50Hz)
3. Need more regularization for small data

Experiments:
- Include elderly subjects (+11 ADL-only subjects)
- Tune embed_dim (48, 64, 96)
- Tune dropout (0.3, 0.5, 0.7)
- Tune Kalman Q/R parameters for 50Hz

Usage:
    python scripts/tune_transformer_wedafall.py --num-gpus 4
    python scripts/tune_transformer_wedafall.py --quick --num-gpus 2
"""

import argparse
import json
import os
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Experiment Configurations
# =============================================================================

# Base config template
BASE_CONFIG = {
    'model': 'Models.encoder_ablation.KalmanConv1dConv1d',
    'dataset': 'wedafall',
    'subjects': [1,2,3,4,5,6,7,8,9,10,11,12,13,14],
    'validation_subjects': [13, 14],
    'model_args': {
        'imu_frames': 192,
        'imu_channels': 7,
        'acc_frames': 192,
        'acc_coords': 7,
        'num_classes': 1,
        'num_heads': 4,
        'num_layers': 2,
        'embed_dim': 48,
        'dropout': 0.5,
        'activation': 'relu',
        'norm_first': True,
        'se_reduction': 4,
        'acc_ratio': 0.65,
    },
    'dataset_args': {
        'base_path': 'other_datasets/WEDA-FALL/dataset',
        'frequency': '50Hz',
        'mode': 'sliding_window',
        'max_length': 192,
        'task': 'fd',
        'stride': 32,
        'loss_type': 'focal',
        'modalities': ['accelerometer', 'gyroscope'],
        'sensors': ['wrist'],
        'age_group': ['young'],
        'use_skeleton': False,
        'include_elderly': False,
        'enable_normalization': True,
        'normalize_modalities': 'acc_only',
        'enable_class_aware_stride': True,
        'fall_stride': 8,
        'adl_stride': 32,
        'convert_gyro_to_rad': True,
        'enable_kalman_fusion': True,
        'kalman_filter_type': 'linear',
        'kalman_output_format': 'euler',
        'kalman_include_smv': True,
        'kalman_include_uncertainty': False,
        'kalman_include_innovation': False,
        'filter_fs': 50.0,
        'kalman_Q_orientation': 0.0124,
        'kalman_Q_rate': 0.1315,
        'kalman_R_acc': 0.2395,
        'kalman_R_gyro': 0.2822,
    },
    'batch_size': 64,
    'test_batch_size': 64,
    'val_batch_size': 64,
    'num_epoch': 80,
    'feeder': 'Feeder.external_datasets.ExternalFallDataset',
    'train_feeder_args': {'batch_size': 64},
    'val_feeder_args': {'batch_size': 64},
    'test_feeder_args': {'batch_size': 64},
    'seed': 2,
    'optimizer': 'adamw',
    'base_lr': 0.001,
    'weight_decay': 0.001,
}

# Experiment variations
EXPERIMENTS = {
    # Baseline: current best config (young only)
    'baseline_young': {
        'include_elderly': False,
        'embed_dim': 48,
        'dropout': 0.5,
    },

    # Add elderly subjects (more training data)
    'with_elderly': {
        'include_elderly': True,
        'embed_dim': 48,
        'dropout': 0.5,
    },

    # Larger model capacity
    'elderly_embed64': {
        'include_elderly': True,
        'embed_dim': 64,
        'dropout': 0.5,
    },

    # Even larger
    'elderly_embed96': {
        'include_elderly': True,
        'embed_dim': 96,
        'dropout': 0.5,
    },

    # More regularization
    'elderly_dropout07': {
        'include_elderly': True,
        'embed_dim': 64,
        'dropout': 0.7,
    },

    # Less regularization
    'elderly_dropout03': {
        'include_elderly': True,
        'embed_dim': 64,
        'dropout': 0.3,
    },

    # More transformer layers
    'elderly_3layers': {
        'include_elderly': True,
        'embed_dim': 64,
        'dropout': 0.5,
        'num_layers': 3,
    },

    # Tuned Kalman for 50Hz (lower process noise)
    'elderly_kalman_tuned': {
        'include_elderly': True,
        'embed_dim': 64,
        'dropout': 0.5,
        'kalman_Q_orientation': 0.005,  # Lower - trust model more
        'kalman_Q_rate': 0.05,
        'kalman_R_acc': 0.1,  # Lower - trust measurements more
        'kalman_R_gyro': 0.15,
    },

    # Combined: best settings
    'elderly_optimal': {
        'include_elderly': True,
        'embed_dim': 64,
        'dropout': 0.5,
        'num_layers': 2,
        'kalman_Q_orientation': 0.008,
        'kalman_Q_rate': 0.08,
        'kalman_R_acc': 0.15,
        'kalman_R_gyro': 0.2,
    },
}


def create_config(exp_name: str, exp_params: dict) -> dict:
    """Create config from base + experiment params."""
    config = deepcopy(BASE_CONFIG)

    # Apply experiment parameters
    if 'include_elderly' in exp_params:
        config['dataset_args']['include_elderly'] = exp_params['include_elderly']

    if 'embed_dim' in exp_params:
        config['model_args']['embed_dim'] = exp_params['embed_dim']

    if 'dropout' in exp_params:
        config['model_args']['dropout'] = exp_params['dropout']

    if 'num_layers' in exp_params:
        config['model_args']['num_layers'] = exp_params['num_layers']

    # Kalman parameters
    for key in ['kalman_Q_orientation', 'kalman_Q_rate', 'kalman_R_acc', 'kalman_R_gyro']:
        if key in exp_params:
            config['dataset_args'][key] = exp_params[key]

    return config


def save_config(config: dict, path: Path):
    """Save config to YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def run_experiment(exp_name: str, config_path: Path, work_dir: Path,
                   num_gpus: int, max_folds: int = None) -> dict:
    """Run single experiment."""
    work_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, 'ray_train.py',
        '--config', str(config_path),
        '--num-gpus', str(num_gpus),
        '--work-dir', str(work_dir),
    ]

    if max_folds:
        cmd.extend(['--max-folds', str(max_folds)])

    print(f"\n{'='*60}")
    print(f"Running: {exp_name}")
    print(f"{'='*60}\n")

    log_file = work_dir / 'train.log'
    with open(log_file, 'w') as f:
        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=f,
            stderr=subprocess.STDOUT,
        )

    # Parse results
    result = {
        'name': exp_name,
        'status': 'success' if proc.returncode == 0 else 'failed',
    }

    # Try to parse summary
    summary_path = work_dir / 'summary_report.txt'
    if summary_path.exists():
        import re
        content = summary_path.read_text()

        f1_match = re.search(r'Test F1:\s+([\d.]+)\s*±\s*([\d.]+)%', content)
        if f1_match:
            result['test_f1'] = float(f1_match.group(1))
            result['test_f1_std'] = float(f1_match.group(2))

        acc_match = re.search(r'Test Accuracy:\s+([\d.]+)\s*±\s*([\d.]+)%', content)
        if acc_match:
            result['test_acc'] = float(acc_match.group(1))
            result['test_acc_std'] = float(acc_match.group(2))

        prec_match = re.search(r'Precision:\s+([\d.]+)\s*±\s*([\d.]+)%', content)
        if prec_match:
            result['precision'] = float(prec_match.group(1))
            result['precision_std'] = float(prec_match.group(2))

        recall_match = re.search(r'Recall:\s+([\d.]+)\s*±\s*([\d.]+)%', content)
        if recall_match:
            result['recall'] = float(recall_match.group(1))
            result['recall_std'] = float(recall_match.group(2))

        auc_match = re.search(r'AUC:\s+([\d.]+)\s*±\s*([\d.]+)%', content)
        if auc_match:
            result['auc'] = float(auc_match.group(1))
            result['auc_std'] = float(auc_match.group(2))

    return result


def generate_report(results: list, output_dir: Path, experiments: dict):
    """Generate markdown report."""
    report_path = output_dir / 'transformer_tuning_report.md'

    lines = [
        "# Transformer Tuning Results for WEDA-FALL",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "## Goal",
        "Make transformer competitive with LSTM-Raw (94.0% F1) on WEDA-FALL.\n",
        "## Hypotheses Tested",
        "1. Small dataset limits transformer - add elderly subjects",
        "2. Model capacity - tune embed_dim (48 → 64 → 96)",
        "3. Regularization - tune dropout (0.3, 0.5, 0.7)",
        "4. Kalman parameters - tune for 50Hz sampling rate",
        "",
        "## Results",
        "",
        "| Experiment | Elderly | Embed | Dropout | Layers | F1 | Std | vs Baseline |",
        "|------------|---------|-------|---------|--------|-----|-----|-------------|",
    ]

    baseline_f1 = None
    for r in results:
        if r['name'] == 'baseline_young' and 'test_f1' in r:
            baseline_f1 = r['test_f1']
            break

    for r in sorted(results, key=lambda x: x.get('test_f1', 0), reverse=True):
        if r['status'] == 'success' and 'test_f1' in r:
            exp_params = experiments.get(r['name'], {})
            elderly = "Yes" if exp_params.get('include_elderly', False) else "No"
            embed = exp_params.get('embed_dim', 48)
            dropout = exp_params.get('dropout', 0.5)
            layers = exp_params.get('num_layers', 2)

            delta = ""
            if baseline_f1 and r['name'] != 'baseline_young':
                diff = r['test_f1'] - baseline_f1
                delta = f"+{diff:.1f}%" if diff > 0 else f"{diff:.1f}%"

            lines.append(
                f"| {r['name']} | {elderly} | {embed} | {dropout} | {layers} | "
                f"{r['test_f1']:.2f}% | ±{r.get('test_f1_std', 0):.2f} | {delta} |"
            )
        else:
            lines.append(f"| {r['name']} | - | - | - | - | FAILED | - | - |")

    # Best result
    successful = [r for r in results if r['status'] == 'success' and 'test_f1' in r]
    if successful:
        best = max(successful, key=lambda x: x['test_f1'])
        lines.extend([
            "",
            "## Best Configuration",
            f"**{best['name']}**: {best['test_f1']:.2f}% ± {best.get('test_f1_std', 0):.2f}%",
            "",
            "### Full Metrics",
            f"- F1: {best['test_f1']:.2f}% ± {best.get('test_f1_std', 0):.2f}%",
            f"- Accuracy: {best.get('test_acc', 0):.2f}% ± {best.get('test_acc_std', 0):.2f}%",
            f"- Precision: {best.get('precision', 0):.2f}% ± {best.get('precision_std', 0):.2f}%",
            f"- Recall: {best.get('recall', 0):.2f}% ± {best.get('recall_std', 0):.2f}%",
            f"- AUC: {best.get('auc', 0):.2f}% ± {best.get('auc_std', 0):.2f}%",
        ])

        # Compare to LSTM-Raw baseline
        lstm_f1 = 94.0  # From previous results
        diff = best['test_f1'] - lstm_f1
        lines.extend([
            "",
            "## Comparison to LSTM-Raw",
            f"- LSTM-Raw (baseline): 94.0% F1",
            f"- Best Transformer: {best['test_f1']:.2f}% F1",
            f"- Difference: {diff:+.2f}%",
        ])

    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\nReport saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Transformer Tuning for WEDA-FALL')
    parser.add_argument('--num-gpus', type=int, default=4, help='Number of GPUs')
    parser.add_argument('--quick', action='store_true', help='Quick test (2 folds)')
    parser.add_argument('--experiments', type=str, default=None,
                       help='Comma-separated experiment names to run')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    args = parser.parse_args()

    # Setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / 'exps' / f'transformer_tuning_wedafall_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine experiments to run
    if args.experiments:
        exp_names = [e.strip() for e in args.experiments.split(',')]
        experiments_to_run = {k: v for k, v in EXPERIMENTS.items() if k in exp_names}
    else:
        experiments_to_run = EXPERIMENTS

    max_folds = 2 if args.quick else None

    print(f"Output: {output_dir}")
    print(f"GPUs: {args.num_gpus}")
    print(f"Experiments: {list(experiments_to_run.keys())}")
    print(f"Quick mode: {args.quick}")

    # Generate configs and run experiments
    results = []
    total = len(experiments_to_run)

    for i, (exp_name, exp_params) in enumerate(experiments_to_run.items()):
        print(f"\n[{i+1}/{total}] {exp_name}")

        # Create config
        config = create_config(exp_name, exp_params)
        config_path = output_dir / 'configs' / f'{exp_name}.yaml'
        save_config(config, config_path)

        # Run experiment
        work_dir = output_dir / exp_name
        result = run_experiment(exp_name, config_path, work_dir, args.num_gpus, max_folds)
        result['params'] = exp_params
        results.append(result)

        # Save partial results
        with open(output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)

    # Generate report
    generate_report(results, output_dir, experiments_to_run)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in sorted(results, key=lambda x: x.get('test_f1', 0), reverse=True):
        if r['status'] == 'success' and 'test_f1' in r:
            print(f"{r['name']}: {r['test_f1']:.2f}% ± {r.get('test_f1_std', 0):.2f}%")
        else:
            print(f"{r['name']}: FAILED")


if __name__ == '__main__':
    main()
