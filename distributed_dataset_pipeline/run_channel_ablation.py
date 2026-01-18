#!/usr/bin/env python3
"""
Channel Ablation Study for Kalman Features.

Tests different input channel configurations on UP-FALL and WEDA-FALL:
1. Orientation only (2ch): roll, pitch
2. Orientation only (3ch): roll, pitch, yaw
3. Hybrid (9ch): smv + acc + roll/pitch + raw gyro
4. Asymmetric embedding: 48:16 vs 32:32 (acc:ori ratio)
5. Baseline Kalman (7ch): reference comparison

Runs experiments sequentially to avoid Ray conflicts.
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent

# Dataset metadata
DATASET_INFO = {
    'upfall': {
        'name': 'UP-FALL',
        'num_folds': 15,
        'kalman_params': {
            'kalman_Q_orientation': 0.0115,
            'kalman_Q_rate': 0.0257,
            'kalman_R_acc': 0.1312,
            'kalman_R_gyro': 0.1074,
        }
    },
    'wedafall': {
        'name': 'WEDA-FALL',
        'num_folds': 12,
        'kalman_params': {
            'kalman_Q_orientation': 0.0124,
            'kalman_Q_rate': 0.1315,
            'kalman_R_acc': 0.2395,
            'kalman_R_gyro': 0.2822,
        }
    }
}

# Ablation configurations
ABLATION_CONFIGS = {
    'upfall': [
        {
            'name': 'orientation_2ch',
            'config': 'config/upfall/channel_ablation/orientation_2ch.yaml',
            'description': 'Orientation only (roll, pitch)',
            'channels': 2,
        },
        {
            'name': 'orientation_3ch',
            'config': 'config/upfall/channel_ablation/orientation_3ch.yaml',
            'description': 'Orientation only (roll, pitch, yaw)',
            'channels': 3,
        },
        {
            'name': 'hybrid_9ch',
            'config': 'config/upfall/channel_ablation/hybrid_9ch.yaml',
            'description': 'Hybrid: smv + acc + roll/pitch + raw gyro',
            'channels': 9,
        },
        {
            'name': 'kalman_balanced_32_32',
            'config': 'config/upfall/channel_ablation/kalman_balanced_32_32.yaml',
            'description': 'Kalman 7ch with balanced 32:32 embedding',
            'channels': 7,
        },
        {
            'name': 'kalman_asymmetric_48_16',
            'config': 'config/upfall/channel_ablation/kalman_asymmetric_48_16.yaml',
            'description': 'Kalman 7ch with asymmetric 48:16 embedding',
            'channels': 7,
        },
        {
            'name': 'kalman_baseline',
            'config': 'config/upfall/kalman_optimal.yaml',
            'description': 'Kalman baseline (7ch, 31:17 embedding)',
            'channels': 7,
        },
    ],
    'wedafall': [
        {
            'name': 'orientation_2ch',
            'config': 'config/wedafall/channel_ablation/orientation_2ch.yaml',
            'description': 'Orientation only (roll, pitch)',
            'channels': 2,
        },
        {
            'name': 'orientation_3ch',
            'config': 'config/wedafall/channel_ablation/orientation_3ch.yaml',
            'description': 'Orientation only (roll, pitch, yaw)',
            'channels': 3,
        },
        {
            'name': 'hybrid_9ch',
            'config': 'config/wedafall/channel_ablation/hybrid_9ch.yaml',
            'description': 'Hybrid: smv + acc + roll/pitch + raw gyro',
            'channels': 9,
        },
        {
            'name': 'kalman_balanced_32_32',
            'config': 'config/wedafall/channel_ablation/kalman_balanced_32_32.yaml',
            'description': 'Kalman 7ch with balanced 32:32 embedding',
            'channels': 7,
        },
        {
            'name': 'kalman_asymmetric_48_16',
            'config': 'config/wedafall/channel_ablation/kalman_asymmetric_48_16.yaml',
            'description': 'Kalman 7ch with asymmetric 48:16 embedding',
            'channels': 7,
        },
        {
            'name': 'kalman_baseline',
            'config': 'config/wedafall/kalman_optimal.yaml',
            'description': 'Kalman baseline (7ch, 31:17 embedding)',
            'channels': 7,
        },
    ],
}


def parse_summary_report(summary_path: Path) -> Dict:
    """Parse summary_report.txt for metrics."""
    results = {}
    if not summary_path.exists():
        return {'error': 'Summary file not found'}

    content = summary_path.read_text()

    # Extract F1
    match = re.search(r'Test F1:\s+([\d.]+)\s*±\s*([\d.]+)%', content)
    if match:
        results['test_f1'] = float(match.group(1))
        results['test_f1_std'] = float(match.group(2))

    # Extract accuracy
    match = re.search(r'Test Accuracy:\s+([\d.]+)\s*±\s*([\d.]+)%', content)
    if match:
        results['test_acc'] = float(match.group(1))
        results['test_acc_std'] = float(match.group(2))

    # Extract validation F1
    match = re.search(r'Val F1:\s+([\d.]+)%', content)
    if match:
        results['val_f1'] = float(match.group(1))

    return results


def run_experiment(
    config_path: str,
    work_dir: str,
    num_gpus: int,
    max_folds: Optional[int] = None,
) -> Dict:
    """Run a single experiment using ray_train.py."""
    cmd = [
        sys.executable, 'ray_train.py',
        '--config', config_path,
        '--num-gpus', str(num_gpus),
        '--work-dir', work_dir,
    ]

    if max_folds:
        cmd.extend(['--max-folds', str(max_folds)])

    print(f"\n{'='*70}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        summary_path = Path(work_dir) / 'summary_report.txt'
        return {'status': 'success', **parse_summary_report(summary_path)}
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Experiment failed with code {e.returncode}")
        return {'status': 'failed', 'error': str(e)}
    except Exception as e:
        print(f"ERROR: {e}")
        return {'status': 'failed', 'error': str(e)}


def run_ablation(
    dataset: str,
    output_dir: Path,
    num_gpus: int,
    max_folds: Optional[int] = None,
    configs: Optional[List[str]] = None,
) -> List[Dict]:
    """Run ablation study for a dataset."""
    results = []
    ablation_configs = ABLATION_CONFIGS[dataset]

    # Filter configs if specified
    if configs:
        ablation_configs = [c for c in ablation_configs if c['name'] in configs]

    for exp_config in ablation_configs:
        exp_name = f"{dataset}_{exp_config['name']}"
        work_dir = str(output_dir / exp_name)

        print(f"\n{'#'*70}")
        print(f"# {dataset.upper()} | {exp_config['name']}")
        print(f"# {exp_config['description']}")
        print(f"# Channels: {exp_config['channels']}")
        print(f"{'#'*70}")

        metrics = run_experiment(
            config_path=exp_config['config'],
            work_dir=work_dir,
            num_gpus=num_gpus,
            max_folds=max_folds,
        )

        result = {
            'name': exp_name,
            'dataset': dataset,
            'config_name': exp_config['name'],
            'config_path': exp_config['config'],
            'description': exp_config['description'],
            'channels': exp_config['channels'],
            'work_dir': work_dir,
            **metrics,
        }
        results.append(result)

        # Save intermediate results
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'channel_ablation_results.json', 'w') as f:
            json.dump(results, f, indent=2)

    return results


def generate_report(results: List[Dict], output_dir: Path):
    """Generate markdown report."""
    report_path = output_dir / 'channel_ablation_report.md'

    with open(report_path, 'w') as f:
        f.write("# Channel Ablation Study\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        for dataset in ['upfall', 'wedafall']:
            dataset_results = [r for r in results if r['dataset'] == dataset]
            if not dataset_results:
                continue

            f.write(f"## {DATASET_INFO[dataset]['name']}\n\n")
            f.write("| Config | Channels | F1 (%) | Std | Description |\n")
            f.write("|--------|----------|--------|-----|-------------|\n")

            # Sort by F1 descending
            sorted_results = sorted(
                dataset_results,
                key=lambda x: x.get('test_f1', 0),
                reverse=True
            )

            for r in sorted_results:
                f1 = f"{r.get('test_f1', 0):.2f}" if 'test_f1' in r else 'N/A'
                std = f"{r.get('test_f1_std', 0):.2f}" if 'test_f1_std' in r else 'N/A'
                status = '' if r.get('status') == 'success' else ' (FAILED)'
                f.write(f"| {r['config_name']}{status} | {r['channels']} | {f1} | {std} | {r['description']} |\n")

            f.write("\n")

        # Summary
        f.write("## Key Findings\n\n")
        for dataset in ['upfall', 'wedafall']:
            dataset_results = [r for r in results if r['dataset'] == dataset and r.get('status') == 'success']
            if dataset_results:
                best = max(dataset_results, key=lambda x: x.get('test_f1', 0))
                f.write(f"**{DATASET_INFO[dataset]['name']}** best: `{best['config_name']}` ({best['channels']}ch) - F1={best.get('test_f1', 0):.2f}%\n\n")

    print(f"\nReport saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Channel ablation study')
    parser.add_argument('--datasets', nargs='+', default=['upfall', 'wedafall'],
                       choices=['upfall', 'wedafall'],
                       help='Datasets to run ablation on')
    parser.add_argument('--configs', nargs='+', default=None,
                       help='Specific config names to run (default: all)')
    parser.add_argument('--num-gpus', type=int, default=6,
                       help='Number of GPUs to use')
    parser.add_argument('--max-folds', type=int, default=None,
                       help='Maximum folds per experiment (for quick testing)')
    parser.add_argument('--output-dir', type=str, default='exps/channel_ablation',
                       help='Output directory for results')

    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for dataset in args.datasets:
        print(f"\n{'='*70}")
        print(f"Dataset: {DATASET_INFO[dataset]['name']}")
        print(f"{'='*70}\n")

        results = run_ablation(
            dataset=dataset,
            output_dir=output_dir,
            num_gpus=args.num_gpus,
            max_folds=args.max_folds,
            configs=args.configs,
        )
        all_results.extend(results)

    # Save final results
    with open(output_dir / 'channel_ablation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Generate report
    generate_report(all_results, output_dir)

    print(f"\n{'='*70}")
    print("Channel ablation study complete!")
    print(f"Results: {output_dir / 'channel_ablation_results.json'}")
    print(f"Report: {output_dir / 'channel_ablation_report.md'}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
