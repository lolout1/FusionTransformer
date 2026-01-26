#!/usr/bin/env python3
"""
Channel Variants Comparison Study.

Compares different input channel configurations:
1. Kalman 7ch: [smv, ax, ay, az, roll, pitch, yaw] - baseline
2. Kalman 6ch: [smv, ax, ay, az, roll, pitch] - no yaw
3. Raw 7ch: [smv, ax, ay, az, gx, gy, gz] - no Kalman fusion

Usage:
    python distributed_dataset_pipeline/run_channel_variants.py --num-gpus 8
    python distributed_dataset_pipeline/run_channel_variants.py --num-gpus 4 --max-folds 2  # Quick test
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
    },
    'wedafall': {
        'name': 'WEDA-FALL',
        'num_folds': 12,
    }
}

# Channel variant configurations
CHANNEL_VARIANTS = {
    'upfall': [
        {
            'name': 'kalman_7ch',
            'config': 'config/upfall/best_kalman.yaml',
            'description': 'Kalman 7ch: smv + acc + roll/pitch/yaw',
            'channels': 7,
            'features': '[smv, ax, ay, az, roll, pitch, yaw]',
        },
        {
            'name': 'kalman_6ch_no_yaw',
            'config': 'config/upfall/channel_ablation/kalman_no_yaw.yaml',
            'description': 'Kalman 6ch: smv + acc + roll/pitch (no yaw)',
            'channels': 6,
            'features': '[smv, ax, ay, az, roll, pitch]',
        },
        {
            'name': 'raw_7ch',
            'config': 'config/upfall/best_raw.yaml',
            'description': 'Raw 7ch: smv + acc + gyro (no Kalman)',
            'channels': 7,
            'features': '[smv, ax, ay, az, gx, gy, gz]',
        },
    ],
    'wedafall': [
        {
            'name': 'kalman_7ch',
            'config': 'config/wedafall/best_kalman.yaml',
            'description': 'Kalman 7ch: smv + acc + roll/pitch/yaw',
            'channels': 7,
            'features': '[smv, ax, ay, az, roll, pitch, yaw]',
        },
        {
            'name': 'kalman_6ch_no_yaw',
            'config': 'config/wedafall/channel_ablation/kalman_no_yaw.yaml',
            'description': 'Kalman 6ch: smv + acc + roll/pitch (no yaw)',
            'channels': 6,
            'features': '[smv, ax, ay, az, roll, pitch]',
        },
        {
            'name': 'raw_7ch',
            'config': 'config/wedafall/best_raw.yaml',
            'description': 'Raw 7ch: smv + acc + gyro (no Kalman)',
            'channels': 7,
            'features': '[smv, ax, ay, az, gx, gy, gz]',
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

    # Extract Macro-F1
    match = re.search(r'Test Macro-F1:\s+([\d.]+)\s*±\s*([\d.]+)%', content)
    if match:
        results['test_macro_f1'] = float(match.group(1))
        results['test_macro_f1_std'] = float(match.group(2))

    # Extract accuracy
    match = re.search(r'Test Accuracy:\s+([\d.]+)\s*±\s*([\d.]+)%', content)
    if match:
        results['test_acc'] = float(match.group(1))
        results['test_acc_std'] = float(match.group(2))

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


def run_channel_comparison(
    dataset: str,
    output_dir: Path,
    num_gpus: int,
    max_folds: Optional[int] = None,
    variants: Optional[List[str]] = None,
) -> List[Dict]:
    """Run channel variant comparison for a dataset."""
    results = []
    configs = CHANNEL_VARIANTS[dataset]

    # Filter if specific variants requested
    if variants:
        configs = [c for c in configs if c['name'] in variants]

    for exp_config in configs:
        exp_name = f"{dataset}_{exp_config['name']}"
        work_dir = str(output_dir / exp_name)

        print(f"\n{'#'*70}")
        print(f"# {dataset.upper()} | {exp_config['name']}")
        print(f"# {exp_config['description']}")
        print(f"# Features: {exp_config['features']}")
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
            'variant': exp_config['name'],
            'config_path': exp_config['config'],
            'description': exp_config['description'],
            'channels': exp_config['channels'],
            'features': exp_config['features'],
            'work_dir': work_dir,
            **metrics,
        }
        results.append(result)

        # Save intermediate results
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'channel_variants_results.json', 'w') as f:
            json.dump(results, f, indent=2)

    return results


def generate_report(results: List[Dict], output_dir: Path):
    """Generate comparison report."""
    report_path = output_dir / 'channel_variants_report.md'

    with open(report_path, 'w') as f:
        f.write("# Channel Variants Comparison\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        # Summary table
        f.write("## Results Summary\n\n")
        f.write("| Dataset | Variant | Channels | Features | F1 (%) | Std | Macro-F1 |\n")
        f.write("|---------|---------|----------|----------|--------|-----|----------|\n")

        for dataset in ['upfall', 'wedafall']:
            dataset_results = [r for r in results if r['dataset'] == dataset]
            # Sort by F1 descending
            dataset_results.sort(key=lambda x: x.get('test_f1', 0), reverse=True)

            for r in dataset_results:
                f1 = f"{r.get('test_f1', 0):.2f}" if 'test_f1' in r else 'N/A'
                std = f"±{r.get('test_f1_std', 0):.2f}" if 'test_f1_std' in r else ''
                macro = f"{r.get('test_macro_f1', 0):.2f}" if 'test_macro_f1' in r else '-'
                status = '' if r.get('status') == 'success' else ' (FAILED)'
                f.write(f"| {DATASET_INFO[dataset]['name']} | {r['variant']}{status} | {r['channels']} | {r['features']} | {f1} | {std} | {macro} |\n")

        f.write("\n")

        # Key findings
        f.write("## Key Findings\n\n")
        for dataset in ['upfall', 'wedafall']:
            dataset_results = [r for r in results if r['dataset'] == dataset and r.get('status') == 'success']
            if dataset_results:
                best = max(dataset_results, key=lambda x: x.get('test_f1', 0))
                f.write(f"**{DATASET_INFO[dataset]['name']}** best: `{best['variant']}` - F1={best.get('test_f1', 0):.2f}%\n\n")

                # Compare Kalman 7ch vs no-yaw
                kalman_7ch = next((r for r in dataset_results if r['variant'] == 'kalman_7ch'), None)
                kalman_no_yaw = next((r for r in dataset_results if r['variant'] == 'kalman_6ch_no_yaw'), None)

                if kalman_7ch and kalman_no_yaw:
                    diff = kalman_no_yaw.get('test_f1', 0) - kalman_7ch.get('test_f1', 0)
                    f.write(f"- Removing yaw: {'+' if diff >= 0 else ''}{diff:.2f}% F1\n")

                # Compare Kalman vs Raw
                raw = next((r for r in dataset_results if r['variant'] == 'raw_7ch'), None)
                if kalman_7ch and raw:
                    diff = kalman_7ch.get('test_f1', 0) - raw.get('test_f1', 0)
                    f.write(f"- Kalman advantage over raw: {'+' if diff >= 0 else ''}{diff:.2f}% F1\n")

                f.write("\n")

    print(f"\nReport saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Channel variants comparison study')
    parser.add_argument('--datasets', nargs='+', default=['upfall', 'wedafall'],
                        choices=['upfall', 'wedafall'],
                        help='Datasets to compare (default: both)')
    parser.add_argument('--variants', nargs='+', default=None,
                        choices=['kalman_7ch', 'kalman_6ch_no_yaw', 'raw_7ch'],
                        help='Specific variants to run (default: all)')
    parser.add_argument('--num-gpus', type=int, default=8,
                        help='Number of GPUs to use (default: 8)')
    parser.add_argument('--max-folds', type=int, default=None,
                        help='Maximum folds per experiment (for quick testing)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: exps/channel_variants_<timestamp>)')

    args = parser.parse_args()

    # Generate timestamped output directory if not specified
    if args.output_dir:
        output_dir = PROJECT_ROOT / args.output_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = PROJECT_ROOT / f'exps/channel_variants_{timestamp}'

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    all_results = []

    for dataset in args.datasets:
        print(f"\n{'='*70}")
        print(f"Dataset: {DATASET_INFO[dataset]['name']}")
        print(f"{'='*70}\n")

        results = run_channel_comparison(
            dataset=dataset,
            output_dir=output_dir,
            num_gpus=args.num_gpus,
            max_folds=args.max_folds,
            variants=args.variants,
        )
        all_results.extend(results)

    # Save final results
    with open(output_dir / 'channel_variants_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Generate report
    generate_report(all_results, output_dir)

    print(f"\n{'='*70}")
    print("Channel variants comparison complete!")
    print(f"{'='*70}")
    print(f"Results: {output_dir / 'channel_variants_results.json'}")
    print(f"Report: {output_dir / 'channel_variants_report.md'}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
