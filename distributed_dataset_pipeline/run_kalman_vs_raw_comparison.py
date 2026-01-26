#!/usr/bin/env python3
"""
Kalman vs Raw IMU Comparison Study.

Compares fall detection performance between:
1. Kalman-fused features (7ch: SMV + acc + orientation)
2. Raw IMU features (6ch: acc + gyro)

Runs 4 experiments:
- UP-FALL Kalman (best_kalman.yaml)
- UP-FALL Raw (best_raw.yaml)
- WEDA-FALL Kalman (best_kalman.yaml)
- WEDA-FALL Raw (best_raw.yaml)

Usage:
    python distributed_dataset_pipeline/run_kalman_vs_raw_comparison.py --num-gpus 8
    python distributed_dataset_pipeline/run_kalman_vs_raw_comparison.py --num-gpus 4 --max-folds 3  # Quick test
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
        'num_subjects': 17,
        'val_subjects': [15, 16],
    },
    'wedafall': {
        'name': 'WEDA-FALL',
        'num_folds': 12,
        'num_subjects': 14,
        'val_subjects': [13, 14],
    }
}

# Comparison configurations
COMPARISON_CONFIGS = {
    'upfall': [
        {
            'name': 'kalman',
            'config': 'config/upfall/best_kalman.yaml',
            'description': 'Kalman-fused features (7ch: SMV + acc + euler)',
            'channels': 7,
            'model': 'KalmanEncoderAblation',
        },
        {
            'name': 'raw',
            'config': 'config/upfall/best_raw.yaml',
            'description': 'Raw IMU features (6ch: acc + gyro)',
            'channels': 6,
            'model': 'DualStreamBaseline',
        },
    ],
    'wedafall': [
        {
            'name': 'kalman',
            'config': 'config/wedafall/best_kalman.yaml',
            'description': 'Kalman-fused features (7ch: SMV + acc + euler)',
            'channels': 7,
            'model': 'KalmanEncoderAblation',
        },
        {
            'name': 'raw',
            'config': 'config/wedafall/best_raw.yaml',
            'description': 'Raw IMU features (6ch: acc + gyro)',
            'channels': 6,
            'model': 'DualStreamBaseline',
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

    # Extract precision
    match = re.search(r'Test Precision:\s+([\d.]+)\s*±\s*([\d.]+)%', content)
    if match:
        results['test_prec'] = float(match.group(1))
        results['test_prec_std'] = float(match.group(2))

    # Extract recall
    match = re.search(r'Test Recall:\s+([\d.]+)\s*±\s*([\d.]+)%', content)
    if match:
        results['test_recall'] = float(match.group(1))
        results['test_recall_std'] = float(match.group(2))

    # Extract AUC
    match = re.search(r'Test AUC:\s+([\d.]+)\s*±\s*([\d.]+)%', content)
    if match:
        results['test_auc'] = float(match.group(1))
        results['test_auc_std'] = float(match.group(2))

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


def run_comparison(
    dataset: str,
    output_dir: Path,
    num_gpus: int,
    max_folds: Optional[int] = None,
    method: Optional[str] = None,
) -> List[Dict]:
    """Run Kalman vs Raw comparison for a dataset."""
    results = []
    configs = COMPARISON_CONFIGS[dataset]

    # Filter if specific method requested
    if method:
        configs = [c for c in configs if c['name'] == method]

    for exp_config in configs:
        exp_name = f"{dataset}_{exp_config['name']}"
        work_dir = str(output_dir / exp_name)

        print(f"\n{'#'*70}")
        print(f"# {dataset.upper()} | {exp_config['name'].upper()}")
        print(f"# {exp_config['description']}")
        print(f"# Model: {exp_config['model']} | Channels: {exp_config['channels']}")
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
            'method': exp_config['name'],
            'config_path': exp_config['config'],
            'description': exp_config['description'],
            'channels': exp_config['channels'],
            'model': exp_config['model'],
            'work_dir': work_dir,
            **metrics,
        }
        results.append(result)

        # Save intermediate results
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'comparison_results.json', 'w') as f:
            json.dump(results, f, indent=2)

    return results


def generate_comparison_table(results: List[Dict], output_dir: Path):
    """Generate comparison table in markdown format."""
    table_path = output_dir / 'comparison_table.md'

    with open(table_path, 'w') as f:
        f.write("# Kalman vs Raw IMU Comparison\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        # Summary table
        f.write("## Summary Results\n\n")
        f.write("| Dataset | Method | Channels | Model | F1 (%) | Std | Accuracy | AUC |\n")
        f.write("|---------|--------|----------|-------|--------|-----|----------|-----|\n")

        for dataset in ['upfall', 'wedafall']:
            dataset_results = [r for r in results if r['dataset'] == dataset]
            for r in dataset_results:
                f1 = f"{r.get('test_f1', 0):.2f}" if 'test_f1' in r else 'N/A'
                std = f"{r.get('test_f1_std', 0):.2f}" if 'test_f1_std' in r else 'N/A'
                acc = f"{r.get('test_acc', 0):.2f}" if 'test_acc' in r else 'N/A'
                auc = f"{r.get('test_auc', 0):.2f}" if 'test_auc' in r else 'N/A'
                status = '' if r.get('status') == 'success' else ' (FAILED)'
                f.write(f"| {DATASET_INFO[dataset]['name']} | {r['method'].upper()}{status} | {r['channels']}ch | {r['model']} | {f1} | ±{std} | {acc}% | {auc}% |\n")

        f.write("\n")

        # Detailed comparison per dataset
        for dataset in ['upfall', 'wedafall']:
            dataset_results = [r for r in results if r['dataset'] == dataset]
            if len(dataset_results) < 2:
                continue

            kalman_result = next((r for r in dataset_results if r['method'] == 'kalman'), None)
            raw_result = next((r for r in dataset_results if r['method'] == 'raw'), None)

            if kalman_result and raw_result and kalman_result.get('status') == 'success' and raw_result.get('status') == 'success':
                f.write(f"## {DATASET_INFO[dataset]['name']} Detailed Comparison\n\n")

                kalman_f1 = kalman_result.get('test_f1', 0)
                raw_f1 = raw_result.get('test_f1', 0)
                improvement = kalman_f1 - raw_f1

                f.write(f"**Kalman F1:** {kalman_f1:.2f}% ± {kalman_result.get('test_f1_std', 0):.2f}%\n\n")
                f.write(f"**Raw F1:** {raw_f1:.2f}% ± {raw_result.get('test_f1_std', 0):.2f}%\n\n")
                f.write(f"**Improvement:** {'+' if improvement >= 0 else ''}{improvement:.2f}%\n\n")

                # Full metrics table
                f.write("| Metric | Kalman | Raw | Difference |\n")
                f.write("|--------|--------|-----|------------|\n")

                for metric, label in [('test_f1', 'F1'), ('test_acc', 'Accuracy'),
                                      ('test_prec', 'Precision'), ('test_recall', 'Recall'),
                                      ('test_auc', 'AUC')]:
                    k_val = kalman_result.get(metric, 0)
                    r_val = raw_result.get(metric, 0)
                    diff = k_val - r_val
                    f.write(f"| {label} | {k_val:.2f}% | {r_val:.2f}% | {'+' if diff >= 0 else ''}{diff:.2f}% |\n")

                f.write("\n")

        # Key findings
        f.write("## Key Findings\n\n")
        for dataset in ['upfall', 'wedafall']:
            dataset_results = [r for r in results if r['dataset'] == dataset]
            kalman_result = next((r for r in dataset_results if r['method'] == 'kalman'), None)
            raw_result = next((r for r in dataset_results if r['method'] == 'raw'), None)

            if kalman_result and raw_result:
                kalman_f1 = kalman_result.get('test_f1', 0)
                raw_f1 = raw_result.get('test_f1', 0)
                improvement = kalman_f1 - raw_f1
                winner = "Kalman" if improvement > 0 else "Raw"
                f.write(f"- **{DATASET_INFO[dataset]['name']}:** {winner} wins by {abs(improvement):.2f}% F1\n")

        f.write("\n")

    print(f"\nComparison table saved: {table_path}")


def generate_text_report(results: List[Dict], output_dir: Path):
    """Generate plain text comparison report."""
    report_path = output_dir / 'comparison_results.txt'

    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("KALMAN vs RAW IMU COMPARISON RESULTS\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * 70 + "\n\n")

        for dataset in ['upfall', 'wedafall']:
            dataset_results = [r for r in results if r['dataset'] == dataset]
            if not dataset_results:
                continue

            f.write(f"{DATASET_INFO[dataset]['name']}\n")
            f.write("-" * 40 + "\n")

            for r in dataset_results:
                f.write(f"\n{r['method'].upper()}:\n")
                f.write(f"  Model: {r['model']}\n")
                f.write(f"  Channels: {r['channels']}\n")
                if r.get('status') == 'success':
                    f.write(f"  F1: {r.get('test_f1', 0):.2f}% ± {r.get('test_f1_std', 0):.2f}%\n")
                    f.write(f"  Accuracy: {r.get('test_acc', 0):.2f}%\n")
                    f.write(f"  Precision: {r.get('test_prec', 0):.2f}%\n")
                    f.write(f"  Recall: {r.get('test_recall', 0):.2f}%\n")
                    f.write(f"  AUC: {r.get('test_auc', 0):.2f}%\n")
                else:
                    f.write(f"  Status: FAILED\n")
                    f.write(f"  Error: {r.get('error', 'Unknown')}\n")

            # Comparison if both succeeded
            kalman = next((r for r in dataset_results if r['method'] == 'kalman' and r.get('status') == 'success'), None)
            raw = next((r for r in dataset_results if r['method'] == 'raw' and r.get('status') == 'success'), None)

            if kalman and raw:
                f.write(f"\nCOMPARISON:\n")
                improvement = kalman.get('test_f1', 0) - raw.get('test_f1', 0)
                f.write(f"  Kalman improvement: {'+' if improvement >= 0 else ''}{improvement:.2f}% F1\n")

            f.write("\n")

        f.write("=" * 70 + "\n")

    print(f"\nText report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Kalman vs Raw IMU comparison study')
    parser.add_argument('--datasets', nargs='+', default=['upfall', 'wedafall'],
                        choices=['upfall', 'wedafall'],
                        help='Datasets to compare (default: both)')
    parser.add_argument('--method', type=str, default=None,
                        choices=['kalman', 'raw'],
                        help='Run only one method (default: both)')
    parser.add_argument('--num-gpus', type=int, default=8,
                        help='Number of GPUs to use (default: 8)')
    parser.add_argument('--max-folds', type=int, default=None,
                        help='Maximum folds per experiment (for quick testing)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: exps/kalman_vs_raw_comparison_<timestamp>)')

    args = parser.parse_args()

    # Generate timestamped output directory if not specified
    if args.output_dir:
        output_dir = PROJECT_ROOT / args.output_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = PROJECT_ROOT / f'exps/kalman_vs_raw_comparison_{timestamp}'

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    all_results = []

    for dataset in args.datasets:
        print(f"\n{'='*70}")
        print(f"Dataset: {DATASET_INFO[dataset]['name']}")
        print(f"Subjects: {DATASET_INFO[dataset]['num_subjects']}")
        print(f"Validation: {DATASET_INFO[dataset]['val_subjects']}")
        print(f"LOSO Folds: {DATASET_INFO[dataset]['num_folds']}")
        print(f"{'='*70}\n")

        results = run_comparison(
            dataset=dataset,
            output_dir=output_dir,
            num_gpus=args.num_gpus,
            max_folds=args.max_folds,
            method=args.method,
        )
        all_results.extend(results)

    # Save final results
    with open(output_dir / 'comparison_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Generate reports
    generate_comparison_table(all_results, output_dir)
    generate_text_report(all_results, output_dir)

    print(f"\n{'='*70}")
    print("Kalman vs Raw comparison study complete!")
    print(f"{'='*70}")
    print(f"Results JSON: {output_dir / 'comparison_results.json'}")
    print(f"Markdown table: {output_dir / 'comparison_table.md'}")
    print(f"Text report: {output_dir / 'comparison_results.txt'}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
