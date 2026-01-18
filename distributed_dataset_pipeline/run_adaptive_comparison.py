#!/usr/bin/env python3
"""
Adaptive Kalman Filter Comparison Study.

Compares fall detection performance between:
1. Raw IMU features (6ch: acc + gyro)
2. Fixed Kalman-fused features (7ch: SMV + acc + orientation)
3. Adaptive Kalman-fused features (7ch: SMV + acc + orientation with IAE)

Runs 6 experiments:
- UP-FALL Raw (best_raw.yaml)
- UP-FALL Kalman (best_kalman.yaml)
- UP-FALL Adaptive (best_adaptive.yaml)
- WEDA-FALL Raw (best_raw.yaml)
- WEDA-FALL Kalman (best_kalman.yaml)
- WEDA-FALL Adaptive (best_adaptive.yaml)

Usage:
    python distributed_dataset_pipeline/run_adaptive_comparison.py --num-gpus 8
    python distributed_dataset_pipeline/run_adaptive_comparison.py --num-gpus 4 --max-folds 3  # Quick test
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
        'sensor_type': 'Research-grade',
        'hypothesis': 'Raw > Kalman (clean sensors), Adaptive ≈ Raw',
    },
    'wedafall': {
        'name': 'WEDA-FALL',
        'num_folds': 12,
        'num_subjects': 14,
        'val_subjects': [13, 14],
        'sensor_type': 'Consumer-grade',
        'hypothesis': 'Kalman > Raw (noisy sensors), Adaptive ≥ Kalman',
    }
}

# Comparison configurations
COMPARISON_CONFIGS = {
    'upfall': [
        {
            'name': 'raw',
            'config': 'config/upfall/best_raw.yaml',
            'description': 'Raw IMU features (6ch: acc + gyro)',
            'channels': 6,
            'model': 'DualStreamBaseline',
            'kalman_type': 'none',
        },
        {
            'name': 'kalman',
            'config': 'config/upfall/best_kalman.yaml',
            'description': 'Fixed Kalman-fused features (7ch: SMV + acc + euler)',
            'channels': 7,
            'model': 'KalmanEncoderAblation',
            'kalman_type': 'fixed',
        },
        {
            'name': 'adaptive',
            'config': 'config/upfall/best_adaptive.yaml',
            'description': 'Adaptive Kalman-fused features (7ch with IAE)',
            'channels': 7,
            'model': 'KalmanEncoderAblation',
            'kalman_type': 'adaptive',
        },
    ],
    'wedafall': [
        {
            'name': 'raw',
            'config': 'config/wedafall/best_raw.yaml',
            'description': 'Raw IMU features (6ch: acc + gyro)',
            'channels': 6,
            'model': 'DualStreamBaseline',
            'kalman_type': 'none',
        },
        {
            'name': 'kalman',
            'config': 'config/wedafall/best_kalman.yaml',
            'description': 'Fixed Kalman-fused features (7ch: SMV + acc + euler)',
            'channels': 7,
            'model': 'KalmanEncoderAblation',
            'kalman_type': 'fixed',
        },
        {
            'name': 'adaptive',
            'config': 'config/wedafall/best_adaptive.yaml',
            'description': 'Adaptive Kalman-fused features (7ch with IAE)',
            'channels': 7,
            'model': 'KalmanEncoderAblation',
            'kalman_type': 'adaptive',
        },
    ],
}


def parse_summary_report(summary_path: Path) -> Dict:
    """Parse summary_report.txt for metrics."""
    results = {}
    if not summary_path.exists():
        return {'error': 'Summary file not found'}

    content = summary_path.read_text()

    # Extract F1 (binary)
    match = re.search(r'Test F1:\s+([\d.]+)\s*±\s*([\d.]+)%', content)
    if match:
        results['test_f1'] = float(match.group(1))
        results['test_f1_std'] = float(match.group(2))

    # Extract Macro F1
    match = re.search(r'Test Macro-F1:\s+([\d.]+)\s*±\s*([\d.]+)%', content)
    if match:
        results['test_macro_f1'] = float(match.group(1))
        results['test_macro_f1_std'] = float(match.group(2))

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

    # Extract validation Macro F1
    match = re.search(r'Val Macro-F1:\s+([\d.]+)%', content)
    if match:
        results['val_macro_f1'] = float(match.group(1))

    return results


def run_experiment(
    config_path: str,
    work_dir: str,
    num_gpus: int,
    max_folds: Optional[int] = None,
) -> Dict:
    """Run a single experiment using ray_train.py."""
    # Check if config exists
    config_file = PROJECT_ROOT / config_path
    if not config_file.exists():
        print(f"WARNING: Config file not found: {config_file}")
        return {'status': 'skipped', 'error': f'Config not found: {config_path}'}

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
    methods: Optional[List[str]] = None,
) -> List[Dict]:
    """Run adaptive comparison for a dataset."""
    results = []
    configs = COMPARISON_CONFIGS[dataset]

    # Filter if specific methods requested
    if methods:
        configs = [c for c in configs if c['name'] in methods]

    for exp_config in configs:
        exp_name = f"{dataset}_{exp_config['name']}"
        work_dir = str(output_dir / exp_name)

        print(f"\n{'#'*70}")
        print(f"# {dataset.upper()} | {exp_config['name'].upper()}")
        print(f"# {exp_config['description']}")
        print(f"# Model: {exp_config['model']} | Channels: {exp_config['channels']} | Kalman: {exp_config['kalman_type']}")
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
            'kalman_type': exp_config['kalman_type'],
            'work_dir': work_dir,
            **metrics,
        }
        results.append(result)

        # Save intermediate results
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'adaptive_comparison_results.json', 'w') as f:
            json.dump(results, f, indent=2)

    return results


def generate_comparison_table(results: List[Dict], output_dir: Path):
    """Generate comparison table in markdown format."""
    table_path = output_dir / 'adaptive_comparison_table.md'

    with open(table_path, 'w') as f:
        f.write("# Adaptive Kalman Filter Comparison\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        f.write("## Hypothesis\n\n")
        f.write("- **UP-FALL (Research-grade):** Raw ≥ Kalman (clean sensors); Adaptive should approach Raw\n")
        f.write("- **WEDA-FALL (Consumer-grade):** Kalman > Raw (noisy sensors); Adaptive should ≥ Kalman\n\n")

        # Summary table
        f.write("## Summary Results\n\n")
        f.write("| Dataset | Method | Kalman Type | F1 (%) | Std | Macro-F1 (%) | Std | Accuracy | AUC |\n")
        f.write("|---------|--------|-------------|--------|-----|--------------|-----|----------|-----|\n")

        for dataset in ['upfall', 'wedafall']:
            dataset_results = [r for r in results if r['dataset'] == dataset]
            for r in sorted(dataset_results, key=lambda x: ['raw', 'kalman', 'adaptive'].index(x['method']) if x['method'] in ['raw', 'kalman', 'adaptive'] else 99):
                f1 = f"{r.get('test_f1', 0):.2f}" if 'test_f1' in r else 'N/A'
                f1_std = f"{r.get('test_f1_std', 0):.2f}" if 'test_f1_std' in r else 'N/A'
                macro_f1 = f"{r.get('test_macro_f1', 0):.2f}" if 'test_macro_f1' in r else 'N/A'
                macro_f1_std = f"{r.get('test_macro_f1_std', 0):.2f}" if 'test_macro_f1_std' in r else 'N/A'
                acc = f"{r.get('test_acc', 0):.2f}" if 'test_acc' in r else 'N/A'
                auc = f"{r.get('test_auc', 0):.2f}" if 'test_auc' in r else 'N/A'
                status = '' if r.get('status') == 'success' else f" ({r.get('status', 'unknown').upper()})"
                f.write(f"| {DATASET_INFO[dataset]['name']} | {r['method'].upper()}{status} | {r['kalman_type']} | {f1} | ±{f1_std} | {macro_f1} | ±{macro_f1_std} | {acc}% | {auc}% |\n")

        f.write("\n")

        # Detailed comparison per dataset
        for dataset in ['upfall', 'wedafall']:
            dataset_results = [r for r in results if r['dataset'] == dataset]
            if len(dataset_results) < 2:
                continue

            raw_result = next((r for r in dataset_results if r['method'] == 'raw'), None)
            kalman_result = next((r for r in dataset_results if r['method'] == 'kalman'), None)
            adaptive_result = next((r for r in dataset_results if r['method'] == 'adaptive'), None)

            f.write(f"## {DATASET_INFO[dataset]['name']} Detailed Comparison\n\n")
            f.write(f"**Sensor Type:** {DATASET_INFO[dataset]['sensor_type']}\n\n")
            f.write(f"**Hypothesis:** {DATASET_INFO[dataset]['hypothesis']}\n\n")

            # Results summary
            for name, result in [('Raw', raw_result), ('Fixed Kalman', kalman_result), ('Adaptive Kalman', adaptive_result)]:
                if result and result.get('status') == 'success':
                    f.write(f"**{name} F1:** {result.get('test_f1', 0):.2f}% ± {result.get('test_f1_std', 0):.2f}%\n\n")
                elif result:
                    f.write(f"**{name}:** {result.get('status', 'unknown').upper()}\n\n")

            # Full metrics table
            f.write("| Metric | Raw | Fixed Kalman | Adaptive | Adaptive vs Raw | Adaptive vs Kalman |\n")
            f.write("|--------|-----|--------------|----------|-----------------|--------------------|\n")

            for metric, label in [('test_f1', 'F1 (Binary)'), ('test_macro_f1', 'Macro-F1'),
                                  ('test_acc', 'Accuracy'), ('test_prec', 'Precision'),
                                  ('test_recall', 'Recall'), ('test_auc', 'AUC')]:
                raw_val = raw_result.get(metric, 0) if raw_result and raw_result.get('status') == 'success' else None
                kalman_val = kalman_result.get(metric, 0) if kalman_result and kalman_result.get('status') == 'success' else None
                adaptive_val = adaptive_result.get(metric, 0) if adaptive_result and adaptive_result.get('status') == 'success' else None

                raw_str = f"{raw_val:.2f}%" if raw_val is not None else "N/A"
                kalman_str = f"{kalman_val:.2f}%" if kalman_val is not None else "N/A"
                adaptive_str = f"{adaptive_val:.2f}%" if adaptive_val is not None else "N/A"

                if adaptive_val is not None and raw_val is not None:
                    diff_raw = adaptive_val - raw_val
                    diff_raw_str = f"{'+' if diff_raw >= 0 else ''}{diff_raw:.2f}%"
                else:
                    diff_raw_str = "N/A"

                if adaptive_val is not None and kalman_val is not None:
                    diff_kalman = adaptive_val - kalman_val
                    diff_kalman_str = f"{'+' if diff_kalman >= 0 else ''}{diff_kalman:.2f}%"
                else:
                    diff_kalman_str = "N/A"

                f.write(f"| {label} | {raw_str} | {kalman_str} | {adaptive_str} | {diff_raw_str} | {diff_kalman_str} |\n")

            f.write("\n")

        # Key findings
        f.write("## Key Findings\n\n")
        for dataset in ['upfall', 'wedafall']:
            dataset_results = [r for r in results if r['dataset'] == dataset]
            raw_result = next((r for r in dataset_results if r['method'] == 'raw' and r.get('status') == 'success'), None)
            kalman_result = next((r for r in dataset_results if r['method'] == 'kalman' and r.get('status') == 'success'), None)
            adaptive_result = next((r for r in dataset_results if r['method'] == 'adaptive' and r.get('status') == 'success'), None)

            f.write(f"### {DATASET_INFO[dataset]['name']}\n\n")

            if raw_result and kalman_result and adaptive_result:
                raw_f1 = raw_result.get('test_f1', 0)
                kalman_f1 = kalman_result.get('test_f1', 0)
                adaptive_f1 = adaptive_result.get('test_f1', 0)

                # Best method
                methods = [('Raw', raw_f1), ('Fixed Kalman', kalman_f1), ('Adaptive Kalman', adaptive_f1)]
                best = max(methods, key=lambda x: x[1])
                f.write(f"- **Best method:** {best[0]} ({best[1]:.2f}% F1)\n")

                # Adaptive vs baselines
                f.write(f"- **Adaptive vs Raw:** {'+' if adaptive_f1 >= raw_f1 else ''}{adaptive_f1 - raw_f1:.2f}% F1\n")
                f.write(f"- **Adaptive vs Fixed Kalman:** {'+' if adaptive_f1 >= kalman_f1 else ''}{adaptive_f1 - kalman_f1:.2f}% F1\n")

                # Hypothesis check
                if dataset == 'upfall':
                    if adaptive_f1 >= raw_f1 - 0.5:
                        f.write("- ✓ Hypothesis supported: Adaptive approaches Raw performance\n")
                    else:
                        f.write("- ✗ Hypothesis not supported: Adaptive underperforms Raw\n")
                else:  # wedafall
                    if adaptive_f1 >= kalman_f1 - 0.5:
                        f.write("- ✓ Hypothesis supported: Adaptive maintains Kalman performance\n")
                    else:
                        f.write("- ✗ Hypothesis not supported: Adaptive underperforms Kalman\n")
            else:
                f.write("- Incomplete results - cannot determine findings\n")

            f.write("\n")

        # Conclusion
        f.write("## Conclusion\n\n")
        f.write("TODO: Add conclusion based on results\n\n")

    print(f"\nComparison table saved: {table_path}")


def generate_text_report(results: List[Dict], output_dir: Path):
    """Generate plain text comparison report."""
    report_path = output_dir / 'adaptive_comparison_results.txt'

    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("ADAPTIVE KALMAN FILTER COMPARISON RESULTS\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * 70 + "\n\n")

        for dataset in ['upfall', 'wedafall']:
            dataset_results = [r for r in results if r['dataset'] == dataset]
            if not dataset_results:
                continue

            f.write(f"{DATASET_INFO[dataset]['name']} ({DATASET_INFO[dataset]['sensor_type']})\n")
            f.write("-" * 40 + "\n")

            for r in sorted(dataset_results, key=lambda x: ['raw', 'kalman', 'adaptive'].index(x['method']) if x['method'] in ['raw', 'kalman', 'adaptive'] else 99):
                f.write(f"\n{r['method'].upper()} ({r['kalman_type']}):\n")
                f.write(f"  Model: {r['model']}\n")
                f.write(f"  Channels: {r['channels']}\n")
                if r.get('status') == 'success':
                    f.write(f"  F1 (Binary): {r.get('test_f1', 0):.2f}% ± {r.get('test_f1_std', 0):.2f}%\n")
                    f.write(f"  Macro-F1: {r.get('test_macro_f1', 0):.2f}% ± {r.get('test_macro_f1_std', 0):.2f}%\n")
                    f.write(f"  Accuracy: {r.get('test_acc', 0):.2f}%\n")
                    f.write(f"  Precision: {r.get('test_prec', 0):.2f}%\n")
                    f.write(f"  Recall: {r.get('test_recall', 0):.2f}%\n")
                    f.write(f"  AUC: {r.get('test_auc', 0):.2f}%\n")
                else:
                    f.write(f"  Status: {r.get('status', 'unknown').upper()}\n")
                    f.write(f"  Error: {r.get('error', 'Unknown')}\n")

            # Comparison summary
            raw = next((r for r in dataset_results if r['method'] == 'raw' and r.get('status') == 'success'), None)
            kalman = next((r for r in dataset_results if r['method'] == 'kalman' and r.get('status') == 'success'), None)
            adaptive = next((r for r in dataset_results if r['method'] == 'adaptive' and r.get('status') == 'success'), None)

            if adaptive:
                f.write(f"\nCOMPARISON:\n")
                if raw:
                    diff = adaptive.get('test_f1', 0) - raw.get('test_f1', 0)
                    f.write(f"  Adaptive vs Raw: {'+' if diff >= 0 else ''}{diff:.2f}% F1\n")
                if kalman:
                    diff = adaptive.get('test_f1', 0) - kalman.get('test_f1', 0)
                    f.write(f"  Adaptive vs Kalman: {'+' if diff >= 0 else ''}{diff:.2f}% F1\n")

            f.write("\n")

        f.write("=" * 70 + "\n")

    print(f"\nText report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Adaptive Kalman Filter comparison study')
    parser.add_argument('--datasets', nargs='+', default=['upfall', 'wedafall'],
                        choices=['upfall', 'wedafall'],
                        help='Datasets to compare (default: both)')
    parser.add_argument('--methods', nargs='+', default=None,
                        choices=['raw', 'kalman', 'adaptive'],
                        help='Run only specific methods (default: all)')
    parser.add_argument('--num-gpus', type=int, default=8,
                        help='Number of GPUs to use (default: 8)')
    parser.add_argument('--max-folds', type=int, default=None,
                        help='Maximum folds per experiment (for quick testing)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: exps/adaptive_comparison_<timestamp>)')

    args = parser.parse_args()

    # Generate timestamped output directory if not specified
    if args.output_dir:
        output_dir = PROJECT_ROOT / args.output_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = PROJECT_ROOT / f'exps/adaptive_comparison_{timestamp}'

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    all_results = []

    for dataset in args.datasets:
        print(f"\n{'='*70}")
        print(f"Dataset: {DATASET_INFO[dataset]['name']}")
        print(f"Sensor Type: {DATASET_INFO[dataset]['sensor_type']}")
        print(f"Subjects: {DATASET_INFO[dataset]['num_subjects']}")
        print(f"Validation: {DATASET_INFO[dataset]['val_subjects']}")
        print(f"LOSO Folds: {DATASET_INFO[dataset]['num_folds']}")
        print(f"Hypothesis: {DATASET_INFO[dataset]['hypothesis']}")
        print(f"{'='*70}\n")

        results = run_comparison(
            dataset=dataset,
            output_dir=output_dir,
            num_gpus=args.num_gpus,
            max_folds=args.max_folds,
            methods=args.methods,
        )
        all_results.extend(results)

    # Save final results
    with open(output_dir / 'adaptive_comparison_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Generate reports
    generate_comparison_table(all_results, output_dir)
    generate_text_report(all_results, output_dir)

    print(f"\n{'='*70}")
    print("Adaptive Kalman comparison study complete!")
    print(f"{'='*70}")
    print(f"Results JSON: {output_dir / 'adaptive_comparison_results.json'}")
    print(f"Markdown table: {output_dir / 'adaptive_comparison_table.md'}")
    print(f"Text report: {output_dir / 'adaptive_comparison_results.txt'}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
