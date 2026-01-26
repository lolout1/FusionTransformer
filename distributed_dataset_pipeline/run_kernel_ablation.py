#!/usr/bin/env python3
"""
Kernel Size Ablation Study.

Tests different Conv1D kernel sizes on UP-FALL and WEDA-FALL:
- kernel_size: 3, 5, 8, 11, 15

Uses raw 6-channel input (best performing) with SingleStreamTransformer.
Runs experiments sequentially to avoid Ray conflicts.
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent

# Dataset metadata
DATASET_INFO = {
    'upfall': {
        'name': 'UP-FALL',
        'num_folds': 15,
        'base_config': 'config/upfall/dual_stream_raw.yaml',
    },
    'wedafall': {
        'name': 'WEDA-FALL',
        'num_folds': 12,
        'base_config': 'config/wedafall/single_stream_raw.yaml',
    }
}

# Kernel sizes to test
KERNEL_SIZES = [3, 5, 8, 11, 15]


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

    # Extract validation F1
    match = re.search(r'Val F1:\s+([\d.]+)%', content)
    if match:
        results['val_f1'] = float(match.group(1))

    return results


def create_kernel_config(base_config_path: str, kernel_size: int, output_dir: Path) -> str:
    """Create a config file with modified kernel size."""
    with open(PROJECT_ROOT / base_config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Modify kernel size in model_args
    if 'model_args' not in config:
        config['model_args'] = {}

    config['model_args']['kernel_size'] = kernel_size

    # Save to temp config file
    config_path = output_dir / f'kernel_{kernel_size}.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return str(config_path)


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
    kernel_sizes: Optional[List[int]] = None,
) -> List[Dict]:
    """Run kernel size ablation study for a dataset."""
    results = []
    kernels = kernel_sizes or KERNEL_SIZES

    # Create config directory
    config_dir = output_dir / 'configs' / dataset
    config_dir.mkdir(parents=True, exist_ok=True)

    for kernel_size in kernels:
        exp_name = f"{dataset}_kernel_{kernel_size}"
        work_dir = str(output_dir / exp_name)

        print(f"\n{'#'*70}")
        print(f"# {dataset.upper()} | Kernel Size: {kernel_size}")
        print(f"{'#'*70}")

        # Create config with this kernel size
        config_path = create_kernel_config(
            DATASET_INFO[dataset]['base_config'],
            kernel_size,
            config_dir
        )

        metrics = run_experiment(
            config_path=config_path,
            work_dir=work_dir,
            num_gpus=num_gpus,
            max_folds=max_folds,
        )

        result = {
            'name': exp_name,
            'dataset': dataset,
            'kernel_size': kernel_size,
            'config_path': config_path,
            'work_dir': work_dir,
            **metrics,
        }
        results.append(result)

        # Save intermediate results
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'kernel_ablation_results.json', 'w') as f:
            json.dump(results, f, indent=2)

    return results


def generate_report(results: List[Dict], output_dir: Path):
    """Generate markdown report."""
    report_path = output_dir / 'kernel_ablation_report.md'

    with open(report_path, 'w') as f:
        f.write("# Kernel Size Ablation Study\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("Tests Conv1D kernel sizes: 3, 5, 8, 11, 15\n\n")

        for dataset in ['upfall', 'wedafall']:
            dataset_results = [r for r in results if r['dataset'] == dataset]
            if not dataset_results:
                continue

            f.write(f"## {DATASET_INFO[dataset]['name']}\n\n")
            f.write("| Kernel | F1 (%) | Std | Macro-F1 | Acc (%) |\n")
            f.write("|--------|--------|-----|----------|--------|\n")

            # Sort by F1 descending
            sorted_results = sorted(
                dataset_results,
                key=lambda x: x.get('test_f1', 0),
                reverse=True
            )

            for r in sorted_results:
                f1 = f"{r.get('test_f1', 0):.2f}" if 'test_f1' in r else 'N/A'
                std = f"{r.get('test_f1_std', 0):.2f}" if 'test_f1_std' in r else 'N/A'
                macro_f1 = f"{r.get('test_macro_f1', 0):.2f}" if 'test_macro_f1' in r else 'N/A'
                acc = f"{r.get('test_acc', 0):.2f}" if 'test_acc' in r else 'N/A'
                status = '' if r.get('status') == 'success' else ' (FAILED)'
                f.write(f"| {r['kernel_size']}{status} | {f1} | {std} | {macro_f1} | {acc} |\n")

            f.write("\n")

        # Summary
        f.write("## Key Findings\n\n")
        for dataset in ['upfall', 'wedafall']:
            dataset_results = [r for r in results if r['dataset'] == dataset and r.get('status') == 'success']
            if dataset_results:
                best = max(dataset_results, key=lambda x: x.get('test_f1', 0))
                f.write(f"**{DATASET_INFO[dataset]['name']}** best kernel: `{best['kernel_size']}` - F1={best.get('test_f1', 0):.2f}%\n\n")

        # Analysis
        f.write("## Analysis\n\n")
        f.write("### Kernel Size Interpretation\n\n")
        f.write("- **Small kernels (3-5)**: Capture fine-grained, high-frequency patterns\n")
        f.write("- **Medium kernels (8)**: Balance between local and broader patterns\n")
        f.write("- **Large kernels (11-15)**: Capture longer temporal dependencies\n\n")

        f.write("### Receptive Field (at 50Hz)\n\n")
        f.write("| Kernel | Samples | Time Window |\n")
        f.write("|--------|---------|-------------|\n")
        f.write("| 3 | 3 | 60ms |\n")
        f.write("| 5 | 5 | 100ms |\n")
        f.write("| 8 | 8 | 160ms |\n")
        f.write("| 11 | 11 | 220ms |\n")
        f.write("| 15 | 15 | 300ms |\n")

    print(f"\nReport saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Kernel size ablation study')
    parser.add_argument('--datasets', nargs='+', default=['upfall', 'wedafall'],
                       choices=['upfall', 'wedafall'],
                       help='Datasets to run ablation on')
    parser.add_argument('--kernel-sizes', nargs='+', type=int, default=None,
                       help='Specific kernel sizes to test (default: 3,5,8,11,15)')
    parser.add_argument('--num-gpus', type=int, default=6,
                       help='Number of GPUs to use')
    parser.add_argument('--max-folds', type=int, default=None,
                       help='Maximum folds per experiment (for quick testing)')
    parser.add_argument('--output-dir', type=str, default='exps/kernel_ablation',
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
            kernel_sizes=args.kernel_sizes,
        )
        all_results.extend(results)

    # Save final results
    with open(output_dir / 'kernel_ablation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Generate report
    generate_report(all_results, output_dir)

    print(f"\n{'='*70}")
    print("Kernel size ablation study complete!")
    print(f"Results: {output_dir / 'kernel_ablation_results.json'}")
    print(f"Report: {output_dir / 'kernel_ablation_report.md'}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
