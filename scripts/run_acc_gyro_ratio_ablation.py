#!/usr/bin/env python3
"""
AccGyroKalman Ratio Ablation Study.

Tests different capacity allocations between acc+orientation and gyro streams:
- 50:50 (equal)
- 65:35 (acc+ori favored)
- 70:30 (current default)

Usage:
    python scripts/run_acc_gyro_ratio_ablation.py --num-gpus 8
    python scripts/run_acc_gyro_ratio_ablation.py --quick --num-gpus 2
    python scripts/run_acc_gyro_ratio_ablation.py --dataset wedafall
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
# Configuration
# =============================================================================

RATIOS = {
    'ratio_50_50': 0.50,  # Equal capacity
    'ratio_65_35': 0.65,  # More for acc+ori
    'ratio_70_30': 0.70,  # Current default
}

DATASETS = {
    'wedafall': {
        'base_config': 'config/best_config/wedafall/acc_gyro_kalman.yaml',
        'num_folds': 12,
        'include_elderly': False,
    },
    'wedafall_all': {
        'base_config': 'config/best_config/wedafall/acc_gyro_kalman.yaml',
        'num_folds': 12,  # Still 12 folds (young test subjects only, but elderly in train)
        'include_elderly': True,
    },
    'upfall': {
        'base_config': 'config/best_config/upfall/acc_gyro_kalman.yaml',
        'num_folds': 15,
        'include_elderly': False,
    },
}


def load_config(path: str) -> dict:
    with open(PROJECT_ROOT / path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def generate_configs(output_dir: Path, datasets: list) -> list:
    """Generate all experiment configs."""
    experiments = []

    for ds_name in datasets:
        ds_info = DATASETS[ds_name]
        base_config = load_config(ds_info['base_config'])

        for ratio_name, ratio_value in RATIOS.items():
            config = deepcopy(base_config)
            config['model_args']['acc_ori_ratio'] = ratio_value

            # Handle include_elderly for wedafall_all
            if ds_info.get('include_elderly', False):
                config['dataset_args']['include_elderly'] = True

            exp_name = f"{ds_name}_{ratio_name}"
            config_path = output_dir / 'configs' / f"{exp_name}.yaml"
            save_config(config, config_path)

            experiments.append({
                'name': exp_name,
                'dataset': ds_name,
                'ratio': ratio_name,
                'ratio_value': ratio_value,
                'config_path': str(config_path),
                'num_folds': ds_info['num_folds'],
                'include_elderly': ds_info.get('include_elderly', False),
            })

    return experiments


def run_experiment(exp: dict, output_dir: Path, num_gpus: int, max_folds: int = None) -> dict:
    """Run single experiment."""
    work_dir = output_dir / exp['name']
    work_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, 'ray_train.py',
        '--config', exp['config_path'],
        '--num-gpus', str(num_gpus),
        '--work-dir', str(work_dir),
    ]

    if max_folds:
        cmd.extend(['--max-folds', str(max_folds)])

    print(f"\n{'='*60}")
    print(f"Running: {exp['name']}")
    print(f"Ratio: {exp['ratio_value']*100:.0f}:{(1-exp['ratio_value'])*100:.0f} (acc+ori:gyro)")
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
        'name': exp['name'],
        'dataset': exp['dataset'],
        'ratio': exp['ratio'],
        'ratio_value': exp['ratio_value'],
        'status': 'success' if proc.returncode == 0 else 'failed',
    }

    # Try to parse summary
    summary_path = work_dir / 'summary_report.txt'
    if summary_path.exists():
        content = summary_path.read_text()
        import re
        f1_match = re.search(r'Test F1:\s+([\d.]+)\s*±\s*([\d.]+)%', content)
        if f1_match:
            result['test_f1'] = float(f1_match.group(1))
            result['test_f1_std'] = float(f1_match.group(2))

        acc_match = re.search(r'Test Accuracy:\s+([\d.]+)\s*±\s*([\d.]+)%', content)
        if acc_match:
            result['test_acc'] = float(acc_match.group(1))
            result['test_acc_std'] = float(acc_match.group(2))

    return result


def generate_report(results: list, output_dir: Path):
    """Generate markdown report."""
    report_path = output_dir / 'ratio_ablation_report.md'

    lines = [
        "# AccGyroKalman Ratio Ablation Results",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "## Configuration",
        "- Stream 1: Acc + Orientation [smv, ax, ay, az, roll, pitch] (6ch)",
        "- Stream 2: Raw Gyroscope [gx, gy, gz] (3ch)",
        "- Ratios tested: 50:50, 65:35, 70:30 (acc+ori:gyro)",
        "",
        "## Results",
        "",
        "| Dataset | Ratio | Acc+Ori Dim | Gyro Dim | Test F1 | Std |",
        "|---------|-------|-------------|----------|---------|-----|",
    ]

    for r in sorted(results, key=lambda x: (x['dataset'], x['ratio_value'])):
        if r['status'] == 'success' and 'test_f1' in r:
            acc_ori_dim = int(48 * r['ratio_value'])
            gyro_dim = 48 - acc_ori_dim
            lines.append(
                f"| {r['dataset']} | {r['ratio']} | {acc_ori_dim} | {gyro_dim} | "
                f"{r['test_f1']:.2f}% | ±{r.get('test_f1_std', 0):.2f} |"
            )
        else:
            lines.append(f"| {r['dataset']} | {r['ratio']} | - | - | FAILED | - |")

    # Best per dataset
    lines.append("\n## Best Results per Dataset\n")
    all_datasets = sorted(set(r['dataset'] for r in results))
    for ds in all_datasets:
        ds_results = [r for r in results if r['dataset'] == ds and r['status'] == 'success' and 'test_f1' in r]
        if ds_results:
            best = max(ds_results, key=lambda x: x['test_f1'])
            elderly_note = " (young+elderly)" if 'all' in ds else " (young only)" if 'wedafall' in ds else ""
            lines.append(f"- **{ds.upper()}**{elderly_note}: {best['ratio']} ({best['ratio_value']*100:.0f}:{(1-best['ratio_value'])*100:.0f}) - {best['test_f1']:.2f}% ± {best.get('test_f1_std', 0):.2f}%")

    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\nReport saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='AccGyroKalman Ratio Ablation')
    parser.add_argument('--num-gpus', type=int, default=8, help='Number of GPUs')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Single dataset (wedafall/wedafall_all/upfall)')
    parser.add_argument('--datasets', type=str, default=None,
                       help='Comma-separated datasets')
    parser.add_argument('--quick', action='store_true', help='Quick test (2 folds)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    args = parser.parse_args()

    # Setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / 'exps' / f'acc_gyro_ratio_ablation_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine datasets
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(',')]
    elif args.dataset:
        datasets = [args.dataset]
    else:
        datasets = ['wedafall', 'wedafall_all', 'upfall']  # Default: all including elderly
    max_folds = 2 if args.quick else None

    print(f"Output: {output_dir}")
    print(f"Datasets: {datasets}")
    print(f"GPUs: {args.num_gpus}")
    print(f"Ratios: {list(RATIOS.keys())}")

    # Generate configs
    experiments = generate_configs(output_dir, datasets)
    print(f"\nGenerated {len(experiments)} experiments")

    # Run experiments
    results = []
    for i, exp in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] {exp['name']}")
        result = run_experiment(exp, output_dir, args.num_gpus, max_folds)
        results.append(result)

        # Save partial results
        with open(output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)

    # Generate report
    generate_report(results, output_dir)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in results:
        if r['status'] == 'success' and 'test_f1' in r:
            print(f"{r['name']}: {r['test_f1']:.2f}% ± {r.get('test_f1_std', 0):.2f}%")
        else:
            print(f"{r['name']}: FAILED")


if __name__ == '__main__':
    main()
