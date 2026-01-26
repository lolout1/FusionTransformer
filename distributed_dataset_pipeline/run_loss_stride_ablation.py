#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent

DATASET_INFO = {
    'upfall': {
        'name': 'UP-FALL',
        'base_config': 'config/upfall/dual_stream_raw.yaml',
        'natural_fall_ratio': 0.13,
        'current_fall_ratio': 0.37,
        'num_folds': 15,
    },
    'wedafall': {
        'name': 'WEDA-FALL',
        'base_config': 'config/wedafall/dual_stream_raw.yaml',
        'natural_fall_ratio': 0.31,
        'current_fall_ratio': 0.68,
        'num_folds': 12,
    }
}

ABLATION_CONFIGS = {
    'loss': [
        {'name': 'focal', 'loss_type': 'focal', 'description': 'Focal Loss (α=0.75, γ=2)'},
        {'name': 'bce', 'loss_type': 'bce', 'description': 'Binary Cross-Entropy'},
        {'name': 'cb_focal', 'loss_type': 'cb_focal', 'description': 'Class-Balanced Focal Loss'},
    ],
    'stride_upfall': [
        {'name': 'original', 'fall_stride': 16, 'adl_stride': 64, 'description': '~37% falls'},
        {'name': 'moderate', 'fall_stride': 24, 'adl_stride': 48, 'description': '~30% falls'},
        {'name': 'equal', 'fall_stride': 32, 'adl_stride': 32, 'description': '~13% falls'},
        {'name': 'aggressive', 'fall_stride': 8, 'adl_stride': 64, 'description': '~50% falls'},
    ],
    'stride_wedafall': [
        {'name': 'original', 'fall_stride': 16, 'adl_stride': 64, 'description': '~68% falls'},
        {'name': 'reduced', 'fall_stride': 32, 'adl_stride': 48, 'description': '~50% falls'},
        {'name': 'equal', 'fall_stride': 32, 'adl_stride': 32, 'description': '~31% falls'},
        {'name': 'inverted', 'fall_stride': 48, 'adl_stride': 24, 'description': '~19% falls'},
        {'name': 'mild_reduce', 'fall_stride': 24, 'adl_stride': 48, 'description': '~55% falls'},
    ],
}


def parse_summary_report(summary_path: Path) -> Dict:
    results = {}
    if not summary_path.exists():
        return {'error': 'Summary file not found'}

    content = summary_path.read_text()

    match = re.search(r'Test F1:\s+([\d.]+)\s*±\s*([\d.]+)%', content)
    if match:
        results['test_f1'] = float(match.group(1))
        results['test_f1_std'] = float(match.group(2))

    match = re.search(r'Test Accuracy:\s+([\d.]+)\s*±\s*([\d.]+)%', content)
    if match:
        results['test_acc'] = float(match.group(1))
        results['test_acc_std'] = float(match.group(2))

    match = re.search(r'Val F1:\s+([\d.]+)%', content)
    if match:
        results['val_f1'] = float(match.group(1))

    return results


def run_experiment(
    base_config: str,
    work_dir: str,
    num_gpus: int,
    max_folds: Optional[int],
    loss_type: Optional[str] = None,
    fall_stride: Optional[int] = None,
    adl_stride: Optional[int] = None,
) -> Tuple[bool, Dict]:
    cmd = [
        sys.executable, 'ray_train.py',
        '--config', base_config,
        '--num-gpus', str(num_gpus),
        '--work-dir', work_dir,
    ]

    if loss_type:
        cmd.extend(['--loss-type', loss_type])
    if fall_stride:
        cmd.extend(['--fall-stride', str(fall_stride)])
    if adl_stride:
        cmd.extend(['--adl-stride', str(adl_stride)])
    if max_folds:
        cmd.extend(['--max-folds', str(max_folds)])

    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        summary_path = Path(work_dir) / 'summary_report.txt'
        return True, parse_summary_report(summary_path)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Experiment failed with code {e.returncode}")
        return False, {'error': str(e)}


def run_ablation(
    ablation_name: str,
    dataset: str,
    configs: List[Dict],
    base_config: str,
    output_dir: Path,
    num_gpus: int,
    max_folds: Optional[int],
) -> List[Dict]:
    results = []

    for config in configs:
        exp_name = f"{dataset}_{ablation_name}_{config['name']}"
        work_dir = str(output_dir / exp_name)

        print(f"\n{'#'*70}")
        print(f"# {ablation_name} | {dataset} | {config['name']}: {config['description']}")
        print(f"{'#'*70}")

        success, metrics = run_experiment(
            base_config=base_config,
            work_dir=work_dir,
            num_gpus=num_gpus,
            max_folds=max_folds,
            loss_type=config.get('loss_type'),
            fall_stride=config.get('fall_stride'),
            adl_stride=config.get('adl_stride'),
        )

        result = {
            'name': exp_name,
            'ablation': ablation_name,
            'dataset': dataset,
            'config': config['name'],
            'description': config['description'],
            'status': 'success' if success else 'failed',
            'work_dir': work_dir,
            **config,
            **metrics,
        }
        results.append(result)

        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'ablation_results.json', 'w') as f:
            json.dump(results, f, indent=2)

    return results


def generate_markdown_report(results: List[Dict], output_dir: Path):
    report_path = output_dir / 'ablation_report.md'

    with open(report_path, 'w') as f:
        f.write("# Ablation Study: Loss Functions and Stride Configurations\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        ablations = {}
        for r in results:
            key = (r['ablation'], r['dataset'])
            ablations.setdefault(key, []).append(r)

        f.write("## Loss Function Comparison\n\n")
        for dataset in ['upfall', 'wedafall']:
            key = ('loss', dataset)
            if key in ablations:
                f.write(f"### {DATASET_INFO[dataset]['name']}\n\n")
                f.write("| Loss | F1 (%) | Std | Description |\n")
                f.write("|------|--------|-----|-------------|\n")
                for r in ablations[key]:
                    f1 = r.get('test_f1', 'N/A')
                    std = r.get('test_f1_std', 'N/A')
                    f.write(f"| {r['config']} | {f1} | {std} | {r['description']} |\n")
                f.write("\n")

        f.write("## Stride Configuration Comparison\n\n")
        for dataset in ['upfall', 'wedafall']:
            key = (f'stride_{dataset}', dataset)
            if key in ablations:
                f.write(f"### {DATASET_INFO[dataset]['name']}\n\n")
                f.write("| Config | Fall Stride | ADL Stride | F1 (%) | Std |\n")
                f.write("|--------|-------------|------------|--------|-----|\n")
                for r in ablations[key]:
                    f1 = r.get('test_f1', 'N/A')
                    std = r.get('test_f1_std', 'N/A')
                    f.write(f"| {r['config']} | {r.get('fall_stride', '-')} | {r.get('adl_stride', '-')} | {f1} | {std} |\n")
                f.write("\n")

        f.write("## Best Configurations\n\n")
        for dataset in ['upfall', 'wedafall']:
            best_f1 = 0
            best_config = None
            for r in results:
                if r['dataset'] == dataset and r.get('test_f1', 0) > best_f1:
                    best_f1 = r.get('test_f1', 0)
                    best_config = r
            if best_config:
                f.write(f"**{DATASET_INFO[dataset]['name']}**: `{best_config['config']}` ({best_config['ablation']}) - F1={best_f1:.2f}%\n\n")

    print(f"Report saved: {report_path}")


def update_exps_readme(results: List[Dict]):
    readme_path = PROJECT_ROOT / 'exps' / 'README.md'

    if readme_path.exists():
        content = readme_path.read_text()
    else:
        content = "# Experiment Results Summary\n\n"

    ablation_header = "## Ablation: Loss & Stride"
    if ablation_header not in content:
        content += f"\n---\n\n{ablation_header}\n\n"

    ablation_content = f"{ablation_header}\n\n"
    ablation_content += f"*Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n"
    ablation_content += "### Loss Function Comparison\n\n"
    ablation_content += "| Dataset | Focal | BCE | CB-Focal | Best |\n"
    ablation_content += "|---------|-------|-----|----------|------|\n"

    for dataset in ['upfall', 'wedafall']:
        loss_results = {r['config']: r.get('test_f1', 0)
                       for r in results
                       if r['dataset'] == dataset and r['ablation'] == 'loss'}
        if loss_results:
            best = max(loss_results, key=loss_results.get)
            ablation_content += f"| {DATASET_INFO[dataset]['name']} | "
            ablation_content += f"{loss_results.get('focal', '-')}% | "
            ablation_content += f"{loss_results.get('bce', '-')}% | "
            ablation_content += f"{loss_results.get('cb_focal', '-')}% | "
            ablation_content += f"**{best}** |\n"

    ablation_content += "\n"

    if ablation_header in content:
        pattern = rf"{re.escape(ablation_header)}.*?(?=\n## |\n---|\Z)"
        content = re.sub(pattern, ablation_content, content, flags=re.DOTALL)
    else:
        content += ablation_content

    readme_path.write_text(content)
    print(f"Updated: {readme_path}")


def main():
    parser = argparse.ArgumentParser(description='Loss and stride ablation study')
    parser.add_argument('--datasets', nargs='+', default=['upfall', 'wedafall'],
                       choices=['upfall', 'wedafall'])
    parser.add_argument('--ablations', nargs='+', default=['loss', 'stride'],
                       choices=['loss', 'stride'])
    parser.add_argument('--num-gpus', type=int, default=4)
    parser.add_argument('--max-folds', type=int, default=None)
    parser.add_argument('--output-dir', type=str, default='exps/ablation_loss_stride')

    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for dataset in args.datasets:
        info = DATASET_INFO[dataset]
        print(f"\n{'='*70}")
        print(f"Dataset: {info['name']}")
        print(f"{'='*70}\n")

        if 'loss' in args.ablations:
            results = run_ablation(
                ablation_name='loss',
                dataset=dataset,
                configs=ABLATION_CONFIGS['loss'],
                base_config=info['base_config'],
                output_dir=output_dir,
                num_gpus=args.num_gpus,
                max_folds=args.max_folds,
            )
            all_results.extend(results)

        if 'stride' in args.ablations:
            stride_key = f'stride_{dataset}'
            if stride_key in ABLATION_CONFIGS:
                results = run_ablation(
                    ablation_name=stride_key,
                    dataset=dataset,
                    configs=ABLATION_CONFIGS[stride_key],
                    base_config=info['base_config'],
                    output_dir=output_dir,
                    num_gpus=args.num_gpus,
                    max_folds=args.max_folds,
                )
                all_results.extend(results)

    with open(output_dir / 'ablation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    generate_markdown_report(all_results, output_dir)
    update_exps_readme(all_results)

    print(f"\n{'='*70}")
    print("Ablation study complete!")
    print(f"Results: {output_dir / 'ablation_results.json'}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
