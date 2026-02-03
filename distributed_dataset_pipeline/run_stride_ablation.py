#!/usr/bin/env python3
"""
Stride Ablation: Fall stride=10, ADL stride varies 10-50

Tests class-aware stride effect with fixed fall_stride=10 and varying adl_stride.

Usage:
    python distributed_dataset_pipeline/run_stride_ablation.py --num-gpus 4 --parallel 2
    python distributed_dataset_pipeline/run_stride_ablation.py --quick
"""

import argparse
import copy
import json
import pickle
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

BASE_CONFIG = 'config/best_config/smartfallmm/kalman.yaml'

# ADL stride values to test (fall stride fixed at 10)
ADL_STRIDES = [10, 20, 30, 40, 50]
FALL_STRIDE = 10


@dataclass
class ExperimentResult:
    name: str
    fall_stride: int
    adl_stride: int
    test_f1: float = 0.0
    test_f1_std: float = 0.0
    test_accuracy: float = 0.0
    test_precision: float = 0.0
    test_recall: float = 0.0
    num_folds: int = 0
    status: str = 'pending'
    error_message: str = ''
    elapsed_time: float = 0.0
    fold_f1s: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def load_base_config() -> Dict[str, Any]:
    with open(BASE_CONFIG) as f:
        return yaml.safe_load(f)


def create_stride_config(fall_stride: int, adl_stride: int) -> Dict[str, Any]:
    cfg = load_base_config()

    # Enable class-aware stride (unless both are equal)
    cfg['dataset_args']['enable_class_aware_stride'] = (fall_stride != adl_stride)
    cfg['dataset_args']['stride'] = adl_stride
    cfg['dataset_args']['fall_stride'] = fall_stride
    cfg['dataset_args']['adl_stride'] = adl_stride

    return cfg


def run_experiment(
    name: str,
    config: Dict[str, Any],
    work_dir: Path,
    num_gpus: int,
    max_folds: Optional[int],
    timeout: int = 7200,
) -> ExperimentResult:
    import time
    start = time.time()

    result = ExperimentResult(
        name=name,
        fall_stride=config['dataset_args']['fall_stride'],
        adl_stride=config['dataset_args']['adl_stride'],
    )

    config_path = work_dir / f'{name}.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    cmd = [
        sys.executable, 'ray_train.py',
        '--config', str(config_path),
        '--work-dir', str(work_dir / name),
        '--num-gpus', str(num_gpus),
    ]
    if max_folds:
        cmd.extend(['--max-folds', str(max_folds)])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=Path.cwd())
        if proc.returncode != 0:
            result.status = 'failed'
            result.error_message = proc.stderr[-500:] if proc.stderr else 'Unknown'
        else:
            result.status = 'completed'
            results_path = work_dir / name / 'fold_results.pkl'
            if results_path.exists():
                result = parse_fold_results(results_path, result)
    except subprocess.TimeoutExpired:
        result.status = 'timeout'
        result.error_message = f'Timeout after {timeout}s'
    except Exception as e:
        result.status = 'error'
        result.error_message = str(e)

    result.elapsed_time = time.time() - start
    return result


def parse_fold_results(path: Path, result: ExperimentResult) -> ExperimentResult:
    with open(path, 'rb') as f:
        data = pickle.load(f)

    if not data:
        return result

    fold_list = list(data.values()) if isinstance(data, dict) else data
    f1s, accs, precs, recs = [], [], [], []

    for fold in fold_list:
        if not isinstance(fold, dict):
            continue
        test = fold.get('test', {})
        if isinstance(test, dict):
            f1 = test.get('f1_score') or test.get('f1') or test.get('macro_f1', 0)
            acc = test.get('accuracy') or test.get('acc', 0)
            prec = test.get('precision', 0)
            rec = test.get('recall', 0)
        else:
            f1 = fold.get('test_f1', 0)
            acc = fold.get('test_accuracy', 0)
            prec = fold.get('test_precision', 0)
            rec = fold.get('test_recall', 0)

        if f1 and f1 <= 1: f1 *= 100
        if acc and acc <= 1: acc *= 100
        if prec and prec <= 1: prec *= 100
        if rec and rec <= 1: rec *= 100

        if f1 > 0:
            f1s.append(f1)
            accs.append(acc)
            precs.append(prec)
            recs.append(rec)

    if f1s:
        result.test_f1 = statistics.mean(f1s)
        result.test_f1_std = statistics.stdev(f1s) if len(f1s) > 1 else 0
        result.test_accuracy = statistics.mean(accs) if accs else 0
        result.test_precision = statistics.mean(precs) if precs else 0
        result.test_recall = statistics.mean(recs) if recs else 0
        result.num_folds = len(f1s)
        result.fold_f1s = f1s

    return result


def run_parallel(experiments, work_dir, num_gpus, parallel, max_folds):
    gpus_per = max(1, num_gpus // parallel)
    results = []

    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futures = {
            ex.submit(run_experiment, e['name'], e['config'], work_dir / 'runs', gpus_per, max_folds): e['name']
            for e in experiments
        }
        for f in as_completed(futures):
            name = futures[f]
            try:
                r = f.result()
                results.append(r)
                status = '✓' if r.status == 'completed' else '✗'
                f1 = f'{r.test_f1:.2f}%' if r.test_f1 > 0 else 'N/A'
                print(f'  {status} {name}: F1={f1}')
            except Exception as e:
                print(f'  ✗ {name}: {e}')
                results.append(ExperimentResult(name=name, fall_stride=0, adl_stride=0, status='error', error_message=str(e)))

    return results


def generate_report(results: List[ExperimentResult], output_path: Path) -> str:
    lines = [
        '# Stride Ablation Results',
        f'\nGenerated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
        '',
        '## Configuration',
        '',
        '- Dataset: SmartFallMM',
        '- Model: KalmanConv1dConv1d',
        f'- Fall stride: {FALL_STRIDE} (fixed)',
        f'- ADL stride: {ADL_STRIDES} (varied)',
        '',
        '## Results',
        '',
        '| Fall Stride | ADL Stride | F1 (%) | Acc (%) | Prec (%) | Rec (%) | Folds |',
        '|-------------|------------|--------|---------|----------|---------|-------|',
    ]

    results.sort(key=lambda x: x.adl_stride)
    best_f1 = max(r.test_f1 for r in results) if results else 0

    for r in results:
        marker = ' **best**' if abs(r.test_f1 - best_f1) < 0.01 else ''
        lines.append(
            f'| {r.fall_stride} | {r.adl_stride} | {r.test_f1:.2f} ± {r.test_f1_std:.2f}{marker} | '
            f'{r.test_accuracy:.2f} | {r.test_precision:.2f} | {r.test_recall:.2f} | {r.num_folds} |'
        )

    lines.extend(['', '## Summary', ''])
    if results:
        best = max(results, key=lambda x: x.test_f1)
        lines.append(f'**Best config**: fall_stride={best.fall_stride}, adl_stride={best.adl_stride} ({best.test_f1:.2f}% F1)')

    report = '\n'.join(lines)
    with open(output_path, 'w') as f:
        f.write(report)
    return report


def main():
    parser = argparse.ArgumentParser(description='Stride ablation (fall=10, ADL=10-50)')
    parser.add_argument('--num-gpus', type=int, default=4)
    parser.add_argument('--parallel', type=int, default=2)
    parser.add_argument('--max-folds', type=int, default=None)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--work-dir', type=Path, default=None)
    parser.add_argument('--results-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    if args.quick:
        args.max_folds = 2
    if args.work_dir is None:
        args.work_dir = Path(f'exps/stride_ablation_{datetime.now().strftime("%Y%m%d_%H%M%S")}')

    if args.results_only:
        results_path = args.work_dir / 'results.json'
        if results_path.exists():
            with open(results_path) as f:
                results = [ExperimentResult(**r) for r in json.load(f)]
        else:
            results = []
            for d in sorted((args.work_dir / 'runs').iterdir()):
                if d.is_dir() and (d / 'fold_results.pkl').exists():
                    parts = d.name.split('_')
                    fall_s = int(parts[1]) if len(parts) > 1 else 10
                    adl_s = int(parts[3]) if len(parts) > 3 else 10
                    r = ExperimentResult(name=d.name, fall_stride=fall_s, adl_stride=adl_s)
                    r = parse_fold_results(d / 'fold_results.pkl', r)
                    r.status = 'completed' if r.test_f1 > 0 else 'failed'
                    results.append(r)
        print(generate_report(results, args.work_dir / 'stride_ablation_report.md'))
        return

    experiments = []
    for adl_stride in ADL_STRIDES:
        name = f'fall_{FALL_STRIDE}_adl_{adl_stride}'
        experiments.append({'name': name, 'config': create_stride_config(FALL_STRIDE, adl_stride)})

    print('Stride Ablation')
    print('=' * 50)
    print(f'Fall stride: {FALL_STRIDE} (fixed)')
    print(f'ADL strides: {ADL_STRIDES}')
    print(f'GPUs: {args.num_gpus}, Parallel: {args.parallel}')
    print(f'Output: {args.work_dir}')
    print()

    if args.dry_run:
        print('Dry run:')
        for e in experiments:
            print(f"  - {e['name']}")
        return

    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / 'runs').mkdir(exist_ok=True)

    with open(args.work_dir / 'spec.json', 'w') as f:
        json.dump({'fall_stride': FALL_STRIDE, 'adl_strides': ADL_STRIDES,
                   'num_gpus': args.num_gpus, 'timestamp': datetime.now().isoformat()}, f, indent=2)

    print('Running experiments...')
    results = run_parallel(experiments, args.work_dir, args.num_gpus, args.parallel, args.max_folds)

    with open(args.work_dir / 'results.json', 'w') as f:
        json.dump([r.to_dict() for r in results], f, indent=2)

    print()
    print(generate_report(results, args.work_dir / 'stride_ablation_report.md'))
    print(f'\nSaved to: {args.work_dir}')


if __name__ == '__main__':
    main()
