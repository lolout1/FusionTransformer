#!/usr/bin/env python3
"""
Loss Function Ablation: BCE vs Focal Loss

Compares class-balanced BCE and Focal loss on SmartFallMM with:
- Uniform stride=10
- Young subjects only
- KalmanConv1dConv1d model

Usage:
    python distributed_dataset_pipeline/run_loss_ablation.py --num-gpus 4 --parallel 2
    python distributed_dataset_pipeline/run_loss_ablation.py --num-gpus 4 --quick
    python distributed_dataset_pipeline/run_loss_ablation.py --results-only --work-dir exps/loss_ablation_XXX
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

BASE_CONFIG = 'config/best_config/smartfallmm/kalman_stride10_young.yaml'


@dataclass
class ExperimentResult:
    name: str
    loss_type: str
    test_f1: float = 0.0
    test_f1_std: float = 0.0
    test_accuracy: float = 0.0
    test_precision: float = 0.0
    test_recall: float = 0.0
    test_auc: float = 0.0
    num_folds: int = 0
    status: str = 'pending'
    error_message: str = ''
    elapsed_time: float = 0.0
    fold_f1s: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'loss_type': self.loss_type,
            'test_f1': self.test_f1,
            'test_f1_std': self.test_f1_std,
            'test_accuracy': self.test_accuracy,
            'test_precision': self.test_precision,
            'test_recall': self.test_recall,
            'test_auc': self.test_auc,
            'num_folds': self.num_folds,
            'status': self.status,
            'error_message': self.error_message,
            'elapsed_time': self.elapsed_time,
            'fold_f1s': self.fold_f1s,
        }


def load_base_config() -> Dict[str, Any]:
    with open(BASE_CONFIG) as f:
        return yaml.safe_load(f)


def create_loss_config(loss_type: str) -> Dict[str, Any]:
    """Create config for specific loss type."""
    cfg = load_base_config()
    cfg['dataset_args']['loss_type'] = loss_type

    # Both use class balancing
    if loss_type == 'bce':
        cfg['loss_type'] = 'bce'
    elif loss_type == 'focal':
        cfg['loss_type'] = 'focal'
        cfg['loss_args'] = {'alpha': 0.75, 'gamma': 2.0}
    elif loss_type == 'cb_focal':
        cfg['loss_type'] = 'cb_focal'
        cfg['loss_args'] = {'beta': 0.9999, 'gamma': 2.0}

    return cfg


def run_experiment(
    name: str,
    config: Dict[str, Any],
    work_dir: Path,
    num_gpus: int,
    max_folds: Optional[int],
    timeout: int = 7200,
) -> ExperimentResult:
    """Run single experiment."""
    import time
    start = time.time()

    result = ExperimentResult(
        name=name,
        loss_type=config.get('loss_type', 'unknown'),
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
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path.cwd(),
        )

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
    """Parse fold_results.pkl."""
    with open(path, 'rb') as f:
        data = pickle.load(f)

    if not data:
        return result

    if isinstance(data, dict):
        fold_list = list(data.values())
    else:
        fold_list = data

    f1s, accs, precs, recs, aucs = [], [], [], [], []

    for fold in fold_list:
        if not isinstance(fold, dict):
            continue

        test_metrics = fold.get('test', {})
        if isinstance(test_metrics, dict):
            f1 = test_metrics.get('f1_score') or test_metrics.get('f1') or test_metrics.get('macro_f1', 0)
            acc = test_metrics.get('accuracy') or test_metrics.get('acc', 0)
            prec = test_metrics.get('precision') or test_metrics.get('prec', 0)
            rec = test_metrics.get('recall') or test_metrics.get('rec', 0)
            auc = test_metrics.get('auc', 0)
        else:
            f1 = fold.get('test_f1') or fold.get('f1', 0)
            acc = fold.get('test_accuracy') or fold.get('acc', 0)
            prec = fold.get('test_precision') or fold.get('prec', 0)
            rec = fold.get('test_recall') or fold.get('rec', 0)
            auc = fold.get('test_auc') or fold.get('auc', 0)

        if f1 and f1 <= 1: f1 *= 100
        if acc and acc <= 1: acc *= 100
        if prec and prec <= 1: prec *= 100
        if rec and rec <= 1: rec *= 100
        if auc and auc <= 1: auc *= 100

        if f1 > 0:
            f1s.append(f1)
            accs.append(acc)
            precs.append(prec)
            recs.append(rec)
            aucs.append(auc)

    if f1s:
        result.test_f1 = statistics.mean(f1s)
        result.test_f1_std = statistics.stdev(f1s) if len(f1s) > 1 else 0
        result.test_accuracy = statistics.mean(accs) if accs else 0
        result.test_precision = statistics.mean(precs) if precs else 0
        result.test_recall = statistics.mean(recs) if recs else 0
        result.test_auc = statistics.mean(aucs) if aucs else 0
        result.num_folds = len(f1s)
        result.fold_f1s = f1s

    return result


def run_experiments_parallel(
    experiments: List[Dict[str, Any]],
    work_dir: Path,
    num_gpus: int,
    parallel: int,
    max_folds: Optional[int],
) -> List[ExperimentResult]:
    """Run experiments in parallel."""
    gpus_per_exp = max(1, num_gpus // parallel)
    results = []

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {}
        for exp in experiments:
            future = executor.submit(
                run_experiment,
                exp['name'],
                exp['config'],
                work_dir / 'runs',
                gpus_per_exp,
                max_folds,
            )
            futures[future] = exp['name']

        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                results.append(result)
                status = '✓' if result.status == 'completed' else '✗'
                f1_str = f'{result.test_f1:.2f}%' if result.test_f1 > 0 else 'N/A'
                print(f'  {status} {name}: F1={f1_str}')
            except Exception as e:
                print(f'  ✗ {name}: {e}')
                results.append(ExperimentResult(
                    name=name,
                    loss_type='unknown',
                    status='error',
                    error_message=str(e),
                ))

    return results


def generate_report(results: List[ExperimentResult], output_path: Path) -> str:
    """Generate markdown report."""
    lines = [
        '# Loss Function Ablation Results',
        f'\nGenerated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
        '',
        '## Configuration',
        '',
        '- Dataset: SmartFallMM (young subjects only)',
        '- Model: KalmanConv1dConv1d',
        '- Stride: 10 (uniform, no class-aware)',
        '- Window: 128 samples (~4s at 30Hz)',
        '',
        '## Results',
        '',
        '| Loss Type | F1 (%) | Accuracy (%) | Precision (%) | Recall (%) | Folds |',
        '|-----------|--------|--------------|---------------|------------|-------|',
    ]

    results.sort(key=lambda x: x.test_f1, reverse=True)
    best_f1 = results[0].test_f1 if results else 0

    for r in results:
        delta = r.test_f1 - best_f1
        delta_str = f' ({delta:+.2f})' if abs(delta) > 0.01 else ' (best)'
        lines.append(
            f'| {r.loss_type} | {r.test_f1:.2f} ± {r.test_f1_std:.2f}{delta_str} | '
            f'{r.test_accuracy:.2f} | {r.test_precision:.2f} | {r.test_recall:.2f} | {r.num_folds} |'
        )

    lines.append('')
    lines.append('## Summary')
    lines.append('')
    if results:
        best = results[0]
        lines.append(f'**Best loss function**: {best.loss_type} ({best.test_f1:.2f}% F1)')

    report = '\n'.join(lines)

    with open(output_path, 'w') as f:
        f.write(report)

    return report


def main():
    parser = argparse.ArgumentParser(description='Loss function ablation (BCE vs Focal)')
    parser.add_argument('--num-gpus', type=int, default=4)
    parser.add_argument('--parallel', type=int, default=2)
    parser.add_argument('--max-folds', type=int, default=None)
    parser.add_argument('--quick', action='store_true', help='2 folds only')
    parser.add_argument('--work-dir', type=Path, default=None)
    parser.add_argument('--results-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')

    args = parser.parse_args()

    if args.quick:
        args.max_folds = 2

    if args.work_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.work_dir = Path(f'exps/loss_ablation_{timestamp}')

    # Results-only mode
    if args.results_only:
        results_path = args.work_dir / 'results.json'
        if results_path.exists():
            with open(results_path) as f:
                data = json.load(f)
            results = [ExperimentResult(**r) for r in data]
        else:
            results = []
            runs_dir = args.work_dir / 'runs'
            if runs_dir.exists():
                for exp_dir in sorted(runs_dir.iterdir()):
                    if not exp_dir.is_dir():
                        continue
                    pkl_path = exp_dir / 'fold_results.pkl'
                    if pkl_path.exists():
                        result = ExperimentResult(
                            name=exp_dir.name,
                            loss_type=exp_dir.name.replace('loss_', ''),
                        )
                        result = parse_fold_results(pkl_path, result)
                        result.status = 'completed' if result.test_f1 > 0 else 'failed'
                        results.append(result)

        report = generate_report(results, args.work_dir / 'loss_ablation_report.md')
        print(report)
        return

    # Build experiments
    loss_types = ['focal', 'bce', 'cb_focal']
    experiments = []

    for loss_type in loss_types:
        config = create_loss_config(loss_type)
        experiments.append({
            'name': f'loss_{loss_type}',
            'config': config,
        })

    print('Loss Function Ablation')
    print('=' * 50)
    print(f'Base config: {BASE_CONFIG}')
    print(f'Loss types: {loss_types}')
    print(f'GPUs: {args.num_gpus}, Parallel: {args.parallel}')
    print(f'Output: {args.work_dir}')
    print()

    if args.dry_run:
        print('Dry run - would execute:')
        for exp in experiments:
            print(f"  - {exp['name']}")
        return

    # Create directories
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / 'runs').mkdir(exist_ok=True)

    # Save spec
    spec = {
        'base_config': BASE_CONFIG,
        'loss_types': loss_types,
        'num_gpus': args.num_gpus,
        'parallel': args.parallel,
        'max_folds': args.max_folds,
        'timestamp': datetime.now().isoformat(),
    }
    with open(args.work_dir / 'spec.json', 'w') as f:
        json.dump(spec, f, indent=2)

    # Run
    print('Running experiments...')
    results = run_experiments_parallel(
        experiments, args.work_dir, args.num_gpus, args.parallel, args.max_folds
    )

    # Save results
    with open(args.work_dir / 'results.json', 'w') as f:
        json.dump([r.to_dict() for r in results], f, indent=2)

    # Report
    report = generate_report(results, args.work_dir / 'loss_ablation_report.md')
    print()
    print(report)
    print(f'\nResults saved to: {args.work_dir}')


if __name__ == '__main__':
    main()
