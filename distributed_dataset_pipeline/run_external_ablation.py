#!/usr/bin/env python3
"""External dataset ablation study for UP-FALL and WEDA-FALL."""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import threading


@dataclass
class ExternalExperiment:
    """Single experiment configuration for external datasets."""
    name: str
    group: str
    dataset: str  # 'upfall' or 'wedafall'
    model: Optional[str] = None
    model_args: Optional[str] = None
    embed_dim: Optional[int] = None
    normalize_mode: Optional[str] = None  # 'acc_only', 'both', 'none'
    enable_kalman: bool = True
    config_override: Optional[str] = None

    def get_config_path(self) -> str:
        """Get base config path for this dataset."""
        if self.config_override:
            return self.config_override
        if self.enable_kalman:
            return f"config/{self.dataset}/kalman_optimal.yaml"
        elif self.model and 'mamba' in self.model.lower():
            return f"config/{self.dataset}/mamba_raw.yaml"
        else:
            return f"config/{self.dataset}/dual_stream_raw.yaml"

    def to_cmd_args(self) -> List[str]:
        """Convert to command-line arguments."""
        args = []
        if self.model:
            args.extend(['--model', self.model])
        if self.model_args:
            args.extend(['--model-args', self.model_args])
        if self.embed_dim:
            args.extend(['--embed-dim', str(self.embed_dim)])
        return args

    def config_str(self) -> str:
        """Human-readable config string."""
        parts = [f"dataset={self.dataset}"]
        if self.model:
            parts.append(f"model={self.model.split('.')[-1]}")
        if self.embed_dim:
            parts.append(f"embed={self.embed_dim}")
        if self.normalize_mode:
            parts.append(f"norm={self.normalize_mode}")
        if not self.enable_kalman:
            parts.append("raw")
        return ', '.join(parts)


def define_experiments() -> List[ExternalExperiment]:
    """Define all ablation experiments for external datasets."""
    experiments = []

    # ========================================================================
    # GROUP A: Kalman Fusion Models (8 experiments)
    # ========================================================================
    for dataset in ['upfall', 'wedafall']:
        # A1/A5: Best config (Conv1d+Linear, embed=48)
        experiments.append(ExternalExperiment(
            name=f'{dataset}_kalman_conv1d_linear_e48',
            group='kalman',
            dataset=dataset,
            model='Models.encoder_ablation.KalmanConv1dLinear',
            embed_dim=48,
            enable_kalman=True,
        ))

        # A2/A6: Larger embedding
        experiments.append(ExternalExperiment(
            name=f'{dataset}_kalman_conv1d_linear_e64',
            group='kalman',
            dataset=dataset,
            model='Models.encoder_ablation.KalmanConv1dLinear',
            embed_dim=64,
            enable_kalman=True,
        ))

        # A3/A7: Conv1d+Conv1d symmetric kernels
        experiments.append(ExternalExperiment(
            name=f'{dataset}_kalman_conv1d_conv1d_k8',
            group='kalman',
            dataset=dataset,
            model='Models.encoder_ablation.KalmanConv1dConv1d',
            embed_dim=48,
            enable_kalman=True,
        ))

        # A4/A8: Asymmetric kernels (5, 13)
        experiments.append(ExternalExperiment(
            name=f'{dataset}_kalman_conv1d_conv1d_k5_13',
            group='kalman',
            dataset=dataset,
            model='Models.encoder_ablation.KalmanEncoderAblation',
            model_args="{'acc_encoder': 'conv1d', 'ori_encoder': 'conv1d', 'acc_kernel_size': 5, 'ori_kernel_size': 13}",
            embed_dim=48,
            enable_kalman=True,
        ))

    # ========================================================================
    # GROUP B: Dual-Stream Transformer (Raw Acc+Gyro) (10 experiments)
    # ========================================================================
    for dataset in ['upfall', 'wedafall']:
        # B1-B3: DualStreamBaseline with different normalization
        for norm in ['acc_only', 'both', 'none']:
            experiments.append(ExternalExperiment(
                name=f'{dataset}_dualstream_baseline_{norm}',
                group='dualstream',
                dataset=dataset,
                model='Models.dual_stream_baseline.DualStreamBaseline',
                normalize_mode=norm,
                enable_kalman=False,
            ))

        # B4: DualStreamSE
        experiments.append(ExternalExperiment(
            name=f'{dataset}_dualstream_se',
            group='dualstream',
            dataset=dataset,
            model='Models.dual_stream_se.DualStreamSE',
            normalize_mode='acc_only',
            enable_kalman=False,
        ))

        # B5: DualStreamBase (minimal)
        experiments.append(ExternalExperiment(
            name=f'{dataset}_dualstream_base',
            group='dualstream',
            dataset=dataset,
            model='Models.dual_stream_base.DualStreamBase',
            normalize_mode='acc_only',
            enable_kalman=False,
        ))

    # ========================================================================
    # GROUP C: Mamba/State-Space Models (6 experiments)
    # ========================================================================
    for dataset in ['upfall', 'wedafall']:
        for norm in ['acc_only', 'both', 'none']:
            experiments.append(ExternalExperiment(
                name=f'{dataset}_mamba_{norm}',
                group='mamba',
                dataset=dataset,
                model='Models.dual_stream_mamba.DualStreamMamba',
                normalize_mode=norm,
                enable_kalman=False,
            ))

    return experiments


@dataclass
class ExperimentResult:
    """Result from a single experiment."""
    name: str
    group: str
    dataset: str
    config: str
    status: str
    test_f1: float
    test_acc: float
    val_f1: float
    elapsed_min: float
    work_dir: str
    # Extended metrics
    avg_best_epoch: float = 0.0
    test_precision: float = 0.0
    test_recall: float = 0.0
    test_auc: float = 0.0
    test_f1_std: float = 0.0
    num_folds: int = 0
    error: Optional[str] = None


def run_experiment(
    exp: ExternalExperiment,
    num_gpus: int,
    output_base: str,
    max_folds: Optional[int] = None,
    gpu_ids: Optional[List[int]] = None,
) -> ExperimentResult:
    """Run a single experiment and return results."""

    work_dir = os.path.join(output_base, exp.dataset, exp.name)
    config_path = exp.get_config_path()

    # Check config exists
    if not os.path.exists(config_path):
        return ExperimentResult(
            name=exp.name,
            group=exp.group,
            dataset=exp.dataset,
            config=exp.config_str(),
            status='config_missing',
            test_f1=0.0,
            test_acc=0.0,
            val_f1=0.0,
            elapsed_min=0.0,
            work_dir=work_dir,
            error=f"Config not found: {config_path}",
        )

    cmd = [
        sys.executable, 'ray_train.py',
        '--config', config_path,
        '--num-gpus', str(num_gpus),
        '--work-dir', work_dir,
    ]
    cmd.extend(exp.to_cmd_args())

    if max_folds:
        cmd.extend(['--max-folds', str(max_folds)])

    env = os.environ.copy()
    if gpu_ids is not None:
        env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
        env['RAY_IGNORE_UNHANDLED_ERRORS'] = '1'
        gpu_str = f"GPUs {gpu_ids}"
    else:
        gpu_str = f"{num_gpus} GPUs"

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {exp.name} [{gpu_str}]")
    print(f"Dataset: {exp.dataset.upper()}")
    print(f"Config: {exp.config_str()}")
    print(f"Command: {' '.join(cmd)}")
    print('='*70)

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            check=True,
            env=env,
        )
        elapsed = (time.time() - start_time) / 60

        # Parse results from output directory
        metrics = parse_experiment_results(work_dir)

        return ExperimentResult(
            name=exp.name,
            group=exp.group,
            dataset=exp.dataset,
            config=exp.config_str(),
            status='success',
            test_f1=metrics.get('test_f1', 0.0),
            test_acc=metrics.get('test_acc', 0.0),
            val_f1=metrics.get('val_f1', 0.0),
            elapsed_min=round(elapsed, 2),
            work_dir=work_dir,
            avg_best_epoch=metrics.get('avg_best_epoch', 0.0),
            test_precision=metrics.get('test_precision', 0.0),
            test_recall=metrics.get('test_recall', 0.0),
            test_auc=metrics.get('test_auc', 0.0),
            test_f1_std=metrics.get('test_f1_std', 0.0),
            num_folds=metrics.get('num_folds', 0),
        )

    except subprocess.CalledProcessError as e:
        elapsed = (time.time() - start_time) / 60

        # Bug fix: Check if results were actually produced despite non-zero exit code
        # Ray/subprocess may return non-zero even when training completed successfully
        metrics = parse_experiment_results(work_dir)

        if metrics.get('num_folds', 0) > 0 and metrics.get('test_f1', 0) > 0:
            # Training completed - mark as success
            return ExperimentResult(
                name=exp.name,
                group=exp.group,
                dataset=exp.dataset,
                config=exp.config_str(),
                status='success',
                test_f1=metrics.get('test_f1', 0.0),
                test_acc=metrics.get('test_acc', 0.0),
                val_f1=metrics.get('val_f1', 0.0),
                elapsed_min=round(elapsed, 2),
                work_dir=work_dir,
                avg_best_epoch=metrics.get('avg_best_epoch', 0.0),
                test_precision=metrics.get('test_precision', 0.0),
                test_recall=metrics.get('test_recall', 0.0),
                test_auc=metrics.get('test_auc', 0.0),
                test_f1_std=metrics.get('test_f1_std', 0.0),
                num_folds=metrics.get('num_folds', 0),
            )

        # No results found - truly failed
        return ExperimentResult(
            name=exp.name,
            group=exp.group,
            dataset=exp.dataset,
            config=exp.config_str(),
            status='failed',
            test_f1=0.0,
            test_acc=0.0,
            val_f1=0.0,
            elapsed_min=round(elapsed, 2),
            work_dir=work_dir,
            error=str(e),
        )


def parse_experiment_results(work_dir: str) -> Dict[str, float]:
    """Parse results from experiment output directory."""
    import numpy as np
    import pandas as pd
    metrics = {}

    test_f1s, test_accs, val_f1s = [], [], []
    test_precisions, test_recalls, test_aucs = [], [], []
    best_epochs = []

    # Primary: scores.csv has best_epoch and per-fold data
    scores_files = list(Path(work_dir).glob('**/scores.csv'))
    if not scores_files:
        scores_files = list(Path(work_dir).glob('**/per_fold*.csv'))

    if scores_files:
        try:
            df = pd.read_csv(scores_files[0])
            if 'test_subject' not in df.columns and len(df.columns) > 0:
                df = df.reset_index()

            for _, row in df.iterrows():
                subj = str(row.get('test_subject', ''))
                if subj in ['Average', 'Std', 'Mean', 'StdDev', '']:
                    continue
                if 'test_f1_score' in row and pd.notna(row.get('test_f1_score')):
                    test_f1s.append(float(row['test_f1_score']))
                if 'test_accuracy' in row and pd.notna(row.get('test_accuracy')):
                    test_accs.append(float(row['test_accuracy']))
                if 'val_f1_score' in row and pd.notna(row.get('val_f1_score')):
                    val_f1s.append(float(row['val_f1_score']))
                if 'test_precision' in row and pd.notna(row.get('test_precision')):
                    test_precisions.append(float(row['test_precision']))
                if 'test_recall' in row and pd.notna(row.get('test_recall')):
                    test_recalls.append(float(row['test_recall']))
                if 'test_auc' in row and pd.notna(row.get('test_auc')):
                    test_aucs.append(float(row['test_auc']))
                if 'best_epoch' in row and pd.notna(row.get('best_epoch')):
                    best_epochs.append(int(row['best_epoch']))
        except Exception:
            pass

    # Fallback: read from metrics.json files
    if not test_f1s:
        metrics_files = list(Path(work_dir).glob('fold_*/metrics.json'))
        for mf in metrics_files:
            try:
                with open(mf) as f:
                    m = json.load(f)
                    if 'test' in m:
                        if 'f1_score' in m['test']:
                            test_f1s.append(m['test']['f1_score'])
                        if 'accuracy' in m['test']:
                            test_accs.append(m['test']['accuracy'])
                        if 'precision' in m['test']:
                            test_precisions.append(m['test']['precision'])
                        if 'recall' in m['test']:
                            test_recalls.append(m['test']['recall'])
                        if 'auc' in m['test']:
                            test_aucs.append(m['test']['auc'])
                    if 'val' in m and 'f1_score' in m['val']:
                        val_f1s.append(m['val']['f1_score'])
                    if 'best_epoch' in m:
                        best_epochs.append(m['best_epoch'])
            except (json.JSONDecodeError, KeyError):
                continue

    # Compute aggregated metrics
    metrics['num_folds'] = len(test_f1s)

    if test_f1s:
        metrics['test_f1'] = np.mean(test_f1s)
        metrics['test_f1_std'] = np.std(test_f1s)
    if test_accs:
        metrics['test_acc'] = np.mean(test_accs)
    if val_f1s:
        metrics['val_f1'] = np.mean(val_f1s)
    if test_precisions:
        metrics['test_precision'] = np.mean(test_precisions)
    if test_recalls:
        metrics['test_recall'] = np.mean(test_recalls)
    if test_aucs:
        metrics['test_auc'] = np.mean(test_aucs)
    if best_epochs:
        metrics['avg_best_epoch'] = np.mean(best_epochs)

    return metrics


def print_results_table(results: List[ExperimentResult]):
    """Print formatted results table."""

    print("\n" + "="*130)
    print("EXTERNAL DATASET ABLATION RESULTS")
    print("="*130)

    # Group by dataset first
    for dataset in ['upfall', 'wedafall']:
        dataset_results = [r for r in results if r.dataset == dataset]
        if not dataset_results:
            continue

        print(f"\n{'='*60}")
        print(f"DATASET: {dataset.upper()}")
        print(f"{'='*60}")

        print(f"\n{'Experiment':<35} {'Group':<12} {'Test F1':>9} {'F1 Std':>7} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'Folds':>6} {'Time':>7}")
        print("-"*110)

        sorted_results = sorted(dataset_results, key=lambda x: (x.group, -x.test_f1))
        current_group = None

        for r in sorted_results:
            if r.group != current_group:
                if current_group is not None:
                    print("-"*110)
                current_group = r.group

            if r.status == 'success':
                print(f"{r.name:<35} {r.group:<12} {r.test_f1:>8.2f}% {r.test_f1_std:>6.2f} {r.test_acc:>6.2f}% {r.test_precision:>6.2f}% {r.test_recall:>6.2f}% {r.num_folds:>6} {r.elapsed_min:>6.1f}m")
            else:
                print(f"{r.name:<35} {r.group:<12} {'FAILED':>9} {'-':>7} {'-':>7} {'-':>7} {'-':>7} {'-':>6} {r.elapsed_min:>6.1f}m")

    # Overall summary
    print("\n" + "="*130)
    print("CROSS-DATASET SUMMARY")
    print("="*130)

    print(f"\n{'Group':<15} {'UP-FALL Best':<25} {'F1':>8} {'WEDA-FALL Best':<25} {'F1':>8}")
    print("-"*90)

    for group in ['kalman', 'dualstream', 'mamba']:
        upfall = [r for r in results if r.dataset == 'upfall' and r.group == group and r.status == 'success']
        wedafall = [r for r in results if r.dataset == 'wedafall' and r.group == group and r.status == 'success']

        up_best = max(upfall, key=lambda x: x.test_f1) if upfall else None
        weda_best = max(wedafall, key=lambda x: x.test_f1) if wedafall else None

        up_name = up_best.name.replace('upfall_', '') if up_best else '-'
        up_f1 = f"{up_best.test_f1:.2f}%" if up_best else '-'
        weda_name = weda_best.name.replace('wedafall_', '') if weda_best else '-'
        weda_f1 = f"{weda_best.test_f1:.2f}%" if weda_best else '-'

        print(f"{group:<15} {up_name:<25} {up_f1:>8} {weda_name:<25} {weda_f1:>8}")

    print("="*130)


def save_results(results: List[ExperimentResult], output_dir: str):
    """Save results to files."""
    import pandas as pd

    # Save as JSON
    json_path = os.path.join(output_dir, 'external_ablation_results.json')
    with open(json_path, 'w') as f:
        json.dump([{
            'name': r.name,
            'group': r.group,
            'dataset': r.dataset,
            'config': r.config,
            'status': r.status,
            'test_f1': r.test_f1,
            'test_f1_std': r.test_f1_std,
            'test_acc': r.test_acc,
            'test_precision': r.test_precision,
            'test_recall': r.test_recall,
            'test_auc': r.test_auc,
            'val_f1': r.val_f1,
            'avg_best_epoch': r.avg_best_epoch,
            'num_folds': r.num_folds,
            'elapsed_min': r.elapsed_min,
            'work_dir': r.work_dir,
            'error': r.error,
        } for r in results], f, indent=2)

    # Save as CSV
    csv_path = os.path.join(output_dir, 'external_ablation_results.csv')
    df = pd.DataFrame([{
        'name': r.name,
        'group': r.group,
        'dataset': r.dataset,
        'status': r.status,
        'test_f1': round(r.test_f1, 2),
        'test_f1_std': round(r.test_f1_std, 2),
        'test_acc': round(r.test_acc, 2),
        'test_precision': round(r.test_precision, 2),
        'test_recall': round(r.test_recall, 2),
        'test_auc': round(r.test_auc, 4),
        'val_f1': round(r.val_f1, 2),
        'avg_best_epoch': round(r.avg_best_epoch, 1),
        'num_folds': r.num_folds,
        'elapsed_min': r.elapsed_min,
    } for r in results])
    df.to_csv(csv_path, index=False)

    # Save markdown report
    md_path = os.path.join(output_dir, 'external_ablation_report.md')
    with open(md_path, 'w') as f:
        f.write("# External Dataset Ablation Study Results\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Summary\n\n")
        f.write("| Dataset | Group | Best Experiment | Test F1 | F1 Std |\n")
        f.write("|---------|-------|-----------------|---------|--------|\n")

        for dataset in ['upfall', 'wedafall']:
            for group in ['kalman', 'dualstream', 'mamba']:
                group_results = [r for r in results
                                if r.dataset == dataset and r.group == group and r.status == 'success']
                if group_results:
                    best = max(group_results, key=lambda x: x.test_f1)
                    f.write(f"| {dataset.upper()} | {group} | {best.name} | {best.test_f1:.2f}% | {best.test_f1_std:.2f} |\n")

        f.write("\n## Detailed Results\n\n")
        for dataset in ['upfall', 'wedafall']:
            f.write(f"### {dataset.upper()}\n\n")
            dataset_results = [r for r in results if r.dataset == dataset]
            for group in ['kalman', 'dualstream', 'mamba']:
                group_results = [r for r in dataset_results if r.group == group and r.status == 'success']
                if group_results:
                    f.write(f"#### {group.title()}\n\n")
                    f.write("| Experiment | Test F1 | F1 Std | Acc | Prec | Rec | AUC |\n")
                    f.write("|------------|---------|--------|-----|------|-----|-----|\n")
                    for r in sorted(group_results, key=lambda x: -x.test_f1):
                        f.write(f"| {r.name} | {r.test_f1:.2f}% | {r.test_f1_std:.2f} | {r.test_acc:.2f}% | {r.test_precision:.2f}% | {r.test_recall:.2f}% | {r.test_auc:.3f} |\n")
                    f.write("\n")

    print(f"\nResults saved to:")
    print(f"  - {json_path}")
    print(f"  - {csv_path}")
    print(f"  - {md_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Run external dataset ablation study',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--num-gpus', '-g',
        type=int,
        default=3,
        help='Number of GPUs (default: 3)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory (default: results/external_ablation_TIMESTAMP)'
    )
    parser.add_argument(
        '--group',
        type=str,
        choices=['kalman', 'dualstream', 'mamba', 'all'],
        default='all',
        help='Run specific experiment group (default: all)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['upfall', 'wedafall', 'all'],
        default='all',
        help='Run specific dataset (default: all)'
    )
    parser.add_argument(
        '--max-folds',
        type=int,
        default=None,
        help='Maximum folds per experiment (for quick testing)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show experiments without running'
    )
    parser.add_argument(
        '--parallel', '-p',
        type=int,
        default=1,
        help='Number of experiments to run in parallel (default: 1)'
    )
    parser.add_argument(
        '--continue-from',
        type=str,
        default=None,
        help='Continue from existing results directory'
    )

    args = parser.parse_args()

    # Change to repo root directory (script is in distributed_dataset_pipeline/)
    script_dir = Path(__file__).parent.absolute()
    repo_root = script_dir.parent
    os.chdir(repo_root)
    print(f"Working directory: {os.getcwd()}")

    # Get experiments
    all_experiments = define_experiments()

    # Filter by group and dataset
    experiments = all_experiments
    if args.group != 'all':
        experiments = [e for e in experiments if e.group == args.group]
    if args.dataset != 'all':
        experiments = [e for e in experiments if e.dataset == args.dataset]

    print(f"\n{'='*70}")
    print("EXTERNAL DATASET ABLATION STUDY")
    print(f"{'='*70}")
    print(f"Total GPUs: {args.num_gpus}")
    print(f"Groups: {args.group}")
    print(f"Datasets: {args.dataset}")
    print(f"Total experiments: {len(experiments)}")
    if args.max_folds:
        print(f"Max folds per experiment: {args.max_folds}")
    print(f"{'='*70}")

    # Show experiment preview
    print("\nExperiments to run:")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i:2d}. [{exp.group}] [{exp.dataset}] {exp.name}")

    if args.dry_run:
        print("\n[DRY RUN] No experiments executed.")
        return

    # Setup output directory
    if args.output:
        output_dir = args.output
    elif args.continue_from:
        output_dir = args.continue_from
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"results/external_ablation_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)

    # Check for completed experiments
    completed = set()
    if args.continue_from or os.path.exists(os.path.join(output_dir, 'external_ablation_results.json')):
        results_file = os.path.join(output_dir, 'external_ablation_results.json')
        if os.path.exists(results_file):
            with open(results_file) as f:
                prev_results = json.load(f)
                completed = {r['name'] for r in prev_results if r['status'] == 'success'}
                print(f"\nFound {len(completed)} completed experiments")

    # Filter out completed experiments
    pending = [e for e in experiments if e.name not in completed]
    total_pending = len(pending)

    if not pending:
        print("\nAll experiments already completed!")
        return

    results = []
    start_time = time.time()

    # Sequential execution
    for i, exp in enumerate(pending, 1):
        print(f"\n[{i}/{total_pending}] Running {exp.name}...")
        result = run_experiment(
            exp, args.num_gpus, output_dir,
            max_folds=args.max_folds
        )
        results.append(result)
        save_results(results, output_dir)

        elapsed = (time.time() - start_time) / 60
        remaining = total_pending - i
        if i > 0:
            eta = (elapsed / i) * remaining
            print(f"\nProgress: {i}/{total_pending} | Elapsed: {elapsed:.1f}m | ETA: {eta:.1f}m")

    # Final results
    total_elapsed = (time.time() - start_time) / 60

    print_results_table(results)
    save_results(results, output_dir)

    print(f"\nTotal time: {total_elapsed:.1f} minutes")
    print(f"Results directory: {output_dir}")


if __name__ == '__main__':
    main()
