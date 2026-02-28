#!/usr/bin/env python3
"""
Queue Aggregation × Stride × Config Ablation

Trains multiple model configs, then runs comprehensive temporal queue evaluation
on each to find optimal deployment parameters.

Two phases:
  Phase 1: Train LOSO folds for each config (parallel via ray_train.py)
  Phase 2: Temporal queue sweep — stride × aggregation × queue params

Eval strides: [2, 4, 5, 8, 10, 15, 20]
Aggregation: average + majority vote (k=0.3..0.7)
Queue sizes: [3, 5, 8, 10, 15, 20]

Usage:
    # Full run (train + eval)
    python distributed_dataset_pipeline/run_queue_ablation.py --num-gpus 4 --parallel 2

    # Quick test (2 folds, subset strides)
    python distributed_dataset_pipeline/run_queue_ablation.py --quick --num-gpus 2

    # Eval-only on existing trained models
    python distributed_dataset_pipeline/run_queue_ablation.py --eval-only --work-dir exps/queue_ablation_XXX

    # Custom configs
    python distributed_dataset_pipeline/run_queue_ablation.py --configs configs_tmp/stride_f8_a32.yaml configs_tmp/ff4x_f8_a32.yaml

    # Skip training (already trained)
    python distributed_dataset_pipeline/run_queue_ablation.py --skip-training --work-dir exps/queue_ablation_XXX
"""

import argparse
import copy
import csv
import json
import pickle
import statistics
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CONFIGS_DIR = Path('configs_tmp')

EVAL_STRIDES = [2, 4, 5, 8, 10, 15, 20]
EVAL_STRIDES_QUICK = [5, 10, 20]

QUEUE_SIZES = [3, 5, 8, 10, 15, 20]
QUEUE_SIZES_QUICK = [5, 10, 15]

QUEUE_THRESHOLDS = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]
QUEUE_THRESHOLDS_QUICK = [0.4, 0.5, 0.6]

QUEUE_RETAINS = [0, 1, 2, 3, 5]
QUEUE_RETAINS_QUICK = [0, 2]

MAJORITY_KS = [0.3, 0.4, 0.5, 0.6, 0.7]
MAJORITY_KS_QUICK = [0.4, 0.5, 0.6]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    name: str
    config_path: str
    run_dir: str = ''
    test_f1: float = 0.0
    test_f1_std: float = 0.0
    test_accuracy: float = 0.0
    test_precision: float = 0.0
    test_recall: float = 0.0
    num_folds: int = 0
    status: str = 'pending'
    error: str = ''
    elapsed: float = 0.0
    fold_f1s: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != 'fold_f1s'}


@dataclass
class QueueResult:
    """Single queue evaluation result row."""
    config_name: str
    stride: int
    method: str
    queue_size: int
    threshold: float
    retain: int
    calibrated: bool
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    specificity: float = 0.0
    accuracy: float = 0.0
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    total_decisions: int = 0
    gt_positive: int = 0
    gt_negative: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


# ---------------------------------------------------------------------------
# Phase 1: Training
# ---------------------------------------------------------------------------

def discover_configs(configs_dir: Path, explicit_configs: Optional[List[str]] = None) -> List[Tuple[str, Path]]:
    if explicit_configs:
        result = []
        for c in explicit_configs:
            p = Path(c)
            if p.exists():
                result.append((p.stem, p))
            else:
                print(f"  WARNING: Config not found: {c}")
        return result

    if not configs_dir.exists():
        raise FileNotFoundError(f"Configs directory not found: {configs_dir}")

    configs = sorted(configs_dir.glob('*.yaml'))
    return [(c.stem, c) for c in configs]


def parse_fold_results(pkl_path: Path) -> Dict[str, float]:
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    fold_list = list(data.values()) if isinstance(data, dict) else data
    f1s, accs, precs, recs = [], [], [], []

    for fold in fold_list:
        if not isinstance(fold, dict):
            continue
        test = fold.get('test', {})
        if isinstance(test, dict):
            f1 = test.get('f1_score') or test.get('f1') or test.get('macro_f1', 0)
            acc = test.get('accuracy', 0)
            prec = test.get('precision', 0)
            rec = test.get('recall', 0)
        else:
            continue

        if f1 and f1 <= 1: f1 *= 100
        if acc and acc <= 1: acc *= 100
        if prec and prec <= 1: prec *= 100
        if rec and rec <= 1: rec *= 100

        if f1 > 0:
            f1s.append(f1)
            accs.append(acc)
            precs.append(prec)
            recs.append(rec)

    if not f1s:
        return {}

    return {
        'test_f1': statistics.mean(f1s),
        'test_f1_std': statistics.stdev(f1s) if len(f1s) > 1 else 0,
        'test_accuracy': statistics.mean(accs),
        'test_precision': statistics.mean(precs),
        'test_recall': statistics.mean(recs),
        'num_folds': len(f1s),
        'fold_f1s': f1s,
    }


def train_single(
    name: str,
    config_path: Path,
    run_dir: Path,
    num_gpus: int,
    max_folds: Optional[int] = None,
    timeout: int = 10800,
) -> TrainResult:
    start = time.time()
    result = TrainResult(name=name, config_path=str(config_path))

    run_dir.mkdir(parents=True, exist_ok=True)
    result.run_dir = str(run_dir)

    cmd = [
        sys.executable, 'ray_train.py',
        '--config', str(config_path),
        '--work-dir', str(run_dir),
        '--num-gpus', str(num_gpus),
    ]
    if max_folds:
        cmd.extend(['--max-folds', str(max_folds)])

    try:
        print(f"  [Train] Starting {name} on {num_gpus} GPUs ...")
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=Path.cwd()
        )
        if proc.returncode != 0:
            result.status = 'failed'
            result.error = (proc.stderr or '')[-500:]
            print(f"  [Train] FAILED {name}: {result.error[:200]}")
        else:
            result.status = 'completed'
            pkl = run_dir / 'fold_results.pkl'
            if pkl.exists():
                metrics = parse_fold_results(pkl)
                result.test_f1 = metrics.get('test_f1', 0)
                result.test_f1_std = metrics.get('test_f1_std', 0)
                result.test_accuracy = metrics.get('test_accuracy', 0)
                result.test_precision = metrics.get('test_precision', 0)
                result.test_recall = metrics.get('test_recall', 0)
                result.num_folds = metrics.get('num_folds', 0)
                result.fold_f1s = metrics.get('fold_f1s', [])
            print(f"  [Train] DONE {name}: F1={result.test_f1:.2f}% ± {result.test_f1_std:.2f}%")
    except subprocess.TimeoutExpired:
        result.status = 'timeout'
        result.error = f'Timeout after {timeout}s'
        print(f"  [Train] TIMEOUT {name}")
    except Exception as e:
        result.status = 'error'
        result.error = str(e)
        print(f"  [Train] ERROR {name}: {e}")

    result.elapsed = time.time() - start
    return result


def train_all(
    configs: List[Tuple[str, Path]],
    work_dir: Path,
    num_gpus: int,
    parallel: int,
    max_folds: Optional[int] = None,
) -> List[TrainResult]:
    gpus_per = max(1, num_gpus // parallel)
    runs_dir = work_dir / 'runs'
    runs_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"PHASE 1: TRAINING ({len(configs)} configs, {num_gpus} GPUs, {parallel} parallel)")
    print(f"{'='*80}")

    results = []
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futures = {
            ex.submit(
                train_single, name, cfg_path, runs_dir / name,
                gpus_per, max_folds
            ): name
            for name, cfg_path in configs
        }
        for f in as_completed(futures):
            name = futures[f]
            try:
                results.append(f.result())
            except Exception as e:
                print(f"  [Train] EXCEPTION {name}: {e}")
                results.append(TrainResult(
                    name=name, config_path='', status='error', error=str(e)
                ))

    results.sort(key=lambda r: r.name)
    return results


# ---------------------------------------------------------------------------
# Phase 2: Temporal Queue Evaluation
# ---------------------------------------------------------------------------

def run_queue_eval_for_config(
    name: str,
    run_dir: Path,
    eval_strides: List[int],
    queue_sizes: List[int],
    queue_thresholds: List[float],
    queue_retains: List[int],
    majority_ks: List[float],
    enable_calibration: bool = True,
    device: str = 'cuda:0',
) -> Dict[str, Any]:
    """Run temporal queue evaluation on a trained model directory."""
    # Ensure project root is on sys.path for imports
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import torch

    if not (run_dir / 'fold_results.pkl').exists():
        # Check for at least one model checkpoint
        ckpts = list(run_dir.glob('model_*.pth'))
        if not ckpts:
            return {'status': 'no_models', 'name': name}

    # Load config
    yaml_files = list(run_dir.glob('*.yaml')) + list(run_dir.glob('*.yml'))
    if not yaml_files:
        return {'status': 'no_config', 'name': name}

    with open(yaml_files[0]) as f:
        config = yaml.safe_load(f)

    device_str = device if torch.cuda.is_available() and 'cuda' in device else 'cpu'

    try:
        from utils.temporal_queue_eval import TemporalQueueEvaluator
        evaluator = TemporalQueueEvaluator(
            work_dir=str(run_dir),
            config=config,
            device=device_str,
            eval_strides=eval_strides,
            queue_sizes=queue_sizes,
            queue_thresholds=queue_thresholds,
            queue_retains=queue_retains,
            majority_ks=majority_ks,
            enable_calibration=enable_calibration,
        )
        output = evaluator.sweep_and_report(print_results=True)
        output['status'] = 'completed'
        output['name'] = name
        return output
    except Exception as e:
        print(f"  [Eval] ERROR {name}: {e}")
        traceback.print_exc()
        return {'status': 'error', 'name': name, 'error': str(e)}


def extract_queue_results(
    name: str,
    eval_output: Dict[str, Any],
) -> List[QueueResult]:
    """Extract flat QueueResult rows from eval output."""
    rows = []
    for r in eval_output.get('all_results', []):
        rows.append(QueueResult(
            config_name=name, calibrated=False, **_extract_fields(r)
        ))
    for r in eval_output.get('all_results_calibrated', []):
        rows.append(QueueResult(
            config_name=name, calibrated=True, **_extract_fields(r)
        ))
    return rows


def _extract_fields(r: dict) -> dict:
    return {
        'stride': r.get('stride', 0),
        'method': r.get('method', ''),
        'queue_size': r.get('queue_size', 0),
        'threshold': r.get('threshold', 0),
        'retain': r.get('retain', 0),
        'f1': r.get('f1', 0),
        'precision': r.get('precision', 0),
        'recall': r.get('recall', 0),
        'specificity': r.get('specificity', 0),
        'accuracy': r.get('accuracy', 0),
        'tp': r.get('tp', 0),
        'fp': r.get('fp', 0),
        'tn': r.get('tn', 0),
        'fn': r.get('fn', 0),
        'total_decisions': r.get('total_decisions', 0),
        'gt_positive': r.get('gt_positive', 0),
        'gt_negative': r.get('gt_negative', 0),
    }


def eval_all(
    train_results: List[TrainResult],
    work_dir: Path,
    eval_strides: List[int],
    queue_sizes: List[int],
    queue_thresholds: List[float],
    queue_retains: List[int],
    majority_ks: List[float],
    enable_calibration: bool = True,
) -> Tuple[List[QueueResult], Dict[str, Dict]]:
    """Run queue eval on all successfully trained configs."""
    completed = [r for r in train_results if r.status == 'completed']

    print(f"\n{'='*80}")
    print(f"PHASE 2: TEMPORAL QUEUE EVALUATION ({len(completed)} configs)")
    print(f"  Strides: {eval_strides}")
    print(f"  Queue sizes: {queue_sizes}")
    print(f"  Thresholds: {queue_thresholds}")
    print(f"  Retains: {queue_retains}")
    print(f"  Majority k: {majority_ks}")
    print(f"  Calibration: {enable_calibration}")

    n_combos = (len(eval_strides) * (1 + len(majority_ks)) *
                len(queue_sizes) * len(queue_thresholds) * len(queue_retains))
    print(f"  Total configs per model: ~{n_combos}")
    print(f"{'='*80}")

    all_queue_results = []
    per_config_outputs = {}

    for tr in completed:
        run_dir = Path(tr.run_dir)
        print(f"\n--- Evaluating: {tr.name} (F1={tr.test_f1:.2f}%) ---")

        output = run_queue_eval_for_config(
            name=tr.name,
            run_dir=run_dir,
            eval_strides=eval_strides,
            queue_sizes=queue_sizes,
            queue_thresholds=queue_thresholds,
            queue_retains=queue_retains,
            majority_ks=majority_ks,
            enable_calibration=enable_calibration,
        )

        per_config_outputs[tr.name] = output

        if output.get('status') == 'completed':
            rows = extract_queue_results(tr.name, output)
            all_queue_results.extend(rows)
            n_uncal = sum(1 for r in rows if not r.calibrated)
            n_cal = sum(1 for r in rows if r.calibrated)
            best_uncal = max([r for r in rows if not r.calibrated], key=lambda r: r.f1, default=None)
            best_cal = max([r for r in rows if r.calibrated], key=lambda r: r.f1, default=None)
            print(f"  Results: {n_uncal} uncalibrated, {n_cal} calibrated")
            if best_uncal:
                print(f"  Best uncal: F1={best_uncal.f1:.2f}% (s={best_uncal.stride}, {best_uncal.method}, "
                      f"q={best_uncal.queue_size}, t={best_uncal.threshold}, r={best_uncal.retain})")
            if best_cal:
                print(f"  Best calib: F1={best_cal.f1:.2f}% (s={best_cal.stride}, {best_cal.method}, "
                      f"q={best_cal.queue_size}, t={best_cal.threshold}, r={best_cal.retain})")

    return all_queue_results, per_config_outputs


# ---------------------------------------------------------------------------
# Phase 3: Reporting
# ---------------------------------------------------------------------------

def generate_report(
    train_results: List[TrainResult],
    queue_results: List[QueueResult],
    eval_strides: List[int],
    majority_ks: List[float],
    output_dir: Path,
) -> str:
    lines = []
    w = lines.append
    ts = datetime.now().strftime('%Y-%m-%d %H:%M')

    w(f'# Queue Aggregation Ablation Report')
    w(f'\nGenerated: {ts}')
    w('')

    # --- Section 1: Training Summary ---
    w('## 1. Training Results')
    w('')
    completed = [r for r in train_results if r.status == 'completed']
    w(f'{len(completed)}/{len(train_results)} configs trained successfully.\n')
    w('| Config | F1 (%) | ± Std | Acc (%) | Prec (%) | Rec (%) | Folds | Time |')
    w('|--------|--------|-------|---------|----------|---------|-------|------|')
    for r in sorted(completed, key=lambda x: -x.test_f1):
        t_min = r.elapsed / 60
        w(f'| {r.name} | {r.test_f1:.2f} | {r.test_f1_std:.2f} | '
          f'{r.test_accuracy:.2f} | {r.test_precision:.2f} | {r.test_recall:.2f} | '
          f'{r.num_folds} | {t_min:.0f}m |')
    failed = [r for r in train_results if r.status != 'completed']
    if failed:
        w('\n**Failed configs:**')
        for r in failed:
            w(f'- {r.name}: {r.status} — {r.error[:100]}')
    w('')

    # --- Section 2: Global Best Queue Configs ---
    if not queue_results:
        w('## 2. No Queue Results')
        report = '\n'.join(lines)
        (output_dir / 'queue_ablation_report.md').write_text(report)
        return report

    uncal = [r for r in queue_results if not r.calibrated]
    cal = [r for r in queue_results if r.calibrated]

    w('## 2. Top 30 Queue Configs (All Configs Combined)')
    w('')
    _write_top_table(w, uncal, 'Uncalibrated', 30)
    if cal:
        w('')
        _write_top_table(w, cal, 'Calibrated', 30)

    # --- Section 3: Best per Config ---
    w('\n## 3. Best Queue Config per Training Config')
    w('')
    w('| Config | Calib | Stride | Method | QSize | Thresh | Retain | F1 (%) | Prec | Rec | Spec |')
    w('|--------|-------|--------|--------|-------|--------|--------|--------|------|-----|------|')
    config_names = sorted(set(r.config_name for r in queue_results))
    for cn in config_names:
        for calib_label, calib_flag in [('No', False), ('Yes', True)]:
            subset = [r for r in queue_results if r.config_name == cn and r.calibrated == calib_flag]
            if not subset:
                continue
            best = max(subset, key=lambda r: r.f1)
            w(f'| {cn} | {calib_label} | {best.stride} | {best.method} | {best.queue_size} | '
              f'{best.threshold} | {best.retain} | {best.f1:.2f} | {best.precision:.2f} | '
              f'{best.recall:.2f} | {best.specificity:.2f} |')

    # --- Section 4: Best per Stride (aggregated across configs) ---
    w('\n## 4. Best F1 per Eval Stride')
    w('')
    w('| Stride | Best Config | Method | QSize | Thresh | Retain | F1 (%) | Prec | Rec | Spec |')
    w('|--------|-------------|--------|-------|--------|--------|--------|------|-----|------|')
    for s in eval_strides:
        subset = [r for r in uncal if r.stride == s]
        if not subset:
            continue
        best = max(subset, key=lambda r: r.f1)
        w(f'| {s} | {best.config_name} | {best.method} | {best.queue_size} | '
          f'{best.threshold} | {best.retain} | {best.f1:.2f} | {best.precision:.2f} | '
          f'{best.recall:.2f} | {best.specificity:.2f} |')

    # --- Section 5: Stride x Method Matrix ---
    w('\n## 5. Stride × Method Best F1 Matrix')
    w('')
    method_keys = ['average'] + [f'majority_k{int(k*100)}' for k in majority_ks]
    method_labels = ['avg'] + [f'k{int(k*100)}' for k in majority_ks]
    header = '| Stride | ' + ' | '.join(method_labels) + ' |'
    sep = '|--------|' + '|'.join(['------'] * len(method_labels)) + '|'
    w(header)
    w(sep)
    for s in eval_strides:
        row = f'| {s:>6} |'
        for mk in method_keys:
            subset = [r for r in uncal if r.stride == s and r.method == mk]
            if subset:
                best = max(subset, key=lambda r: r.f1)
                row += f' {best.f1:>5.2f} |'
            else:
                row += '    -- |'
        w(row)

    # --- Section 6: Config x Stride Matrix ---
    w('\n## 6. Config × Stride Best F1 Matrix')
    w('')
    header = '| Config | ' + ' | '.join(f's={s}' for s in eval_strides) + ' | Best |'
    sep = '|--------|' + '|'.join(['------'] * len(eval_strides)) + '|------|'
    w(header)
    w(sep)
    for cn in config_names:
        row = f'| {cn} |'
        best_overall = 0
        for s in eval_strides:
            subset = [r for r in uncal if r.config_name == cn and r.stride == s]
            if subset:
                best = max(subset, key=lambda r: r.f1)
                row += f' {best.f1:>5.2f} |'
                best_overall = max(best_overall, best.f1)
            else:
                row += '    -- |'
        row += f' {best_overall:>5.2f} |'
        w(row)

    # --- Section 7: Average vs Majority Vote ---
    w('\n## 7. Average vs Majority Vote (Best per Stride)')
    w('')
    w('| Stride | Avg F1 | Best Maj F1 | Best k | Winner | Delta |')
    w('|--------|--------|-------------|--------|--------|-------|')
    for s in eval_strides:
        avg_sub = [r for r in uncal if r.stride == s and r.method == 'average']
        maj_sub = [r for r in uncal if r.stride == s and r.method.startswith('majority')]
        if not avg_sub:
            continue
        avg_best = max(avg_sub, key=lambda r: r.f1)
        if maj_sub:
            maj_best = max(maj_sub, key=lambda r: r.f1)
            delta = maj_best.f1 - avg_best.f1
            winner = 'Majority' if delta > 0 else 'Average'
            w(f'| {s} | {avg_best.f1:.2f} | {maj_best.f1:.2f} | {maj_best.method} | '
              f'{winner} | {delta:+.2f} |')
        else:
            w(f'| {s} | {avg_best.f1:.2f} | -- | -- | Average | -- |')

    # --- Section 8: High-Specificity Configs ---
    w('\n## 8. High-Specificity Configs (Spec >= 90%)')
    w('')
    high_spec = sorted([r for r in uncal if r.specificity >= 90], key=lambda r: -r.f1)
    if high_spec:
        _write_top_table(w, high_spec[:20], 'High Specificity (uncalibrated)', 20)
    else:
        w('No configs achieved specificity >= 90%.')

    # --- Section 9: Calibration Impact ---
    if cal:
        w('\n## 9. Calibration Impact')
        w('')
        w('| Config | Stride | F1(raw) | F1(cal) | Delta | Method(raw) | Method(cal) |')
        w('|--------|--------|---------|---------|-------|-------------|-------------|')
        for cn in config_names:
            for s in eval_strides:
                raw_sub = [r for r in uncal if r.config_name == cn and r.stride == s]
                cal_sub = [r for r in cal if r.config_name == cn and r.stride == s]
                if not raw_sub or not cal_sub:
                    continue
                raw_best = max(raw_sub, key=lambda r: r.f1)
                cal_best = max(cal_sub, key=lambda r: r.f1)
                delta = cal_best.f1 - raw_best.f1
                w(f'| {cn} | {s} | {raw_best.f1:.2f} | {cal_best.f1:.2f} | {delta:+.2f} | '
                  f'{raw_best.method} | {cal_best.method} |')

    # --- Section 10: Deployment Recommendation ---
    w('\n## 10. Deployment Recommendation (Stride=5, App Default)')
    w('')
    s5_uncal = [r for r in uncal if r.stride == 5]
    s5_cal = [r for r in cal if r.stride == 5]
    if s5_uncal:
        best_s5 = max(s5_uncal, key=lambda r: r.f1)
        w(f'**Best uncalibrated** (stride=5): F1={best_s5.f1:.2f}%')
        w(f'  Config: {best_s5.config_name}, Method: {best_s5.method}, '
          f'QSize: {best_s5.queue_size}, Thresh: {best_s5.threshold}, Retain: {best_s5.retain}')
        w(f'  Prec={best_s5.precision:.2f}%, Rec={best_s5.recall:.2f}%, Spec={best_s5.specificity:.2f}%')
    if s5_cal:
        best_s5c = max(s5_cal, key=lambda r: r.f1)
        w(f'\n**Best calibrated** (stride=5): F1={best_s5c.f1:.2f}%')
        w(f'  Config: {best_s5c.config_name}, Method: {best_s5c.method}, '
          f'QSize: {best_s5c.queue_size}, Thresh: {best_s5c.threshold}, Retain: {best_s5c.retain}')

    # Queue timing at stride=5 (30Hz)
    w('\n### Decision Timing (stride=5, 30Hz)')
    w('| QSize | Sec/Decision | Decisions/Min |')
    w('|-------|-------------|---------------|')
    for qs in QUEUE_SIZES:
        sec_per_win = 5 / 30.0
        sec_per_dec = qs * sec_per_win
        dec_per_min = 60.0 / sec_per_dec
        w(f'| {qs} | {sec_per_dec:.2f}s | {dec_per_min:.1f} |')

    report = '\n'.join(lines)
    report_path = output_dir / 'queue_ablation_report.md'
    report_path.write_text(report)
    print(f"\n  [Report] Saved: {report_path}")
    return report


def _write_top_table(w, results: List[QueueResult], title: str, n: int):
    w(f'### {title} (Top {n})')
    w('')
    w('| # | Config | Stride | Method | QSize | Thresh | Ret | F1 | Prec | Rec | Spec | Acc | Dec | GT+ | GT- |')
    w('|---|--------|--------|--------|-------|--------|-----|-----|------|-----|------|-----|-----|-----|-----|')
    top = sorted(results, key=lambda r: -r.f1)[:n]
    for i, r in enumerate(top, 1):
        w(f'| {i} | {r.config_name} | {r.stride} | {r.method} | {r.queue_size} | '
          f'{r.threshold} | {r.retain} | {r.f1:.2f} | {r.precision:.2f} | {r.recall:.2f} | '
          f'{r.specificity:.2f} | {r.accuracy:.2f} | {r.total_decisions} | {r.gt_positive} | {r.gt_negative} |')


def save_csv(queue_results: List[QueueResult], output_dir: Path):
    if not queue_results:
        return

    fields = [
        'config_name', 'calibrated', 'stride', 'method', 'queue_size', 'threshold', 'retain',
        'f1', 'precision', 'recall', 'specificity', 'accuracy',
        'tp', 'fp', 'tn', 'fn', 'total_decisions', 'gt_positive', 'gt_negative',
    ]

    # Full results
    csv_path = output_dir / 'queue_ablation_full.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in sorted(queue_results, key=lambda r: (-r.f1)):
            writer.writerow({k: getattr(r, k) for k in fields})
    print(f"  [CSV] Saved: {csv_path} ({len(queue_results)} rows)")

    # Best per config
    best_path = output_dir / 'queue_ablation_best_per_config.csv'
    config_names = sorted(set(r.config_name for r in queue_results))
    best_rows = []
    for cn in config_names:
        for calib in [False, True]:
            subset = [r for r in queue_results if r.config_name == cn and r.calibrated == calib]
            if subset:
                best_rows.append(max(subset, key=lambda r: r.f1))
    with open(best_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in best_rows:
            writer.writerow({k: getattr(r, k) for k in fields})
    print(f"  [CSV] Saved: {best_path} ({len(best_rows)} rows)")

    # Best per stride
    stride_path = output_dir / 'queue_ablation_best_per_stride.csv'
    stride_rows = []
    strides = sorted(set(r.stride for r in queue_results))
    for s in strides:
        for calib in [False, True]:
            subset = [r for r in queue_results if r.stride == s and r.calibrated == calib]
            if subset:
                stride_rows.append(max(subset, key=lambda r: r.f1))
    with open(stride_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in stride_rows:
            writer.writerow({k: getattr(r, k) for k in fields})
    print(f"  [CSV] Saved: {stride_path} ({len(stride_rows)} rows)")


def save_summary_json(
    train_results: List[TrainResult],
    queue_results: List[QueueResult],
    output_dir: Path,
):
    summary = {
        'generated': datetime.now().isoformat(),
        'training': {
            'total_configs': len(train_results),
            'completed': sum(1 for r in train_results if r.status == 'completed'),
            'results': [r.to_dict() for r in train_results],
        },
        'queue_eval': {
            'total_results': len(queue_results),
            'uncalibrated': sum(1 for r in queue_results if not r.calibrated),
            'calibrated': sum(1 for r in queue_results if r.calibrated),
        },
    }

    # Best overall
    uncal = [r for r in queue_results if not r.calibrated]
    cal = [r for r in queue_results if r.calibrated]
    if uncal:
        best = max(uncal, key=lambda r: r.f1)
        summary['best_uncalibrated'] = best.to_dict()
    if cal:
        best = max(cal, key=lambda r: r.f1)
        summary['best_calibrated'] = best.to_dict()

    # Best at stride=5 (deployment)
    s5 = [r for r in uncal if r.stride == 5]
    if s5:
        summary['best_stride_5'] = max(s5, key=lambda r: r.f1).to_dict()

    path = output_dir / 'queue_ablation_summary.json'
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  [JSON] Saved: {path}")


# ---------------------------------------------------------------------------
# Eval-only mode (reuse trained models)
# ---------------------------------------------------------------------------

def discover_trained_runs(work_dir: Path) -> List[TrainResult]:
    """Discover already-trained runs in work_dir/runs/."""
    runs_dir = work_dir / 'runs'
    if not runs_dir.exists():
        return []

    results = []
    for run_path in sorted(runs_dir.iterdir()):
        if not run_path.is_dir():
            continue
        pkl = run_path / 'fold_results.pkl'
        if not pkl.exists():
            continue

        name = run_path.name
        metrics = parse_fold_results(pkl)
        results.append(TrainResult(
            name=name,
            config_path=str(next(run_path.glob('*.yaml'), '')),
            run_dir=str(run_path),
            test_f1=metrics.get('test_f1', 0),
            test_f1_std=metrics.get('test_f1_std', 0),
            test_accuracy=metrics.get('test_accuracy', 0),
            test_precision=metrics.get('test_precision', 0),
            test_recall=metrics.get('test_recall', 0),
            num_folds=metrics.get('num_folds', 0),
            fold_f1s=metrics.get('fold_f1s', []),
            status='completed',
        ))

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description='Queue Aggregation × Stride × Config Ablation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--configs', nargs='+', default=None,
                        help='Explicit config paths (default: all in configs_tmp/)')
    parser.add_argument('--configs-dir', type=str, default='configs_tmp',
                        help='Directory containing config YAMLs (default: configs_tmp/)')
    parser.add_argument('--work-dir', type=str, default=None,
                        help='Output directory (default: auto-generated)')
    parser.add_argument('--num-gpus', '-g', type=int, default=4)
    parser.add_argument('--parallel', '-p', type=int, default=2,
                        help='Number of parallel training runs')
    parser.add_argument('--max-folds', type=int, default=None,
                        help='Limit LOSO folds (for testing)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: 2 folds, subset strides/params')
    parser.add_argument('--eval-only', action='store_true',
                        help='Skip training, eval existing runs in --work-dir')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training phase (alias for --eval-only)')
    parser.add_argument('--no-calibration', action='store_true',
                        help='Disable temperature scaling')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--timeout', type=int, default=10800,
                        help='Per-experiment timeout in seconds (default: 3h)')

    return parser.parse_args()


def main():
    args = get_args()

    # Work directory
    if args.work_dir:
        work_dir = Path(args.work_dir)
    else:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        work_dir = Path(f'exps/queue_ablation_{ts}')
    work_dir.mkdir(parents=True, exist_ok=True)

    # Quick mode overrides
    if args.quick:
        args.max_folds = args.max_folds or 2
        eval_strides = EVAL_STRIDES_QUICK
        queue_sizes = QUEUE_SIZES_QUICK
        queue_thresholds = QUEUE_THRESHOLDS_QUICK
        queue_retains = QUEUE_RETAINS_QUICK
        majority_ks = MAJORITY_KS_QUICK
    else:
        eval_strides = EVAL_STRIDES
        queue_sizes = QUEUE_SIZES
        queue_thresholds = QUEUE_THRESHOLDS
        queue_retains = QUEUE_RETAINS
        majority_ks = MAJORITY_KS

    enable_calibration = not args.no_calibration

    print(f"\n{'='*80}")
    print(f"QUEUE AGGREGATION ABLATION")
    print(f"  Work dir: {work_dir}")
    print(f"  Eval strides: {eval_strides}")
    print(f"  Quick mode: {args.quick}")
    print(f"  Calibration: {enable_calibration}")
    print(f"{'='*80}")

    # Phase 1: Training
    eval_only = args.eval_only or args.skip_training

    if eval_only:
        print(f"\n[Skip training] Discovering existing runs in {work_dir}/runs/ ...")
        train_results = discover_trained_runs(work_dir)
        if not train_results:
            print("ERROR: No trained runs found. Run training first.")
            sys.exit(1)
        print(f"  Found {len(train_results)} trained configs")
        for r in train_results:
            print(f"    {r.name}: F1={r.test_f1:.2f}% ({r.num_folds} folds)")
    else:
        configs = discover_configs(Path(args.configs_dir), args.configs)
        if not configs:
            print("ERROR: No configs found.")
            sys.exit(1)
        print(f"\nConfigs to train ({len(configs)}):")
        for name, path in configs:
            print(f"  {name}: {path}")

        train_results = train_all(
            configs=configs,
            work_dir=work_dir,
            num_gpus=args.num_gpus,
            parallel=args.parallel,
            max_folds=args.max_folds,
        )

        # Save training results
        train_json = [r.to_dict() for r in train_results]
        with open(work_dir / 'train_results.json', 'w') as f:
            json.dump(train_json, f, indent=2, default=str)

    # Phase 2: Queue Eval
    queue_results, per_config = eval_all(
        train_results=train_results,
        work_dir=work_dir,
        eval_strides=eval_strides,
        queue_sizes=queue_sizes,
        queue_thresholds=queue_thresholds,
        queue_retains=queue_retains,
        majority_ks=majority_ks,
        enable_calibration=enable_calibration,
    )

    # Phase 3: Reports
    print(f"\n{'='*80}")
    print(f"PHASE 3: GENERATING REPORTS")
    print(f"{'='*80}")

    report = generate_report(
        train_results=train_results,
        queue_results=queue_results,
        eval_strides=eval_strides,
        majority_ks=majority_ks,
        output_dir=work_dir,
    )

    save_csv(queue_results, work_dir)
    save_summary_json(train_results, queue_results, work_dir)

    # Print final summary
    uncal = [r for r in queue_results if not r.calibrated]
    cal = [r for r in queue_results if r.calibrated]

    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"  Configs trained: {sum(1 for r in train_results if r.status == 'completed')}/{len(train_results)}")
    print(f"  Queue results: {len(uncal)} uncalibrated, {len(cal)} calibrated")

    if uncal:
        best = max(uncal, key=lambda r: r.f1)
        print(f"\n  BEST UNCALIBRATED:")
        print(f"    Config: {best.config_name}")
        print(f"    F1={best.f1:.2f}%, Prec={best.precision:.2f}%, Rec={best.recall:.2f}%, Spec={best.specificity:.2f}%")
        print(f"    Stride={best.stride}, Method={best.method}, QSize={best.queue_size}, Thresh={best.threshold}, Retain={best.retain}")

    if cal:
        best = max(cal, key=lambda r: r.f1)
        print(f"\n  BEST CALIBRATED:")
        print(f"    Config: {best.config_name}")
        print(f"    F1={best.f1:.2f}%, Prec={best.precision:.2f}%, Rec={best.recall:.2f}%, Spec={best.specificity:.2f}%")
        print(f"    Stride={best.stride}, Method={best.method}, QSize={best.queue_size}, Thresh={best.threshold}, Retain={best.retain}")

    # Deployment recommendation (stride=5)
    s5 = [r for r in uncal if r.stride == 5]
    if s5:
        best_s5 = max(s5, key=lambda r: r.f1)
        print(f"\n  DEPLOYMENT (stride=5):")
        print(f"    Config: {best_s5.config_name}")
        print(f"    F1={best_s5.f1:.2f}%, Method={best_s5.method}, QSize={best_s5.queue_size}, Thresh={best_s5.threshold}")

    print(f"\n  Output: {work_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
