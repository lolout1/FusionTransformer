"""Pooled micro-averaged metrics and alpha queue simulation for LOSO cross-validation."""

import json
import csv
import traceback
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, roc_auc_score, matthews_corrcoef,
    balanced_accuracy_score, average_precision_score
)


def extract_fold_predictions(fold_results: List[Dict]) -> Tuple[List[int], List[float], Dict[str, Dict]]:
    """Extract pooled and per-subject predictions from fold results."""
    all_targets = []
    all_probs = []
    per_subject = {}
    n_missing = 0

    for result in fold_results:
        if result.get('status') == 'failed':
            continue

        ta = result.get('threshold_analysis')
        if not ta or 'targets' not in ta or 'probabilities' not in ta:
            n_missing += 1
            continue

        targets = ta['targets']
        probs = ta['probabilities']
        if not targets or not probs:
            n_missing += 1
            continue

        all_targets.extend(targets)
        all_probs.extend(probs)

        subject = str(result.get('test_subject', 'unknown'))
        per_subject[subject] = {'targets': targets, 'probabilities': probs}

    if n_missing > 0:
        print(f"  [Queue] {n_missing} fold(s) missing raw predictions, skipped")

    return all_targets, all_probs, per_subject


def compute_pooled_metrics(targets: List[int], probs: List[float], threshold: float = 0.5) -> Dict:
    """Compute metrics on pooled predictions from all LOSO folds (micro-average)."""
    if not targets:
        return {'error': 'No predictions available'}

    y_true = np.array(targets)
    y_prob = np.array(probs)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        'n_samples': len(y_true),
        'n_positive': int(y_true.sum()),
        'n_negative': int((y_true == 0).sum()),
        'threshold': threshold,
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
    }

    if len(np.unique(y_true)) > 1:
        metrics['auc'] = roc_auc_score(y_true, y_prob)
        metrics['pr_auc'] = average_precision_score(y_true, y_prob)
    else:
        metrics['auc'] = None
        metrics['pr_auc'] = None

    return metrics


def simulate_queue_subject(
    targets: List[int],
    probabilities: List[float],
    queue_size: int,
    threshold: float,
    retain: int,
) -> Dict:
    """Run alpha queue on one subject's test predictions.

    Ground truth: if ANY window in the queue batch has label=1, queue GT = 1.
    """
    from collections import deque

    queue = deque(maxlen=queue_size)
    label_buf = deque(maxlen=queue_size)
    # Track which labels entered since last decision for GT computation
    batch_labels = []

    queue_preds = []
    queue_gt = []
    queue_avg_probs = []

    for prob, label in zip(probabilities, targets):
        queue.append(prob)
        label_buf.append(label)
        batch_labels.append(label)

        if len(queue) < queue_size:
            continue

        avg = sum(queue) / len(queue)
        gt = 1 if any(l == 1 for l in batch_labels) else 0
        decision = 1 if avg > threshold else 0

        queue_preds.append(decision)
        queue_gt.append(gt)
        queue_avg_probs.append(avg)

        if avg > threshold:
            # FALL: flush
            queue.clear()
            label_buf.clear()
            batch_labels = []
        else:
            # ADL: retain last N
            kept_probs = list(queue)[-retain:] if retain > 0 else []
            kept_labels = list(label_buf)[-retain:] if retain > 0 else []
            queue.clear()
            label_buf.clear()
            queue.extend(kept_probs)
            label_buf.extend(kept_labels)
            batch_labels = list(kept_labels)

    if not queue_preds:
        return {
            'skipped': True,
            'n_windows': len(targets),
            'n_decisions': 0,
        }

    y_true = np.array(queue_gt)
    y_pred = np.array(queue_preds)

    if len(y_true) == 0:
        return {'skipped': True, 'n_windows': len(targets), 'n_decisions': 0}

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        'skipped': False,
        'n_windows': len(targets),
        'n_decisions': len(queue_preds),
        'queue_targets': queue_gt,
        'queue_predictions': queue_preds,
        'queue_avg_probs': queue_avg_probs,
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
    }


def simulate_queue_all_subjects(
    per_subject: Dict[str, Dict],
    queue_size: int,
    threshold: float,
    retain: int,
) -> Dict:
    """Run queue per subject, pool decisions for micro-averaged metrics."""
    all_gt = []
    all_preds = []
    n_skipped = 0
    total_windows = 0
    per_subject_results = {}

    for subject, data in per_subject.items():
        result = simulate_queue_subject(
            data['targets'], data['probabilities'],
            queue_size, threshold, retain,
        )
        per_subject_results[subject] = result
        total_windows += result['n_windows']

        if result.get('skipped'):
            n_skipped += 1
            continue

        all_gt.extend(result['queue_targets'])
        all_preds.extend(result['queue_predictions'])

    if not all_gt:
        return {
            'params': {'queue_size': queue_size, 'threshold': threshold, 'retain': retain},
            'error': 'No queue decisions produced',
            'n_subjects': len(per_subject),
            'n_subjects_skipped': n_skipped,
        }

    y_true = np.array(all_gt)
    y_pred = np.array(all_preds)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        'params': {'queue_size': queue_size, 'threshold': threshold, 'retain': retain},
        'n_subjects': len(per_subject),
        'n_subjects_skipped': n_skipped,
        'total_windows': total_windows,
        'total_decisions': len(all_gt),
        'pooled_metrics': {
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'accuracy': accuracy_score(y_true, y_pred),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        },
        'per_subject': per_subject_results,
    }


DEFAULT_QUEUE_SIZES = [3, 5, 10, 15, 20]
DEFAULT_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
DEFAULT_RETAINS = [0, 2, 5]


def sweep_queue_params(
    per_subject: Dict[str, Dict],
    sizes: Optional[List[int]] = None,
    thresholds: Optional[List[float]] = None,
    retains: Optional[List[int]] = None,
) -> List[Dict]:
    """Sweep queue parameters, return results sorted by F1 descending."""
    sizes = sizes or DEFAULT_QUEUE_SIZES
    thresholds = thresholds or DEFAULT_THRESHOLDS
    retains = retains or DEFAULT_RETAINS

    results = []
    for size in sizes:
        for thresh in thresholds:
            for retain in retains:
                if retain >= size:
                    continue
                r = simulate_queue_all_subjects(per_subject, size, thresh, retain)
                if 'error' not in r:
                    results.append(r)

    results.sort(key=lambda x: x['pooled_metrics']['f1'], reverse=True)
    return results


def print_pooled_and_queue_results(
    pooled: Dict,
    sweep_results: List[Dict],
    n_folds: int,
    n_top: int = 5,
) -> None:
    """Print formatted comparison table."""
    print("\n" + "=" * 90)
    print(f"POOLED MICRO-AVERAGED METRICS ({pooled['n_samples']} windows across {n_folds} folds)")
    print("=" * 90)

    print(f"\nWindow-Level (\u03c4={pooled['threshold']:.2f}):")
    print(f"  F1:          {pooled['f1']*100:>6.2f}%    Precision: {pooled['precision']*100:>6.2f}%    Recall: {pooled['recall']*100:>6.2f}%")
    print(f"  Accuracy:    {pooled['accuracy']*100:>6.2f}%    Specificity: {pooled['specificity']*100:>6.2f}%")
    auc_str = f"{pooled['auc']*100:.2f}%" if pooled.get('auc') is not None else "N/A"
    print(f"  AUC:         {auc_str:>7}    MCC: {pooled['mcc']:.4f}")
    print(f"  Confusion:   TP={pooled['tp']}  FP={pooled['fp']}  TN={pooled['tn']}  FN={pooled['fn']}")

    if not sweep_results:
        print("\n  [Queue] No valid queue configurations produced results")
        print("=" * 90)
        return

    n_configs = len(sweep_results)
    n_subjects = sweep_results[0].get('n_subjects', 0) if sweep_results else 0

    print(f"\n{'='*90}")
    print(f"ALPHA QUEUE PARAMETER SWEEP ({n_configs} configs, {n_subjects} subjects)")
    print(f"{'='*90}")

    # Header
    print(f"\nTop {min(n_top, len(sweep_results))} by F1:")
    print(f"  {'Size':>4}  {'Thresh':>6}  {'Retain':>6} | {'F1':>7}  {'Prec':>7}  {'Recall':>7}  {'Acc':>7} | {'Dec':>5}  {'TP':>3}  {'FP':>3}  {'TN':>3}  {'FN':>3}")
    print(f"  {'-'*80}")

    for r in sweep_results[:n_top]:
        p = r['params']
        m = r['pooled_metrics']
        print(f"  {p['queue_size']:>4}  {p['threshold']:>6.2f}  {p['retain']:>6} | "
              f"{m['f1']*100:>6.2f}% {m['precision']*100:>6.2f}% {m['recall']*100:>6.2f}% {m['accuracy']*100:>6.2f}% | "
              f"{r['total_decisions']:>5}  {m['tp']:>3}  {m['fp']:>3}  {m['tn']:>3}  {m['fn']:>3}")

    # Comparison: window vs best queue
    best = sweep_results[0]
    bm = best['pooled_metrics']
    bp = best['params']

    print(f"\n{'-'*90}")
    print(f"WINDOW vs QUEUE COMPARISON")
    print(f"{'-'*90}")
    print(f"  {'':>18} {'F1':>8}  {'Prec':>8}  {'Recall':>8}  {'Acc':>8}  {'Spec':>8}")
    win_label = 'Window (\u03c4=0.5)'
    queue_label = 'Queue (best)'
    print(f"  {win_label:>18} {pooled['f1']*100:>7.2f}% {pooled['precision']*100:>7.2f}% {pooled['recall']*100:>7.2f}% {pooled['accuracy']*100:>7.2f}% {pooled['specificity']*100:>7.2f}%")
    print(f"  {queue_label:>18} {bm['f1']*100:>7.2f}% {bm['precision']*100:>7.2f}% {bm['recall']*100:>7.2f}% {bm['accuracy']*100:>7.2f}% {bm['specificity']*100:>7.2f}%")

    d_f1 = (bm['f1'] - pooled['f1']) * 100
    d_prec = (bm['precision'] - pooled['precision']) * 100
    d_rec = (bm['recall'] - pooled['recall']) * 100
    d_acc = (bm['accuracy'] - pooled['accuracy']) * 100
    d_spec = (bm['specificity'] - pooled['specificity']) * 100
    print(f"  {'Delta':>18} {d_f1:>+7.2f}% {d_prec:>+7.2f}% {d_rec:>+7.2f}% {d_acc:>+7.2f}% {d_spec:>+7.2f}%")
    print(f"  Best queue config: size={bp['queue_size']}, threshold={bp['threshold']}, retain={bp['retain']}")
    print("=" * 90)


def save_pooled_and_queue_results(
    pooled: Dict,
    sweep_results: List[Dict],
    per_subject: Dict[str, Dict],
    output_dir: str,
) -> None:
    """Save all results to disk."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Pooled metrics JSON
    pooled_path = out / 'pooled_metrics.json'
    # Convert numpy types for JSON serialization
    serializable_pooled = {
        k: float(v) if isinstance(v, (np.floating, float)) and v is not None
        else int(v) if isinstance(v, (np.integer, int)) and not isinstance(v, bool)
        else v
        for k, v in pooled.items()
    }
    with open(pooled_path, 'w') as f:
        json.dump(serializable_pooled, f, indent=2)
    print(f"  [Queue] Saved pooled metrics: {pooled_path}")

    # Queue sweep CSV
    if sweep_results:
        csv_path = out / 'queue_sweep.csv'
        fieldnames = [
            'queue_size', 'threshold', 'retain',
            'f1', 'precision', 'recall', 'accuracy', 'specificity',
            'tp', 'fp', 'tn', 'fn',
            'total_decisions', 'n_subjects', 'n_subjects_skipped',
        ]
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in sweep_results:
                row = {**r['params'], **r['pooled_metrics']}
                row['total_decisions'] = r['total_decisions']
                row['n_subjects'] = r['n_subjects']
                row['n_subjects_skipped'] = r['n_subjects_skipped']
                # Convert to percentages for CSV readability
                for k in ['f1', 'precision', 'recall', 'accuracy', 'specificity']:
                    if k in row:
                        row[k] = round(row[k] * 100, 2)
                writer.writerow(row)
        print(f"  [Queue] Saved queue sweep: {csv_path}")

        # Best config JSON (without per-subject raw data to keep it small)
        best = sweep_results[0]
        best_info = {
            'params': best['params'],
            'pooled_metrics': {k: float(v) if isinstance(v, float) else v
                               for k, v in best['pooled_metrics'].items()},
            'total_decisions': best['total_decisions'],
            'n_subjects': best['n_subjects'],
            'n_subjects_skipped': best['n_subjects_skipped'],
            'per_subject_summary': {},
        }
        for subj, sr in best.get('per_subject', {}).items():
            if sr.get('skipped'):
                best_info['per_subject_summary'][subj] = {'skipped': True}
            else:
                best_info['per_subject_summary'][subj] = {
                    'n_windows': sr['n_windows'],
                    'n_decisions': sr['n_decisions'],
                    'f1': round(sr['f1'] * 100, 2),
                    'precision': round(sr['precision'] * 100, 2),
                    'recall': round(sr['recall'] * 100, 2),
                    'accuracy': round(sr['accuracy'] * 100, 2),
                }
        best_path = out / 'queue_best_config.json'
        with open(best_path, 'w') as f:
            json.dump(best_info, f, indent=2)
        print(f"  [Queue] Saved best queue config: {best_path}")


def compute_pooled_and_queue_metrics(
    fold_results: List[Dict],
    output_dir: Optional[str] = None,
    print_results: bool = True,
) -> Dict:
    """Main entry point. Called from ray_distributed.py after all folds complete."""
    print("\n[Queue Metrics] Computing pooled and queue metrics...")

    # 1. Extract predictions
    all_targets, all_probs, per_subject = extract_fold_predictions(fold_results)

    n_folds_with = len(per_subject)
    n_folds_total = sum(1 for r in fold_results if r.get('status') != 'failed')

    if not all_targets:
        print("  [Queue] WARNING: No raw predictions found in any fold. "
              "Ensure threshold_analysis data is present.")
        return {'error': 'No predictions available', 'n_folds_with_predictions': 0}

    print(f"  [Queue] Extracted predictions from {n_folds_with}/{n_folds_total} folds "
          f"({len(all_targets)} total windows, {len(per_subject)} subjects)")

    # 2. Pooled window-level metrics
    pooled = compute_pooled_metrics(all_targets, all_probs, threshold=0.5)

    # 3. Queue parameter sweep
    sweep_results = sweep_queue_params(per_subject)
    print(f"  [Queue] Swept {len(sweep_results)} queue configurations")

    # 4. Print
    if print_results:
        print_pooled_and_queue_results(pooled, sweep_results, n_folds_with)

    # 5. Save
    if output_dir:
        save_pooled_and_queue_results(pooled, sweep_results, per_subject, output_dir)

    return {
        'pooled_window_metrics': pooled,
        'queue_sweep_results': sweep_results,
        'best_queue_config': sweep_results[0] if sweep_results else None,
        'per_subject_predictions': per_subject,
        'n_folds_with_predictions': n_folds_with,
        'n_folds_missing': n_folds_total - n_folds_with,
    }
