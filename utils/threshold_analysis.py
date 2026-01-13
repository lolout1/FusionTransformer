"""Threshold optimization for binary classification."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    f1_score, fbeta_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, roc_auc_score, roc_curve
)


class ThresholdAnalyzer:
    """Systematic threshold optimization for binary fall detection."""

    def __init__(self, targets: np.ndarray, probabilities: np.ndarray):
        self.targets = np.asarray(targets).ravel()
        self.probs = np.asarray(probabilities).ravel()
        self._validate_inputs()

    def _validate_inputs(self):
        if len(self.targets) != len(self.probs):
            raise ValueError(f"Length mismatch: targets={len(self.targets)}, probs={len(self.probs)}")
        if len(self.targets) == 0:
            raise ValueError("Empty inputs")

    def sweep_thresholds(
        self,
        start: float = 0.3,
        end: float = 0.9,
        step: float = 0.05
    ) -> List[Dict]:
        """Compute metrics across threshold range."""
        thresholds = np.arange(start, end + step / 2, step)
        results = []
        for t in thresholds:
            metrics = self._compute_metrics_at_threshold(t)
            results.append(metrics)
        return results

    def _compute_metrics_at_threshold(self, threshold: float) -> Dict:
        preds = (self.probs >= threshold).astype(int)
        tn, fp, fn, tp = self._safe_confusion_matrix(self.targets, preds)

        return {
            'threshold': round(threshold, 3),
            'f1': f1_score(self.targets, preds, zero_division=0),
            'f2': fbeta_score(self.targets, preds, beta=2, zero_division=0),
            'precision': precision_score(self.targets, preds, zero_division=0),
            'recall': recall_score(self.targets, preds, zero_division=0),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'accuracy': accuracy_score(self.targets, preds),
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        }

    def _safe_confusion_matrix(self, y_true, y_pred) -> Tuple[int, int, int, int]:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
        return tn, fp, fn, tp

    def find_optimal_threshold(self, criterion: str = 'f1') -> Dict:
        """Find optimal threshold by specified criterion."""
        results = self.sweep_thresholds()

        if criterion == 'f1':
            return max(results, key=lambda x: x['f1'])
        elif criterion == 'f2':
            return max(results, key=lambda x: x['f2'])
        elif criterion == 'youden':
            return max(results, key=lambda x: x['recall'] + x['specificity'] - 1)
        elif criterion == 'gmean':
            return max(results, key=lambda x: np.sqrt(x['recall'] * x['specificity']))
        elif criterion == 'recall_95':
            valid = [r for r in results if r['recall'] >= 0.95]
            return min(valid, key=lambda x: x['threshold']) if valid else results[0]
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

    def get_all_optimal_thresholds(self) -> Dict[str, Dict]:
        """Return optimal threshold for each criterion."""
        criteria = ['f1', 'f2', 'youden', 'gmean']
        return {c: self.find_optimal_threshold(c) for c in criteria}

    def compute_auc(self) -> float:
        """Compute proper ROC-AUC using probabilities."""
        if len(np.unique(self.targets)) < 2:
            return 0.5
        return roc_auc_score(self.targets, self.probs)

    def summary(self) -> Dict:
        """Generate comprehensive threshold analysis summary."""
        sweep = self.sweep_thresholds()
        optimal_f1 = self.find_optimal_threshold('f1')
        default_05 = self._compute_metrics_at_threshold(0.5)

        return {
            'auc': self.compute_auc(),
            'n_samples': len(self.targets),
            'n_positive': int(self.targets.sum()),
            'n_negative': int(len(self.targets) - self.targets.sum()),
            'default_threshold': default_05,
            'optimal_f1_threshold': optimal_f1,
            'f1_improvement': optimal_f1['f1'] - default_05['f1'],
            'sweep_results': sweep,
        }


def analyze_fold_thresholds(
    fold_results: List[Dict],
    criterion: str = 'f1'
) -> Dict:
    """Aggregate threshold analysis across LOSO folds for Ray distributed training."""
    optimal_thresholds = []
    f1_at_optimal = []
    f1_at_default = []

    for fold in fold_results:
        if 'threshold_analysis' not in fold:
            continue
        analysis = fold['threshold_analysis']
        optimal_thresholds.append(analysis['optimal_f1_threshold']['threshold'])
        f1_at_optimal.append(analysis['optimal_f1_threshold']['f1'])
        f1_at_default.append(analysis['default_threshold']['f1'])

    if not optimal_thresholds:
        return {'error': 'No threshold analysis data in fold results'}

    return {
        'mean_optimal_threshold': np.mean(optimal_thresholds),
        'std_optimal_threshold': np.std(optimal_thresholds),
        'median_optimal_threshold': np.median(optimal_thresholds),
        'mean_f1_at_optimal': np.mean(f1_at_optimal),
        'mean_f1_at_default': np.mean(f1_at_default),
        'mean_f1_improvement': np.mean(f1_at_optimal) - np.mean(f1_at_default),
        'per_fold_thresholds': optimal_thresholds,
    }


def format_threshold_table(sweep_results: List[Dict]) -> str:
    """Format threshold sweep as ASCII table."""
    header = f"{'Thresh':>6} | {'F1':>6} | {'Prec':>6} | {'Recall':>6} | {'Spec':>6} | {'Acc':>6}"
    sep = "-" * len(header)
    lines = [header, sep]

    for r in sweep_results:
        lines.append(
            f"{r['threshold']:>6.2f} | {r['f1']*100:>5.1f}% | {r['precision']*100:>5.1f}% | "
            f"{r['recall']*100:>5.1f}% | {r['specificity']*100:>5.1f}% | {r['accuracy']*100:>5.1f}%"
        )
    return '\n'.join(lines)


def compute_metrics_at_threshold(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float
) -> Dict:
    """
    Compute classification metrics at a specific threshold.

    This is a standalone function for recomputing metrics at a global threshold
    across all folds (for deployment-realistic evaluation).

    Args:
        targets: Ground truth labels (0/1)
        probabilities: Predicted probabilities
        threshold: Classification threshold

    Returns:
        Dict with f1, precision, recall, specificity, accuracy
    """
    targets = np.asarray(targets).ravel()
    probs = np.asarray(probabilities).ravel()
    preds = (probs >= threshold).astype(int)

    cm = confusion_matrix(targets, preds, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    return {
        'threshold': round(threshold, 3),
        'f1': f1_score(targets, preds, zero_division=0),
        'precision': precision_score(targets, preds, zero_division=0),
        'recall': recall_score(targets, preds, zero_division=0),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'accuracy': accuracy_score(targets, preds),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
    }


def evaluate_fixed_thresholds(
    fold_results: List[Dict],
    thresholds: List[float] = [0.5, 0.55, 0.6, 0.7, 0.9]
) -> Dict:
    """
    Evaluate metrics at fixed thresholds across all folds.

    This is useful for comparing model performance at standard operating points
    without per-fold optimization bias.

    Args:
        fold_results: List of fold result dicts containing 'threshold_analysis'
        thresholds: List of fixed thresholds to evaluate

    Returns:
        Dict mapping threshold -> aggregate metrics across folds
    """
    # Collect folds with raw probability data
    valid_folds = []
    for fold in fold_results:
        ta = fold.get('threshold_analysis')
        if ta and 'targets' in ta and 'probabilities' in ta:
            valid_folds.append({
                'test_subject': fold.get('test_subject'),
                'targets': ta['targets'],
                'probabilities': ta['probabilities'],
            })

    if not valid_folds:
        return {'error': 'No folds with raw probability data'}

    results = {}
    for threshold in thresholds:
        fold_metrics = []
        for fold in valid_folds:
            metrics = compute_metrics_at_threshold(
                fold['targets'],
                fold['probabilities'],
                threshold
            )
            metrics['test_subject'] = fold['test_subject']
            fold_metrics.append(metrics)

        # Aggregate
        f1s = [m['f1'] for m in fold_metrics]
        precs = [m['precision'] for m in fold_metrics]
        recs = [m['recall'] for m in fold_metrics]
        specs = [m['specificity'] for m in fold_metrics]
        accs = [m['accuracy'] for m in fold_metrics]

        results[threshold] = {
            'threshold': threshold,
            'n_folds': len(valid_folds),
            'mean_f1': float(np.mean(f1s)),
            'std_f1': float(np.std(f1s)),
            'mean_precision': float(np.mean(precs)),
            'std_precision': float(np.std(precs)),
            'mean_recall': float(np.mean(recs)),
            'std_recall': float(np.std(recs)),
            'mean_specificity': float(np.mean(specs)),
            'std_specificity': float(np.std(specs)),
            'mean_accuracy': float(np.mean(accs)),
            'std_accuracy': float(np.std(accs)),
            'per_fold_metrics': fold_metrics,
        }

    return results


def compute_global_threshold_metrics(
    fold_results: List[Dict],
    global_threshold: Optional[float] = None
) -> Dict:
    """
    Compute metrics at a global (fixed) threshold across all folds.

    For deployment, we need a single threshold used across all subjects.
    This function:
    1. Computes the global threshold (mean of per-fold optimal) if not provided
    2. Recomputes metrics at this threshold for each fold using stored probabilities
    3. Returns aggregate statistics

    Args:
        fold_results: List of fold result dicts containing 'threshold_analysis'
        global_threshold: Fixed threshold to use (if None, computed as mean of per-fold optimal)

    Returns:
        Dict with global threshold and metrics at that threshold
    """
    # Collect threshold analyses with raw data
    valid_folds = []
    for fold in fold_results:
        ta = fold.get('threshold_analysis')
        if ta and 'targets' in ta and 'probabilities' in ta:
            valid_folds.append({
                'test_subject': fold.get('test_subject'),
                'targets': ta['targets'],
                'probabilities': ta['probabilities'],
                'per_fold_optimal': ta['optimal_f1_threshold']['threshold'],
            })

    if not valid_folds:
        return {'error': 'No folds with raw probability data for global threshold computation'}

    # Compute global threshold if not provided
    per_fold_thresholds = [f['per_fold_optimal'] for f in valid_folds]
    if global_threshold is None:
        global_threshold = float(np.mean(per_fold_thresholds))

    # Recompute metrics at global threshold for each fold
    global_metrics = []
    for fold in valid_folds:
        metrics = compute_metrics_at_threshold(
            fold['targets'],
            fold['probabilities'],
            global_threshold
        )
        metrics['test_subject'] = fold['test_subject']
        metrics['per_fold_optimal'] = fold['per_fold_optimal']
        global_metrics.append(metrics)

    # Aggregate
    f1s = [m['f1'] for m in global_metrics]
    precs = [m['precision'] for m in global_metrics]
    recs = [m['recall'] for m in global_metrics]
    specs = [m['specificity'] for m in global_metrics]
    accs = [m['accuracy'] for m in global_metrics]

    return {
        'global_threshold': round(global_threshold, 3),
        'threshold_std': round(float(np.std(per_fold_thresholds)), 3),
        'n_folds': len(valid_folds),
        'mean_f1': float(np.mean(f1s)),
        'std_f1': float(np.std(f1s)),
        'mean_precision': float(np.mean(precs)),
        'std_precision': float(np.std(precs)),
        'mean_recall': float(np.mean(recs)),
        'std_recall': float(np.std(recs)),
        'mean_specificity': float(np.mean(specs)),
        'std_specificity': float(np.std(specs)),
        'mean_accuracy': float(np.mean(accs)),
        'std_accuracy': float(np.std(accs)),
        'per_fold_metrics': global_metrics,
    }
