"""Comprehensive metrics computation."""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, matthews_corrcoef, balanced_accuracy_score,
    log_loss, brier_score_loss
)

from ..data.schema import TestWindow, PredictionResult


class MetricsCalculator:
    """Calculate all classification metrics."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def compute_all(
        self,
        predictions: List[PredictionResult],
        windows: List[TestWindow]
    ) -> Dict[str, float]:
        """Compute all metrics."""
        y_true = np.array([w.ground_truth for w in windows])
        y_prob = np.array([p.probability for p in predictions])
        y_pred = (y_prob > self.threshold).astype(int)

        metrics = {}

        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity
        metrics['sensitivity'] = metrics['recall']
        metrics['specificity'] = self._specificity(y_true, y_pred)
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)

        # Confusion matrix values
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        metrics['tp'] = int(tp)
        metrics['fp'] = int(fp)
        metrics['tn'] = int(tn)
        metrics['fn'] = int(fn)

        # Derived metrics
        metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative predictive value
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0  # False positive rate
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0  # False negative rate

        # Probability-based metrics (if both classes present)
        if len(np.unique(y_true)) > 1:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            metrics['pr_auc'] = average_precision_score(y_true, y_prob)
            metrics['brier_score'] = brier_score_loss(y_true, y_prob)
            try:
                metrics['log_loss'] = log_loss(y_true, y_prob)
            except ValueError:
                metrics['log_loss'] = None
        else:
            metrics['roc_auc'] = None
            metrics['pr_auc'] = None
            metrics['brier_score'] = None
            metrics['log_loss'] = None

        # Per-class F1
        metrics['f1_fall'] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        metrics['f1_adl'] = f1_score(y_true, y_pred, pos_label=0, zero_division=0)

        return metrics

    def compute_confusion_matrix(
        self,
        predictions: List[PredictionResult],
        windows: List[TestWindow]
    ) -> np.ndarray:
        """Compute confusion matrix."""
        y_true = [w.ground_truth for w in windows]
        y_pred = [1 if p.probability > self.threshold else 0 for p in predictions]
        return confusion_matrix(y_true, y_pred, labels=[0, 1])

    def compute_roc_curve(
        self,
        predictions: List[PredictionResult],
        windows: List[TestWindow]
    ) -> Dict:
        """Compute ROC curve data."""
        y_true = np.array([w.ground_truth for w in windows])
        y_prob = np.array([p.probability for p in predictions])

        if len(np.unique(y_true)) < 2:
            return {'fpr': [], 'tpr': [], 'thresholds': [], 'auc': None}

        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)

        return {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist(),
            'auc': auc
        }

    def compute_pr_curve(
        self,
        predictions: List[PredictionResult],
        windows: List[TestWindow]
    ) -> Dict:
        """Compute precision-recall curve data."""
        y_true = np.array([w.ground_truth for w in windows])
        y_prob = np.array([p.probability for p in predictions])

        if len(np.unique(y_true)) < 2:
            return {'precision': [], 'recall': [], 'thresholds': [], 'auc': None}

        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        auc = average_precision_score(y_true, y_prob)

        return {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': thresholds.tolist(),
            'auc': auc
        }

    def compute_threshold_sweep(
        self,
        predictions: List[PredictionResult],
        windows: List[TestWindow],
        thresholds: Optional[List[float]] = None
    ) -> Dict[float, Dict]:
        """Compute metrics at different thresholds."""
        if thresholds is None:
            thresholds = np.arange(0.1, 0.95, 0.05).tolist()

        y_true = np.array([w.ground_truth for w in windows])
        y_prob = np.array([p.probability for p in predictions])

        results = {}
        for thresh in thresholds:
            y_pred = (y_prob > thresh).astype(int)

            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()

            results[thresh] = {
                'f1': f1_score(y_true, y_pred, zero_division=0),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'accuracy': accuracy_score(y_true, y_pred),
                'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
            }

        return results

    def compute_per_subject(
        self,
        predictions: List[PredictionResult],
        windows: List[TestWindow]
    ) -> Dict[str, Dict]:
        """Compute metrics per subject."""
        # Group by UUID
        by_uuid = {}
        for pred, window in zip(predictions, windows):
            uuid = window.uuid
            if uuid not in by_uuid:
                by_uuid[uuid] = {'preds': [], 'windows': []}
            by_uuid[uuid]['preds'].append(pred)
            by_uuid[uuid]['windows'].append(window)

        # Compute per-subject
        results = {}
        for uuid, data in by_uuid.items():
            preds, wins = data['preds'], data['windows']
            y_true = np.array([w.ground_truth for w in wins])
            y_pred = np.array([1 if p.probability > self.threshold else 0 for p in preds])

            results[uuid] = {
                'n_windows': len(wins),
                'n_falls': int(sum(y_true)),
                'n_adl': int(len(y_true) - sum(y_true)),
                'accuracy': accuracy_score(y_true, y_pred) if len(y_true) > 0 else 0,
                'f1': f1_score(y_true, y_pred, zero_division=0) if len(np.unique(y_true)) > 1 else 0,
            }

        return results

    def find_optimal_threshold(
        self,
        predictions: List[PredictionResult],
        windows: List[TestWindow],
        metric: str = 'f1'
    ) -> Tuple[float, float]:
        """Find threshold that maximizes given metric."""
        sweep = self.compute_threshold_sweep(predictions, windows)

        best_thresh = 0.5
        best_value = 0.0

        for thresh, metrics in sweep.items():
            if metrics[metric] > best_value:
                best_value = metrics[metric]
                best_thresh = thresh

        return best_thresh, best_value

    @staticmethod
    def _specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute specificity (true negative rate)."""
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0

    @staticmethod
    def format_metrics(metrics: Dict, precision: int = 4) -> Dict[str, str]:
        """Format metrics for display."""
        formatted = {}
        for key, value in metrics.items():
            if value is None:
                formatted[key] = "N/A"
            elif isinstance(value, float):
                if key in ['tp', 'fp', 'tn', 'fn']:
                    formatted[key] = str(int(value))
                else:
                    formatted[key] = f"{value:.{precision}f}"
            else:
                formatted[key] = str(value)
        return formatted
