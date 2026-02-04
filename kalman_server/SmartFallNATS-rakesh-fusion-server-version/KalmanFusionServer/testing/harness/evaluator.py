"""Metrics computation and evaluation."""

from __future__ import annotations
from typing import Optional, List, Dict
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, classification_report, roc_auc_score, precision_recall_curve
)

from ..data.schema import TestWindow, WindowPrediction, SessionPrediction


class Evaluator:
    """Compute classification metrics."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def compute_window_metrics(
        self,
        predictions: list[WindowPrediction],
        windows: list[TestWindow]
    ) -> dict:
        """Compute metrics at window level.

        Args:
            predictions: Model predictions
            windows: Ground truth windows

        Returns:
            Dictionary of metrics
        """
        if len(predictions) != len(windows):
            raise ValueError(f"Predictions ({len(predictions)}) != windows ({len(windows)})")

        y_true = np.array([w.ground_truth for w in windows])
        y_prob = np.array([p.probability for p in predictions])
        y_pred = (y_prob > self.threshold).astype(int)

        metrics = {
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'accuracy': accuracy_score(y_true, y_pred),
            'specificity': self._specificity(y_true, y_pred),
        }

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        metrics['confusion_matrix'] = cm.tolist()
        metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp'] = cm.ravel()

        # ROC-AUC if both classes present
        if len(np.unique(y_true)) > 1:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        else:
            metrics['roc_auc'] = None

        # Error analysis
        metrics['false_negatives'] = self._get_error_indices(y_true, y_pred, 'FN')
        metrics['false_positives'] = self._get_error_indices(y_true, y_pred, 'FP')

        return metrics

    def compute_session_metrics(self, predictions: list[SessionPrediction]) -> dict:
        """Compute metrics at session level (after alpha queue).

        Args:
            predictions: Session-level predictions

        Returns:
            Dictionary of metrics
        """
        if not predictions:
            return {
                'f1': 0.0, 'precision': 0.0, 'recall': 0.0,
                'accuracy': 0.0, 'total_decisions': 0
            }

        y_true = np.array([1 if p.ground_truth == 'Fall' else 0 for p in predictions])
        y_pred = np.array([1 if p.decision == 'FALL' else 0 for p in predictions])
        y_prob = np.array([p.avg_probability for p in predictions])

        metrics = {
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'accuracy': accuracy_score(y_true, y_pred),
            'specificity': self._specificity(y_true, y_pred),
            'total_decisions': len(predictions),
        }

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        metrics['confusion_matrix'] = cm.tolist()
        if cm.size == 4:
            metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp'] = cm.ravel()

        # Error breakdown
        metrics['false_negatives'] = [
            i for i, p in enumerate(predictions)
            if p.decision == 'ADL' and p.ground_truth == 'Fall'
        ]
        metrics['false_positives'] = [
            i for i, p in enumerate(predictions)
            if p.decision == 'FALL' and p.ground_truth == 'ADL'
        ]

        return metrics

    def compute_per_subject_metrics(
        self,
        predictions: list[WindowPrediction],
        windows: list[TestWindow]
    ) -> dict[str, dict]:
        """Compute metrics grouped by subject UUID.

        Returns:
            {uuid: {metrics}} dictionary
        """
        # Group by UUID
        uuid_data = {}
        for pred, window in zip(predictions, windows):
            uuid = window.uuid
            if uuid not in uuid_data:
                uuid_data[uuid] = {'preds': [], 'windows': []}
            uuid_data[uuid]['preds'].append(pred)
            uuid_data[uuid]['windows'].append(window)

        # Compute per-UUID metrics
        results = {}
        for uuid, data in uuid_data.items():
            results[uuid] = self.compute_window_metrics(data['preds'], data['windows'])
            results[uuid]['num_windows'] = len(data['windows'])
            results[uuid]['num_falls'] = sum(1 for w in data['windows'] if w.ground_truth == 1)

        return results

    @staticmethod
    def _specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute specificity (true negative rate)."""
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0

    @staticmethod
    def _get_error_indices(y_true: np.ndarray, y_pred: np.ndarray, error_type: str) -> list[int]:
        """Get indices of specific error type."""
        if error_type == 'FN':
            return list(np.where((y_true == 1) & (y_pred == 0))[0])
        elif error_type == 'FP':
            return list(np.where((y_true == 0) & (y_pred == 1))[0])
        return []

    def get_classification_report(
        self,
        predictions: list[WindowPrediction],
        windows: list[TestWindow]
    ) -> str:
        """Get sklearn classification report as string."""
        y_true = [w.ground_truth for w in windows]
        y_pred = [1 if p.probability > self.threshold else 0 for p in predictions]
        return classification_report(y_true, y_pred, target_names=['ADL', 'Fall'])

    def get_roc_curve_data(
        self,
        predictions: list[WindowPrediction],
        windows: list[TestWindow]
    ) -> dict:
        """Get ROC curve data for plotting."""
        from sklearn.metrics import roc_curve
        y_true = np.array([w.ground_truth for w in windows])
        y_prob = np.array([p.probability for p in predictions])
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        return {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'thresholds': thresholds.tolist()}

    def get_pr_curve_data(
        self,
        predictions: list[WindowPrediction],
        windows: list[TestWindow]
    ) -> dict:
        """Get precision-recall curve data for plotting."""
        y_true = np.array([w.ground_truth for w in windows])
        y_prob = np.array([p.probability for p in predictions])
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        return {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': thresholds.tolist()
        }
