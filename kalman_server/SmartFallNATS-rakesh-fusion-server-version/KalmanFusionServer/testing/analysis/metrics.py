"""Extended metrics computation."""

from __future__ import annotations
from typing import Optional, List, Dict
import numpy as np

from ..data.schema import TestResults, WindowPrediction, SessionPrediction, TestWindow


class MetricsCalculator:
    """Compute and compare metrics across models."""

    @staticmethod
    def compare_configs(results: list[TestResults]) -> dict:
        """Compare metrics across multiple config results.

        Args:
            results: List of TestResults from different configs

        Returns:
            Comparison dictionary
        """
        comparison = {
            'configs': [],
            'window_f1': [],
            'window_precision': [],
            'window_recall': [],
            'session_f1': [],
            'session_precision': [],
            'session_recall': [],
            'avg_latency_ms': [],
        }

        for r in results:
            comparison['configs'].append(r.config_name)
            comparison['window_f1'].append(r.window_metrics.get('f1', 0))
            comparison['window_precision'].append(r.window_metrics.get('precision', 0))
            comparison['window_recall'].append(r.window_metrics.get('recall', 0))
            comparison['session_f1'].append(r.session_metrics.get('f1', 0))
            comparison['session_precision'].append(r.session_metrics.get('precision', 0))
            comparison['session_recall'].append(r.session_metrics.get('recall', 0))
            comparison['avg_latency_ms'].append(r.avg_preprocessing_ms + r.avg_inference_ms)

        # Find best config for each metric
        comparison['best'] = {
            'window_f1': comparison['configs'][np.argmax(comparison['window_f1'])],
            'session_f1': comparison['configs'][np.argmax(comparison['session_f1'])],
            'latency': comparison['configs'][np.argmin(comparison['avg_latency_ms'])],
        }

        return comparison

    @staticmethod
    def get_error_samples(
        predictions: list[WindowPrediction],
        windows: list[TestWindow],
        error_type: str = 'FN',
        threshold: float = 0.5
    ) -> list[dict]:
        """Get detailed info about error samples.

        Args:
            predictions: Model predictions
            windows: Ground truth windows
            error_type: 'FN' or 'FP'
            threshold: Decision threshold

        Returns:
            List of error sample info
        """
        errors = []

        for pred, window in zip(predictions, windows):
            is_predicted_fall = pred.probability > threshold
            is_actual_fall = window.ground_truth == 1

            if error_type == 'FN' and is_actual_fall and not is_predicted_fall:
                errors.append({
                    'uuid': window.uuid,
                    'timestamp_ms': window.timestamp_ms,
                    'probability': pred.probability,
                    'original_prediction': window.original_prediction,
                    'acc': window.acc,
                    'gyro': window.gyro,
                })
            elif error_type == 'FP' and not is_actual_fall and is_predicted_fall:
                errors.append({
                    'uuid': window.uuid,
                    'timestamp_ms': window.timestamp_ms,
                    'probability': pred.probability,
                    'original_prediction': window.original_prediction,
                    'acc': window.acc,
                    'gyro': window.gyro,
                })

        return errors

    @staticmethod
    def compute_threshold_sweep(
        predictions: list[WindowPrediction],
        windows: list[TestWindow],
        thresholds: Optional[list[float]] = None
    ) -> dict:
        """Compute metrics at different thresholds.

        Returns:
            {threshold: {metrics}} dictionary
        """
        from .visualization import Visualizer  # Avoid circular import
        from ..harness.evaluator import Evaluator

        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.05).tolist()

        results = {}
        for thresh in thresholds:
            evaluator = Evaluator(threshold=thresh)
            metrics = evaluator.compute_window_metrics(predictions, windows)
            results[thresh] = {
                'f1': metrics['f1'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'fp': metrics['fp'],
                'fn': metrics['fn'],
            }

        return results

    @staticmethod
    def probability_distribution(
        predictions: list[WindowPrediction],
        windows: list[TestWindow]
    ) -> dict:
        """Get probability distributions for falls and ADLs.

        Returns:
            {'fall_probs': [...], 'adl_probs': [...]}
        """
        fall_probs = []
        adl_probs = []

        for pred, window in zip(predictions, windows):
            if window.ground_truth == 1:
                fall_probs.append(pred.probability)
            else:
                adl_probs.append(pred.probability)

        return {
            'fall_probs': fall_probs,
            'adl_probs': adl_probs,
            'fall_mean': np.mean(fall_probs) if fall_probs else 0,
            'fall_std': np.std(fall_probs) if fall_probs else 0,
            'adl_mean': np.mean(adl_probs) if adl_probs else 0,
            'adl_std': np.std(adl_probs) if adl_probs else 0,
        }
