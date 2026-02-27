"""Error analysis utilities."""

from __future__ import annotations
from typing import List, Dict, Optional
from collections import Counter
import numpy as np

from ..data.schema import TestWindow, PredictionResult


class ErrorAnalyzer:
    """Analyze prediction errors."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def get_errors(
        self,
        predictions: List[PredictionResult],
        windows: List[TestWindow],
        error_type: Optional[str] = None
    ) -> List[Dict]:
        """Get error samples with details.

        Args:
            predictions: Model predictions
            windows: Ground truth windows
            error_type: Filter by type ('FN', 'FP', or None for all errors)

        Returns:
            List of error dictionaries with sample details
        """
        errors = []

        for idx, (pred, window) in enumerate(zip(predictions, windows)):
            is_predicted_fall = pred.probability > self.threshold
            is_actual_fall = window.ground_truth == 1

            current_type = None
            if is_actual_fall and not is_predicted_fall:
                current_type = 'FN'
            elif not is_actual_fall and is_predicted_fall:
                current_type = 'FP'

            if current_type is None:
                continue

            if error_type is not None and current_type != error_type:
                continue

            errors.append({
                'idx': idx,
                'type': current_type,
                'uuid': window.uuid,
                'timestamp_ms': window.timestamp_ms,
                'probability': pred.probability,
                'confidence': pred.confidence,
                'label': window.label,
                'adl_type': window.adl_type,
                'acc': window.acc,
                'gyro': window.gyro,
            })

        return errors

    def error_summary(
        self,
        predictions: List[PredictionResult],
        windows: List[TestWindow]
    ) -> Dict:
        """Get summary statistics about errors."""
        errors = self.get_errors(predictions, windows)

        fn_errors = [e for e in errors if e['type'] == 'FN']
        fp_errors = [e for e in errors if e['type'] == 'FP']

        # Confidence distribution
        fn_confidences = [e['confidence'] for e in fn_errors]
        fp_confidences = [e['confidence'] for e in fp_errors]

        # By subject
        fn_by_subject = Counter(e['uuid'] for e in fn_errors)
        fp_by_subject = Counter(e['uuid'] for e in fp_errors)

        return {
            'total_errors': len(errors),
            'fn_count': len(fn_errors),
            'fp_count': len(fp_errors),
            'fn_mean_confidence': np.mean(fn_confidences) if fn_confidences else 0,
            'fp_mean_confidence': np.mean(fp_confidences) if fp_confidences else 0,
            'fn_by_subject': dict(fn_by_subject),
            'fp_by_subject': dict(fp_by_subject),
            'subjects_with_fn': len(fn_by_subject),
            'subjects_with_fp': len(fp_by_subject),
        }

    def analyze_signal_characteristics(
        self,
        windows: List[TestWindow],
        labels: Optional[List[str]] = None
    ) -> Dict:
        """Analyze signal characteristics by class.

        Returns statistics about accelerometer and gyroscope signals
        for falls vs ADLs.
        """
        if labels is None:
            labels = [w.label for w in windows]

        fall_acc, fall_gyro = [], []
        adl_acc, adl_gyro = [], []

        for window, label in zip(windows, labels):
            if label.lower() == 'fall':
                fall_acc.append(window.acc)
                fall_gyro.append(window.gyro)
            else:
                adl_acc.append(window.acc)
                adl_gyro.append(window.gyro)

        def compute_stats(arrays):
            if not arrays:
                return {'mean': 0, 'std': 0, 'max': 0, 'min': 0}
            data = np.concatenate(arrays, axis=0)
            return {
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'max': float(np.max(data)),
                'min': float(np.min(data)),
            }

        def compute_smv_stats(acc_arrays):
            if not acc_arrays:
                return {'mean': 0, 'std': 0, 'max': 0, 'min': 0}
            smvs = [np.linalg.norm(a, axis=1) for a in acc_arrays]
            data = np.concatenate(smvs)
            return {
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'max': float(np.max(data)),
                'min': float(np.min(data)),
            }

        return {
            'fall': {
                'n_samples': len(fall_acc),
                'acc': compute_stats(fall_acc),
                'gyro': compute_stats(fall_gyro),
                'smv': compute_smv_stats(fall_acc),
            },
            'adl': {
                'n_samples': len(adl_acc),
                'acc': compute_stats(adl_acc),
                'gyro': compute_stats(adl_gyro),
                'smv': compute_smv_stats(adl_acc),
            }
        }

    def probability_analysis(
        self,
        predictions: List[PredictionResult],
        windows: List[TestWindow]
    ) -> Dict:
        """Analyze probability distributions by class."""
        fall_probs = []
        adl_probs = []

        for pred, window in zip(predictions, windows):
            if window.ground_truth == 1:
                fall_probs.append(pred.probability)
            else:
                adl_probs.append(pred.probability)

        def stats(probs):
            if not probs:
                return {'mean': 0, 'std': 0, 'median': 0, 'q25': 0, 'q75': 0}
            arr = np.array(probs)
            return {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'median': float(np.median(arr)),
                'q25': float(np.percentile(arr, 25)),
                'q75': float(np.percentile(arr, 75)),
            }

        # Separation analysis
        if fall_probs and adl_probs:
            overlap = self._compute_overlap(fall_probs, adl_probs)
        else:
            overlap = None

        return {
            'fall': {'n': len(fall_probs), 'probs': fall_probs, **stats(fall_probs)},
            'adl': {'n': len(adl_probs), 'probs': adl_probs, **stats(adl_probs)},
            'separation': overlap,
        }

    @staticmethod
    def _compute_overlap(a: List[float], b: List[float]) -> float:
        """Compute distribution overlap (0 = no overlap, 1 = complete overlap)."""
        a_arr = np.array(a)
        b_arr = np.array(b)

        # Use histogram-based overlap
        bins = np.linspace(0, 1, 50)
        hist_a, _ = np.histogram(a_arr, bins=bins, density=True)
        hist_b, _ = np.histogram(b_arr, bins=bins, density=True)

        # Overlap as minimum of normalized histograms
        hist_a = hist_a / (hist_a.sum() + 1e-10)
        hist_b = hist_b / (hist_b.sum() + 1e-10)

        overlap = np.minimum(hist_a, hist_b).sum()
        return float(overlap)
