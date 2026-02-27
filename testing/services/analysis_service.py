"""Analysis service - business logic for metrics and error analysis.

This service layer is framework-agnostic and can be used by:
- Streamlit app
- FastAPI endpoints
- CLI
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from ..data.schema import TestWindow, PredictionResult, AnalysisResult
from ..analysis.metrics import MetricsCalculator
from ..analysis.errors import ErrorAnalyzer


@dataclass
class AnalysisRequest:
    """Request for analysis."""
    predictions: List[PredictionResult]
    windows: List[TestWindow]
    threshold: float = 0.5
    compute_curves: bool = True
    compute_per_subject: bool = True


@dataclass
class ComparisonRequest:
    """Request for multi-config comparison."""
    results: Dict[str, AnalysisResult]
    metrics_to_compare: List[str] = None


class AnalysisService:
    """Service for running analysis on predictions."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._metrics_calc = MetricsCalculator(threshold)
        self._error_analyzer = ErrorAnalyzer(threshold)

    def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """Run full analysis on predictions."""
        from datetime import datetime

        calc = MetricsCalculator(request.threshold)
        analyzer = ErrorAnalyzer(request.threshold)

        # Core metrics
        metrics = calc.compute_all(request.predictions, request.windows)
        cm = calc.compute_confusion_matrix(request.predictions, request.windows)

        # Optional: curves
        roc_data = None
        pr_data = None
        threshold_sweep = None

        if request.compute_curves:
            roc_data = calc.compute_roc_curve(request.predictions, request.windows)
            pr_data = calc.compute_pr_curve(request.predictions, request.windows)
            threshold_sweep = calc.compute_threshold_sweep(request.predictions, request.windows)

        # Optional: per-subject
        per_subject = {}
        if request.compute_per_subject:
            per_subject = calc.compute_per_subject(request.predictions, request.windows)

        # Timing
        preprocess_times = [p.preprocessing_ms for p in request.predictions if p.preprocessing_ms > 0]
        inference_times = [p.inference_ms for p in request.predictions if p.inference_ms > 0]

        return AnalysisResult(
            config_name="",
            timestamp=datetime.now(),
            threshold=request.threshold,
            metrics=metrics,
            confusion_matrix=cm,
            predictions=request.predictions,
            per_subject_metrics=per_subject,
            roc_data=roc_data,
            pr_data=pr_data,
            threshold_sweep=threshold_sweep,
            total_windows=len(request.windows),
            avg_preprocessing_ms=np.mean(preprocess_times) if preprocess_times else 0,
            avg_inference_ms=np.mean(inference_times) if inference_times else 0,
        )

    def get_errors(
        self,
        predictions: List[PredictionResult],
        windows: List[TestWindow],
        error_type: Optional[str] = None
    ) -> List[Dict]:
        """Get error samples."""
        return self._error_analyzer.get_errors(predictions, windows, error_type)

    def error_summary(
        self,
        predictions: List[PredictionResult],
        windows: List[TestWindow]
    ) -> Dict:
        """Get error summary statistics."""
        return self._error_analyzer.error_summary(predictions, windows)

    def find_optimal_threshold(
        self,
        predictions: List[PredictionResult],
        windows: List[TestWindow],
        metric: str = 'f1'
    ) -> tuple:
        """Find optimal threshold for given metric."""
        return self._metrics_calc.find_optimal_threshold(predictions, windows, metric)

    def compare_configs(self, request: ComparisonRequest) -> Dict:
        """Compare multiple configurations."""
        metrics = request.metrics_to_compare or ['f1', 'precision', 'recall', 'accuracy', 'specificity']

        comparison = {}
        for config_name, result in request.results.items():
            comparison[config_name] = {
                m: result.metrics.get(m, 0) for m in metrics
            }

        # Find best by F1
        best_config = max(comparison.keys(), key=lambda k: comparison[k].get('f1', 0))

        return {
            'comparison': comparison,
            'best_config': best_config,
            'best_f1': comparison[best_config]['f1']
        }
