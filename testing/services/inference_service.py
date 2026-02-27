"""Inference service - business logic for model inference.

This service layer is framework-agnostic and can be used by:
- Streamlit app
- FastAPI endpoints
- CLI
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from pathlib import Path

from ..data.schema import TestWindow, PredictionResult
from ..data.loader import DataLoader
from ..inference.pipeline import InferencePipeline
from ..inference.queue import AlphaQueueSimulator, QueueDecision


@dataclass
class InferenceRequest:
    """Request for inference."""
    windows: List[TestWindow]
    config_path: str
    threshold: float = 0.5
    device: str = 'cpu'
    use_alpha_queue: bool = False


@dataclass
class InferenceResponse:
    """Response from inference."""
    predictions: List[PredictionResult]
    queue_decisions: List[QueueDecision] = field(default_factory=list)
    config_name: str = ""
    total_windows: int = 0
    avg_preprocessing_ms: float = 0.0
    avg_inference_ms: float = 0.0


@dataclass
class DataLoadRequest:
    """Request for data loading."""
    path: str
    format: Optional[str] = None  # Auto-detect if None


@dataclass
class DataLoadResponse:
    """Response from data loading."""
    windows: List[TestWindow]
    stats: Dict


class InferenceService:
    """Service for running model inference."""

    def __init__(self):
        self._pipelines: Dict[str, InferencePipeline] = {}

    def load_data(self, request: DataLoadRequest) -> DataLoadResponse:
        """Load data from file."""
        loader = DataLoader(request.path)
        windows = loader.load()
        stats = loader.get_stats(windows)

        return DataLoadResponse(windows=windows, stats=stats)

    def run_inference(self, request: InferenceRequest) -> InferenceResponse:
        """Run inference on windows."""
        # Get or create pipeline
        config_key = f"{request.config_path}:{request.device}"
        if config_key not in self._pipelines:
            self._pipelines[config_key] = InferencePipeline(
                config_path=request.config_path,
                device=request.device,
                threshold=request.threshold
            )

        pipeline = self._pipelines[config_key]
        pipeline.threshold = request.threshold

        # Run predictions
        predictions = pipeline.predict_batch(request.windows)

        # Alpha queue simulation
        queue_decisions = []
        if request.use_alpha_queue:
            queue = AlphaQueueSimulator(threshold=request.threshold)
            probs = [p.probability for p in predictions]
            queue_decisions = queue.process_session(probs)

        # Compute averages
        preprocess_times = [p.preprocessing_ms for p in predictions if p.preprocessing_ms > 0]
        inference_times = [p.inference_ms for p in predictions if p.inference_ms > 0]

        return InferenceResponse(
            predictions=predictions,
            queue_decisions=queue_decisions,
            config_name=Path(request.config_path).stem,
            total_windows=len(request.windows),
            avg_preprocessing_ms=sum(preprocess_times) / len(preprocess_times) if preprocess_times else 0,
            avg_inference_ms=sum(inference_times) / len(inference_times) if inference_times else 0,
        )

    def clear_cache(self) -> None:
        """Clear cached pipelines."""
        self._pipelines.clear()
