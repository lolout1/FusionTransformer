"""Inference pipeline wrapping server preprocessing."""

from __future__ import annotations
import time
from pathlib import Path
from typing import List, Optional
import numpy as np
import torch

from ..config import setup_server_imports, SERVER_DIR, DEFAULT_THRESHOLD
from ..data.schema import TestWindow, PredictionResult

# Setup server imports before importing server modules
setup_server_imports()


class InferencePipeline:
    """Inference pipeline using server preprocessing.

    Wraps the server's PreprocessingPipeline and model for offline testing.
    """

    def __init__(
        self,
        config_path: str,
        device: str = 'cpu',
        threshold: float = DEFAULT_THRESHOLD
    ):
        self.config_path = Path(config_path)
        self.device = torch.device(device)
        self.threshold = threshold

        self._config = None
        self._pipeline = None
        self._model = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize pipeline and model from config."""
        if self._initialized:
            return

        # Import server modules (after path setup)
        from config import load_config
        from preprocessing import PreprocessingPipeline
        from models import get_model_class

        # Load config
        self._config = load_config(str(self.config_path))

        # Initialize preprocessing pipeline
        self._pipeline = PreprocessingPipeline(self._config)

        # Initialize model
        model_cls = get_model_class(self._config.model.architecture)
        model_args = {
            'imu_frames': self._config.model.imu_frames,
            'imu_channels': self._config.model.imu_channels,
            'embed_dim': self._config.model.embed_dim,
            'acc_ratio': self._config.model.acc_ratio,
            'num_heads': self._config.model.num_heads,
            'num_layers': self._config.model.num_layers,
        }
        self._model = model_cls(**model_args)

        # Load weights
        weights_path = self.config_path.parent / self._config.model.weights_path
        if not weights_path.exists():
            weights_path = SERVER_DIR / self._config.model.weights_path

        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location=self.device)
            self._model.load_state_dict(state_dict)

        self._model.to(self.device)
        self._model.eval()
        self._initialized = True

    def preprocess(self, window: TestWindow, user_id: str = "test") -> np.ndarray:
        """Preprocess window using server pipeline."""
        if not self._initialized:
            self.initialize()

        # Server pipeline expects timestamps in ms
        timestamps = np.arange(window.acc.shape[0]) * (1000 / 32)  # ~32Hz

        features = self._pipeline.process(
            user_id=user_id,
            acc=window.acc,
            gyro=window.gyro,
            timestamps=timestamps
        )
        return features

    def predict(
        self,
        window: TestWindow,
        idx: int = 0,
        return_features: bool = False
    ) -> PredictionResult:
        """Run inference on single window."""
        if not self._initialized:
            self.initialize()

        # Preprocess
        t0 = time.perf_counter()
        features = self.preprocess(window, user_id=window.uuid)
        preprocess_ms = (time.perf_counter() - t0) * 1000

        # Inference
        t0 = time.perf_counter()
        with torch.no_grad():
            x = torch.from_numpy(features).unsqueeze(0).float().to(self.device)
            logits, _ = self._model(x)
            prob = torch.sigmoid(logits).item()
        inference_ms = (time.perf_counter() - t0) * 1000

        predicted_label = 'Fall' if prob > self.threshold else 'ADL'
        ground_truth = window.label
        is_correct = (
            (predicted_label == 'Fall' and window.ground_truth == 1) or
            (predicted_label == 'ADL' and window.ground_truth == 0)
        )

        return PredictionResult(
            window_idx=idx,
            uuid=window.uuid,
            probability=prob,
            predicted_label=predicted_label,
            ground_truth=ground_truth,
            is_correct=is_correct,
            preprocessing_ms=preprocess_ms,
            inference_ms=inference_ms,
            features=features if return_features else None
        )

    def predict_batch(
        self,
        windows: List[TestWindow],
        return_features: bool = False
    ) -> List[PredictionResult]:
        """Run inference on batch of windows."""
        return [
            self.predict(w, idx=i, return_features=return_features)
            for i, w in enumerate(windows)
        ]

    @property
    def num_channels(self) -> int:
        """Get number of output channels from preprocessing."""
        if self._pipeline is None:
            self.initialize()
        return self._pipeline.num_output_channels

    @property
    def config(self):
        """Get loaded config."""
        if self._config is None:
            self.initialize()
        return self._config
