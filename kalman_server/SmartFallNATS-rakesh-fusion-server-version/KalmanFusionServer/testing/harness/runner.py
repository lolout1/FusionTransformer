"""Test runner for model evaluation."""

from __future__ import annotations
import sys
import time
from pathlib import Path
from typing import Optional, List, Tuple, TYPE_CHECKING

import numpy as np

# Add parent directory to path for imports
_server_root = Path(__file__).parent.parent.parent
if str(_server_root) not in sys.path:
    sys.path.insert(0, str(_server_root))

# Lazy imports to avoid import errors during package init
if TYPE_CHECKING:
    import torch
    from config import ServerConfig
    from preprocessing.pipeline import PreprocessingPipeline

from ..data.schema import TestWindow, TestSession, WindowPrediction, SessionPrediction, TestResults
from .queue_simulator import AlphaQueueSimulator


class TestRunner:
    """Run model inference on test data with full preprocessing."""

    def __init__(self, config_path: str | Path, device: str = 'cpu'):
        # Lazy import to avoid circular imports
        from config import load_config

        self.config_path = Path(config_path)
        self.config = load_config(str(config_path))
        self.device = device

        self.pipeline = None
        self.model = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize preprocessing pipeline and model."""
        if self._initialized:
            return

        # Lazy imports
        from preprocessing.pipeline import PreprocessingPipeline

        # Initialize preprocessing
        self.pipeline = PreprocessingPipeline(self.config)

        # Load model
        self._load_model()
        self._initialized = True

    def _load_model(self) -> None:
        """Load model weights."""
        import torch
        from models import get_model_class

        weights_path = self.config.model.weights_path
        arch_name = getattr(self.config.model, 'architecture', 'KalmanBalancedFlexible')

        ModelClass = get_model_class(arch_name)

        # Get model args from config
        model_args = getattr(self.config.model, 'model_args', {})
        if hasattr(model_args, '__dict__'):
            model_args = vars(model_args)

        # Build model with config params
        self.model = ModelClass(
            imu_frames=model_args.get('imu_frames', 128),
            imu_channels=model_args.get('imu_channels', 7),
            embed_dim=model_args.get('embed_dim', 48),
            num_heads=model_args.get('num_heads', 4),
            num_layers=model_args.get('num_layers', 2),
            dropout=model_args.get('dropout', 0.5),
            acc_ratio=model_args.get('acc_ratio', 0.65),
        )

        # Load weights
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict_window(
        self,
        window: TestWindow,
        uuid: str = "test",
        return_features: bool = False
    ) -> WindowPrediction:
        """Run inference on a single window.

        Args:
            window: TestWindow with acc and gyro data
            uuid: User ID for Kalman state tracking
            return_features: If True, include preprocessed features

        Returns:
            WindowPrediction with probability and timing
        """
        import torch

        if not self._initialized:
            self.initialize()

        # Preprocess
        t0 = time.perf_counter()
        features = self.pipeline.process(
            uuid=uuid,
            acc=window.acc,
            gyro=window.gyro,
            timestamps=np.arange(128) * (1000.0 / 31.25),  # Default timestamps
        )
        preprocess_ms = (time.perf_counter() - t0) * 1000

        # Inference
        t1 = time.perf_counter()
        with torch.no_grad():
            x = torch.from_numpy(features).unsqueeze(0).to(self.device)
            logits, _ = self.model(x)
            prob = torch.sigmoid(logits).item()
        inference_ms = (time.perf_counter() - t1) * 1000

        return WindowPrediction(
            window_idx=window.sequence_idx,
            probability=prob,
            preprocessing_time_ms=preprocess_ms,
            inference_time_ms=inference_ms,
            features=features if return_features else None,
        )

    def run_session(
        self,
        session: TestSession,
        queue: Optional[AlphaQueueSimulator] = None
    ) -> tuple[list[WindowPrediction], list[SessionPrediction]]:
        """Run inference on all windows in a session.

        Args:
            session: TestSession with ordered windows
            queue: Alpha queue simulator (None for window-level only)

        Returns:
            (window_predictions, session_predictions)
        """
        if not self._initialized:
            self.initialize()

        # Reset Kalman state for new session
        if hasattr(self.pipeline, 'state_manager'):
            self.pipeline.state_manager.clear_user(session.uuid)

        window_preds = []
        session_preds = []

        if queue:
            queue.reset()

        for window in session.windows:
            pred = self.predict_window(window, uuid=session.uuid)
            window_preds.append(pred)

            if queue:
                decision = queue.add_prediction(pred.probability)
                if decision.is_decision_made:
                    session_preds.append(SessionPrediction(
                        decision=decision.decision,
                        avg_probability=decision.avg_probability,
                        window_probabilities=decision.window_probabilities,
                        window_indices=list(range(
                            len(window_preds) - len(decision.window_probabilities),
                            len(window_preds)
                        )),
                        ground_truth=session.ground_truth_label,
                        is_correct=(decision.decision == "FALL") == (session.ground_truth_label == "Fall"),
                    ))

        return window_preds, session_preds

    def run_all(
        self,
        sessions: list[TestSession],
        use_alpha_queue: bool = True,
        threshold: float = 0.5,
        verbose: bool = False
    ) -> TestResults:
        """Run inference on all sessions.

        Args:
            sessions: List of test sessions
            use_alpha_queue: Whether to simulate alpha queue
            threshold: Decision threshold
            verbose: Print progress

        Returns:
            TestResults with all predictions and metrics
        """
        if not self._initialized:
            self.initialize()

        queue = AlphaQueueSimulator(threshold=threshold) if use_alpha_queue else None

        all_window_preds = []
        all_session_preds = []

        for i, session in enumerate(sessions):
            if verbose and i % 10 == 0:
                print(f"Processing session {i+1}/{len(sessions)}...")

            window_preds, session_preds = self.run_session(session, queue)
            all_window_preds.extend(window_preds)
            all_session_preds.extend(session_preds)

        # Compute metrics
        from .evaluator import Evaluator
        evaluator = Evaluator(threshold=threshold)

        window_metrics = evaluator.compute_window_metrics(
            all_window_preds,
            [w for s in sessions for w in s.windows]
        )
        session_metrics = evaluator.compute_session_metrics(all_session_preds)

        return TestResults(
            config_name=self.config_path.stem,
            window_predictions=all_window_preds,
            session_predictions=all_session_preds,
            window_metrics=window_metrics,
            session_metrics=session_metrics,
            total_windows=len(all_window_preds),
            total_sessions=len(sessions),
            avg_preprocessing_ms=np.mean([p.preprocessing_time_ms for p in all_window_preds]),
            avg_inference_ms=np.mean([p.inference_time_ms for p in all_window_preds]),
        )

    @property
    def config_name(self) -> str:
        return self.config_path.stem
