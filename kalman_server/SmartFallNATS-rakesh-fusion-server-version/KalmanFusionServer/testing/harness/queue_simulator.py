"""Alpha/Beta queue simulation matching watch behavior."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np


@dataclass
class QueueDecision:
    """Result of alpha queue decision."""
    is_decision_made: bool
    decision: Optional[str] = None       # "FALL" or "ADL"
    avg_probability: Optional[float] = None
    window_probabilities: list[float] = field(default_factory=list)
    queue_was_flushed: bool = False


class AlphaQueueSimulator:
    """Simulate watch alpha queue behavior.

    Watch behavior:
    - Beta queue: 128 frames → single prediction (handled by model)
    - Alpha queue: Collect 10 predictions → average → decision
    - If FALL (avg > threshold): flush entire queue
    - If ADL (avg <= threshold): retain last 5, add 5 new
    """

    def __init__(self, threshold: float = 0.5, queue_size: int = 10, retain_on_adl: int = 5):
        self.threshold = threshold
        self.queue_size = queue_size
        self.retain_on_adl = retain_on_adl
        self.alpha_queue: list[float] = []
        self.total_decisions = 0
        self.fall_decisions = 0
        self.adl_decisions = 0

    def reset(self) -> None:
        """Reset queue state."""
        self.alpha_queue.clear()
        self.total_decisions = 0
        self.fall_decisions = 0
        self.adl_decisions = 0

    def add_prediction(self, probability: float) -> QueueDecision:
        """Add a window prediction and check if decision is made.

        Args:
            probability: Model output probability [0, 1]

        Returns:
            QueueDecision with decision info if queue is full
        """
        self.alpha_queue.append(probability)

        if len(self.alpha_queue) >= self.queue_size:
            return self._make_decision()

        return QueueDecision(is_decision_made=False)

    def _make_decision(self) -> QueueDecision:
        """Make decision when queue is full."""
        avg_prob = float(np.mean(self.alpha_queue))
        probs_copy = self.alpha_queue.copy()

        self.total_decisions += 1

        if avg_prob > self.threshold:
            # FALL detected - flush entire queue
            self.alpha_queue.clear()
            self.fall_decisions += 1
            return QueueDecision(
                is_decision_made=True,
                decision="FALL",
                avg_probability=avg_prob,
                window_probabilities=probs_copy,
                queue_was_flushed=True,
            )
        else:
            # ADL - retain last N predictions for continuity
            self.alpha_queue = self.alpha_queue[self.retain_on_adl:]
            self.adl_decisions += 1
            return QueueDecision(
                is_decision_made=True,
                decision="ADL",
                avg_probability=avg_prob,
                window_probabilities=probs_copy,
                queue_was_flushed=False,
            )

    @property
    def current_queue_size(self) -> int:
        """Get current number of predictions in queue."""
        return len(self.alpha_queue)

    @property
    def needs_more(self) -> int:
        """Get number of predictions needed for next decision."""
        return max(0, self.queue_size - len(self.alpha_queue))

    def get_stats(self) -> dict:
        """Get queue statistics."""
        return {
            'total_decisions': self.total_decisions,
            'fall_decisions': self.fall_decisions,
            'adl_decisions': self.adl_decisions,
            'fall_rate': self.fall_decisions / self.total_decisions if self.total_decisions else 0,
            'current_queue_size': self.current_queue_size,
        }

    def process_session(self, probabilities: list[float]) -> list[QueueDecision]:
        """Process a full session of window predictions.

        Args:
            probabilities: List of window probabilities in order

        Returns:
            List of decisions made during session
        """
        self.reset()
        decisions = []

        for prob in probabilities:
            result = self.add_prediction(prob)
            if result.is_decision_made:
                decisions.append(result)

        return decisions


class WindowLevelEvaluator:
    """Simple window-level evaluation (no queue simulation)."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def evaluate(self, probability: float) -> str:
        """Evaluate single window prediction."""
        return "FALL" if probability > self.threshold else "ADL"

    def evaluate_batch(self, probabilities: list[float]) -> list[str]:
        """Evaluate batch of predictions."""
        return [self.evaluate(p) for p in probabilities]
