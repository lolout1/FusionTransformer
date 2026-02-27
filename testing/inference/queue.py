"""Alpha queue simulation for fall detection decisions."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
from collections import deque


@dataclass
class QueueDecision:
    """Decision made by alpha queue."""
    window_idx: int
    decision: str  # 'FALL' or 'ADL'
    avg_probability: float
    queue_size: int


class AlphaQueueSimulator:
    """Simulate watch-side alpha queue decision logic.

    Queue behavior:
    - Collect predictions until queue has `size` items
    - Compute average probability
    - If avg > threshold: FALL decision, flush queue
    - If avg <= threshold: ADL decision, retain last `retain` items
    """

    def __init__(
        self,
        size: int = 10,
        threshold: float = 0.5,
        retain: int = 5
    ):
        self.size = size
        self.threshold = threshold
        self.retain = retain
        self._queue: deque = deque(maxlen=size)
        self._window_idx = 0
        self._decisions: List[QueueDecision] = []

    def add(self, probability: float) -> Optional[QueueDecision]:
        """Add prediction to queue, return decision if triggered."""
        self._queue.append(probability)
        current_idx = self._window_idx
        self._window_idx += 1

        if len(self._queue) < self.size:
            return None

        avg = sum(self._queue) / len(self._queue)

        if avg > self.threshold:
            decision = QueueDecision(
                window_idx=current_idx,
                decision='FALL',
                avg_probability=avg,
                queue_size=len(self._queue)
            )
            self._queue.clear()
        else:
            decision = QueueDecision(
                window_idx=current_idx,
                decision='ADL',
                avg_probability=avg,
                queue_size=len(self._queue)
            )
            # Retain last N items
            retained = list(self._queue)[-self.retain:]
            self._queue.clear()
            self._queue.extend(retained)

        self._decisions.append(decision)
        return decision

    def process_session(self, probabilities: List[float]) -> List[QueueDecision]:
        """Process all probabilities in a session."""
        self.reset()
        decisions = []
        for prob in probabilities:
            decision = self.add(prob)
            if decision is not None:
                decisions.append(decision)
        return decisions

    def reset(self) -> None:
        """Reset queue state."""
        self._queue.clear()
        self._window_idx = 0
        self._decisions.clear()

    @property
    def decisions(self) -> List[QueueDecision]:
        """Get all decisions made."""
        return self._decisions.copy()

    @property
    def stats(self) -> dict:
        """Get queue statistics."""
        fall_count = sum(1 for d in self._decisions if d.decision == 'FALL')
        adl_count = len(self._decisions) - fall_count
        return {
            'total_decisions': len(self._decisions),
            'fall_decisions': fall_count,
            'adl_decisions': adl_count,
            'current_queue_size': len(self._queue),
        }
