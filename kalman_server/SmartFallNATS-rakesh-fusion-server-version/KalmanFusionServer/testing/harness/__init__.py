"""Test harness components."""

from .queue_simulator import AlphaQueueSimulator
from .evaluator import Evaluator

# Lazy import to avoid circular imports with server preprocessing
def get_test_runner():
    from .runner import TestRunner
    return TestRunner

__all__ = ['AlphaQueueSimulator', 'Evaluator', 'get_test_runner']
