"""Automated model testing framework for fall detection models."""

from .data.loader import TestDataLoader
from .harness.queue_simulator import AlphaQueueSimulator
from .harness.evaluator import Evaluator

# Lazy import for TestRunner to avoid circular imports
def get_test_runner():
    from .harness.runner import TestRunner
    return TestRunner

__all__ = ['TestDataLoader', 'AlphaQueueSimulator', 'Evaluator', 'get_test_runner']
