"""Unit tests for threshold analysis utilities."""

import pytest
import numpy as np
from utils.threshold_analysis import ThresholdAnalyzer


class TestThresholdAnalyzer:
    @pytest.fixture
    def balanced_data(self):
        np.random.seed(42)
        n = 200
        targets = np.concatenate([np.zeros(n//2), np.ones(n//2)])
        probs = np.where(targets == 1,
                        np.random.beta(5, 2, n),
                        np.random.beta(2, 5, n))
        return targets, probs

    @pytest.fixture
    def imbalanced_data(self):
        np.random.seed(42)
        n_neg, n_pos = 180, 20
        targets = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])
        probs = np.where(targets == 1,
                        np.random.beta(4, 2, len(targets)),
                        np.random.beta(2, 4, len(targets)))
        return targets, probs

    def test_init(self, balanced_data):
        targets, probs = balanced_data
        analyzer = ThresholdAnalyzer(targets, probs)
        assert analyzer is not None

    def test_find_optimal_f1(self, balanced_data):
        targets, probs = balanced_data
        analyzer = ThresholdAnalyzer(targets, probs)
        result = analyzer.find_optimal_threshold('f1')
        assert 'threshold' in result
        assert 0 < result['threshold'] < 1

    def test_find_optimal_youden(self, balanced_data):
        targets, probs = balanced_data
        analyzer = ThresholdAnalyzer(targets, probs)
        result = analyzer.find_optimal_threshold('youden')
        assert 'threshold' in result
        assert 0 < result['threshold'] < 1

    def test_metrics_at_threshold(self, balanced_data):
        targets, probs = balanced_data
        analyzer = ThresholdAnalyzer(targets, probs)
        metrics = analyzer._compute_metrics_at_threshold(0.5)

        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'specificity' in metrics

        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1'] <= 1

    def test_optimal_better_than_default(self, balanced_data):
        targets, probs = balanced_data
        analyzer = ThresholdAnalyzer(targets, probs)

        default_metrics = analyzer._compute_metrics_at_threshold(0.5)
        optimal_result = analyzer.find_optimal_threshold('f1')

        assert optimal_result['f1'] >= default_metrics['f1'] - 0.01

    def test_imbalanced_threshold_shift(self, imbalanced_data):
        targets, probs = imbalanced_data
        analyzer = ThresholdAnalyzer(targets, probs)
        result = analyzer.find_optimal_threshold('f1')
        assert result['threshold'] != 0.5

    def test_perfect_predictions(self):
        targets = np.array([0, 0, 0, 1, 1, 1])
        probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        analyzer = ThresholdAnalyzer(targets, probs)

        result = analyzer.find_optimal_threshold('f1')
        assert result['f1'] == 1.0

    def test_threshold_sweep(self, balanced_data):
        targets, probs = balanced_data
        analyzer = ThresholdAnalyzer(targets, probs)

        results = analyzer.sweep_thresholds(start=0.1, end=0.9, step=0.1)

        assert len(results) >= 8
        assert all(0 <= r['f1'] <= 1 for r in results)
