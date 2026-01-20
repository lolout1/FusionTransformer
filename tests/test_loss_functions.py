"""Unit tests for loss functions."""

import pytest
import torch
import numpy as np
from utils.loss import BinaryFocalLoss, ClassBalancedFocalLoss


class TestBinaryFocalLoss:
    def test_output_shape(self):
        loss_fn = BinaryFocalLoss()
        logits = torch.randn(32)
        targets = torch.randint(0, 2, (32,))
        loss = loss_fn(logits, targets)
        assert loss.shape == ()

    def test_positive_loss(self):
        loss_fn = BinaryFocalLoss()
        logits = torch.randn(32)
        targets = torch.randint(0, 2, (32,))
        loss = loss_fn(logits, targets)
        assert loss.item() > 0

    def test_perfect_prediction(self):
        loss_fn = BinaryFocalLoss()
        logits = torch.tensor([10.0, -10.0, 10.0, -10.0])
        targets = torch.tensor([1, 0, 1, 0])
        loss = loss_fn(logits, targets)
        assert loss.item() < 0.01

    def test_worst_prediction(self):
        loss_fn = BinaryFocalLoss()
        logits = torch.tensor([-10.0, 10.0, -10.0, 10.0])
        targets = torch.tensor([1, 0, 1, 0])
        loss = loss_fn(logits, targets)
        assert loss.item() > 1.0

    def test_reduction_none(self):
        loss_fn = BinaryFocalLoss(reduction='none')
        logits = torch.randn(32)
        targets = torch.randint(0, 2, (32,))
        loss = loss_fn(logits, targets)
        assert loss.shape == (32,)

    def test_reduction_sum(self):
        loss_fn = BinaryFocalLoss(reduction='sum')
        logits = torch.randn(32)
        targets = torch.randint(0, 2, (32,))
        loss = loss_fn(logits, targets)
        assert loss.shape == ()

    def test_alpha_effect(self):
        logits = torch.zeros(100)
        targets = torch.ones(100)

        loss_high_alpha = BinaryFocalLoss(alpha=0.9)(logits, targets)
        loss_low_alpha = BinaryFocalLoss(alpha=0.1)(logits, targets)
        assert loss_high_alpha > loss_low_alpha

    def test_gamma_effect(self):
        logits = torch.zeros(100)
        targets = torch.ones(100)

        loss_high_gamma = BinaryFocalLoss(gamma=5)(logits, targets)
        loss_low_gamma = BinaryFocalLoss(gamma=0)(logits, targets)
        assert loss_high_gamma < loss_low_gamma

    def test_gradient_flow(self):
        loss_fn = BinaryFocalLoss()
        logits = torch.randn(32, requires_grad=True)
        targets = torch.randint(0, 2, (32,))
        loss = loss_fn(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()


class TestCallbacks:
    def test_early_stopping_init(self):
        from utils.callbacks import EarlyStopping
        es = EarlyStopping(patience=10, min_delta=0.001)
        assert es.patience == 10
        assert es.min_delta == 0.001
        assert es.early_stop == False

    def test_early_stopping_improvement(self):
        from utils.callbacks import EarlyStopping
        es = EarlyStopping(patience=3)
        es(1.0)
        assert es.best_loss == 1.0
        es(0.9)
        assert es.best_loss == 0.9
        assert es.counter == 0
        assert es.early_stop == False

    def test_early_stopping_no_improvement(self):
        from utils.callbacks import EarlyStopping
        es = EarlyStopping(patience=3)
        es(1.0)
        es(1.0)
        assert es.counter == 1
        es(1.0)
        assert es.counter == 2
        es(1.0)
        assert es.counter == 3
        assert es.early_stop == True

    def test_early_stopping_min_delta(self):
        from utils.callbacks import EarlyStopping
        es = EarlyStopping(patience=3, min_delta=0.1)
        es(1.0)
        es(0.95)
        assert es.counter == 1
        es(0.85)
        assert es.counter == 0
        assert es.best_loss == 0.85

    def test_early_stopping_reset_on_improvement(self):
        from utils.callbacks import EarlyStopping
        es = EarlyStopping(patience=5)
        es(1.0)
        es(1.0)
        es(1.0)
        assert es.counter == 2
        es(0.5)
        assert es.counter == 0
        assert es.best_loss == 0.5
