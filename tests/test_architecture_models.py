"""Comprehensive unit tests for all architecture models.

Tests all models used in architecture comparison:
- DualStreamCNN (Kalman/Raw)
- DualStreamLSTM (Kalman/Raw)
- DualStreamMamba (Kalman/Raw)
- Transformer Single Stream (Kalman/Raw)
- Transformer Dual Stream (Kalman/Raw)

Run with: pytest tests/test_architecture_models.py -v
"""

import pytest
import torch
import torch.nn as nn
import numpy as np


# =============================================================================
# Test Configuration
# =============================================================================

BATCH_SIZES = [1, 4, 16]
SEQ_LENGTHS = [64, 128, 256]
EMBED_DIMS = [32, 48, 64, 96]
KALMAN_CHANNELS = 7  # [smv, ax, ay, az, roll, pitch, yaw]
RAW_CHANNELS = 6     # [ax, ay, az, gx, gy, gz]


def get_device():
    """Get available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Check which models are available
try:
    from Models.dual_stream_cnn_lstm import DualStreamCNNKalman, DualStreamCNNRaw
    from Models.dual_stream_cnn_lstm import DualStreamLSTMKalman, DualStreamLSTMRaw
    HAS_CNN_LSTM = True
except ImportError:
    HAS_CNN_LSTM = False

try:
    from Models.dual_stream_base import DualStreamBaseline
    HAS_DUAL_STREAM_BASE = True
except ImportError:
    HAS_DUAL_STREAM_BASE = False

try:
    from Models.single_stream_transformer import KalmanSingleStream
    HAS_SINGLE_STREAM = True
except ImportError:
    HAS_SINGLE_STREAM = False

try:
    from Models.encoder_ablation import KalmanConv1dLinear
    HAS_ENCODER_ABLATION = True
except ImportError:
    HAS_ENCODER_ABLATION = False


# =============================================================================
# CNN Model Tests
# =============================================================================

@pytest.mark.skipif(not HAS_CNN_LSTM, reason="Models.dual_stream_cnn_lstm not available")
class TestDualStreamCNN:
    """Tests for DualStreamCNN models."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from Models.dual_stream_cnn_lstm import DualStreamCNNKalman, DualStreamCNNRaw
        self.KalmanModel = DualStreamCNNKalman
        self.RawModel = DualStreamCNNRaw

    @pytest.mark.parametrize("embed_dim", EMBED_DIMS)
    def test_kalman_output_shape(self, embed_dim):
        """Test CNN Kalman model output shapes."""
        model = self.KalmanModel(imu_frames=128, embed_dim=embed_dim)
        x = torch.randn(4, 128, KALMAN_CHANNELS)
        logits, features = model(x)

        assert logits.shape == (4, 1), f"Expected (4, 1), got {logits.shape}"
        assert features.shape == (4, 128, embed_dim), f"Expected (4, 128, {embed_dim}), got {features.shape}"

    @pytest.mark.parametrize("embed_dim", EMBED_DIMS)
    def test_raw_output_shape(self, embed_dim):
        """Test CNN Raw model output shapes."""
        model = self.RawModel(imu_frames=128, embed_dim=embed_dim)
        x = torch.randn(4, 128, RAW_CHANNELS)
        logits, features = model(x)

        assert logits.shape == (4, 1), f"Expected (4, 1), got {logits.shape}"
        assert features.shape == (4, 128, embed_dim), f"Expected (4, 128, {embed_dim}), got {features.shape}"

    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_batch_size_invariance(self, batch_size):
        """Test CNN handles different batch sizes."""
        model = self.KalmanModel(imu_frames=128, embed_dim=48)
        x = torch.randn(batch_size, 128, KALMAN_CHANNELS)
        logits, features = model(x)

        assert logits.shape[0] == batch_size
        assert features.shape[0] == batch_size

    @pytest.mark.parametrize("seq_len", SEQ_LENGTHS)
    def test_sequence_length(self, seq_len):
        """Test CNN handles different sequence lengths."""
        model = self.KalmanModel(imu_frames=seq_len, embed_dim=48)
        x = torch.randn(4, seq_len, KALMAN_CHANNELS)
        logits, features = model(x)

        assert logits.shape == (4, 1)
        assert features.shape[1] == seq_len

    def test_gradient_flow_kalman(self):
        """Test gradients flow through CNN Kalman model."""
        model = self.KalmanModel(imu_frames=128, embed_dim=48)
        model.train()
        x = torch.randn(4, 128, KALMAN_CHANNELS, requires_grad=True)
        logits, _ = model(x)
        loss = logits.sum()
        loss.backward()

        assert x.grad is not None, "No gradient for input"
        assert not torch.isnan(x.grad).any(), "NaN in input gradient"

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

    def test_gradient_flow_raw(self):
        """Test gradients flow through CNN Raw model."""
        model = self.RawModel(imu_frames=128, embed_dim=48)
        model.train()
        x = torch.randn(4, 128, RAW_CHANNELS, requires_grad=True)
        logits, _ = model(x)
        loss = logits.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_eval_mode(self):
        """Test CNN works in eval mode."""
        model = self.KalmanModel(imu_frames=128, embed_dim=48)
        model.eval()
        with torch.no_grad():
            x = torch.randn(4, 128, KALMAN_CHANNELS)
            logits, features = model(x)

        assert not torch.isnan(logits).any()
        assert not torch.isnan(features).any()

    def test_deterministic_eval(self):
        """Test CNN is deterministic in eval mode."""
        model = self.KalmanModel(imu_frames=128, embed_dim=48)
        model.eval()
        x = torch.randn(4, 128, KALMAN_CHANNELS)

        with torch.no_grad():
            out1, _ = model(x)
            out2, _ = model(x)

        assert torch.allclose(out1, out2), "Model not deterministic in eval mode"


# =============================================================================
# LSTM Model Tests
# =============================================================================

@pytest.mark.skipif(not HAS_CNN_LSTM, reason="Models.dual_stream_cnn_lstm not available")
class TestDualStreamLSTM:
    """Tests for DualStreamLSTM models."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from Models.dual_stream_cnn_lstm import DualStreamLSTMKalman, DualStreamLSTMRaw
        self.KalmanModel = DualStreamLSTMKalman
        self.RawModel = DualStreamLSTMRaw

    @pytest.mark.parametrize("embed_dim", EMBED_DIMS)
    def test_kalman_output_shape(self, embed_dim):
        """Test LSTM Kalman model output shapes."""
        model = self.KalmanModel(imu_frames=128, embed_dim=embed_dim)
        x = torch.randn(4, 128, KALMAN_CHANNELS)
        logits, features = model(x)

        # LSTM uses fused_dim = acc_dim + ori_dim which equals embed_dim
        fused_dim = model.acc_dim + model.ori_dim
        assert logits.shape == (4, 1), f"Expected (4, 1), got {logits.shape}"
        assert features.shape == (4, 128, fused_dim), f"Expected (4, 128, {fused_dim}), got {features.shape}"

    @pytest.mark.parametrize("embed_dim", EMBED_DIMS)
    def test_raw_output_shape(self, embed_dim):
        """Test LSTM Raw model output shapes."""
        model = self.RawModel(imu_frames=128, embed_dim=embed_dim)
        x = torch.randn(4, 128, RAW_CHANNELS)
        logits, features = model(x)

        fused_dim = model.acc_dim + model.ori_dim
        assert logits.shape == (4, 1), f"Expected (4, 1), got {logits.shape}"
        assert features.shape == (4, 128, fused_dim), f"Expected (4, 128, {fused_dim}), got {features.shape}"

    @pytest.mark.parametrize("embed_dim", EMBED_DIMS)
    def test_dimension_calculation(self, embed_dim):
        """Test LSTM dimension calculation is correct (no rounding errors)."""
        model = self.KalmanModel(imu_frames=128, embed_dim=embed_dim)

        # Verify dimensions add up correctly
        total_hidden = embed_dim // 2
        expected_total = total_hidden * 2  # bidirectional doubles
        actual_total = model.acc_dim + model.ori_dim

        assert actual_total == expected_total, \
            f"Dimension mismatch: {actual_total} != {expected_total}"

    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_batch_size_invariance(self, batch_size):
        """Test LSTM handles different batch sizes."""
        model = self.KalmanModel(imu_frames=128, embed_dim=48)
        x = torch.randn(batch_size, 128, KALMAN_CHANNELS)
        logits, features = model(x)

        assert logits.shape[0] == batch_size
        assert features.shape[0] == batch_size

    @pytest.mark.parametrize("seq_len", SEQ_LENGTHS)
    def test_sequence_length(self, seq_len):
        """Test LSTM handles different sequence lengths."""
        model = self.KalmanModel(imu_frames=seq_len, embed_dim=48)
        x = torch.randn(4, seq_len, KALMAN_CHANNELS)
        logits, features = model(x)

        assert logits.shape == (4, 1)
        assert features.shape[1] == seq_len

    def test_gradient_flow_kalman(self):
        """Test gradients flow through LSTM Kalman model."""
        model = self.KalmanModel(imu_frames=128, embed_dim=48)
        model.train()
        x = torch.randn(4, 128, KALMAN_CHANNELS, requires_grad=True)
        logits, _ = model(x)
        loss = logits.sum()
        loss.backward()

        assert x.grad is not None, "No gradient for input"
        assert not torch.isnan(x.grad).any(), "NaN in input gradient"

    def test_gradient_flow_raw(self):
        """Test gradients flow through LSTM Raw model."""
        model = self.RawModel(imu_frames=128, embed_dim=48)
        model.train()
        x = torch.randn(4, 128, RAW_CHANNELS, requires_grad=True)
        logits, _ = model(x)
        loss = logits.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_hidden_state_handling(self):
        """Test LSTM properly handles hidden states."""
        model = self.KalmanModel(imu_frames=128, embed_dim=48)
        model.eval()

        # Process same input twice - should get same output
        x = torch.randn(4, 128, KALMAN_CHANNELS)
        with torch.no_grad():
            out1, _ = model(x)
            out2, _ = model(x)

        assert torch.allclose(out1, out2, atol=1e-6), "LSTM not resetting hidden states"


# =============================================================================
# Mamba Model Tests
# =============================================================================

class TestDualStreamMamba:
    """Tests for DualStreamMamba models."""

    @pytest.fixture(autouse=True)
    def setup(self):
        try:
            from Models.dual_stream_mamba import DualStreamMamba
            self.MambaModel = DualStreamMamba
            self.available = True
        except ImportError:
            self.available = False

    def test_kalman_output_shape(self):
        """Test Mamba Kalman model output shapes."""
        if not self.available:
            pytest.skip("Mamba model not available")

        model = self.MambaModel(
            imu_frames=128,
            embed_dim=48,
            acc_coords=4,  # Kalman: smv + ax + ay + az
            gyro_coords=3  # roll + pitch + yaw
        )
        x = torch.randn(4, 128, KALMAN_CHANNELS)
        logits, *_ = model(x)

        assert logits.shape == (4, 1), f"Expected (4, 1), got {logits.shape}"

    def test_raw_output_shape(self):
        """Test Mamba Raw model output shapes."""
        if not self.available:
            pytest.skip("Mamba model not available")

        model = self.MambaModel(
            imu_frames=128,
            embed_dim=48,
            acc_coords=3,  # Raw: ax + ay + az
            gyro_coords=3  # gx + gy + gz
        )
        x = torch.randn(4, 128, RAW_CHANNELS)
        logits, *_ = model(x)

        assert logits.shape == (4, 1), f"Expected (4, 1), got {logits.shape}"

    @pytest.mark.parametrize("embed_dim", EMBED_DIMS)
    def test_embed_dims(self, embed_dim):
        """Test Mamba with different embedding dimensions."""
        if not self.available:
            pytest.skip("Mamba model not available")

        model = self.MambaModel(
            imu_frames=128,
            embed_dim=embed_dim,
            acc_coords=4,
            gyro_coords=3
        )
        x = torch.randn(4, 128, KALMAN_CHANNELS)
        logits, *_ = model(x)

        assert logits.shape == (4, 1)

    def test_gradient_flow(self):
        """Test gradients flow through Mamba model."""
        if not self.available:
            pytest.skip("Mamba model not available")

        model = self.MambaModel(
            imu_frames=128,
            embed_dim=48,
            acc_coords=4,
            gyro_coords=3
        )
        model.train()
        x = torch.randn(4, 128, KALMAN_CHANNELS, requires_grad=True)
        logits, *_ = model(x)
        loss = logits.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


# =============================================================================
# Transformer Single Stream Tests
# =============================================================================

@pytest.mark.skipif(not HAS_SINGLE_STREAM, reason="Models.single_stream_transformer not available")
class TestTransformerSingleStream:
    """Tests for Transformer Single Stream models."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from Models.single_stream_transformer import KalmanSingleStream
        self.Model = KalmanSingleStream

    @pytest.mark.parametrize("embed_dim", EMBED_DIMS)
    def test_kalman_output_shape(self, embed_dim):
        """Test Single Stream Kalman model output shapes."""
        model = self.Model(
            imu_frames=128,
            imu_channels=KALMAN_CHANNELS,
            embed_dim=embed_dim
        )
        x = torch.randn(4, 128, KALMAN_CHANNELS)
        logits, features = model(x)

        assert logits.shape == (4, 1), f"Expected (4, 1), got {logits.shape}"

    @pytest.mark.parametrize("embed_dim", EMBED_DIMS)
    def test_raw_output_shape(self, embed_dim):
        """Test Single Stream Raw model output shapes."""
        model = self.Model(
            imu_frames=128,
            imu_channels=RAW_CHANNELS,
            embed_dim=embed_dim
        )
        x = torch.randn(4, 128, RAW_CHANNELS)
        logits, features = model(x)

        assert logits.shape == (4, 1), f"Expected (4, 1), got {logits.shape}"

    def test_gradient_flow(self):
        """Test gradients flow through Single Stream model."""
        model = self.Model(imu_frames=128, imu_channels=KALMAN_CHANNELS, embed_dim=48)
        model.train()
        x = torch.randn(4, 128, KALMAN_CHANNELS, requires_grad=True)
        logits, _ = model(x)
        loss = logits.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


# =============================================================================
# Transformer Dual Stream Tests
# =============================================================================

@pytest.mark.skipif(not HAS_ENCODER_ABLATION or not HAS_DUAL_STREAM_BASE,
                    reason="Required models not available")
class TestTransformerDualStream:
    """Tests for Transformer Dual Stream models."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from Models.encoder_ablation import KalmanConv1dLinear
        from Models.dual_stream_base import DualStreamBaseline
        self.KalmanModel = KalmanConv1dLinear
        self.RawModel = DualStreamBaseline

    @pytest.mark.parametrize("embed_dim", EMBED_DIMS)
    def test_kalman_output_shape(self, embed_dim):
        """Test Dual Stream Kalman model output shapes."""
        model = self.KalmanModel(
            imu_frames=128,
            imu_channels=KALMAN_CHANNELS,
            embed_dim=embed_dim
        )
        x = torch.randn(4, 128, KALMAN_CHANNELS)
        logits, features = model(x)

        assert logits.shape == (4, 1), f"Expected (4, 1), got {logits.shape}"

    @pytest.mark.parametrize("embed_dim", EMBED_DIMS)
    def test_raw_output_shape(self, embed_dim):
        """Test Dual Stream Raw model output shapes."""
        model = self.RawModel(
            imu_frames=128,
            imu_channels=RAW_CHANNELS,
            embed_dim=embed_dim
        )
        x = torch.randn(4, 128, RAW_CHANNELS)
        logits, features = model(x)

        assert logits.shape == (4, 1), f"Expected (4, 1), got {logits.shape}"

    def test_gradient_flow_kalman(self):
        """Test gradients flow through Dual Stream Kalman model."""
        model = self.KalmanModel(imu_frames=128, imu_channels=KALMAN_CHANNELS, embed_dim=48)
        model.train()
        x = torch.randn(4, 128, KALMAN_CHANNELS, requires_grad=True)
        logits, _ = model(x)
        loss = logits.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_gradient_flow_raw(self):
        """Test gradients flow through Dual Stream Raw model."""
        model = self.RawModel(imu_frames=128, imu_channels=RAW_CHANNELS, embed_dim=48)
        model.train()
        x = torch.randn(4, 128, RAW_CHANNELS, requires_grad=True)
        logits, _ = model(x)
        loss = logits.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


# =============================================================================
# Cross-Model Consistency Tests
# =============================================================================

@pytest.mark.skipif(not HAS_CNN_LSTM or not HAS_ENCODER_ABLATION or not HAS_SINGLE_STREAM,
                    reason="Required models not available")
class TestModelConsistency:
    """Cross-model consistency and integration tests."""

    def test_all_models_same_interface(self):
        """Test all models have consistent forward interface."""
        from Models.dual_stream_cnn_lstm import DualStreamCNNKalman, DualStreamLSTMKalman
        from Models.encoder_ablation import KalmanConv1dLinear
        from Models.single_stream_transformer import KalmanSingleStream

        models = [
            DualStreamCNNKalman(imu_frames=128, embed_dim=48),
            DualStreamLSTMKalman(imu_frames=128, embed_dim=48),
            KalmanConv1dLinear(imu_frames=128, imu_channels=7, embed_dim=48),
            KalmanSingleStream(imu_frames=128, imu_channels=7, embed_dim=48),
        ]

        x = torch.randn(4, 128, KALMAN_CHANNELS)

        for model in models:
            model.eval()
            with torch.no_grad():
                output = model(x)

            # All models should return (logits, features) or (logits, features, ...)
            assert isinstance(output, tuple), f"{model.__class__.__name__} should return tuple"
            assert len(output) >= 2, f"{model.__class__.__name__} should return at least 2 values"

            logits = output[0]
            assert logits.shape == (4, 1), f"{model.__class__.__name__} logits shape wrong"

    def test_model_parameter_counts(self):
        """Test model parameter counts are reasonable."""
        from Models.dual_stream_cnn_lstm import DualStreamCNNKalman, DualStreamLSTMKalman
        from Models.encoder_ablation import KalmanConv1dLinear
        from Models.single_stream_transformer import KalmanSingleStream

        models = {
            'CNN': DualStreamCNNKalman(imu_frames=128, embed_dim=48),
            'LSTM': DualStreamLSTMKalman(imu_frames=128, embed_dim=48),
            'Transformer-Dual': KalmanConv1dLinear(imu_frames=128, imu_channels=7, embed_dim=48),
            'Transformer-Single': KalmanSingleStream(imu_frames=128, imu_channels=7, embed_dim=48),
        }

        for name, model in models.items():
            n_params = sum(p.numel() for p in model.parameters())
            # All models should be under 1M params for efficiency
            assert n_params < 1_000_000, f"{name} has {n_params:,} params (too many)"
            # All models should have at least some params
            assert n_params > 1000, f"{name} has {n_params:,} params (too few)"
            print(f"{name}: {n_params:,} parameters")

    def test_no_nan_outputs(self):
        """Test no model produces NaN outputs."""
        from Models.dual_stream_cnn_lstm import DualStreamCNNKalman, DualStreamLSTMKalman
        from Models.encoder_ablation import KalmanConv1dLinear
        from Models.single_stream_transformer import KalmanSingleStream

        models = [
            DualStreamCNNKalman(imu_frames=128, embed_dim=48),
            DualStreamLSTMKalman(imu_frames=128, embed_dim=48),
            KalmanConv1dLinear(imu_frames=128, imu_channels=7, embed_dim=48),
            KalmanSingleStream(imu_frames=128, imu_channels=7, embed_dim=48),
        ]

        # Test with various inputs including edge cases
        inputs = [
            torch.randn(4, 128, KALMAN_CHANNELS),            # Normal
            torch.zeros(4, 128, KALMAN_CHANNELS),            # All zeros
            torch.ones(4, 128, KALMAN_CHANNELS),             # All ones
            torch.randn(4, 128, KALMAN_CHANNELS) * 10,       # Large values
            torch.randn(4, 128, KALMAN_CHANNELS) * 0.001,    # Small values
        ]

        for model in models:
            model.eval()
            for x in inputs:
                with torch.no_grad():
                    logits, *rest = model(x)
                assert not torch.isnan(logits).any(), \
                    f"{model.__class__.__name__} produced NaN with input range [{x.min():.2f}, {x.max():.2f}]"


# =============================================================================
# Performance Tests
# =============================================================================

@pytest.mark.skipif(not HAS_CNN_LSTM, reason="Models.dual_stream_cnn_lstm not available")
class TestModelPerformance:
    """Performance and efficiency tests."""

    @pytest.mark.parametrize("model_name,model_class,channels", [
        ("CNN-Kalman", "DualStreamCNNKalman", 7),
        ("LSTM-Kalman", "DualStreamLSTMKalman", 7),
    ])
    def test_inference_speed(self, model_name, model_class, channels):
        """Test model inference is reasonably fast."""
        import time

        if model_class == "DualStreamCNNKalman":
            from Models.dual_stream_cnn_lstm import DualStreamCNNKalman
            model = DualStreamCNNKalman(imu_frames=128, embed_dim=48)
        elif model_class == "DualStreamLSTMKalman":
            from Models.dual_stream_cnn_lstm import DualStreamLSTMKalman
            model = DualStreamLSTMKalman(imu_frames=128, embed_dim=48)

        model.eval()
        x = torch.randn(32, 128, channels)

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(x)

        # Benchmark
        start = time.time()
        with torch.no_grad():
            for _ in range(20):
                _ = model(x)
        elapsed = time.time() - start

        avg_time = elapsed / 20
        # Should process batch of 32 in under 100ms on CPU
        assert avg_time < 0.1, f"{model_name} too slow: {avg_time*1000:.1f}ms per batch"


# =============================================================================
# Integration Test with Training Pipeline
# =============================================================================

@pytest.mark.skipif(not HAS_CNN_LSTM, reason="Models.dual_stream_cnn_lstm not available")
class TestTrainingIntegration:
    """Test models work with training pipeline components."""

    def test_loss_computation(self):
        """Test models work with loss functions."""
        from Models.dual_stream_cnn_lstm import DualStreamLSTMKalman
        from utils.loss import BinaryFocalLoss

        model = DualStreamLSTMKalman(imu_frames=128, embed_dim=48)
        loss_fn = BinaryFocalLoss()

        model.train()
        x = torch.randn(16, 128, KALMAN_CHANNELS)
        targets = torch.randint(0, 2, (16,)).float()

        logits, _ = model(x)
        loss = loss_fn(logits.squeeze(), targets)

        assert not torch.isnan(loss), "Loss is NaN"
        assert loss.item() > 0, "Loss should be positive"

        loss.backward()

        # Check all gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_optimizer_step(self):
        """Test models work with optimizer."""
        from Models.dual_stream_cnn_lstm import DualStreamCNNKalman

        model = DualStreamCNNKalman(imu_frames=128, embed_dim=48)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        model.train()
        x = torch.randn(8, 128, KALMAN_CHANNELS)
        targets = torch.randint(0, 2, (8,)).float()

        # Get initial params
        initial_params = {name: p.clone() for name, p in model.named_parameters()}

        # Forward + backward + step
        logits, _ = model(x)
        loss = nn.functional.binary_cross_entropy_with_logits(logits.squeeze(), targets)
        loss.backward()
        optimizer.step()

        # Check params changed
        params_changed = False
        for name, p in model.named_parameters():
            if not torch.allclose(initial_params[name], p):
                params_changed = True
                break

        assert params_changed, "Optimizer step did not update parameters"


# =============================================================================
# Encoder Ablation Tests (always available)
# =============================================================================

@pytest.mark.skipif(not HAS_ENCODER_ABLATION, reason="Models.encoder_ablation not available")
class TestEncoderAblation:
    """Tests for encoder ablation models that should always be available."""

    def test_kalman_conv1d_linear(self):
        """Test KalmanConv1dLinear model."""
        from Models.encoder_ablation import KalmanConv1dLinear

        model = KalmanConv1dLinear(imu_frames=128, imu_channels=7, embed_dim=48)
        x = torch.randn(4, 128, 7)
        logits, features = model(x)

        assert logits.shape == (4, 1)
        assert not torch.isnan(logits).any()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
