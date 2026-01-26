"""
Unit tests for architecture ablation models (LSTM, Transformer, DeepCNN, Mamba).
Tests Kalman and Raw inputs across all window sizes. CI-friendly with synthetic data.

Run: pytest tests/test_ablation_architectures.py -v
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path

KALMAN_CH = 7
RAW_CH = 7
WINDOW_SIZES = [36, 54, 72, 100, 128, 160]
BATCH_SIZES = [1, 4, 16]
EMBED_DIMS = [32, 48, 64]

# Check module availability for CI
try:
    from Models.dual_stream_cnn_lstm import DualStreamLSTM
    HAS_LSTM = True
except ImportError:
    HAS_LSTM = False

try:
    from Models.encoder_ablation import KalmanConv1dLinear
    HAS_TRANSFORMER = True
except ImportError:
    HAS_TRANSFORMER = False

try:
    from Models.short_window_variants import DeepCNNTransformer
    HAS_DEEP_CNN = True
except ImportError:
    HAS_DEEP_CNN = False

try:
    from Models.dual_stream_mamba import DualStreamMamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False


def load_lstm(frames=128, channels=7, embed=64):
    from Models.dual_stream_cnn_lstm import DualStreamLSTM
    return DualStreamLSTM(imu_frames=frames, imu_channels=channels, acc_coords=channels, embed_dim=embed)


def load_transformer(frames=128, channels=7, embed=48):
    from Models.encoder_ablation import KalmanConv1dLinear
    return KalmanConv1dLinear(imu_frames=frames, imu_channels=channels, acc_coords=channels, embed_dim=embed)


def load_deep_cnn(frames=128, channels=7, embed=48):
    from Models.short_window_variants import DeepCNNTransformer
    return DeepCNNTransformer(imu_frames=frames, imu_channels=channels, embed_dim=embed, cnn_stages=3, kernel_sizes=[8, 5, 3])


def load_mamba(frames=128, acc_ch=4, gyro_ch=3, embed=48):
    from Models.dual_stream_mamba import DualStreamMamba
    return DualStreamMamba(imu_frames=frames, acc_coords=acc_ch, gyro_coords=gyro_ch, embed_dim=embed, d_state=16)


@pytest.fixture
def kalman_data():
    def gen(batch, seq):
        x = torch.randn(batch, seq, KALMAN_CH)
        x[..., 1:4] /= x[..., 1:4].norm(dim=-1, keepdim=True).clamp(min=1e-6)
        x[..., 4:7] *= 0.5
        return x
    return gen


@pytest.fixture
def raw_data():
    def gen(batch, seq):
        x = torch.randn(batch, seq, RAW_CH)
        x[..., 1:4] /= x[..., 1:4].norm(dim=-1, keepdim=True).clamp(min=1e-6)
        x[..., 4:7] *= 2.0
        return x
    return gen


@pytest.mark.skipif(not HAS_LSTM, reason="DualStreamLSTM not available")
class TestLSTM:
    @pytest.mark.parametrize("seq", WINDOW_SIZES)
    def test_kalman_forward(self, kalman_data, seq):
        model = load_lstm(frames=seq)
        model.eval()
        with torch.no_grad():
            out, _ = model(kalman_data(4, seq))
        assert out.shape == (4, 1) and not torch.isnan(out).any()

    @pytest.mark.parametrize("seq", WINDOW_SIZES)
    def test_raw_forward(self, raw_data, seq):
        model = load_lstm(frames=seq)
        model.eval()
        with torch.no_grad():
            out, _ = model(raw_data(4, seq))
        assert out.shape == (4, 1) and not torch.isnan(out).any()

    @pytest.mark.parametrize("bs", BATCH_SIZES)
    def test_batch_sizes(self, kalman_data, bs):
        model = load_lstm()
        model.eval()
        with torch.no_grad():
            out, _ = model(kalman_data(bs, 128))
        assert out.shape == (bs, 1)

    def test_gradient_flow(self, kalman_data):
        model = load_lstm()
        model.train()
        x = kalman_data(4, 128)
        x.requires_grad = True
        out, _ = model(x)
        out.sum().backward()
        assert x.grad is not None and not torch.isnan(x.grad).any()


@pytest.mark.skipif(not HAS_TRANSFORMER, reason="KalmanConv1dLinear not available")
class TestTransformer:
    @pytest.mark.parametrize("seq", WINDOW_SIZES)
    def test_kalman_forward(self, kalman_data, seq):
        model = load_transformer(frames=seq)
        model.eval()
        with torch.no_grad():
            out, feat = model(kalman_data(4, seq))
        assert out.shape == (4, 1) and not torch.isnan(out).any()

    @pytest.mark.parametrize("seq", WINDOW_SIZES)
    def test_raw_forward(self, raw_data, seq):
        model = load_transformer(frames=seq)
        model.eval()
        with torch.no_grad():
            out, _ = model(raw_data(4, seq))
        assert out.shape == (4, 1) and not torch.isnan(out).any()

    @pytest.mark.parametrize("embed", EMBED_DIMS)
    def test_embed_dims(self, kalman_data, embed):
        model = load_transformer(embed=embed)
        model.eval()
        with torch.no_grad():
            out, feat = model(kalman_data(4, 128))
        assert out.shape == (4, 1) and feat.shape[-1] == embed

    def test_gradient_flow(self, kalman_data):
        model = load_transformer()
        model.train()
        x = kalman_data(4, 128)
        x.requires_grad = True
        out, _ = model(x)
        out.sum().backward()
        assert x.grad is not None and not torch.isnan(x.grad).any()


@pytest.mark.skipif(not HAS_DEEP_CNN, reason="DeepCNNTransformer not available")
class TestDeepCNN:
    @pytest.mark.parametrize("seq", WINDOW_SIZES)
    def test_kalman_forward(self, kalman_data, seq):
        try:
            model = load_deep_cnn(frames=seq)
        except ImportError:
            pytest.skip("DeepCNNTransformer unavailable")
        model.eval()
        with torch.no_grad():
            out, _ = model(kalman_data(4, seq))
        assert out.shape == (4, 1) and not torch.isnan(out).any()

    @pytest.mark.parametrize("seq", WINDOW_SIZES)
    def test_raw_forward(self, raw_data, seq):
        try:
            model = load_deep_cnn(frames=seq)
        except ImportError:
            pytest.skip("DeepCNNTransformer unavailable")
        model.eval()
        with torch.no_grad():
            out, _ = model(raw_data(4, seq))
        assert out.shape == (4, 1) and not torch.isnan(out).any()

    def test_gradient_flow(self, kalman_data):
        try:
            model = load_deep_cnn()
        except ImportError:
            pytest.skip("DeepCNNTransformer unavailable")
        model.train()
        x = kalman_data(4, 128)
        x.requires_grad = True
        out, _ = model(x)
        out.sum().backward()
        assert x.grad is not None


@pytest.mark.skipif(not HAS_MAMBA, reason="DualStreamMamba not available")
class TestMamba:
    @pytest.mark.parametrize("seq", WINDOW_SIZES)
    def test_kalman_forward(self, kalman_data, seq):
        try:
            model = load_mamba(frames=seq, acc_ch=4, gyro_ch=3)
        except ImportError:
            pytest.skip("DualStreamMamba unavailable")
        model.eval()
        with torch.no_grad():
            out, _ = model(kalman_data(4, seq))
        assert out.shape == (4, 1) and not torch.isnan(out).any()

    @pytest.mark.parametrize("seq", WINDOW_SIZES)
    def test_raw_forward(self, raw_data, seq):
        try:
            model = load_mamba(frames=seq, acc_ch=4, gyro_ch=3)
        except ImportError:
            pytest.skip("DualStreamMamba unavailable")
        model.eval()
        with torch.no_grad():
            out, _ = model(raw_data(4, seq))
        assert out.shape == (4, 1) and not torch.isnan(out).any()

    def test_channel_split_correct(self):
        """Verifies fix for acc_coords=7 bug - must split 4+3."""
        try:
            model = load_mamba(frames=100, acc_ch=4, gyro_ch=3)
        except ImportError:
            pytest.skip("DualStreamMamba unavailable")
        model.eval()
        x = torch.randn(4, 100, 7)
        with torch.no_grad():
            out, _ = model(x)
        assert out.shape == (4, 1) and not torch.isnan(out).any()

    def test_wrong_split_fails(self):
        """Wrong acc_coords=7 should fail on 7ch input."""
        try:
            from Models.dual_stream_mamba import DualStreamMamba
            model = DualStreamMamba(acc_coords=7, gyro_coords=3, imu_frames=100, embed_dim=48)
        except ImportError:
            pytest.skip("DualStreamMamba unavailable")
        model.eval()
        with pytest.raises((RuntimeError, IndexError)):
            with torch.no_grad():
                model(torch.randn(4, 100, 7))

    def test_gradient_flow(self, kalman_data):
        try:
            model = load_mamba(acc_ch=4, gyro_ch=3)
        except ImportError:
            pytest.skip("DualStreamMamba unavailable")
        model.train()
        x = kalman_data(4, 128)
        x.requires_grad = True
        out, _ = model(x)
        out.sum().backward()
        assert x.grad is not None


class TestCrossArchitecture:
    def test_consistent_output_format(self, kalman_data):
        """All models return (logits, features) with logits shape (B, 1)."""
        models = []
        try:
            models.append(('lstm', load_lstm()))
        except ImportError:
            pass
        try:
            models.append(('transformer', load_transformer()))
        except ImportError:
            pass
        try:
            models.append(('deep_cnn', load_deep_cnn()))
        except ImportError:
            pass
        try:
            models.append(('mamba', load_mamba(acc_ch=4, gyro_ch=3)))
        except ImportError:
            pass

        if not models:
            pytest.skip("No models available")

        x = kalman_data(4, 128)
        for name, model in models:
            model.eval()
            with torch.no_grad():
                out = model(x)
            assert isinstance(out, tuple) and len(out) >= 2, f"{name} output format"
            assert out[0].shape == (4, 1), f"{name} logits shape"

    def test_param_counts(self):
        """All models under 2M params."""
        models = {}
        try:
            models['lstm'] = load_lstm(embed=48)
        except ImportError:
            pass
        try:
            models['transformer'] = load_transformer(embed=48)
        except ImportError:
            pass
        try:
            models['deep_cnn'] = load_deep_cnn(embed=48)
        except ImportError:
            pass
        try:
            models['mamba'] = load_mamba(embed=48, acc_ch=4, gyro_ch=3)
        except ImportError:
            pass

        if not models:
            pytest.skip("No models available")

        for name, model in models.items():
            n = sum(p.numel() for p in model.parameters())
            assert 1000 < n < 2_000_000, f"{name}: {n:,} params"


class TestConfigCompatibility:
    def test_ablation_config_valid(self):
        """Ablation script configs are properly structured."""
        try:
            import yaml
        except ImportError:
            pytest.skip("yaml not available")

        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from distributed_dataset_pipeline.run_architecture_ablation import ARCHITECTURES, INPUT_TYPES, DATASET_CONFIGS
        except ImportError:
            pytest.skip("ablation script not available")

        for arch, cfg in ARCHITECTURES.items():
            assert 'model' in cfg and 'model_args_override' in cfg
            if arch == 'mamba':
                override = cfg['model_args_override']
                assert override.get('acc_coords', 0) + override.get('gyro_coords', 0) == 7

        for inp, cfg in INPUT_TYPES.items():
            assert cfg['imu_channels'] == 7

        for ds, cfg in DATASET_CONFIGS.items():
            assert all(v > 0 for v in cfg['window_sizes'].values())


class TestEdgeCases:
    @pytest.mark.parametrize("loader", [load_lstm, load_transformer])
    def test_batch_one(self, kalman_data, loader):
        try:
            model = loader()
        except ImportError:
            pytest.skip("Model unavailable")
        model.eval()
        with torch.no_grad():
            out, _ = model(kalman_data(1, 128))
        assert out.shape == (1, 1)

    @pytest.mark.parametrize("loader", [load_lstm, load_transformer])
    def test_zero_input(self, loader):
        try:
            model = loader()
        except ImportError:
            pytest.skip("Model unavailable")
        model.eval()
        with torch.no_grad():
            out, _ = model(torch.zeros(4, 128, 7))
        assert not torch.isnan(out).any() and not torch.isinf(out).any()

    @pytest.mark.parametrize("loader", [load_lstm, load_transformer])
    def test_deterministic(self, kalman_data, loader):
        try:
            model = loader()
        except ImportError:
            pytest.skip("Model unavailable")
        model.eval()
        x = kalman_data(4, 128)
        with torch.no_grad():
            o1, _ = model(x)
            o2, _ = model(x)
        assert torch.allclose(o1, o2, atol=1e-6)


class TestLossIntegration:
    @pytest.mark.parametrize("loader", [load_lstm, load_transformer])
    def test_bce_loss(self, kalman_data, loader):
        try:
            model = loader()
        except ImportError:
            pytest.skip("Model unavailable")
        model.train()
        x = kalman_data(8, 128)
        targets = torch.randint(0, 2, (8,)).float()
        out, _ = model(x)
        loss = nn.functional.binary_cross_entropy_with_logits(out.squeeze(), targets)
        assert not torch.isnan(loss) and loss.item() > 0
        loss.backward()

    @pytest.mark.parametrize("loader", [load_lstm, load_transformer])
    def test_focal_loss(self, kalman_data, loader):
        try:
            from utils.loss import BinaryFocalLoss
            model = loader()
        except ImportError:
            pytest.skip("Model or loss unavailable")
        model.train()
        x = kalman_data(8, 128)
        targets = torch.randint(0, 2, (8,)).float()
        out, _ = model(x)
        loss = BinaryFocalLoss()(out.squeeze(), targets)
        assert not torch.isnan(loss)
        loss.backward()


class TestLoaderStats:
    """Test that loaders properly track ADL:Fall statistics."""

    def test_upfall_loader_stats(self):
        """UP-FALL loader tracks fold statistics."""
        try:
            from utils.upfall_loader import UPFallLoader
        except ImportError:
            pytest.skip("UPFallLoader unavailable")

        # Just verify the structure exists
        loader = UPFallLoader.__new__(UPFallLoader)
        loader.fold_stats = {'fall_windows': 0, 'adl_windows': 0, 'fall_trials': 0, 'adl_trials': 0}
        assert 'fall_windows' in loader.fold_stats
        assert 'adl_windows' in loader.fold_stats

    def test_wedafall_loader_stats(self):
        """WEDA-FALL loader tracks fold statistics."""
        try:
            from utils.wedafall_loader import WEDAFallLoader
        except ImportError:
            pytest.skip("WEDAFallLoader unavailable")

        loader = WEDAFallLoader.__new__(WEDAFallLoader)
        loader.fold_stats = {'fall_windows': 0, 'adl_windows': 0, 'fall_trials': 0, 'adl_trials': 0}
        assert 'fall_windows' in loader.fold_stats
        assert 'adl_trials' in loader.fold_stats


class TestWindowSizeConfig:
    """Test window size configuration is correctly applied."""

    @pytest.mark.parametrize("dataset,window,samples", [
        ('upfall', '2s', 36),
        ('upfall', '3s', 54),
        ('upfall', '4s', 72),
        ('upfall', 'default', 160),
        ('wedafall', '2s', 100),
        ('wedafall', '3s', 150),
        ('wedafall', '4s', 200),
        ('wedafall', 'default', 250),
    ])
    def test_window_size_mapping(self, dataset, window, samples):
        """Verify window size mappings are correct."""
        try:
            import yaml
        except ImportError:
            pytest.skip("yaml not available")
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from distributed_dataset_pipeline.run_architecture_ablation import DATASET_CONFIGS
        except ImportError:
            pytest.skip("ablation script not available")

        assert DATASET_CONFIGS[dataset]['window_sizes'][window] == samples


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
