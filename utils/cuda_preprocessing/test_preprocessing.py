"""Unit tests for CUDA preprocessing module."""

import pytest
import torch
import numpy as np

from . import CUDAPreprocessor
from .ops import sliding_window_gpu, normalize_gpu, compute_smv_gpu, butterworth_coeffs
from .fallback import sliding_window_cpu, normalize_cpu, compute_smv_cpu, fir_filter_cpu


DEVICES = ['cpu']
if torch.cuda.is_available():
    DEVICES.append('cuda')


class TestSlidingWindow:
    @pytest.mark.parametrize('device', DEVICES)
    def test_basic_shape(self, device):
        data = torch.randn(4, 256, 7)
        prep = CUDAPreprocessor(device=device)
        windows = prep.sliding_window(data, window_size=128, stride=32)
        n_windows = (256 - 128) // 32 + 1
        assert windows.shape == (4 * n_windows, 128, 7)

    @pytest.mark.parametrize('device', DEVICES)
    def test_single_batch(self, device):
        data = torch.randn(1, 128, 3)
        prep = CUDAPreprocessor(device=device)
        windows = prep.sliding_window(data, window_size=64, stride=32)
        n_windows = (128 - 64) // 32 + 1
        assert windows.shape == (n_windows, 64, 3)

    @pytest.mark.parametrize('device', DEVICES)
    def test_2d_input(self, device):
        data = torch.randn(256, 7)
        prep = CUDAPreprocessor(device=device)
        windows = prep.sliding_window(data, window_size=128, stride=32)
        n_windows = (256 - 128) // 32 + 1
        assert windows.shape == (n_windows, 128, 7)

    @pytest.mark.parametrize('device', DEVICES)
    def test_class_aware_stride(self, device):
        data = torch.randn(2, 256, 3)
        labels = torch.tensor([0, 1])
        prep = CUDAPreprocessor(device=device)
        windows = prep.sliding_window(
            data, window_size=128, stride=32,
            class_aware=True, labels=labels,
            fall_stride=16, adl_stride=64
        )
        n_adl = (256 - 128) // 64 + 1
        n_fall = (256 - 128) // 16 + 1
        assert windows.shape[0] == n_adl + n_fall

    @pytest.mark.parametrize('device', DEVICES)
    def test_content_correctness(self, device):
        data = torch.arange(20).float().view(1, 20, 1)
        prep = CUDAPreprocessor(device=device)
        windows = prep.sliding_window(data, window_size=5, stride=5)
        assert windows.shape == (4, 5, 1)
        windows = windows.cpu()
        assert torch.allclose(windows[0, :, 0], torch.arange(0, 5).float())
        assert torch.allclose(windows[1, :, 0], torch.arange(5, 10).float())

    @pytest.mark.parametrize('device', DEVICES)
    def test_short_sequence_error(self, device):
        data = torch.randn(1, 50, 3)
        prep = CUDAPreprocessor(device=device)
        with pytest.raises(ValueError):
            prep.sliding_window(data, window_size=128, stride=32)


class TestNormalize:
    @pytest.mark.parametrize('device', DEVICES)
    def test_zscore_shape(self, device):
        data = torch.randn(4, 128, 7) * 10 + 5
        prep = CUDAPreprocessor(device=device)
        normalized = prep.normalize(data, mode='zscore')
        assert normalized.shape == data.shape

    @pytest.mark.parametrize('device', DEVICES)
    def test_zscore_stats(self, device):
        data = torch.randn(4, 128, 7) * 10 + 5
        prep = CUDAPreprocessor(device=device)
        normalized = prep.normalize(data, mode='zscore')
        mean = normalized.mean(dim=-2)
        std = normalized.std(dim=-2)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
        assert torch.allclose(std, torch.ones_like(std), atol=0.1)

    @pytest.mark.parametrize('device', DEVICES)
    def test_minmax_range(self, device):
        data = torch.randn(4, 128, 7) * 10 + 5
        prep = CUDAPreprocessor(device=device)
        normalized = prep.normalize(data, mode='minmax')
        assert normalized.min() >= -1e-5
        assert normalized.max() <= 1 + 1e-5

    @pytest.mark.parametrize('device', DEVICES)
    def test_none_mode(self, device):
        data = torch.randn(4, 128, 7)
        prep = CUDAPreprocessor(device=device)
        result = prep.normalize(data, mode='none')
        assert torch.equal(result, data)

    @pytest.mark.parametrize('device', DEVICES)
    def test_2d_input(self, device):
        data = torch.randn(128, 7) * 10 + 5
        prep = CUDAPreprocessor(device=device)
        normalized = prep.normalize(data, mode='zscore')
        assert normalized.shape == data.shape


class TestFIRFilter:
    @pytest.mark.parametrize('device', DEVICES)
    def test_lowpass_shape(self, device):
        data = torch.randn(4, 128, 3)
        prep = CUDAPreprocessor(device=device)
        filtered = prep.apply_fir_filter(data, filter_type='lowpass', cutoff=5.0, fs=30.0)
        assert filtered.shape == data.shape

    @pytest.mark.parametrize('device', DEVICES)
    def test_highpass_shape(self, device):
        data = torch.randn(4, 128, 3)
        prep = CUDAPreprocessor(device=device)
        filtered = prep.apply_fir_filter(data, filter_type='highpass', cutoff=0.5, fs=30.0)
        assert filtered.shape == data.shape

    def test_smoothing_effect(self):
        """Test FIR filter smoothing on CPU (deterministic scipy backend)."""
        torch.manual_seed(42)
        signal = torch.sin(torch.linspace(0, 4 * np.pi, 128)).unsqueeze(0).unsqueeze(-1)
        noise = torch.randn_like(signal) * 0.5
        noisy = signal + noise

        prep = CUDAPreprocessor(device='cpu')
        filtered = prep.apply_fir_filter(noisy, filter_type='lowpass', cutoff=2.0, fs=30.0)

        noise_before = (noisy - signal).abs().mean()
        noise_after = (filtered - signal).abs().mean()
        assert noise_after < noise_before

    @pytest.mark.parametrize('device', DEVICES)
    def test_coeffs_cache(self, device):
        prep = CUDAPreprocessor(device=device)
        data = torch.randn(4, 128, 3)
        _ = prep.apply_fir_filter(data, filter_type='lowpass', cutoff=5.0, fs=30.0)
        assert ('lowpass', 5.0, 30.0, 4) in prep._filter_cache
        prep.clear_cache()
        assert len(prep._filter_cache) == 0


class TestSMV:
    @pytest.mark.parametrize('device', DEVICES)
    def test_shape(self, device):
        data = torch.randn(4, 128, 3)
        prep = CUDAPreprocessor(device=device)
        smv = prep.compute_smv(data)
        assert smv.shape == (4, 128, 1)

    @pytest.mark.parametrize('device', DEVICES)
    def test_unit_vector(self, device):
        data = torch.ones(4, 128, 3)
        prep = CUDAPreprocessor(device=device)
        smv = prep.compute_smv(data)
        expected = np.sqrt(3)
        assert torch.allclose(smv, torch.full_like(smv, expected), atol=1e-5)

    @pytest.mark.parametrize('device', DEVICES)
    def test_zero_mean_smv(self, device):
        data = torch.ones(4, 128, 3) * 5
        prep = CUDAPreprocessor(device=device)
        smv = prep.compute_smv_zero_mean(data)
        assert torch.allclose(smv, torch.zeros_like(smv), atol=1e-5)

    @pytest.mark.parametrize('device', DEVICES)
    def test_2d_input(self, device):
        data = torch.randn(128, 3)
        prep = CUDAPreprocessor(device=device)
        smv = prep.compute_smv(data)
        assert smv.shape == (128, 1)


class TestPreprocessBatch:
    @pytest.mark.parametrize('device', DEVICES)
    def test_acc_only(self, device):
        acc = torch.randn(4, 128, 3)
        prep = CUDAPreprocessor(device=device)
        result = prep.preprocess_batch(acc, include_smv=True)
        assert result.shape == (4, 128, 4)

    @pytest.mark.parametrize('device', DEVICES)
    def test_acc_gyro(self, device):
        acc = torch.randn(4, 128, 3)
        gyro = torch.randn(4, 128, 3)
        prep = CUDAPreprocessor(device=device)
        result = prep.preprocess_batch(acc, gyro, include_smv=True)
        assert result.shape == (4, 128, 7)

    @pytest.mark.parametrize('device', DEVICES)
    def test_no_smv(self, device):
        acc = torch.randn(4, 128, 3)
        prep = CUDAPreprocessor(device=device)
        result = prep.preprocess_batch(acc, include_smv=False)
        assert result.shape == (4, 128, 3)

    @pytest.mark.parametrize('device', DEVICES)
    def test_with_filter(self, device):
        acc = torch.randn(4, 128, 3)
        gyro = torch.randn(4, 128, 3)
        prep = CUDAPreprocessor(device=device)
        result = prep.preprocess_batch(
            acc, gyro,
            apply_filter=True,
            acc_filter_cutoff=5.0,
            gyro_filter_cutoff=0.5
        )
        assert result.shape == (4, 128, 7)

    @pytest.mark.parametrize('device', DEVICES)
    def test_normalize_modalities(self, device):
        acc = torch.randn(4, 128, 3) * 10 + 5
        gyro = torch.randn(4, 128, 3) * 2 + 1
        prep = CUDAPreprocessor(device=device)

        result = prep.preprocess_batch(
            acc, gyro,
            normalize_mode='zscore',
            normalize_modalities='acc_only',
            include_smv=False
        )
        acc_part = result[..., :3].cpu()

        assert torch.allclose(acc_part.mean(dim=-2), torch.zeros(4, 3), atol=1e-4)


class TestDeviceHandling:
    def test_cpu_default(self):
        prep = CUDAPreprocessor()
        assert prep.device == torch.device('cpu')

    def test_explicit_cpu(self):
        prep = CUDAPreprocessor(device='cpu')
        assert prep.device == torch.device('cpu')

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_explicit_cuda(self):
        prep = CUDAPreprocessor(device='cuda')
        assert prep.device.type == 'cuda'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_auto_device(self):
        prep = CUDAPreprocessor(device='auto')
        assert prep.device.type == 'cuda'

    def test_auto_device_cpu_only(self):
        if not torch.cuda.is_available():
            prep = CUDAPreprocessor(device='auto')
            assert prep.device == torch.device('cpu')

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_tensor_moved_to_device(self):
        prep = CUDAPreprocessor(device='cuda')
        data = torch.randn(4, 128, 3)
        result = prep.normalize(data)
        assert result.device.type == 'cuda'


class TestEdgeCases:
    @pytest.mark.parametrize('device', DEVICES)
    def test_single_sample(self, device):
        data = torch.randn(1, 128, 3)
        prep = CUDAPreprocessor(device=device)
        normalized = prep.normalize(data)
        assert normalized.shape == data.shape

    @pytest.mark.parametrize('device', DEVICES)
    def test_single_channel(self, device):
        data = torch.randn(4, 128, 1)
        prep = CUDAPreprocessor(device=device)
        normalized = prep.normalize(data)
        assert normalized.shape == data.shape

    @pytest.mark.parametrize('device', DEVICES)
    def test_large_batch(self, device):
        data = torch.randn(64, 256, 7)
        prep = CUDAPreprocessor(device=device)
        windows = prep.sliding_window(data, window_size=128, stride=64)
        n_windows = (256 - 128) // 64 + 1
        assert windows.shape == (64 * n_windows, 128, 7)

    @pytest.mark.parametrize('device', DEVICES)
    def test_stride_equals_window(self, device):
        data = torch.randn(4, 256, 3)
        prep = CUDAPreprocessor(device=device)
        windows = prep.sliding_window(data, window_size=128, stride=128)
        assert windows.shape == (4 * 2, 128, 3)


class TestConsistency:
    def test_cpu_gpu_normalize_consistency(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        data = torch.randn(4, 128, 7)
        cpu_prep = CUDAPreprocessor(device='cpu')
        gpu_prep = CUDAPreprocessor(device='cuda')

        cpu_result = cpu_prep.normalize(data)
        gpu_result = gpu_prep.normalize(data).cpu()

        assert torch.allclose(cpu_result, gpu_result, atol=1e-5)

    def test_cpu_gpu_smv_consistency(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        data = torch.randn(4, 128, 3)
        cpu_prep = CUDAPreprocessor(device='cpu')
        gpu_prep = CUDAPreprocessor(device='cuda')

        cpu_result = cpu_prep.compute_smv(data)
        gpu_result = gpu_prep.compute_smv(data).cpu()

        assert torch.allclose(cpu_result, gpu_result, atol=1e-5)

    def test_cpu_gpu_window_consistency(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        data = torch.randn(4, 128, 3)
        cpu_prep = CUDAPreprocessor(device='cpu')
        gpu_prep = CUDAPreprocessor(device='cuda')

        cpu_result = cpu_prep.sliding_window(data, 64, 16)
        gpu_result = gpu_prep.sliding_window(data, 64, 16).cpu()

        assert torch.allclose(cpu_result, gpu_result, atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
