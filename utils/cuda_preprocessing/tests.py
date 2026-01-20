"""Tests for CUDA preprocessing kernels."""

import torch
import numpy as np
import time


def test_sliding_window():
    from . import CUDAPreprocessor

    print("Testing sliding window...")
    data = torch.randn(4, 256, 7)

    for device in ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']:
        prep = CUDAPreprocessor(device=device)
        windows = prep.sliding_window(data, window_size=128, stride=32)
        expected_n = (256 - 128) // 32 + 1
        assert windows.shape == (4 * expected_n, 128, 7), f"Shape mismatch on {device}"
        print(f"  {device}: {windows.shape} ✓")


def test_normalize():
    from . import CUDAPreprocessor

    print("Testing normalization...")
    data = torch.randn(4, 128, 7) * 10 + 5

    for device in ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']:
        prep = CUDAPreprocessor(device=device)
        normalized = prep.normalize(data, mode='zscore')
        mean = normalized.mean(dim=-2)
        std = normalized.std(dim=-2)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5), f"Mean not zero on {device}"
        assert torch.allclose(std, torch.ones_like(std), atol=1e-1), f"Std not one on {device}"
        print(f"  {device}: mean={mean.mean():.6f}, std={std.mean():.4f} ✓")


def test_fir_filter():
    from . import CUDAPreprocessor

    print("Testing FIR filter...")
    data = torch.randn(4, 128, 3)

    for device in ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']:
        prep = CUDAPreprocessor(device=device)
        filtered = prep.apply_fir_filter(data, filter_type='lowpass', cutoff=5.0, fs=30.0)
        assert filtered.shape == data.shape, f"Shape mismatch on {device}"
        print(f"  {device}: output shape {filtered.shape} ✓")


def test_smv():
    from . import CUDAPreprocessor

    print("Testing SMV...")
    data = torch.ones(4, 128, 3)

    for device in ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']:
        prep = CUDAPreprocessor(device=device)
        smv = prep.compute_smv(data)
        expected = np.sqrt(3)
        assert torch.allclose(smv, torch.full_like(smv, expected), atol=1e-5), f"SMV incorrect on {device}"
        print(f"  {device}: SMV={smv[0, 0, 0]:.4f} (expected {expected:.4f}) ✓")


def test_preprocess_batch():
    from . import CUDAPreprocessor

    print("Testing full pipeline...")
    acc = torch.randn(4, 128, 3)
    gyro = torch.randn(4, 128, 3)

    for device in ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']:
        prep = CUDAPreprocessor(device=device)
        result = prep.preprocess_batch(acc, gyro, include_smv=True)
        assert result.shape == (4, 128, 7), f"Shape mismatch on {device}: {result.shape}"
        print(f"  {device}: output shape {result.shape} ✓")


def benchmark():
    from . import CUDAPreprocessor

    print("\nBenchmarking...")
    data = torch.randn(64, 512, 7)
    n_iters = 100

    for device in ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']:
        prep = CUDAPreprocessor(device=device)
        data_d = data.to(device)

        if device == 'cuda':
            torch.cuda.synchronize()

        start = time.time()
        for _ in range(n_iters):
            _ = prep.normalize(data_d)
            _ = prep.sliding_window(data_d, 128, 32)
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.time() - start

        print(f"  {device}: {elapsed / n_iters * 1000:.2f} ms/iter")


def run_all():
    print("=" * 50)
    print("CUDA Preprocessing Tests")
    print("=" * 50)

    test_sliding_window()
    test_normalize()
    test_fir_filter()
    test_smv()
    test_preprocess_batch()
    benchmark()

    print("\nAll tests passed!")


if __name__ == "__main__":
    run_all()
