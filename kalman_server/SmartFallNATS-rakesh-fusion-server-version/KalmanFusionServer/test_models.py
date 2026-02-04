#!/usr/bin/env python3
"""Configurable model testing for deployed variants.

Usage:
    # List available models
    python test_models.py --list

    # Test single model with synthetic data
    python test_models.py --config configs/s8_16_kalman_gyromag_norm.yaml

    # Test all models
    python test_models.py --all

    # Benchmark latency
    python test_models.py --config configs/... --benchmark --iterations 100

    # Verbose output
    python test_models.py --all --verbose
"""

import argparse
import time
from pathlib import Path

import numpy as np

from config import load_config
from server import FallDetectionServer


def list_models(configs_dir: Path, verbose: bool = False, stride_filter: str = None):
    """List all available model configs with metadata."""
    if stride_filter:
        configs = sorted(configs_dir.glob(f"{stride_filter}_*.yaml"))
    else:
        configs = sorted(configs_dir.glob("s*_*.yaml"))

    if not configs:
        print(f"No s8_16_* configs found in {configs_dir}")
        return

    print(f"\nAvailable models ({len(configs)}):\n")
    print(f"{'Model':<35} {'Arch':<25} {'Ch':>3} {'Feature':<15} {'Norm':<8}")
    print("-" * 90)

    for cfg_path in configs:
        try:
            cfg = load_config(str(cfg_path))
            name = cfg_path.stem
            arch = cfg.model.architecture
            channels = cfg.model.imu_channels
            feature = cfg.preprocessing.feature_mode.value
            norm = cfg.preprocessing.normalization_mode.value

            print(f"{name:<35} {arch:<25} {channels:>3} {feature:<15} {norm:<8}")

            if verbose:
                print(f"    weights: {cfg.model.weights_path}")
                if cfg.preprocessing.scaler_path:
                    print(f"    scaler: {cfg.preprocessing.scaler_path}")
                print()
        except Exception as e:
            print(f"{cfg_path.stem:<35} ERROR: {e}")

    print()


def generate_synthetic_data(window_size: int = 128, scenario: str = "standing"):
    """Generate synthetic IMU data for testing.

    Args:
        window_size: Number of samples
        scenario: "standing", "walking", "fall"

    Returns:
        acc (T, 3), gyro (T, 3), timestamps (T,)
    """
    t = np.arange(window_size) / 30.0  # 30 Hz

    if scenario == "standing":
        # Quiet standing: mostly gravity on z-axis, small noise
        acc = np.random.randn(window_size, 3).astype(np.float32) * 0.05
        acc[:, 2] += 9.8  # gravity
        gyro = np.random.randn(window_size, 3).astype(np.float32) * 0.02

    elif scenario == "walking":
        # Walking: periodic accelerations
        acc = np.zeros((window_size, 3), dtype=np.float32)
        acc[:, 0] = np.sin(2 * np.pi * 2 * t) * 2  # forward/back at 2Hz
        acc[:, 2] = 9.8 + np.sin(2 * np.pi * 4 * t) * 1.5  # up/down at 4Hz
        acc += np.random.randn(window_size, 3).astype(np.float32) * 0.1

        gyro = np.zeros((window_size, 3), dtype=np.float32)
        gyro[:, 1] = np.sin(2 * np.pi * 2 * t) * 0.5  # pitch rotation
        gyro += np.random.randn(window_size, 3).astype(np.float32) * 0.05

    elif scenario == "fall":
        # Fall: sudden high acceleration spike then impact
        acc = np.zeros((window_size, 3), dtype=np.float32)
        acc[:, 2] = 9.8  # start upright

        # Free fall phase (frames 40-60): near-zero acceleration
        fall_start, fall_end = 40, 60
        acc[fall_start:fall_end, 2] = 0.5  # near zero-g

        # Impact phase (frames 60-80): high spike
        impact_start, impact_end = 60, 80
        acc[impact_start:impact_end, :] = np.random.randn(20, 3) * 15
        acc[impact_start:impact_end, 2] += 20  # large z impact

        # Post-fall (lying): different orientation
        acc[impact_end:, 0] = 9.8  # lying on side
        acc[impact_end:, 2] = 0.5

        acc += np.random.randn(window_size, 3).astype(np.float32) * 0.3

        gyro = np.zeros((window_size, 3), dtype=np.float32)
        gyro[fall_start:fall_end, 0] = np.linspace(0, 3, fall_end - fall_start)  # rotation during fall
        gyro += np.random.randn(window_size, 3).astype(np.float32) * 0.1

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    timestamps = np.arange(window_size) * (1000.0 / 30.0)  # ms
    return acc, gyro, timestamps


def test_model(config_path: str, iterations: int = 5, verbose: bool = False, benchmark: bool = False):
    """Test a single model with synthetic data."""
    config = load_config(config_path)
    name = Path(config_path).stem

    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    print(f"  Architecture: {config.model.architecture}")
    print(f"  Channels: {config.model.imu_channels}")
    print(f"  Feature mode: {config.preprocessing.feature_mode.value}")
    print(f"  Normalization: {config.preprocessing.normalization_mode.value}")
    print(f"  Weights: {config.model.weights_path}")

    # Check weights exist
    if not Path(config.model.weights_path).exists():
        print(f"\n  ERROR: Weights not found: {config.model.weights_path}")
        return {"name": name, "status": "FAIL", "error": "weights_not_found"}

    # Check scaler exists if needed
    if config.preprocessing.normalization_mode.value != "none":
        if config.preprocessing.scaler_path and not Path(config.preprocessing.scaler_path).exists():
            print(f"\n  WARNING: Scaler not found: {config.preprocessing.scaler_path}")

    try:
        server = FallDetectionServer(config)
        server.initialize()
    except Exception as e:
        print(f"\n  ERROR during initialization: {e}")
        return {"name": name, "status": "FAIL", "error": str(e)}

    # Test with different scenarios
    scenarios = ["standing", "walking", "fall"]
    results = {}
    latencies = []

    print(f"\n  Testing scenarios ({iterations} iterations each):")

    for scenario in scenarios:
        probs = []
        scenario_latencies = []

        for i in range(iterations):
            acc, gyro, timestamps = generate_synthetic_data(scenario=scenario)

            t0 = time.perf_counter()
            prob = server.process_request(f"test_{scenario}_{i}", acc, gyro, timestamps)
            latency = (time.perf_counter() - t0) * 1000

            probs.append(prob)
            scenario_latencies.append(latency)
            latencies.append(latency)

            if verbose:
                print(f"    {scenario}[{i}]: prob={prob:.4f}, latency={latency:.2f}ms")

        mean_prob = np.mean(probs)
        mean_lat = np.mean(scenario_latencies)
        results[scenario] = {"mean_prob": mean_prob, "mean_latency": mean_lat}

        print(f"    {scenario:>10}: prob={mean_prob:.4f} (latency={mean_lat:.1f}ms)")

    # Overall latency stats
    print(f"\n  Latency: mean={np.mean(latencies):.1f}ms, std={np.std(latencies):.1f}ms, max={np.max(latencies):.1f}ms")

    # Sanity check: fall scenario should generally have higher prob
    fall_prob = results["fall"]["mean_prob"]
    standing_prob = results["standing"]["mean_prob"]

    if fall_prob > standing_prob:
        print(f"  Sanity check: PASS (fall prob > standing prob)")
    else:
        print(f"  Sanity check: WARNING (fall prob <= standing prob - synthetic data may not trigger)")

    if benchmark:
        print(f"\n  Benchmark ({iterations * 10} iterations):")
        bench_latencies = []
        acc, gyro, timestamps = generate_synthetic_data()
        for _ in range(iterations * 10):
            t0 = time.perf_counter()
            server.process_request("bench", acc, gyro, timestamps)
            bench_latencies.append((time.perf_counter() - t0) * 1000)
        print(f"    mean={np.mean(bench_latencies):.2f}ms, p50={np.percentile(bench_latencies, 50):.2f}ms, p99={np.percentile(bench_latencies, 99):.2f}ms")

    return {
        "name": name,
        "status": "PASS",
        "results": results,
        "mean_latency": np.mean(latencies),
    }


def test_all(configs_dir: Path, iterations: int = 5, verbose: bool = False, benchmark: bool = False, stride_filter: str = None):
    """Test all models in configs directory."""
    if stride_filter:
        configs = sorted(configs_dir.glob(f"{stride_filter}_*.yaml"))
    else:
        configs = sorted(configs_dir.glob("s*_*.yaml"))

    if not configs:
        print(f"No s8_16_* configs found in {configs_dir}")
        return

    all_results = []

    for cfg_path in configs:
        try:
            result = test_model(str(cfg_path), iterations, verbose, benchmark)
            all_results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results.append({"name": cfg_path.stem, "status": "FAIL", "error": str(e)})

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print('='*60)

    passed = [r for r in all_results if r["status"] == "PASS"]
    failed = [r for r in all_results if r["status"] == "FAIL"]

    print(f"\nPassed: {len(passed)}/{len(all_results)}")
    for r in passed:
        lat = r.get("mean_latency", 0)
        print(f"  {r['name']}: latency={lat:.1f}ms")

    if failed:
        print(f"\nFailed: {len(failed)}")
        for r in failed:
            print(f"  {r['name']}: {r.get('error', 'unknown')}")

    print()
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Model Testing CLI")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--config", type=str, help="Config YAML path to test")
    parser.add_argument("--all", action="store_true", help="Test all models")
    parser.add_argument("--stride", type=str, help="Filter by stride (e.g., s8_16, s16_32)")
    parser.add_argument("--benchmark", action="store_true", help="Run extended latency benchmark")
    parser.add_argument("--iterations", type=int, default=5, help="Iterations per scenario (default: 5)")
    parser.add_argument("--configs-dir", type=str, default="configs", help="Configs directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    configs_dir = Path(args.configs_dir)

    if args.list:
        list_models(configs_dir, verbose=args.verbose, stride_filter=args.stride)
    elif args.all:
        test_all(configs_dir, args.iterations, args.verbose, args.benchmark, stride_filter=args.stride)
    elif args.config:
        test_model(args.config, args.iterations, args.verbose, args.benchmark)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
