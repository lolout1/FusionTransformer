#!/usr/bin/env python3
"""Command-line interface for model testing.

Usage:
    # List configs
    python -m testing.cli list

    # Run test
    python -m testing.cli run --config configs/s8_16_kalman_gyromag_norm.yaml --data logs/prediction-data-couchbase.json

    # Compare configs
    python -m testing.cli compare --configs "configs/s8_16_*.yaml" --data data.json --output results.html

    # Start web UI
    python -m testing.cli app
"""

import argparse
import json
import sys
from glob import glob
from pathlib import Path


def cmd_list(args):
    """List available configs."""
    configs_dir = Path(args.configs_dir)
    configs = sorted(configs_dir.glob("s*_*.yaml"))

    if not configs:
        print(f"No configs found in {configs_dir}")
        return

    print(f"\nAvailable configs ({len(configs)}):\n")
    for cfg in configs:
        print(f"  {cfg.name}")
    print()


def cmd_run(args):
    """Run test on single config."""
    from .data.loader import TestDataLoader
    from .harness.runner import TestRunner

    print(f"Loading data from {args.data}...")
    loader = TestDataLoader(args.data)
    windows = loader.load_windows()
    sessions = loader.group_into_sessions(windows)

    print(f"  {len(windows)} windows, {len(sessions)} sessions")
    print(f"  Stats: {loader.get_stats()}")

    print(f"\nLoading model from {args.config}...")
    runner = TestRunner(args.config)
    runner.initialize()

    print("\nRunning inference...")
    results = runner.run_all(
        sessions,
        use_alpha_queue=not args.no_queue,
        threshold=args.threshold,
        verbose=args.verbose
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"Results: {results.config_name}")
    print('='*60)

    print("\nWindow-level metrics:")
    for key, val in results.window_metrics.items():
        if key not in ['confusion_matrix', 'false_negatives', 'false_positives']:
            print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")

    if results.session_metrics:
        print("\nSession-level metrics (with alpha queue):")
        for key, val in results.session_metrics.items():
            if key not in ['confusion_matrix', 'false_negatives', 'false_positives']:
                print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")

    print(f"\nLatency: preprocessing={results.avg_preprocessing_ms:.2f}ms, inference={results.avg_inference_ms:.2f}ms")

    # Save if output specified
    if args.output:
        output_data = {
            'config_name': results.config_name,
            'window_metrics': results.window_metrics,
            'session_metrics': results.session_metrics,
            'total_windows': results.total_windows,
            'total_sessions': results.total_sessions,
            'avg_preprocessing_ms': results.avg_preprocessing_ms,
            'avg_inference_ms': results.avg_inference_ms,
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


def cmd_compare(args):
    """Compare multiple configs."""
    from .data.loader import TestDataLoader
    from .harness.runner import TestRunner
    from .analysis.metrics import MetricsCalculator

    # Expand config patterns
    config_files = []
    for pattern in args.configs.split(','):
        config_files.extend(glob(pattern.strip()))

    if not config_files:
        print(f"No configs found matching: {args.configs}")
        return

    print(f"Comparing {len(config_files)} configs...")

    # Load data once
    print(f"Loading data from {args.data}...")
    loader = TestDataLoader(args.data)
    windows = loader.load_windows()
    sessions = loader.group_into_sessions(windows)

    # Run each config
    all_results = []
    for cfg_path in sorted(config_files):
        print(f"\n  Testing: {Path(cfg_path).stem}")
        runner = TestRunner(cfg_path)
        results = runner.run_all(sessions, use_alpha_queue=not args.no_queue, threshold=args.threshold)
        all_results.append(results)
        print(f"    Window F1: {results.window_metrics['f1']:.4f}, Session F1: {results.session_metrics.get('f1', 0):.4f}")

    # Compare
    comparison = MetricsCalculator.compare_configs(all_results)

    print(f"\n{'='*60}")
    print("Comparison Summary")
    print('='*60)
    print(f"\nBest window F1: {comparison['best']['window_f1']}")
    print(f"Best session F1: {comparison['best']['session_f1']}")
    print(f"Best latency: {comparison['best']['latency']}")

    # Save comparison
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"\nComparison saved to {args.output}")


def cmd_app(args):
    """Launch Streamlit app."""
    import subprocess
    app_path = Path(__file__).parent / 'app.py'
    subprocess.run([sys.executable, '-m', 'streamlit', 'run', str(app_path)])


def main():
    parser = argparse.ArgumentParser(
        description="Fall Detection Model Testing CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # list
    list_parser = subparsers.add_parser('list', help='List available configs')
    list_parser.add_argument('--configs-dir', default='configs', help='Configs directory')

    # run
    run_parser = subparsers.add_parser('run', help='Run test on single config')
    run_parser.add_argument('--config', '-c', required=True, help='Config YAML path')
    run_parser.add_argument('--data', '-d', required=True, help='Test data path')
    run_parser.add_argument('--output', '-o', help='Output JSON path')
    run_parser.add_argument('--threshold', '-t', type=float, default=0.5, help='Decision threshold')
    run_parser.add_argument('--no-queue', action='store_true', help='Disable alpha queue simulation')
    run_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    # compare
    compare_parser = subparsers.add_parser('compare', help='Compare multiple configs')
    compare_parser.add_argument('--configs', required=True, help='Comma-separated config patterns')
    compare_parser.add_argument('--data', '-d', required=True, help='Test data path')
    compare_parser.add_argument('--output', '-o', help='Output JSON path')
    compare_parser.add_argument('--threshold', '-t', type=float, default=0.5, help='Decision threshold')
    compare_parser.add_argument('--no-queue', action='store_true', help='Disable alpha queue simulation')

    # app
    app_parser = subparsers.add_parser('app', help='Launch Streamlit web app')

    args = parser.parse_args()

    if args.command == 'list':
        cmd_list(args)
    elif args.command == 'run':
        cmd_run(args)
    elif args.command == 'compare':
        cmd_compare(args)
    elif args.command == 'app':
        cmd_app(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
