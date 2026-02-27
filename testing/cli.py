"""CLI interface for testing framework."""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from glob import glob
from typing import List

from .config import DEFAULT_DATA_DIR, DEFAULT_CONFIGS_DIR, DEFAULT_THRESHOLD
from .data import DataLoader
from .analysis import MetricsCalculator


def run_inference(args):
    """Run inference on single config."""
    from .inference import InferencePipeline

    # Load data
    print(f"Loading data from {args.data}...")
    loader = DataLoader(args.data)
    windows = loader.load()
    stats = loader.get_stats(windows)
    print(f"Loaded {stats['total_windows']} windows ({stats['fall_windows']} falls, {stats['adl_windows']} ADLs)")

    # Initialize pipeline
    print(f"Initializing pipeline with {args.config}...")
    pipeline = InferencePipeline(args.config, threshold=args.threshold)

    # Run inference
    print("Running inference...")
    predictions = pipeline.predict_batch(windows)

    # Compute metrics
    calc = MetricsCalculator(threshold=args.threshold)
    metrics = calc.compute_all(predictions, windows)

    # Display results
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"F1 Score:    {metrics['f1']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"TP: {metrics['tp']} | FP: {metrics['fp']} | TN: {metrics['tn']} | FN: {metrics['fn']}")

    if metrics.get('roc_auc'):
        print(f"ROC-AUC:     {metrics['roc_auc']:.4f}")

    # Save output if specified
    if args.output:
        output_data = {
            'config': args.config,
            'data': args.data,
            'threshold': args.threshold,
            'metrics': {k: v for k, v in metrics.items() if not isinstance(v, (list, dict))},
            'stats': stats
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


def run_compare(args):
    """Compare multiple configs."""
    from .inference import InferencePipeline

    # Expand config patterns
    config_paths: List[Path] = []
    for pattern in args.configs.split(','):
        pattern = pattern.strip()
        if '*' in pattern:
            config_paths.extend(Path(p) for p in glob(pattern))
        else:
            config_paths.append(Path(pattern))

    if not config_paths:
        print("No configs found matching patterns")
        return

    print(f"Found {len(config_paths)} configs to compare")

    # Load data
    print(f"Loading data from {args.data}...")
    loader = DataLoader(args.data)
    windows = loader.load()
    print(f"Loaded {len(windows)} windows")

    # Run each config
    results = {}
    for config_path in config_paths:
        print(f"\nRunning {config_path.name}...")
        try:
            pipeline = InferencePipeline(str(config_path), threshold=args.threshold)
            predictions = pipeline.predict_batch(windows)

            calc = MetricsCalculator(threshold=args.threshold)
            metrics = calc.compute_all(predictions, windows)
            results[config_path.name] = metrics
        except Exception as e:
            print(f"  ERROR: {e}")
            results[config_path.name] = {'error': str(e)}

    # Display comparison table
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    # Header
    print(f"{'Config':<40} {'F1':>8} {'Prec':>8} {'Recall':>8} {'Acc':>8}")
    print("-" * 80)

    # Sort by F1
    sorted_results = sorted(
        [(k, v) for k, v in results.items() if 'error' not in v],
        key=lambda x: x[1].get('f1', 0),
        reverse=True
    )

    for config, metrics in sorted_results:
        print(f"{config:<40} {metrics['f1']:>8.4f} {metrics['precision']:>8.4f} {metrics['recall']:>8.4f} {metrics['accuracy']:>8.4f}")

    # Best config
    if sorted_results:
        best = sorted_results[0]
        print(f"\nBest config: {best[0]} (F1: {best[1]['f1']:.4f})")

    # Save output if specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


def run_app(args):
    """Launch Streamlit web app."""
    import subprocess

    app_path = Path(__file__).parent / "app" / "main.py"
    cmd = ["streamlit", "run", str(app_path)]

    if args.port:
        cmd.extend(["--server.port", str(args.port)])

    if args.headless:
        cmd.extend(["--server.headless", "true"])

    print(f"Launching Streamlit app on port {args.port or 8501}...")
    print("Press Ctrl+C to stop\n")

    subprocess.run(cmd)


def list_configs(args):
    """List available configs."""
    config_dir = Path(args.configs_dir)

    if not config_dir.exists():
        print(f"Config directory not found: {config_dir}")
        return

    configs = sorted(config_dir.glob("*.yaml"))

    if not configs:
        print("No YAML configs found")
        return

    print(f"Found {len(configs)} configs in {config_dir}:\n")
    for c in configs:
        print(f"  {c.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Fall Detection Testing Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run inference on single config")
    run_parser.add_argument("--config", "-c", required=True, help="Path to config YAML")
    run_parser.add_argument("--data", "-d", required=True, help="Path to data file")
    run_parser.add_argument("--output", "-o", help="Output JSON file")
    run_parser.add_argument("--threshold", "-t", type=float, default=DEFAULT_THRESHOLD, help="Classification threshold")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare multiple configs")
    compare_parser.add_argument("--configs", "-c", required=True, help="Comma-separated config paths or patterns")
    compare_parser.add_argument("--data", "-d", required=True, help="Path to data file")
    compare_parser.add_argument("--output", "-o", help="Output JSON file")
    compare_parser.add_argument("--threshold", "-t", type=float, default=DEFAULT_THRESHOLD, help="Classification threshold")

    # App command
    app_parser = subparsers.add_parser("app", help="Launch web app")
    app_parser.add_argument("--port", "-p", type=int, default=8501, help="Port number")
    app_parser.add_argument("--headless", action="store_true", help="Run in headless mode")

    # List command
    list_parser = subparsers.add_parser("list", help="List available configs")
    list_parser.add_argument("--configs-dir", default=str(DEFAULT_CONFIGS_DIR), help="Configs directory")

    args = parser.parse_args()

    if args.command == "run":
        run_inference(args)
    elif args.command == "compare":
        run_compare(args)
    elif args.command == "app":
        run_app(args)
    elif args.command == "list":
        list_configs(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
