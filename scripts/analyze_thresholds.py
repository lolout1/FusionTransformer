#!/usr/bin/env python3
"""
Standalone threshold analysis for trained fall detection models.

Supports multiple input formats:
1. NPZ file with targets and probabilities (single dataset)
2. JSON file with fold results from Ray distributed training
3. Pickle file with fold results

Usage:
    # Single dataset analysis
    python scripts/analyze_thresholds.py --probabilities results/probs.npz

    # Analyze Ray distributed training results
    python scripts/analyze_thresholds.py --fold-results results/fold_results.json

    # Evaluate specific fixed thresholds
    python scripts/analyze_thresholds.py --fold-results results/fold_results.json --thresholds 0.5,0.7,0.9

    # Full sweep with custom range
    python scripts/analyze_thresholds.py --probabilities results/probs.npz --start 0.3 --end 0.9 --step 0.05
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.threshold_analysis import (
    ThresholdAnalyzer,
    format_threshold_table,
    compute_metrics_at_threshold,
    evaluate_fixed_thresholds,
    compute_global_threshold_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Threshold Analysis for Fall Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    # Input sources (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--probabilities', type=str,
                            help='NPZ file with targets and probabilities arrays')
    input_group.add_argument('--fold-results', type=str,
                            help='JSON/pickle file with fold results from Ray training')

    # Output
    parser.add_argument('--output', '-o', type=str, default='threshold_analysis',
                       help='Output directory')

    # Threshold options
    parser.add_argument('--thresholds', type=str, default=None,
                       help='Comma-separated fixed thresholds to evaluate (e.g., 0.5,0.7,0.9)')
    parser.add_argument('--start', type=float, default=0.3, help='Start threshold for sweep')
    parser.add_argument('--end', type=float, default=0.9, help='End threshold for sweep')
    parser.add_argument('--step', type=float, default=0.05, help='Threshold step for sweep')

    # Output options
    parser.add_argument('--no-plot', action='store_true', help='Skip generating plots')
    parser.add_argument('--json-output', action='store_true',
                       help='Output results as JSON (for scripting)')

    return parser.parse_args()


def load_fold_results(path: str) -> list:
    """Load fold results from JSON or pickle file."""
    path = Path(path)
    if path.suffix == '.json':
        with open(path) as f:
            return json.load(f)
    elif path.suffix in ['.pkl', '.pickle']:
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def plot_threshold_curves(sweep_results, output_path):
    """Generate threshold vs metrics visualization."""
    thresholds = [r['threshold'] for r in sweep_results]
    f1s = [r['f1'] * 100 for r in sweep_results]
    precisions = [r['precision'] * 100 for r in sweep_results]
    recalls = [r['recall'] * 100 for r in sweep_results]
    specificities = [r['specificity'] * 100 for r in sweep_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: F1, Precision, Recall
    ax1.plot(thresholds, f1s, 'b-o', label='F1', linewidth=2, markersize=6)
    ax1.plot(thresholds, precisions, 'g--s', label='Precision', linewidth=1.5, markersize=5)
    ax1.plot(thresholds, recalls, 'r--^', label='Recall', linewidth=1.5, markersize=5)

    best_f1_idx = np.argmax(f1s)
    ax1.axvline(thresholds[best_f1_idx], color='blue', linestyle=':', alpha=0.7,
                label=f'Optimal (τ={thresholds[best_f1_idx]:.2f})')
    ax1.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='Default (τ=0.5)')

    ax1.set_xlabel('Threshold', fontsize=12)
    ax1.set_ylabel('Score (%)', fontsize=12)
    ax1.set_title('Threshold vs Classification Metrics', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(thresholds[0] - 0.02, thresholds[-1] + 0.02)

    # Right: ROC-like (Recall vs Specificity)
    ax2.plot(recalls, specificities, 'b-o', linewidth=2, markersize=6)
    for i, t in enumerate(thresholds):
        if i % 2 == 0:
            ax2.annotate(f'{t:.2f}', (recalls[i], specificities[i]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax2.set_xlabel('Recall (Sensitivity) %', fontsize=12)
    ax2.set_ylabel('Specificity %', fontsize=12)
    ax2.set_title('Recall vs Specificity Trade-off', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def analyze_single_dataset(args):
    """Analyze a single dataset from NPZ file."""
    data = np.load(args.probabilities)
    targets = data['targets']
    probs = data['probabilities']

    print(f"Loaded {len(targets)} samples")
    print(f"Positive class: {int(targets.sum())} ({targets.mean()*100:.1f}%)")
    print(f"Negative class: {int(len(targets) - targets.sum())} ({(1-targets.mean())*100:.1f}%)")

    analyzer = ThresholdAnalyzer(targets, probs)

    # Threshold sweep
    sweep = analyzer.sweep_thresholds(args.start, args.end, args.step)
    print("\n" + format_threshold_table(sweep))

    # Fixed thresholds if specified
    if args.thresholds:
        fixed = [float(t.strip()) for t in args.thresholds.split(',')]
        print(f"\n--- Fixed Threshold Evaluation ---")
        print(f"{'Threshold':<10} | {'F1':>8} | {'Precision':>10} | {'Recall':>8} | {'Accuracy':>10}")
        print("-" * 60)
        for t in fixed:
            m = compute_metrics_at_threshold(targets, probs, t)
            print(f"τ = {t:<6.2f} | {m['f1']*100:>6.2f}% | {m['precision']*100:>8.2f}% | "
                  f"{m['recall']*100:>6.2f}% | {m['accuracy']*100:>8.2f}%")

    # Optimal thresholds by different criteria
    print("\n--- Optimal Thresholds ---")
    for criterion in ['f1', 'f2', 'youden', 'gmean']:
        opt = analyzer.find_optimal_threshold(criterion)
        print(f"{criterion.upper():>8}: τ={opt['threshold']:.3f} "
              f"(F1={opt['f1']*100:.1f}%, P={opt['precision']*100:.1f}%, R={opt['recall']*100:.1f}%)")

    return {
        'sweep': sweep,
        'summary': analyzer.summary(),
        'targets': targets,
        'probabilities': probs,
    }


def analyze_fold_results(args):
    """Analyze fold results from Ray distributed training."""
    fold_results = load_fold_results(args.fold_results)
    print(f"Loaded {len(fold_results)} fold results")

    # Count folds with probability data
    valid_folds = [f for f in fold_results
                   if f.get('threshold_analysis') and 'targets' in f.get('threshold_analysis', {})]
    print(f"Folds with probability data: {len(valid_folds)}")

    if not valid_folds:
        print("ERROR: No folds contain probability data for threshold analysis")
        print("Make sure training was run with threshold analysis enabled")
        sys.exit(1)

    # Fixed threshold evaluation
    if args.thresholds:
        fixed = [float(t.strip()) for t in args.thresholds.split(',')]
    else:
        fixed = [0.5, 0.55, 0.6, 0.7, 0.9]  # Default fixed thresholds

    fixed_results = evaluate_fixed_thresholds(fold_results, fixed)

    print(f"\n{'='*95}")
    print(f"FIXED THRESHOLD COMPARISON ({len(valid_folds)} folds)")
    print(f"{'='*95}")
    print(f"{'Threshold':<10} | {'F1':>18} | {'Precision':>18} | {'Recall':>18} | {'Accuracy':>18}")
    print(f"{'-'*95}")

    for t in fixed:
        if t in fixed_results and 'error' not in fixed_results:
            r = fixed_results[t]
            print(f"τ = {t:<6.2f} | {r['mean_f1']*100:>6.2f}% ± {r['std_f1']*100:>5.2f}% | "
                  f"{r['mean_precision']*100:>6.2f}% ± {r['std_precision']*100:>5.2f}% | "
                  f"{r['mean_recall']*100:>6.2f}% ± {r['std_recall']*100:>5.2f}% | "
                  f"{r['mean_accuracy']*100:>6.2f}% ± {r['std_accuracy']*100:>5.2f}%")

    # Global threshold analysis
    global_results = compute_global_threshold_metrics(fold_results)
    if 'error' not in global_results:
        print(f"\n{'='*95}")
        print(f"GLOBAL THRESHOLD ANALYSIS")
        print(f"{'='*95}")
        print(f"Global Threshold: τ = {global_results['global_threshold']:.3f} "
              f"(mean of per-fold optimal, std={global_results['threshold_std']:.3f})")
        print(f"\n{'Metric':<12} | {'@ Global τ':>24}")
        print(f"{'-'*45}")
        print(f"{'F1':<12} | {global_results['mean_f1']*100:>6.2f}% ± {global_results['std_f1']*100:>5.2f}%")
        print(f"{'Precision':<12} | {global_results['mean_precision']*100:>6.2f}% ± {global_results['std_precision']*100:>5.2f}%")
        print(f"{'Recall':<12} | {global_results['mean_recall']*100:>6.2f}% ± {global_results['std_recall']*100:>5.2f}%")
        print(f"{'Specificity':<12} | {global_results['mean_specificity']*100:>6.2f}% ± {global_results['std_specificity']*100:>5.2f}%")
        print(f"{'Accuracy':<12} | {global_results['mean_accuracy']*100:>6.2f}% ± {global_results['std_accuracy']*100:>5.2f}%")

    # Per-fold optimal (for comparison)
    opt_thresholds = [f['threshold_analysis']['optimal_f1_threshold']['threshold']
                      for f in valid_folds]
    opt_f1s = [f['threshold_analysis']['optimal_f1_threshold']['f1'] * 100
               for f in valid_folds]

    print(f"\n{'='*95}")
    print(f"PER-FOLD OPTIMAL (Upper Bound)")
    print(f"{'='*95}")
    print(f"Mean Optimal Threshold: {np.mean(opt_thresholds):.3f} ± {np.std(opt_thresholds):.3f}")
    print(f"Mean F1 @ Per-Fold Optimal: {np.mean(opt_f1s):.2f}% ± {np.std(opt_f1s):.2f}%")

    print(f"\n{'='*95}")
    print(f"DEPLOYMENT RECOMMENDATION")
    print(f"{'='*95}")
    if 'error' not in global_results:
        print(f"Use threshold τ = {global_results['global_threshold']:.3f} for deployment")
        print(f"Expected F1: {global_results['mean_f1']*100:.2f}% ± {global_results['std_f1']*100:.2f}%")
        gap = np.mean(opt_f1s) - global_results['mean_f1'] * 100
        print(f"Gap from per-fold optimal: {gap:.2f}% (expected due to using fixed threshold)")
    print(f"{'='*95}")

    return {
        'fixed_results': fixed_results,
        'global_results': global_results,
        'fold_results': fold_results,
    }


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.probabilities:
        results = analyze_single_dataset(args)

        # Save sweep results
        sweep_df = pd.DataFrame(results['sweep'])
        sweep_df.to_csv(output_dir / 'threshold_sweep.csv', index=False)
        print(f"\nSaved: {output_dir / 'threshold_sweep.csv'}")

        # Save summary
        summary = results['summary']
        # Remove non-serializable items
        summary_clean = {k: v for k, v in summary.items()
                        if k not in ['sweep_results']}
        with open(output_dir / 'optimal_thresholds.json', 'w') as f:
            json.dump(summary_clean, f, indent=2, default=float)
        print(f"Saved: {output_dir / 'optimal_thresholds.json'}")

        # Plot
        if not args.no_plot:
            plot_threshold_curves(results['sweep'], output_dir / 'threshold_curve.png')

        # Summary
        print("\n--- Summary ---")
        print(f"AUC: {summary['auc']*100:.2f}%")
        print(f"Default (τ=0.5): F1={summary['default_threshold']['f1']*100:.1f}%")
        print(f"Optimal (τ={summary['optimal_f1_threshold']['threshold']:.2f}): "
              f"F1={summary['optimal_f1_threshold']['f1']*100:.1f}%")
        print(f"F1 Improvement: +{summary['f1_improvement']*100:.2f}%")

    elif args.fold_results:
        results = analyze_fold_results(args)

        # Save fixed threshold results
        if 'error' not in results['fixed_results']:
            fixed_df = pd.DataFrame([
                {
                    'threshold': t,
                    'mean_f1': r['mean_f1'],
                    'std_f1': r['std_f1'],
                    'mean_precision': r['mean_precision'],
                    'std_precision': r['std_precision'],
                    'mean_recall': r['mean_recall'],
                    'std_recall': r['std_recall'],
                    'mean_accuracy': r['mean_accuracy'],
                    'std_accuracy': r['std_accuracy'],
                }
                for t, r in results['fixed_results'].items()
            ])
            fixed_df.to_csv(output_dir / 'fixed_threshold_results.csv', index=False)
            print(f"\nSaved: {output_dir / 'fixed_threshold_results.csv'}")

        # Save global threshold results
        if 'error' not in results['global_results']:
            global_clean = {k: v for k, v in results['global_results'].items()
                          if k != 'per_fold_metrics'}
            with open(output_dir / 'global_threshold.json', 'w') as f:
                json.dump(global_clean, f, indent=2)
            print(f"Saved: {output_dir / 'global_threshold.json'}")

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()
