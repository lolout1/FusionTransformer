#!/usr/bin/env python3
"""
Universal Results Aggregation Script

Aggregates results from the modular experiment framework.
Handles multiple experiments, models, and HP configurations.

Usage:
    python -m utils.aggregate_results <results_dir>
    python -m utils.aggregate_results results/modality_tests_v1_20241215_120000
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.summary_generator import SummaryGenerator


def find_summary_stats(results_dir: Path) -> List[Dict[str, Any]]:
    """
    Find all summary_stats.json files in the results directory.

    Args:
        results_dir: Root results directory

    Returns:
        List of dicts with model, hp, and stats
    """
    results = []

    for stats_file in results_dir.rglob('summary_stats.json'):
        try:
            with open(stats_file) as f:
                stats = json.load(f)

            # Extract model and HP names from path
            # Structure: results_dir / model_name / hp_name / summary_stats.json
            hp_name = stats_file.parent.name
            model_name = stats_file.parent.parent.name

            results.append({
                'model': model_name,
                'hp': hp_name,
                'stats': stats,
                'path': str(stats_file.parent)
            })

        except Exception as e:
            print(f'Warning: Failed to load {stats_file}: {e}')

    return results


def find_fold_metrics(results_dir: Path) -> List[Dict[str, Any]]:
    """
    Find all fold_metrics.csv files in the results directory.

    Args:
        results_dir: Root results directory

    Returns:
        List of dicts with model, hp, and DataFrame
    """
    results = []

    for csv_file in results_dir.rglob('fold_metrics.csv'):
        try:
            df = pd.read_csv(csv_file)

            hp_name = csv_file.parent.name
            model_name = csv_file.parent.parent.name

            results.append({
                'model': model_name,
                'hp': hp_name,
                'df': df,
                'path': str(csv_file)
            })

        except Exception as e:
            print(f'Warning: Failed to load {csv_file}: {e}')

    return results


def aggregate_experiment(results_dir: Path) -> Dict[str, Any]:
    """
    Aggregate all results in an experiment directory.

    Args:
        results_dir: Root results directory

    Returns:
        Dict with aggregated results
    """
    print(f'Aggregating results from: {results_dir}')

    # Find all summary stats
    all_stats = find_summary_stats(results_dir)

    if not all_stats:
        print('No summary_stats.json files found!')
        return {}

    print(f'Found {len(all_stats)} model/HP combinations')

    # Group by model
    model_results = {}
    for item in all_stats:
        model_name = item['model']
        if model_name not in model_results:
            model_results[model_name] = []
        model_results[model_name].append(item)

    print(f'Models found: {list(model_results.keys())}')

    # Find best HP for each model
    best_per_model = []
    for model_name, results in model_results.items():
        best = max(
            results,
            key=lambda x: x['stats'].get('test_f1', {}).get('mean', 0)
        )
        best_per_model.append({
            'model': model_name,
            'best_hp': best['hp'],
            'stats': best['stats']
        })

    # Sort by F1
    best_per_model.sort(
        key=lambda x: x['stats'].get('test_f1', {}).get('mean', 0),
        reverse=True
    )

    # Generate comparison table
    comparison_data = []
    for rank, item in enumerate(best_per_model, 1):
        stats = item['stats']
        comparison_data.append({
            'rank': rank,
            'model': item['model'],
            'best_hp': item['best_hp'],
            'test_f1_mean': stats.get('test_f1', {}).get('mean', 0),
            'test_f1_std': stats.get('test_f1', {}).get('std', 0),
            'test_acc_mean': stats.get('test_acc', {}).get('mean', 0),
            'test_acc_std': stats.get('test_acc', {}).get('std', 0),
            'test_precision': stats.get('test_precision', {}).get('mean', 0),
            'test_recall': stats.get('test_recall', {}).get('mean', 0),
            'test_auc': stats.get('test_auc', {}).get('mean', 0),
            'overfit_gap': stats.get('overfit_gap', {}).get('mean', 0),
            'validation_subjects': str(stats.get('validation_subjects', [])),
            'val_falls': stats.get('val_class_balance', {}).get('falls', 'N/A'),
            'val_adls': stats.get('val_class_balance', {}).get('adls', 'N/A'),
        })

    # Save comparison CSV
    comparison_df = pd.DataFrame(comparison_data)
    aggregated_dir = results_dir / 'aggregated'
    aggregated_dir.mkdir(exist_ok=True)

    comparison_path = aggregated_dir / 'model_comparison.csv'
    comparison_df.to_csv(comparison_path, index=False)
    print(f'Saved comparison to: {comparison_path}')

    # Generate summary markdown
    experiment_name = results_dir.name.split('_v1_')[0] if '_v1_' in results_dir.name else results_dir.name

    summary_lines = [
        f'# {experiment_name} - Aggregated Results',
        '',
        f'**Generated from**: {results_dir}',
        f'**Models**: {len(model_results)}',
        f'**Total HP combinations**: {len(all_stats)}',
        '',
        '---',
        '',
        '## Model Ranking (Best HP per Model)',
        '',
        '| Rank | Model | Best HP | Test F1 (%) | Test Acc (%) | Overfit Gap |',
        '|------|-------|---------|-------------|--------------|-------------|',
    ]

    for item in comparison_data:
        summary_lines.append(
            f"| {item['rank']} | {item['model']} | {item['best_hp']} | "
            f"{item['test_f1_mean']*100:.2f} +/- {item['test_f1_std']*100:.2f} | "
            f"{item['test_acc_mean']*100:.2f} +/- {item['test_acc_std']*100:.2f} | "
            f"{item['overfit_gap']:.4f} |"
        )

    # Add validation info
    if comparison_data:
        sample_stats = best_per_model[0]['stats']
        summary_lines.extend([
            '',
            '---',
            '',
            '## Validation Configuration',
            '',
            f"**Validation Subjects**: {sample_stats.get('validation_subjects', [])}",
            f"**Class Balance**: Falls={sample_stats.get('val_class_balance', {}).get('falls', 'N/A')}, "
            f"ADLs={sample_stats.get('val_class_balance', {}).get('adls', 'N/A')}",
            '',
        ])

    # Add HP comparison for top model
    if best_per_model:
        top_model = best_per_model[0]['model']
        top_model_results = model_results[top_model]

        summary_lines.extend([
            '---',
            '',
            f'## HP Comparison for Top Model: {top_model}',
            '',
            '| HP Config | Test F1 (%) | Overfit Gap |',
            '|-----------|-------------|-------------|',
        ])

        for hp_result in sorted(top_model_results,
                                key=lambda x: x['stats'].get('test_f1', {}).get('mean', 0),
                                reverse=True):
            stats = hp_result['stats']
            summary_lines.append(
                f"| {hp_result['hp']} | "
                f"{stats.get('test_f1', {}).get('mean', 0)*100:.2f} +/- "
                f"{stats.get('test_f1', {}).get('std', 0)*100:.2f} | "
                f"{stats.get('overfit_gap', {}).get('mean', 0):.4f} |"
            )

    summary_lines.extend([
        '',
        '---',
        '',
        '## Notes',
        '',
        '- All results from 22-fold LOSO cross-validation',
        '- Overfit Gap = train_f1 - val_f1 at best validation epoch (lower = better)',
        '- Test subjects exclude validation subjects and train-only subjects',
        '',
    ])

    summary_path = aggregated_dir / f'{experiment_name}_summary.md'
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    print(f'Saved summary to: {summary_path}')

    return {
        'experiment': experiment_name,
        'models': list(model_results.keys()),
        'comparison_csv': str(comparison_path),
        'summary_md': str(summary_path),
        'best_model': best_per_model[0] if best_per_model else None
    }


def print_summary(result: Dict[str, Any]):
    """Print aggregation summary to console."""
    if not result:
        return

    print('\n' + '='*60)
    print('AGGREGATION COMPLETE')
    print('='*60)
    print(f"Experiment: {result.get('experiment', 'Unknown')}")
    print(f"Models: {len(result.get('models', []))}")

    if result.get('best_model'):
        best = result['best_model']
        stats = best['stats']
        print(f"\nTop Model: {best['model']}")
        print(f"  Best HP: {best['best_hp']}")
        print(f"  Test F1: {stats.get('test_f1', {}).get('mean', 0)*100:.2f}%")
        print(f"  Test Acc: {stats.get('test_acc', {}).get('mean', 0)*100:.2f}%")

    print(f"\nOutputs:")
    print(f"  Comparison CSV: {result.get('comparison_csv', 'N/A')}")
    print(f"  Summary MD: {result.get('summary_md', 'N/A')}")
    print('='*60)


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate results from modular experiment framework'
    )

    parser.add_argument('results_dir', type=str,
                        help='Results directory to aggregate')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f'Error: Results directory not found: {results_dir}')
        sys.exit(1)

    result = aggregate_experiment(results_dir)
    print_summary(result)

    return result


if __name__ == '__main__':
    main()
