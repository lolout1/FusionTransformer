#!/usr/bin/env python3
"""
Aggregate and analyze Kalman filter ablation results.

Generates:
- Summary table (model × filter)
- Statistical comparisons
- LaTeX table for paper

Usage:
    python scripts/ablation/aggregate_results.py --results-dir results/kalman_ablation/
    python scripts/ablation/aggregate_results.py --wandb-project smartfall-mm --wandb-group kalman-filter-ablation
"""

import os
import sys
import argparse
import yaml
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


def load_local_results(results_dir: str) -> pd.DataFrame:
    """Load results from local YAML/JSON files."""
    results_dir = Path(results_dir)
    rows = []

    for result_file in results_dir.glob('**/scores.yaml'):
        with open(result_file) as f:
            data = yaml.safe_load(f)

        # Parse variant name from path
        variant = result_file.parent.name
        parts = variant.split('_')

        if len(parts) >= 3:
            model = parts[0]
            filter_type = '_'.join(parts[1:3]) if len(parts) >= 3 else parts[1]
        else:
            model = variant
            filter_type = 'unknown'

        rows.append({
            'variant': variant,
            'model': model,
            'filter': filter_type,
            'test_f1': data.get('mean_test_f1', data.get('test_f1', 0)),
            'test_f1_std': data.get('std_test_f1', 0),
            'val_f1': data.get('best_val_f1', data.get('val_f1', 0)),
            'fold_scores': data.get('fold_scores', []),
        })

    return pd.DataFrame(rows)


def load_wandb_results(project: str, group: str = None, entity: str = None) -> pd.DataFrame:
    """Load results from W&B."""
    try:
        import wandb
    except ImportError:
        print('Error: wandb not installed. Run: pip install wandb')
        sys.exit(1)

    api = wandb.Api()

    if entity is None:
        entity = 'abheek-texas-state-university'

    runs = api.runs(f'{entity}/{project}', filters={'group': group} if group else None)

    rows = []
    for run in runs:
        config = run.config
        summary = run.summary

        rows.append({
            'variant': config.get('variant', run.name),
            'model': config.get('model', 'unknown').split('.')[-1],
            'filter': config.get('kalman_filter_type', 'unknown'),
            'test_f1': summary.get('mean_test_f1', 0),
            'val_f1': summary.get('best_val_f1', summary.get('val_f1', 0)),
            'run_id': run.id,
        })

    return pd.DataFrame(rows)


def compute_statistics(df: pd.DataFrame) -> dict:
    """Compute statistical comparisons."""
    from scipy import stats

    results = {}

    # Best config
    best_idx = df['test_f1'].idxmax()
    results['best'] = df.loc[best_idx].to_dict()

    # Model comparison
    model_stats = df.groupby('model')['test_f1'].agg(['mean', 'std', 'count'])
    results['model_comparison'] = model_stats.to_dict()

    # Filter comparison
    filter_stats = df.groupby('filter')['test_f1'].agg(['mean', 'std', 'count'])
    results['filter_comparison'] = filter_stats.to_dict()

    # Pairwise t-tests (if fold scores available)
    if 'fold_scores' in df.columns and df['fold_scores'].apply(len).min() > 1:
        pairwise = {}
        variants = df['variant'].tolist()
        for i, v1 in enumerate(variants):
            for v2 in variants[i+1:]:
                scores1 = df[df['variant'] == v1]['fold_scores'].values[0]
                scores2 = df[df['variant'] == v2]['fold_scores'].values[0]
                if len(scores1) == len(scores2):
                    t_stat, p_val = stats.ttest_rel(scores1, scores2)
                    pairwise[f'{v1}_vs_{v2}'] = {'t': t_stat, 'p': p_val}
        results['pairwise_tests'] = pairwise

    return results


def generate_latex_table(df: pd.DataFrame) -> str:
    """Generate LaTeX table for paper."""
    # Pivot: models as rows, filters as columns
    pivot = df.pivot_table(
        index='model',
        columns='filter',
        values='test_f1',
        aggfunc='first'
    )

    # Format as percentages
    pivot = pivot * 100

    # Find best per row and column
    best_per_row = pivot.idxmax(axis=1)
    best_per_col = pivot.idxmax(axis=0)
    global_best = pivot.max().max()

    latex = []
    latex.append(r'\begin{table}[h]')
    latex.append(r'\centering')
    latex.append(r'\caption{Kalman Filter Ablation Study Results (Test F1 \%)}')
    latex.append(r'\label{tab:kalman_ablation}')

    cols = ['Model'] + list(pivot.columns)
    latex.append(r'\begin{tabular}{l' + 'c' * len(pivot.columns) + '}')
    latex.append(r'\toprule')
    latex.append(' & '.join(cols) + r' \\')
    latex.append(r'\midrule')

    for model in pivot.index:
        row = [model]
        for filter_name in pivot.columns:
            val = pivot.loc[model, filter_name]
            if pd.isna(val):
                row.append('-')
            elif val == global_best:
                row.append(r'\textbf{' + f'{val:.2f}' + r'}')
            elif filter_name == best_per_row[model]:
                row.append(r'\underline{' + f'{val:.2f}' + r'}')
            else:
                row.append(f'{val:.2f}')
        latex.append(' & '.join(row) + r' \\')

    latex.append(r'\bottomrule')
    latex.append(r'\end{tabular}')
    latex.append(r'\end{table}')

    return '\n'.join(latex)


def generate_summary(df: pd.DataFrame, output_dir: str = None):
    """Generate full summary report."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print('\n' + '='*70)
    print('KALMAN FILTER ABLATION STUDY - SUMMARY')
    print('='*70)

    # Sort by test F1
    df_sorted = df.sort_values('test_f1', ascending=False)

    print('\nRanked Results:')
    print('-'*70)
    for _, row in df_sorted.iterrows():
        print(f"{row['variant']:35s}  Test F1: {row['test_f1']*100:5.2f}%  Val F1: {row['val_f1']*100:5.2f}%")

    # Statistics
    stats = compute_statistics(df)

    print('\nBest Configuration:')
    print(f"  {stats['best']['variant']}: {stats['best']['test_f1']*100:.2f}%")

    print('\nModel Comparison (Mean Test F1):')
    for model, mean in stats['model_comparison']['mean'].items():
        std = stats['model_comparison']['std'].get(model, 0)
        print(f"  {model}: {mean*100:.2f}% +/- {std*100:.2f}%")

    print('\nFilter Comparison (Mean Test F1):')
    for filt, mean in stats['filter_comparison']['mean'].items():
        std = stats['filter_comparison']['std'].get(filt, 0)
        print(f"  {filt}: {mean*100:.2f}% +/- {std*100:.2f}%")

    # LaTeX table
    latex = generate_latex_table(df)
    print('\nLaTeX Table:')
    print('-'*70)
    print(latex)

    if output_dir:
        # Save CSV
        df_sorted.to_csv(f'{output_dir}/results.csv', index=False)

        # Save LaTeX
        with open(f'{output_dir}/table.tex', 'w') as f:
            f.write(latex)

        # Save stats
        with open(f'{output_dir}/statistics.yaml', 'w') as f:
            yaml.dump(stats, f, default_flow_style=False)

        print(f'\nResults saved to: {output_dir}/')


def main():
    parser = argparse.ArgumentParser(description='Aggregate ablation results')
    parser.add_argument('--results-dir', type=str, default='results/kalman_ablation/',
                        help='Local results directory')
    parser.add_argument('--wandb-project', type=str, default=None,
                        help='W&B project name')
    parser.add_argument('--wandb-group', type=str, default=None,
                        help='W&B run group')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='W&B entity')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for reports')
    args = parser.parse_args()

    if args.wandb_project:
        df = load_wandb_results(args.wandb_project, args.wandb_group, args.wandb_entity)
    else:
        df = load_local_results(args.results_dir)

    if len(df) == 0:
        print('No results found.')
        sys.exit(1)

    generate_summary(df, args.output_dir)


if __name__ == '__main__':
    main()
