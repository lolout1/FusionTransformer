"""
Enhanced Metrics Reporting Module
Provides comprehensive per-fold analysis and summary statistics for model evaluation.

Author: Claude Code
Date: 2025-11-15
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json


def format_per_fold_table(fold_metrics: List[Dict]) -> pd.DataFrame:
    """
    Convert fold_metrics list to detailed DataFrame with all subjects.

    Args:
        fold_metrics: List of dictionaries, one per fold, containing:
                     {'test_subject': str, 'train': {...}, 'val': {...}, 'test': {...}}

    Returns:
        DataFrame with columns: test_subject, train_*, val_*, test_* metrics
    """
    rows = []

    for fold in fold_metrics:
        row = {'test_subject': fold['test_subject']}

        # Add train metrics
        for key, value in fold['train'].items():
            row[f'train_{key}'] = value

        # Add val metrics
        for key, value in fold['val'].items():
            row[f'val_{key}'] = value

        # Add test metrics
        for key, value in fold['test'].items():
            row[f'test_{key}'] = value

        # Calculate overfitting gap (val - test for accuracy and F1)
        if 'val_accuracy' in row and 'test_accuracy' in row:
            row['overfitting_gap_accuracy'] = row['val_accuracy'] - row['test_accuracy']
        if 'val_f1_score' in row and 'test_f1_score' in row:
            row['overfitting_gap_f1'] = row['val_f1_score'] - row['test_f1_score']

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by test_subject for consistent ordering
    df['test_subject_int'] = pd.to_numeric(df['test_subject'], errors='coerce')
    df = df.sort_values('test_subject_int').drop('test_subject_int', axis=1)

    return df


def calculate_summary_stats(df: pd.DataFrame, metrics: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calculate mean, std, min, max across all subjects for specified metrics.

    Args:
        df: DataFrame from format_per_fold_table()
        metrics: List of column names to summarize. If None, summarizes all numeric columns.

    Returns:
        DataFrame with rows: mean, std, min, max for each metric
    """
    if metrics is None:
        # Auto-detect numeric columns (exclude test_subject)
        metrics = df.select_dtypes(include=[np.number]).columns.tolist()

    summary_data = {
        'mean': df[metrics].mean(),
        'std': df[metrics].std(),
        'min': df[metrics].min(),
        'max': df[metrics].max()
    }

    summary_df = pd.DataFrame(summary_data).T
    summary_df.index.name = 'statistic'

    return summary_df


def identify_outliers(df: pd.DataFrame,
                     metric: str = 'test_f1_score',
                     n_top: int = 5,
                     n_bottom: int = 5) -> Dict[str, List[Tuple[str, float]]]:
    """
    Identify best and worst performing subjects for a given metric.

    Args:
        df: DataFrame from format_per_fold_table()
        metric: Column name to rank by
        n_top: Number of top performers to return
        n_bottom: Number of bottom performers to return

    Returns:
        Dict with keys 'best' and 'worst', each containing list of (subject, value) tuples
    """
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in DataFrame. Available: {df.columns.tolist()}")

    # Sort by metric
    sorted_df = df.sort_values(metric, ascending=False)

    # Get top and bottom subjects
    best = [(row['test_subject'], row[metric])
            for _, row in sorted_df.head(n_top).iterrows()]

    worst = [(row['test_subject'], row[metric])
             for _, row in sorted_df.tail(n_bottom).iterrows()][::-1]  # Reverse to show worst first

    return {'best': best, 'worst': worst}


def calculate_overfitting_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate and rank subjects by overfitting gap (val - test performance).

    Args:
        df: DataFrame from format_per_fold_table()

    Returns:
        DataFrame sorted by overfitting gap, highest first
    """
    gap_df = df[['test_subject']].copy()

    # Calculate gaps for all metrics
    for col in df.columns:
        if col.startswith('val_'):
            metric_name = col.replace('val_', '')
            test_col = f'test_{metric_name}'

            if test_col in df.columns:
                gap_df[f'gap_{metric_name}'] = df[col] - df[test_col]

    # Sort by F1 gap (most common metric for analysis)
    if 'gap_f1_score' in gap_df.columns:
        gap_df = gap_df.sort_values('gap_f1_score', ascending=False)

    return gap_df


def generate_text_report(fold_metrics: List[Dict], model_name: str) -> str:
    """
    Generate human-readable text summary of results.

    Args:
        fold_metrics: List of fold results
        model_name: Name of the model

    Returns:
        Formatted string report
    """
    df = format_per_fold_table(fold_metrics)

    # Calculate key statistics
    test_f1_mean = df['test_f1_score'].mean()
    test_f1_std = df['test_f1_score'].std()
    test_acc_mean = df['test_accuracy'].mean()
    test_acc_std = df['test_accuracy'].std()
    val_f1_mean = df['val_f1_score'].mean()
    val_acc_mean = df['val_accuracy'].mean()

    overfitting_gap_f1 = val_f1_mean - test_f1_mean
    overfitting_gap_acc = val_acc_mean - test_acc_mean

    # Get outliers
    outliers_f1 = identify_outliers(df, 'test_f1_score', n_top=3, n_bottom=3)

    # Build report
    lines = []
    lines.append("=" * 70)
    lines.append(f"{model_name.upper()} MODEL - Results Summary")
    lines.append("=" * 70)
    lines.append("")

    lines.append("Average Performance:")
    lines.append(f"  Test F1:       {test_f1_mean:.2f} ± {test_f1_std:.2f}%")
    lines.append(f"  Test Accuracy: {test_acc_mean:.2f} ± {test_acc_std:.2f}%")
    lines.append(f"  Val F1:        {val_f1_mean:.2f}%")
    lines.append(f"  Val Accuracy:  {val_acc_mean:.2f}%")
    lines.append("")

    lines.append("Overfitting Analysis:")
    lines.append(f"  Gap (Val-Test) F1:  {overfitting_gap_f1:.2f}%")
    lines.append(f"  Gap (Val-Test) Acc: {overfitting_gap_acc:.2f}%")
    lines.append("")

    lines.append("Per-Subject Performance (Test F1):")
    lines.append("  Best subjects:")
    for subject, f1 in outliers_f1['best']:
        lines.append(f"    Subject {subject}: {f1:.2f}%")

    lines.append("  Worst subjects:")
    for subject, f1 in outliers_f1['worst']:
        lines.append(f"    Subject {subject}: {f1:.2f}%")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def generate_per_fold_summary_table(df: pd.DataFrame) -> str:
    """
    Generate formatted table showing key metrics per subject.

    Args:
        df: DataFrame from format_per_fold_table()

    Returns:
        Formatted string table
    """
    # Select key columns for display
    display_cols = ['test_subject', 'test_accuracy', 'test_f1_score',
                   'test_precision', 'test_recall', 'val_accuracy', 'val_f1_score']

    # Check which columns exist
    display_cols = [col for col in display_cols if col in df.columns]

    display_df = df[display_cols].copy()

    # Add overfitting gap if possible
    if 'overfitting_gap_f1' in df.columns:
        display_df['gap_f1'] = df['overfitting_gap_f1']

    # Format as string table
    table_str = display_df.to_string(index=False, float_format=lambda x: f'{x:.2f}')

    # Add summary row
    summary_row = "\n" + "-" * len(table_str.split('\n')[0]) + "\n"
    summary_vals = []
    summary_vals.append("AVERAGE".ljust(len(str(display_df['test_subject'].iloc[0]))))

    for col in display_df.columns:
        if col != 'test_subject' and pd.api.types.is_numeric_dtype(display_df[col]):
            summary_vals.append(f"{display_df[col].mean():.2f}".rjust(15))

    summary_row += "  ".join(summary_vals)

    return table_str + summary_row


def save_enhanced_results(fold_metrics: List[Dict],
                         output_dir: str,
                         model_name: str,
                         save_validation_report: bool = True) -> None:
    """
    Save comprehensive results including per-fold details and summaries.

    Args:
        fold_metrics: List of fold results from training
        output_dir: Directory to save results
        model_name: Name of the model
        save_validation_report: Whether to save data validation info
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Per-fold detailed CSV
    df = format_per_fold_table(fold_metrics)
    per_fold_csv = output_path / 'per_fold_detailed.csv'
    df.to_csv(per_fold_csv, index=False, float_format='%.2f')
    print(f"[Reporting] Saved per-fold details: {per_fold_csv}")

    # 2. Summary statistics
    summary_stats = calculate_summary_stats(df)
    summary_stats_file = output_path / 'summary_statistics.csv'
    summary_stats.to_csv(summary_stats_file, float_format='%.2f')
    print(f"[Reporting] Saved summary statistics: {summary_stats_file}")

    # 3. Text report
    text_report = generate_text_report(fold_metrics, model_name)
    report_file = output_path / 'summary_report.txt'
    with open(report_file, 'w') as f:
        f.write(text_report)
    print(f"[Reporting] Saved text report: {report_file}")

    # 4. Outlier analysis
    outliers_f1 = identify_outliers(df, 'test_f1_score', n_top=5, n_bottom=5)
    gap_analysis = calculate_overfitting_gaps(df)

    outlier_file = output_path / 'outlier_analysis.txt'
    with open(outlier_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("OUTLIER ANALYSIS\n")
        f.write("=" * 70 + "\n\n")

        f.write("Top 5 Best Subjects (Test F1):\n")
        for subject, f1 in outliers_f1['best']:
            f.write(f"  Subject {subject}: {f1:.2f}%\n")

        f.write("\nTop 5 Worst Subjects (Test F1):\n")
        for subject, f1 in outliers_f1['worst']:
            f.write(f"  Subject {subject}: {f1:.2f}%\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("OVERFITTING GAP ANALYSIS (Val - Test)\n")
        f.write("=" * 70 + "\n\n")
        f.write("Subjects with Highest Overfitting (Top 5):\n")
        top_overfit = gap_analysis.head(5)
        for _, row in top_overfit.iterrows():
            if 'gap_f1_score' in row:
                f.write(f"  Subject {row['test_subject']}: Gap = {row['gap_f1_score']:.2f}%\n")

    print(f"[Reporting] Saved outlier analysis: {outlier_file}")

    # 5. Per-fold summary table (human-readable)
    table_str = generate_per_fold_summary_table(df)
    table_file = output_path / 'per_fold_summary_table.txt'
    with open(table_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(f"{model_name.upper()} - Per-Fold Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(table_str)

    print(f"[Reporting] Saved per-fold table: {table_file}")

    print(f"\n[Reporting] All enhanced reports saved to: {output_dir}")


def merge_model_results(model_data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge per-fold results from multiple models into a single comparison table.

    Args:
        model_data_dict: Dict mapping model_name -> per_fold DataFrame

    Returns:
        Wide-format DataFrame with all models side-by-side
    """
    if not model_data_dict:
        return pd.DataFrame()

    # Start with first model
    first_model = list(model_data_dict.keys())[0]
    merged_df = model_data_dict[first_model][['test_subject']].copy()

    # Add metrics from each model with prefixed column names
    for model_name, model_df in model_data_dict.items():
        # Select key metrics
        key_metrics = ['test_accuracy', 'test_f1_score', 'test_precision', 'test_recall',
                      'val_accuracy', 'val_f1_score']

        available_metrics = [m for m in key_metrics if m in model_df.columns]

        for metric in available_metrics:
            col_name = f"{model_name}_{metric}"
            merged_df[col_name] = model_df[metric]

    return merged_df


# Utility function for backward compatibility
def create_scores_csv_compatible(fold_metrics: List[Dict], output_file: str) -> None:
    """
    Create scores.csv in the original format for backward compatibility.

    Args:
        fold_metrics: List of fold results
        output_file: Path to save scores.csv
    """
    df = format_per_fold_table(fold_metrics)

    # Calculate average row
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    avg_row = df[numeric_cols].mean().to_dict()
    avg_row['test_subject'] = 'Average'

    # Append average row
    df_with_avg = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    # Save
    df_with_avg.to_csv(output_file, index=False, float_format='%.2f')
    print(f"[Reporting] Saved backward-compatible scores.csv: {output_file}")


if __name__ == "__main__":
    # Test with dummy data
    dummy_fold_metrics = [
        {
            'test_subject': '29',
            'train': {'loss': 0.3, 'accuracy': 90.0, 'f1_score': 89.0},
            'val': {'loss': 0.4, 'accuracy': 82.0, 'f1_score': 84.0},
            'test': {'loss': 0.5, 'accuracy': 73.0, 'f1_score': 72.0}
        },
        {
            'test_subject': '30',
            'train': {'loss': 0.3, 'accuracy': 91.0, 'f1_score': 90.0},
            'val': {'loss': 0.4, 'accuracy': 85.0, 'f1_score': 86.0},
            'test': {'loss': 0.5, 'accuracy': 68.0, 'f1_score': 67.0}
        },
    ]

    # Test functions
    print("Testing format_per_fold_table:")
    df = format_per_fold_table(dummy_fold_metrics)
    print(df)

    print("\nTesting calculate_summary_stats:")
    summary = calculate_summary_stats(df)
    print(summary)

    print("\nTesting identify_outliers:")
    outliers = identify_outliers(df, 'test_f1_score', n_top=1, n_bottom=1)
    print(outliers)

    print("\nTesting generate_text_report:")
    report = generate_text_report(dummy_fold_metrics, 'TestModel')
    print(report)
