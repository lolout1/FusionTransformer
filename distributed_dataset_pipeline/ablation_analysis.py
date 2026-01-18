#!/usr/bin/env python3
"""Statistical analysis of ablation study results."""

import json
import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Statistical tests will be skipped.")


class AblationAnalyzer:
    """Comprehensive analysis of hyperparameter ablation results."""

    def __init__(self, results_dir: Path):
        """
        Initialize analyzer with results directory.

        Args:
            results_dir: Directory containing ablation results
        """
        self.results_dir = Path(results_dir)
        self.results_df: Optional[pd.DataFrame] = None
        self.raw_results: Optional[Dict] = None

        self._load_results()

    def _load_results(self):
        """Load results from comprehensive_results.csv or JSON."""
        csv_path = self.results_dir / 'comprehensive_results.csv'
        json_path = self.results_dir / 'comprehensive_results.json'

        if csv_path.exists():
            self.results_df = pd.read_csv(csv_path)
            # Filter to successful experiments only
            self.results_df = self.results_df[self.results_df['status'] == 'success']
            print(f"Loaded {len(self.results_df)} successful experiments from CSV")

        if json_path.exists():
            with open(json_path, 'r') as f:
                self.raw_results = json.load(f)
            print(f"Loaded raw results from JSON")

        if self.results_df is None or len(self.results_df) == 0:
            print("Warning: No results found to analyze")

    def compute_confidence_interval(
        self,
        mean: float,
        std: float,
        n: int,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """
        Compute confidence interval.

        Args:
            mean: Sample mean
            std: Sample standard deviation
            n: Sample size
            confidence: Confidence level (default 0.95)

        Returns:
            Tuple of (lower, upper) CI bounds
        """
        if not SCIPY_AVAILABLE or n <= 1:
            # Fall back to simple estimate
            se = std / np.sqrt(n) if n > 0 else 0
            margin = 1.96 * se  # Approximate for 95%
            return mean - margin, mean + margin

        se = std / np.sqrt(n)
        t_critical = stats.t.ppf((1 + confidence) / 2, df=n - 1)
        margin = t_critical * se

        return mean - margin, mean + margin

    def paired_t_test(
        self,
        group1: pd.DataFrame,
        group2: pd.DataFrame,
        metric: str = 'test_f1',
    ) -> Dict:
        """
        Perform paired t-test between two experiment groups.

        Args:
            group1: First group DataFrame
            group2: Second group DataFrame
            metric: Metric to compare

        Returns:
            Dictionary with test results
        """
        if not SCIPY_AVAILABLE:
            return {'error': 'scipy not available'}

        # Match by common configuration dimensions
        merged = group1.merge(
            group2,
            on=['window_size', 'stride_name', 'embed_dim'],
            suffixes=('_1', '_2'),
        )

        if len(merged) == 0:
            return {'error': 'No matching pairs found'}

        vals1 = merged[f'{metric}_1'].values
        vals2 = merged[f'{metric}_2'].values

        t_stat, p_value = stats.ttest_rel(vals1, vals2)

        # Effect size (Cohen's d for paired samples)
        diff = vals1 - vals2
        d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0

        return {
            'n_pairs': int(len(merged)),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'mean_diff': float(np.mean(diff)),
            'std_diff': float(np.std(diff, ddof=1)),
            'cohens_d': float(d),
            'significant_005': bool(p_value < 0.05),
            'significant_001': bool(p_value < 0.01),
        }

    def one_way_anova(
        self,
        df: pd.DataFrame,
        factor: str,
        metric: str = 'test_f1',
    ) -> Dict:
        """
        Perform one-way ANOVA for a factor.

        Args:
            df: DataFrame with results
            factor: Factor column to analyze
            metric: Metric to compare

        Returns:
            Dictionary with ANOVA results
        """
        if not SCIPY_AVAILABLE:
            return {'error': 'scipy not available'}

        groups = [group[metric].values for name, group in df.groupby(factor)]

        if len(groups) < 2:
            return {'error': 'Need at least 2 groups'}

        f_stat, p_value = stats.f_oneway(*groups)

        return {
            'factor': factor,
            'n_groups': int(len(groups)),
            'f_statistic': float(f_stat),
            'p_value': float(p_value),
            'significant_005': bool(p_value < 0.05),
            'significant_001': bool(p_value < 0.01),
        }

    def analyze_model_variants(self, dataset: str) -> Dict:
        """
        Compare model variants for a dataset.

        Args:
            dataset: Dataset name ('upfall' or 'wedafall')

        Returns:
            Dictionary with model comparison results
        """
        df = self.results_df[self.results_df['dataset'] == dataset]

        if len(df) == 0:
            return {'error': f'No results for {dataset}'}

        results = {
            'dataset': dataset,
            'model_performance': {},
            'pairwise_tests': {},
        }

        # Per-model summary
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            results['model_performance'][model] = {
                'n_experiments': len(model_df),
                'mean_f1': float(model_df['test_f1'].mean()),
                'std_f1': float(model_df['test_f1'].std()),
                'mean_accuracy': float(model_df['test_accuracy'].mean()),
                'mean_precision': float(model_df['test_precision'].mean()),
                'mean_recall': float(model_df['test_recall'].mean()),
            }

        # Pairwise comparisons
        models = list(df['model'].unique())
        for i, m1 in enumerate(models):
            for m2 in models[i+1:]:
                df1 = df[df['model'] == m1]
                df2 = df[df['model'] == m2]
                test_result = self.paired_t_test(df1, df2)
                results['pairwise_tests'][f'{m1}_vs_{m2}'] = test_result

        return results

    def analyze_window_effect(self, dataset: str) -> Dict:
        """
        Analyze effect of window size on performance.

        Args:
            dataset: Dataset name

        Returns:
            Dictionary with window size analysis
        """
        df = self.results_df[self.results_df['dataset'] == dataset]

        if len(df) == 0:
            return {'error': f'No results for {dataset}'}

        results = {
            'dataset': dataset,
            'window_performance': {},
            'anova': self.one_way_anova(df, 'window_size'),
        }

        # Per-window summary
        for ws in sorted(df['window_size'].unique()):
            ws_df = df[df['window_size'] == ws]
            ci_low, ci_high = self.compute_confidence_interval(
                ws_df['test_f1'].mean(),
                ws_df['test_f1'].std(),
                len(ws_df),
            )
            results['window_performance'][int(ws)] = {
                'n_experiments': len(ws_df),
                'mean_f1': float(ws_df['test_f1'].mean()),
                'std_f1': float(ws_df['test_f1'].std()),
                'ci_95_low': float(ci_low),
                'ci_95_high': float(ci_high),
                'mean_accuracy': float(ws_df['test_accuracy'].mean()),
            }

        return results

    def analyze_stride_effect(self, dataset: str) -> Dict:
        """
        Analyze effect of stride configuration on performance.

        Args:
            dataset: Dataset name

        Returns:
            Dictionary with stride analysis
        """
        df = self.results_df[self.results_df['dataset'] == dataset]

        if len(df) == 0:
            return {'error': f'No results for {dataset}'}

        results = {
            'dataset': dataset,
            'stride_performance': {},
            'anova': self.one_way_anova(df, 'stride_name'),
        }

        # Per-stride summary
        for stride in df['stride_name'].unique():
            stride_df = df[df['stride_name'] == stride]
            ci_low, ci_high = self.compute_confidence_interval(
                stride_df['test_f1'].mean(),
                stride_df['test_f1'].std(),
                len(stride_df),
            )
            results['stride_performance'][stride] = {
                'n_experiments': len(stride_df),
                'mean_f1': float(stride_df['test_f1'].mean()),
                'std_f1': float(stride_df['test_f1'].std()),
                'ci_95_low': float(ci_low),
                'ci_95_high': float(ci_high),
                'mean_accuracy': float(stride_df['test_accuracy'].mean()),
                'mean_recall': float(stride_df['test_recall'].mean()),
            }

        return results

    def analyze_embed_effect(self, dataset: str) -> Dict:
        """
        Analyze effect of embedding dimension on performance.

        Args:
            dataset: Dataset name

        Returns:
            Dictionary with embedding dimension analysis
        """
        df = self.results_df[self.results_df['dataset'] == dataset]

        if len(df) == 0:
            return {'error': f'No results for {dataset}'}

        results = {
            'dataset': dataset,
            'embed_performance': {},
        }

        # Per-embed summary
        for ed in sorted(df['embed_dim'].unique()):
            ed_df = df[df['embed_dim'] == ed]
            results['embed_performance'][int(ed)] = {
                'n_experiments': len(ed_df),
                'mean_f1': float(ed_df['test_f1'].mean()),
                'std_f1': float(ed_df['test_f1'].std()),
                'mean_accuracy': float(ed_df['test_accuracy'].mean()),
            }

        # Paired t-test if two embed dims
        embeds = sorted(df['embed_dim'].unique())
        if len(embeds) == 2:
            df1 = df[df['embed_dim'] == embeds[0]]
            df2 = df[df['embed_dim'] == embeds[1]]
            results['pairwise_test'] = self.paired_t_test(df1, df2)

        return results

    def find_best_config(self, dataset: str) -> Dict:
        """
        Find the best configuration for a dataset.

        Args:
            dataset: Dataset name

        Returns:
            Dictionary with best config details
        """
        df = self.results_df[self.results_df['dataset'] == dataset]

        if len(df) == 0:
            return {'error': f'No results for {dataset}'}

        # Find best by F1
        best_idx = df['test_f1'].idxmax()
        best_row = df.loc[best_idx]

        return {
            'dataset': dataset,
            'best_config': {
                'window_size': int(best_row['window_size']),
                'stride_name': best_row['stride_name'],
                'fall_stride': int(best_row['fall_stride']),
                'adl_stride': int(best_row['adl_stride']),
                'embed_dim': int(best_row['embed_dim']),
                'model': best_row['model'],
            },
            'best_metrics': {
                'test_f1': float(best_row['test_f1']),
                'test_f1_std': float(best_row['test_f1_std']),
                'test_accuracy': float(best_row['test_accuracy']),
                'test_precision': float(best_row['test_precision']),
                'test_recall': float(best_row['test_recall']),
            },
        }

    def generate_statistical_report(self) -> str:
        """
        Generate comprehensive statistical analysis report.

        Returns:
            Report as string
        """
        lines = []
        lines.append("=" * 70)
        lines.append("STATISTICAL SIGNIFICANCE ANALYSIS")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("=" * 70)

        for dataset in self.results_df['dataset'].unique():
            lines.append(f"\n{'='*70}")
            lines.append(f"DATASET: {dataset.upper()}")
            lines.append(f"{'='*70}")

            # Best configuration
            best = self.find_best_config(dataset)
            if 'error' not in best:
                lines.append(f"\n1. BEST CONFIGURATION")
                lines.append("-" * 40)
                lines.append(f"   Window: {best['best_config']['window_size']}")
                lines.append(f"   Stride: {best['best_config']['stride_name']} "
                           f"(fall={best['best_config']['fall_stride']}, adl={best['best_config']['adl_stride']})")
                lines.append(f"   Embed: {best['best_config']['embed_dim']}")
                lines.append(f"   Model: {best['best_config']['model']}")
                lines.append(f"   F1: {best['best_metrics']['test_f1']:.2f}% +/- {best['best_metrics']['test_f1_std']:.2f}%")

            # Model comparison
            model_analysis = self.analyze_model_variants(dataset)
            if 'error' not in model_analysis:
                lines.append(f"\n2. MODEL VARIANT COMPARISON")
                lines.append("-" * 40)
                for model, perf in model_analysis['model_performance'].items():
                    lines.append(f"   {model}: F1 = {perf['mean_f1']:.2f}% +/- {perf['std_f1']:.2f}%")

                for comparison, test in model_analysis['pairwise_tests'].items():
                    if 'error' not in test:
                        sig = "*" if test['significant_005'] else ""
                        sig += "*" if test['significant_001'] else ""
                        lines.append(f"   {comparison}: Delta={test['mean_diff']:+.2f}%, "
                                   f"p={test['p_value']:.4f}{sig}, d={test['cohens_d']:.2f}")

            # Window effect
            window_analysis = self.analyze_window_effect(dataset)
            if 'error' not in window_analysis:
                lines.append(f"\n3. WINDOW SIZE EFFECT")
                lines.append("-" * 40)
                for ws, perf in sorted(window_analysis['window_performance'].items()):
                    lines.append(f"   {ws} samples: F1 = {perf['mean_f1']:.2f}% "
                               f"[{perf['ci_95_low']:.2f}, {perf['ci_95_high']:.2f}]")

                anova = window_analysis['anova']
                if 'error' not in anova:
                    sig = "*" if anova['significant_005'] else ""
                    sig += "*" if anova['significant_001'] else ""
                    lines.append(f"   ANOVA: F={anova['f_statistic']:.2f}, p={anova['p_value']:.4f}{sig}")

            # Stride effect
            stride_analysis = self.analyze_stride_effect(dataset)
            if 'error' not in stride_analysis:
                lines.append(f"\n4. STRIDE CONFIGURATION EFFECT")
                lines.append("-" * 40)
                for stride, perf in stride_analysis['stride_performance'].items():
                    lines.append(f"   {stride}: F1 = {perf['mean_f1']:.2f}% +/- {perf['std_f1']:.2f}%, "
                               f"Recall = {perf['mean_recall']:.2f}%")

                anova = stride_analysis['anova']
                if 'error' not in anova:
                    sig = "*" if anova['significant_005'] else ""
                    sig += "*" if anova['significant_001'] else ""
                    lines.append(f"   ANOVA: F={anova['f_statistic']:.2f}, p={anova['p_value']:.4f}{sig}")

            # Embedding effect
            embed_analysis = self.analyze_embed_effect(dataset)
            if 'error' not in embed_analysis:
                lines.append(f"\n5. EMBEDDING DIMENSION EFFECT")
                lines.append("-" * 40)
                for ed, perf in sorted(embed_analysis['embed_performance'].items()):
                    lines.append(f"   {ed}d: F1 = {perf['mean_f1']:.2f}% +/- {perf['std_f1']:.2f}%")

                if 'pairwise_test' in embed_analysis and 'error' not in embed_analysis['pairwise_test']:
                    test = embed_analysis['pairwise_test']
                    sig = "*" if test['significant_005'] else ""
                    lines.append(f"   t-test: Delta={test['mean_diff']:+.2f}%, p={test['p_value']:.4f}{sig}")

        lines.append(f"\n{'='*70}")
        lines.append("Note: * p < 0.05, ** p < 0.01")
        lines.append("=" * 70)

        return "\n".join(lines)

    def generate_latex_tables(self) -> str:
        """
        Generate publication-ready LaTeX tables.

        Returns:
            LaTeX code as string
        """
        lines = []
        lines.append("% Hyperparameter Ablation Results")
        lines.append(f"% Generated: {datetime.now().strftime('%Y-%m-%d')}")
        lines.append("")

        for dataset in self.results_df['dataset'].unique():
            df = self.results_df[self.results_df['dataset'] == dataset]
            dataset_name = "UP-FALL" if dataset == 'upfall' else "WEDA-FALL"

            # Table 1: Window Size Results
            lines.append(f"% {dataset_name} - Window Size Ablation")
            lines.append("\\begin{table}[h]")
            lines.append("\\centering")
            lines.append(f"\\caption{{Window Size Ablation: {dataset_name} Dataset}}")
            lines.append(f"\\label{{tab:{dataset}_window}}")
            lines.append("\\begin{tabular}{lccccc}")
            lines.append("\\toprule")
            lines.append("Window & F1 (\\%) & Macro-F1 (\\%) & Acc (\\%) & Prec (\\%) & Rec (\\%) \\\\")
            lines.append("\\midrule")

            # Find best window size by mean F1
            ws_means = df.groupby('window_size')['test_f1'].mean()
            best_ws = ws_means.idxmax()

            for ws in sorted(df['window_size'].unique()):
                ws_df = df[df['window_size'] == ws]
                f1 = ws_df['test_f1'].mean()
                f1_std = ws_df['test_f1'].std()
                macro_f1 = ws_df['test_macro_f1'].mean() if 'test_macro_f1' in ws_df.columns else f1
                acc = ws_df['test_accuracy'].mean()
                prec = ws_df['test_precision'].mean()
                rec = ws_df['test_recall'].mean()

                # Bold best
                is_best = ws == best_ws
                f1_str = f"\\textbf{{{f1:.1f}}} $\\pm$ {f1_std:.1f}" if is_best else f"{f1:.1f} $\\pm$ {f1_std:.1f}"

                lines.append(f"{int(ws)} & {f1_str} & {macro_f1:.1f} & {acc:.1f} & {prec:.1f} & {rec:.1f} \\\\")

            lines.append("\\bottomrule")
            lines.append("\\end{tabular}")
            lines.append("\\end{table}")
            lines.append("")

            # Table 2: Stride Configuration Results
            lines.append(f"% {dataset_name} - Stride Configuration Ablation")
            lines.append("\\begin{table}[h]")
            lines.append("\\centering")
            lines.append(f"\\caption{{Stride Configuration Ablation: {dataset_name} Dataset}}")
            lines.append(f"\\label{{tab:{dataset}_stride}}")
            lines.append("\\begin{tabular}{lccccc}")
            lines.append("\\toprule")
            lines.append("Stride & F1 (\\%) & Acc (\\%) & Prec (\\%) & Rec (\\%) \\\\")
            lines.append("\\midrule")

            # Find best stride by mean F1
            stride_means = df.groupby('stride_name')['test_f1'].mean()
            best_stride = stride_means.idxmax()

            for stride in ['aggressive', 'standard', 'moderate', 'equal']:
                if stride not in df['stride_name'].values:
                    continue
                stride_df = df[df['stride_name'] == stride]
                f1 = stride_df['test_f1'].mean()
                f1_std = stride_df['test_f1'].std()
                acc = stride_df['test_accuracy'].mean()
                prec = stride_df['test_precision'].mean()
                rec = stride_df['test_recall'].mean()

                fall_stride = int(stride_df['fall_stride'].iloc[0])
                adl_stride = int(stride_df['adl_stride'].iloc[0])

                stride_label = f"{stride} ({fall_stride}/{adl_stride})"
                is_best = stride == best_stride
                f1_str = f"\\textbf{{{f1:.1f}}} $\\pm$ {f1_std:.1f}" if is_best else f"{f1:.1f} $\\pm$ {f1_std:.1f}"

                lines.append(f"{stride_label} & {f1_str} & {acc:.1f} & {prec:.1f} & {rec:.1f} \\\\")

            lines.append("\\bottomrule")
            lines.append("\\end{tabular}")
            lines.append("\\end{table}")
            lines.append("")

            # Table 3: Model Variant Results
            lines.append(f"% {dataset_name} - Model Variant Ablation")
            lines.append("\\begin{table}[h]")
            lines.append("\\centering")
            lines.append(f"\\caption{{Model Variant Ablation: {dataset_name} Dataset}}")
            lines.append(f"\\label{{tab:{dataset}_model}}")
            lines.append("\\begin{tabular}{lccccc}")
            lines.append("\\toprule")
            lines.append("Model & F1 (\\%) & Acc (\\%) & Prec (\\%) & Rec (\\%) \\\\")
            lines.append("\\midrule")

            model_labels = {
                'kalman_conv1d_linear': 'Kalman + C1D + Lin',
                'kalman_conv1d_conv1d': 'Kalman + C1D + C1D',
                'raw_dual_stream': 'Raw Dual Stream',
            }

            # Find best model by mean F1
            model_means = df.groupby('model')['test_f1'].mean()
            best_model = model_means.idxmax()

            for model in df['model'].unique():
                model_df = df[df['model'] == model]
                f1 = model_df['test_f1'].mean()
                f1_std = model_df['test_f1'].std()
                acc = model_df['test_accuracy'].mean()
                prec = model_df['test_precision'].mean()
                rec = model_df['test_recall'].mean()

                model_label = model_labels.get(model, model)
                is_best = model == best_model
                f1_str = f"\\textbf{{{f1:.1f}}} $\\pm$ {f1_std:.1f}" if is_best else f"{f1:.1f} $\\pm$ {f1_std:.1f}"

                lines.append(f"{model_label} & {f1_str} & {acc:.1f} & {prec:.1f} & {rec:.1f} \\\\")

            lines.append("\\bottomrule")
            lines.append("\\end{tabular}")
            lines.append("\\end{table}")
            lines.append("")

        return "\n".join(lines)

    def run_full_analysis(self):
        """Run complete analysis and save all outputs."""
        if self.results_df is None or len(self.results_df) == 0:
            print("No results to analyze")
            return

        print("\n" + "=" * 70)
        print("RUNNING COMPREHENSIVE ABLATION ANALYSIS")
        print("=" * 70)

        # 1. Statistical report
        stat_report = self.generate_statistical_report()
        stat_path = self.results_dir / 'statistical_analysis.txt'
        with open(stat_path, 'w') as f:
            f.write(stat_report)
        print(f"\nStatistical report saved: {stat_path}")

        # Print to console
        print("\n" + stat_report)

        # 2. LaTeX tables
        latex_tables = self.generate_latex_tables()
        latex_path = self.results_dir / 'latex_tables.tex'
        with open(latex_path, 'w') as f:
            f.write(latex_tables)
        print(f"\nLaTeX tables saved: {latex_path}")

        # 3. Summary JSON with all analysis
        summary = {
            'timestamp': datetime.now().isoformat(),
            'datasets': {},
        }

        for dataset in self.results_df['dataset'].unique():
            summary['datasets'][dataset] = {
                'best_config': self.find_best_config(dataset),
                'model_analysis': self.analyze_model_variants(dataset),
                'window_analysis': self.analyze_window_effect(dataset),
                'stride_analysis': self.analyze_stride_effect(dataset),
                'embed_analysis': self.analyze_embed_effect(dataset),
            }

        summary_path = self.results_dir / 'analysis_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nAnalysis summary saved: {summary_path}")

        # 4. Best configurations summary
        self._print_best_configs(summary)

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"All outputs saved to: {self.results_dir}")

    def _print_best_configs(self, summary: Dict):
        """Print best configurations to console."""
        print("\n" + "=" * 70)
        print("BEST CONFIGURATIONS PER DATASET")
        print("=" * 70)

        for dataset, analysis in summary['datasets'].items():
            best = analysis['best_config']
            if 'error' in best:
                continue

            print(f"\n{dataset.upper()}:")
            print(f"  Window: {best['best_config']['window_size']} samples")
            print(f"  Stride: {best['best_config']['stride_name']} "
                  f"(fall={best['best_config']['fall_stride']}, adl={best['best_config']['adl_stride']})")
            print(f"  Embed: {best['best_config']['embed_dim']}")
            print(f"  Model: {best['best_config']['model']}")
            print(f"  F1: {best['best_metrics']['test_f1']:.2f}% +/- {best['best_metrics']['test_f1_std']:.2f}%")
            print(f"  Accuracy: {best['best_metrics']['test_accuracy']:.2f}%")
            print(f"  Recall: {best['best_metrics']['test_recall']:.2f}%")


def main():
    """Run analysis on specified results directory."""
    import argparse

    parser = argparse.ArgumentParser(description='Analyze hyperparameter ablation results')
    parser.add_argument('results_dir', type=str, help='Path to results directory')

    args = parser.parse_args()

    analyzer = AblationAnalyzer(Path(args.results_dir))
    analyzer.run_full_analysis()


if __name__ == '__main__':
    main()
