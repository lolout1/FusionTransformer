#!/usr/bin/env python3
"""
Stream-Kalman Ablation Study - Results Aggregation and Analysis

Aggregates results from 6-model ablation study:
  A: Single-stream, No Kalman, No SE/TAP (baseline)
  B: Dual-stream, No Kalman, No SE/TAP
  C: Single-stream + Kalman smoothing, No SE/TAP
  D: Dual-stream + Kalman smoothing, No SE/TAP
  E: Single-stream + Kalman + SE + TAP
  F: Dual-stream + Kalman + SE + TAP (reference)

Scientific Questions Answered:
  1. Single vs Dual-stream: A vs B (effect of modality separation)
  2. Effect of Kalman smoothing: A vs C, B vs D
  3. Effect of SE+TAP: C vs E, D vs F
  4. Full comparison: expected ordering A < B < C < D < E < F

Features:
  - Supports both ongoing (log.txt) and completed (scores.csv) experiments
  - Real-time progress monitoring during training
  - Per-fold metric extraction and comparison
  - Statistical significance testing (paired t-test, Cohen's d)
  - Publication-ready summary tables

Usage:
    python aggregate_stream_kalman_ablation.py <results_dir>
    python aggregate_stream_kalman_ablation.py  # Auto-detect latest results
"""

import os
import sys
import re
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class ExperimentConfig:
    key: str
    display_name: str
    short_name: str
    model_class: str
    stream_type: str
    kalman: bool
    se_tap: bool
    color: str = '#333333'


EXPERIMENTS = {
    'model_a': ExperimentConfig(
        key='model_a',
        display_name='Single-stream Raw',
        short_name='Single_Raw',
        model_class='SingleStreamTransformer',
        stream_type='single',
        kalman=False,
        se_tap=False,
        color='#E74C3C'
    ),
    'model_b': ExperimentConfig(
        key='model_b',
        display_name='Dual-stream Raw',
        short_name='Dual_Raw',
        model_class='DualStreamBase',
        stream_type='dual',
        kalman=False,
        se_tap=False,
        color='#3498DB'
    ),
    'model_c': ExperimentConfig(
        key='model_c',
        display_name='Single-stream + Kalman',
        short_name='Single_Kalman',
        model_class='SingleStreamTransformer',
        stream_type='single',
        kalman=True,
        se_tap=False,
        color='#E67E22'
    ),
    'model_d': ExperimentConfig(
        key='model_d',
        display_name='Dual-stream + Kalman',
        short_name='Dual_Kalman',
        model_class='DualStreamBase',
        stream_type='dual',
        kalman=True,
        se_tap=False,
        color='#2ECC71'
    ),
    'model_e': ExperimentConfig(
        key='model_e',
        display_name='Single + Kalman + SE',
        short_name='Single_Full',
        model_class='SingleStreamTransformerSE',
        stream_type='single',
        kalman=True,
        se_tap=True,
        color='#9B59B6'
    ),
    'model_f': ExperimentConfig(
        key='model_f',
        display_name='Dual + Kalman + SE',
        short_name='Dual_Full',
        model_class='DualStreamRobust',
        stream_type='dual',
        kalman=True,
        se_tap=True,
        color='#1ABC9C'
    ),
}


@dataclass
class FoldMetrics:
    subject: str
    val_acc: Optional[float] = None
    val_f1: Optional[float] = None
    test_acc: Optional[float] = None
    test_f1: Optional[float] = None
    val_loss: Optional[float] = None
    test_loss: Optional[float] = None
    test_precision: Optional[float] = None
    test_recall: Optional[float] = None
    test_auc: Optional[float] = None

    def is_complete(self) -> bool:
        return all([
            self.val_acc is not None,
            self.val_f1 is not None,
            self.test_acc is not None,
            self.test_f1 is not None
        ])

    def has_any_data(self) -> bool:
        return any([
            self.val_acc is not None,
            self.val_f1 is not None,
            self.test_acc is not None,
            self.test_f1 is not None
        ])


@dataclass
class ExperimentResults:
    name: str
    config: ExperimentConfig
    folds: Dict[str, FoldMetrics] = field(default_factory=dict)
    status: str = "NOT_FOUND"
    source_path: str = ""

    def add_fold(self, fold: FoldMetrics) -> None:
        self.folds[fold.subject] = fold

    def n_complete_folds(self) -> int:
        return sum(1 for f in self.folds.values() if f.is_complete())

    def n_partial_folds(self) -> int:
        return sum(1 for f in self.folds.values() if f.has_any_data() and not f.is_complete())

    def get_metric_array(self, metric: str) -> np.ndarray:
        values = []
        for fold in self.folds.values():
            val = getattr(fold, metric, None)
            if val is not None:
                values.append(val)
        return np.array(values)

    def compute_stats(self, metric: str) -> Tuple[float, float]:
        arr = self.get_metric_array(metric)
        if len(arr) == 0:
            return (np.nan, np.nan)
        return (float(np.mean(arr)), float(np.std(arr)))


class LogParser:
    """Parse log.txt files to extract fold metrics during/after training."""

    @classmethod
    def parse_log(cls, log_path: Path) -> Dict[str, FoldMetrics]:
        """Parse a log.txt file and extract metrics for each fold."""
        folds = {}
        if not log_path.exists():
            return folds

        try:
            with open(log_path, 'r', errors='ignore') as f:
                lines = f.readlines()
        except Exception:
            return folds

        current_subject = None
        best_val_acc = None
        best_val_f1 = None

        for line in lines:
            # Match fold header: "Test subjects: [31]" or "Test Subjects: [31]"
            match = re.search(r'Test Subjects?[:\s]+\[?(\d+)', line, re.IGNORECASE)
            if not match:
                match = re.search(r'Fold \d+/\d+.*Test subjects?[:\s]+\[?(\d+)', line, re.IGNORECASE)
            if match:
                # Save previous fold if exists
                if current_subject and current_subject in folds:
                    if best_val_acc is not None:
                        folds[current_subject].val_acc = best_val_acc
                    if best_val_f1 is not None:
                        folds[current_subject].val_f1 = best_val_f1

                current_subject = match.group(1)
                if current_subject not in folds:
                    folds[current_subject] = FoldMetrics(subject=current_subject)
                best_val_acc = None
                best_val_f1 = None
                continue

            if current_subject is None:
                continue

            # Match validation metrics: "Val Loss: 0.000627. Val Acc: 87.058824% F1 score: 90.265487%"
            val_match = re.search(
                r'Val Loss:\s*([\d.]+)\.\s*Val Acc:\s*([\d.]+)%.*?F1 score:\s*([\d.]+)%',
                line, re.IGNORECASE
            )
            if val_match:
                val_acc = float(val_match.group(2))
                val_f1 = float(val_match.group(3))
                # Keep best validation F1
                if best_val_f1 is None or val_f1 > best_val_f1:
                    best_val_acc = val_acc
                    best_val_f1 = val_f1
                folds[current_subject].val_acc = val_acc
                folds[current_subject].val_f1 = val_f1
                continue

            # Match test accuracy: "Test accuracy on subject [31]: 87.50"
            test_acc_match = re.search(r'Test accuracy[^:]*:\s*([\d.]+)', line, re.IGNORECASE)
            if test_acc_match:
                folds[current_subject].test_acc = float(test_acc_match.group(1))
                continue

            # Match test F1: "Test F-Score: 90.32"
            test_f1_match = re.search(r'Test F-?Score:\s*([\d.]+)', line, re.IGNORECASE)
            if test_f1_match:
                folds[current_subject].test_f1 = float(test_f1_match.group(1))
                continue

            # Match test precision: "Test Precision: 85.00"
            test_prec_match = re.search(r'Test Precision:\s*([\d.]+)', line, re.IGNORECASE)
            if test_prec_match:
                folds[current_subject].test_precision = float(test_prec_match.group(1))
                continue

            # Match test recall: "Test Recall: 96.15"
            test_rec_match = re.search(r'Test Recall:\s*([\d.]+)', line, re.IGNORECASE)
            if test_rec_match:
                folds[current_subject].test_recall = float(test_rec_match.group(1))
                continue

        # Save last fold
        if current_subject and current_subject in folds:
            if best_val_acc is not None:
                folds[current_subject].val_acc = best_val_acc
            if best_val_f1 is not None:
                folds[current_subject].val_f1 = best_val_f1

        return folds


class StreamKalmanAggregator:
    """Aggregates and analyzes stream-kalman ablation study results."""

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / 'aggregated'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiments: Dict[str, ExperimentResults] = {}
        self.all_subjects: List[str] = []

    def find_experiment_dir(self, exp_key: str) -> Optional[Path]:
        """Find the experiment directory for a given experiment key."""
        metrics_dir = self.results_dir / 'metrics'
        if not metrics_dir.exists():
            return None

        matching_dirs = []
        for subdir in metrics_dir.iterdir():
            if not subdir.is_dir():
                continue
            name_lower = subdir.name.lower()
            if exp_key.lower() in name_lower:
                has_log = (subdir / 'log.txt').exists()
                has_scores = (subdir / 'scores.csv').exists()
                has_data = has_log or has_scores
                matching_dirs.append((subdir, has_data, len(subdir.name)))

        if not matching_dirs:
            return None

        # Prefer dirs with data, then longest name (most recent timestamp)
        matching_dirs.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return matching_dirs[0][0]

    def load_from_scores_csv(self, csv_path: Path) -> Dict[str, FoldMetrics]:
        """Load metrics from completed scores.csv file."""
        folds = {}
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  Warning: Could not read {csv_path}: {e}")
            return folds

        for _, row in df.iterrows():
            subject = str(row.get('test_subject', ''))
            if subject.lower() == 'average' or subject == '':
                continue

            fold = FoldMetrics(
                subject=subject,
                val_acc=row.get('val_accuracy'),
                val_f1=row.get('val_f1_score'),
                test_acc=row.get('test_accuracy'),
                test_f1=row.get('test_f1_score'),
                val_loss=row.get('val_loss'),
                test_loss=row.get('test_loss'),
                test_precision=row.get('test_precision'),
                test_recall=row.get('test_recall'),
                test_auc=row.get('test_auc'),
            )
            folds[subject] = fold

        return folds

    def load_experiment(self, exp_key: str) -> ExperimentResults:
        """Load results for a single experiment."""
        config = EXPERIMENTS[exp_key]
        result = ExperimentResults(name=exp_key, config=config)

        exp_dir = self.find_experiment_dir(exp_key)
        if exp_dir is None:
            result.status = "NOT_FOUND"
            return result

        result.source_path = str(exp_dir)

        # Try scores.csv first (completed experiment)
        scores_csv = exp_dir / 'scores.csv'
        if scores_csv.exists():
            folds = self.load_from_scores_csv(scores_csv)
            if folds:
                for fold in folds.values():
                    result.add_fold(fold)
                result.status = "COMPLETED"
                return result

        # Try log.txt (ongoing or completed experiment)
        log_file = exp_dir / 'log.txt'
        if log_file.exists():
            folds = LogParser.parse_log(log_file)
            if folds:
                for fold in folds.values():
                    result.add_fold(fold)
                n_complete = result.n_complete_folds()
                n_partial = result.n_partial_folds()
                result.status = "ONGOING" if n_partial > 0 or n_complete < 22 else "COMPLETED"
                return result

        result.status = "NOT_FOUND"
        return result

    def load_all_experiments(self) -> None:
        """Load all experiments and collect results."""
        print("\n" + "=" * 80)
        print("LOADING EXPERIMENTS")
        print("=" * 80)

        for exp_key in EXPERIMENTS:
            result = self.load_experiment(exp_key)
            self.experiments[exp_key] = result

            status_icon = {"COMPLETED": "[OK]", "ONGOING": "[..]", "NOT_FOUND": "[--]"}.get(result.status, "[??]")
            n_complete = result.n_complete_folds()
            n_partial = result.n_partial_folds()

            print(f"\n{status_icon} {result.config.short_name} ({result.status})")
            if result.source_path:
                print(f"    Source: {Path(result.source_path).name}")
            print(f"    Folds: {n_complete}/22 complete, {n_partial} partial")

            # Show current metrics if available
            if n_complete > 0 or n_partial > 0:
                tf_mean, tf_std = result.compute_stats('test_f1')
                vf_mean, vf_std = result.compute_stats('val_f1')
                if not np.isnan(tf_mean):
                    print(f"    Test F1:  {tf_mean:.2f} ± {tf_std:.2f}%")
                if not np.isnan(vf_mean):
                    print(f"    Val F1:   {vf_mean:.2f} ± {vf_std:.2f}%")

        # Collect all subjects
        all_subjects = set()
        for exp in self.experiments.values():
            all_subjects.update(exp.folds.keys())
        self.all_subjects = sorted(all_subjects, key=lambda x: int(x) if x.isdigit() else 0)

        print(f"\nTotal unique subjects: {len(self.all_subjects)}")

    def generate_ablation_table(self) -> str:
        """Generate the main ablation comparison table."""
        lines = []
        lines.append("\n" + "=" * 90)
        lines.append("6-MODEL ABLATION STUDY RESULTS")
        lines.append("=" * 90)
        lines.append("")

        lines.append(f"{'Model':<6} {'Description':<25} {'Stream':<8} {'Kalman':<8} {'SE+TAP':<8} "
                    f"{'Folds':<6} {'TestF1':<16}")
        lines.append("-" * 90)

        for exp_key in ['model_a', 'model_b', 'model_c', 'model_d', 'model_e', 'model_f']:
            exp = self.experiments.get(exp_key)
            if not exp:
                continue

            n_folds = exp.n_complete_folds()
            tf_mean, tf_std = exp.compute_stats('test_f1')

            def fmt(m, s):
                return f"{m:.2f} ± {s:.2f}" if not np.isnan(m) else "N/A"

            kalman_str = "Yes" if exp.config.kalman else "No"
            se_str = "Yes" if exp.config.se_tap else "No"

            lines.append(
                f"{exp_key.upper():<6} {exp.config.display_name:<25} {exp.config.stream_type:<8} "
                f"{kalman_str:<8} {se_str:<8} {n_folds:<6} {fmt(tf_mean, tf_std):<16}"
            )

        lines.append("-" * 90)
        return "\n".join(lines)

    def generate_effect_analysis(self) -> str:
        """Analyze the effect of each architectural component."""
        lines = []
        lines.append("\n" + "=" * 80)
        lines.append("COMPONENT EFFECT ANALYSIS")
        lines.append("=" * 80)

        # Effect 1: Single vs Dual Stream (no Kalman)
        lines.append("\n1. SINGLE vs DUAL STREAM (baseline, no Kalman)")
        lines.append("-" * 50)
        exp_a = self.experiments.get('model_a')
        exp_b = self.experiments.get('model_b')

        if exp_a and exp_b:
            for metric in ['test_f1', 'test_acc']:
                m1, s1 = exp_a.compute_stats(metric)
                m2, s2 = exp_b.compute_stats(metric)
                if not np.isnan(m1) and not np.isnan(m2):
                    diff = m2 - m1
                    better = "Dual" if diff > 0 else "Single" if diff < 0 else "Tie"
                    metric_name = metric.replace('_', ' ').title()
                    lines.append(f"  {metric_name}: Single {m1:.2f}±{s1:.2f} vs Dual {m2:.2f}±{s2:.2f} "
                               f"(Δ={diff:+.2f}% → {better})")
        else:
            lines.append("  (Insufficient data)")

        # Effect 2: Kalman Smoothing
        lines.append("\n2. EFFECT OF KALMAN SMOOTHING")
        lines.append("-" * 50)

        # Single: A vs C
        lines.append("  Single-stream (A vs C):")
        exp_a = self.experiments.get('model_a')
        exp_c = self.experiments.get('model_c')
        if exp_a and exp_c:
            m1, _ = exp_a.compute_stats('test_f1')
            m2, _ = exp_c.compute_stats('test_f1')
            if not np.isnan(m1) and not np.isnan(m2):
                diff = m2 - m1
                lines.append(f"    F1: {m1:.2f} → {m2:.2f} (Δ={diff:+.2f}%)")

        # Dual: B vs D
        lines.append("  Dual-stream (B vs D):")
        exp_b = self.experiments.get('model_b')
        exp_d = self.experiments.get('model_d')
        if exp_b and exp_d:
            m1, _ = exp_b.compute_stats('test_f1')
            m2, _ = exp_d.compute_stats('test_f1')
            if not np.isnan(m1) and not np.isnan(m2):
                diff = m2 - m1
                lines.append(f"    F1: {m1:.2f} → {m2:.2f} (Δ={diff:+.2f}%)")

        # Effect 3: SE + TAP
        lines.append("\n3. EFFECT OF SE + TEMPORAL ATTENTION POOLING")
        lines.append("-" * 50)

        # Single: C vs E
        lines.append("  Single-stream (C vs E):")
        exp_c = self.experiments.get('model_c')
        exp_e = self.experiments.get('model_e')
        if exp_c and exp_e:
            m1, _ = exp_c.compute_stats('test_f1')
            m2, _ = exp_e.compute_stats('test_f1')
            if not np.isnan(m1) and not np.isnan(m2):
                diff = m2 - m1
                lines.append(f"    F1: {m1:.2f} → {m2:.2f} (Δ={diff:+.2f}%)")

        # Dual: D vs F
        lines.append("  Dual-stream (D vs F):")
        exp_d = self.experiments.get('model_d')
        exp_f = self.experiments.get('model_f')
        if exp_d and exp_f:
            m1, _ = exp_d.compute_stats('test_f1')
            m2, _ = exp_f.compute_stats('test_f1')
            if not np.isnan(m1) and not np.isnan(m2):
                diff = m2 - m1
                lines.append(f"    F1: {m1:.2f} → {m2:.2f} (Δ={diff:+.2f}%)")

        # Cumulative improvement
        lines.append("\n4. CUMULATIVE IMPROVEMENT (A → F)")
        lines.append("-" * 50)
        exp_a = self.experiments.get('model_a')
        exp_f = self.experiments.get('model_f')
        if exp_a and exp_f:
            m1, _ = exp_a.compute_stats('test_f1')
            m2, _ = exp_f.compute_stats('test_f1')
            if not np.isnan(m1) and not np.isnan(m2):
                diff = m2 - m1
                pct_improve = (diff / m1) * 100 if m1 > 0 else 0
                lines.append(f"  Baseline (A): {m1:.2f}%")
                lines.append(f"  Full Model (F): {m2:.2f}%")
                lines.append(f"  Total Improvement: {diff:+.2f}% ({pct_improve:+.1f}% relative)")

        return "\n".join(lines)

    def perform_statistical_tests(self) -> str:
        """Perform paired t-tests for key comparisons."""
        lines = []
        lines.append("\n" + "=" * 80)
        lines.append("STATISTICAL SIGNIFICANCE TESTS")
        lines.append("=" * 80)

        if not HAS_SCIPY:
            lines.append("(scipy not available - skipping statistical tests)")
            return "\n".join(lines)

        lines.append("")
        lines.append("Paired t-test on Test F1 (22 folds)")
        lines.append("Significance: * p<0.05, ** p<0.01, *** p<0.001")

        comparisons = [
            ('model_a', 'model_b', 'Single vs Dual (raw)'),
            ('model_a', 'model_c', 'Effect of Kalman (single)'),
            ('model_b', 'model_d', 'Effect of Kalman (dual)'),
            ('model_c', 'model_e', 'Effect of SE+TAP (single)'),
            ('model_d', 'model_f', 'Effect of SE+TAP (dual)'),
            ('model_a', 'model_f', 'Full improvement (A→F)'),
        ]

        for key1, key2, desc in comparisons:
            exp1 = self.experiments.get(key1)
            exp2 = self.experiments.get(key2)

            if not exp1 or not exp2:
                continue

            # Find common subjects with complete data
            common = set(exp1.folds.keys()) & set(exp2.folds.keys())
            common = [s for s in common if exp1.folds[s].is_complete() and exp2.folds[s].is_complete()]

            if len(common) < 3:
                lines.append(f"\n{desc}: Insufficient paired data ({len(common)} pairs)")
                continue

            v1 = np.array([exp1.folds[s].test_f1 for s in common])
            v2 = np.array([exp2.folds[s].test_f1 for s in common])

            mask = ~(np.isnan(v1) | np.isnan(v2))
            v1, v2 = v1[mask], v2[mask]

            if len(v1) < 3:
                continue

            t_stat, t_pval = stats.ttest_rel(v1, v2)
            diff = v2 - v1
            cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff) > 0 else 0

            sig = ""
            if t_pval < 0.001:
                sig = "***"
            elif t_pval < 0.01:
                sig = "**"
            elif t_pval < 0.05:
                sig = "*"

            winner = key2.upper() if np.mean(diff) > 0 else key1.upper()
            lines.append(f"\n{desc}:")
            lines.append(f"  Δ={np.mean(diff):+.2f}% → {winner} {sig}")
            lines.append(f"  p={t_pval:.4f}, Cohen's d={cohens_d:.3f}, n={len(v1)}")

        return "\n".join(lines)

    def generate_summary_table(self) -> str:
        """Generate publication-ready summary table."""
        lines = []
        lines.append("\n" + "=" * 100)
        lines.append("SUMMARY TABLE (Publication Ready)")
        lines.append("=" * 100)
        lines.append("")
        lines.append(f"{'Model':<6} {'Description':<25} {'Folds':<6} "
                    f"{'Test Acc':<14} {'Test F1':<14} {'Val Acc':<14} {'Val F1':<14}")
        lines.append("-" * 100)

        for exp_key in ['model_a', 'model_b', 'model_c', 'model_d', 'model_e', 'model_f']:
            exp = self.experiments.get(exp_key)
            if not exp:
                continue

            n_folds = exp.n_complete_folds()
            va_mean, va_std = exp.compute_stats('val_acc')
            vf_mean, vf_std = exp.compute_stats('val_f1')
            ta_mean, ta_std = exp.compute_stats('test_acc')
            tf_mean, tf_std = exp.compute_stats('test_f1')

            def fmt(m, s):
                return f"{m:.2f}±{s:.2f}" if not np.isnan(m) else "N/A"

            lines.append(
                f"{exp_key.upper():<6} {exp.config.display_name:<25} {n_folds:<6} "
                f"{fmt(ta_mean, ta_std):<14} {fmt(tf_mean, tf_std):<14} "
                f"{fmt(va_mean, va_std):<14} {fmt(vf_mean, vf_std):<14}"
            )

        lines.append("-" * 100)

        # Best configuration
        lines.append("\nRanking by Test F1:")
        ranked = []
        for exp_key, exp in self.experiments.items():
            m, s = exp.compute_stats('test_f1')
            if not np.isnan(m):
                ranked.append((exp_key, exp.config.short_name, m, s))

        ranked.sort(key=lambda x: x[2], reverse=True)
        for i, (key, name, mean, std) in enumerate(ranked, 1):
            lines.append(f"  {i}. {key.upper()} ({name}): {mean:.2f} ± {std:.2f}%")

        return "\n".join(lines)

    def generate_per_fold_csv(self) -> pd.DataFrame:
        """Generate per-fold comparison CSV."""
        rows = []

        for subject in self.all_subjects:
            row = {'fold': subject}

            for exp_key in EXPERIMENTS:
                exp = self.experiments.get(exp_key)
                prefix = exp_key.upper()

                if exp and subject in exp.folds:
                    fold = exp.folds[subject]
                    row[f'{prefix}_val_acc'] = fold.val_acc
                    row[f'{prefix}_val_f1'] = fold.val_f1
                    row[f'{prefix}_test_acc'] = fold.test_acc
                    row[f'{prefix}_test_f1'] = fold.test_f1
                else:
                    row[f'{prefix}_val_acc'] = None
                    row[f'{prefix}_val_f1'] = None
                    row[f'{prefix}_test_acc'] = None
                    row[f'{prefix}_test_f1'] = None

            rows.append(row)

        # Add average row
        avg_row = {'fold': 'AVERAGE'}
        for exp_key in EXPERIMENTS:
            exp = self.experiments.get(exp_key)
            prefix = exp_key.upper()

            if exp:
                for metric in ['val_acc', 'val_f1', 'test_acc', 'test_f1']:
                    m, _ = exp.compute_stats(metric)
                    avg_row[f'{prefix}_{metric}'] = m

        rows.append(avg_row)
        return pd.DataFrame(rows)

    def save_results(self) -> None:
        """Save all aggregated results to files."""
        print("\n" + "=" * 80)
        print("SAVING RESULTS")
        print("=" * 80)

        # Check if we have any data
        total_folds = sum(exp.n_complete_folds() + exp.n_partial_folds()
                         for exp in self.experiments.values())
        if total_folds == 0:
            print("No data found to save.")
            return

        # Generate reports
        ablation_table = self.generate_ablation_table()
        effect_analysis = self.generate_effect_analysis()
        stats_tests = self.perform_statistical_tests()
        summary_table = self.generate_summary_table()

        full_report = "\n\n".join([
            "=" * 80,
            "STREAM-KALMAN ABLATION STUDY - ANALYSIS REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Results Dir: {self.results_dir}",
            "",
            ablation_table,
            effect_analysis,
            stats_tests,
            summary_table
        ])

        # Save report
        report_path = self.output_dir / 'ablation_report.txt'
        with open(report_path, 'w') as f:
            f.write(full_report)
        print(f"[OK] Saved: {report_path}")

        # Save per-fold CSV
        df = self.generate_per_fold_csv()
        csv_path = self.output_dir / 'per_fold_comparison.csv'
        df.to_csv(csv_path, index=False, float_format='%.2f')
        print(f"[OK] Saved: {csv_path}")

        # Save JSON summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'results_dir': str(self.results_dir),
            'experiments': {}
        }

        for exp_key, exp in self.experiments.items():
            summary['experiments'][exp_key] = {
                'display_name': exp.config.display_name,
                'model_class': exp.config.model_class,
                'stream_type': exp.config.stream_type,
                'kalman': exp.config.kalman,
                'se_tap': exp.config.se_tap,
                'status': exp.status,
                'n_complete_folds': exp.n_complete_folds(),
                'n_partial_folds': exp.n_partial_folds(),
                'metrics': {
                    'val_acc': {'mean': exp.compute_stats('val_acc')[0], 'std': exp.compute_stats('val_acc')[1]},
                    'val_f1': {'mean': exp.compute_stats('val_f1')[0], 'std': exp.compute_stats('val_f1')[1]},
                    'test_acc': {'mean': exp.compute_stats('test_acc')[0], 'std': exp.compute_stats('test_acc')[1]},
                    'test_f1': {'mean': exp.compute_stats('test_f1')[0], 'std': exp.compute_stats('test_f1')[1]},
                }
            }

        def nan_to_none(obj):
            if isinstance(obj, float) and np.isnan(obj):
                return None
            return obj

        json_path = self.output_dir / 'summary.json'
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2, default=nan_to_none)
        print(f"[OK] Saved: {json_path}")

        # Save model comparison CSV
        model_comparison_path = self.output_dir / 'model_comparison.csv'
        model_rows = []
        for exp_key in EXPERIMENTS:
            exp = self.experiments.get(exp_key)
            if not exp:
                continue
            ta_mean, ta_std = exp.compute_stats('test_acc')
            tf_mean, tf_std = exp.compute_stats('test_f1')
            model_rows.append({
                'model': exp_key.upper(),
                'description': exp.config.display_name,
                'stream_type': exp.config.stream_type,
                'kalman': exp.config.kalman,
                'se_tap': exp.config.se_tap,
                'n_folds': exp.n_complete_folds(),
                'test_acc_mean': ta_mean,
                'test_acc_std': ta_std,
                'test_f1_mean': tf_mean,
                'test_f1_std': tf_std,
            })
        pd.DataFrame(model_rows).to_csv(model_comparison_path, index=False, float_format='%.4f')
        print(f"[OK] Saved: {model_comparison_path}")

        # Print report
        print("\n")
        print(full_report)

    def run(self) -> None:
        """Run the full aggregation pipeline."""
        print("\n" + "=" * 80)
        print("STREAM-KALMAN ABLATION STUDY AGGREGATOR")
        print("=" * 80)
        print(f"Results Directory: {self.results_dir}")
        print(f"Output Directory: {self.output_dir}")

        self.load_all_experiments()
        self.save_results()

        print("\n" + "=" * 80)
        print("AGGREGATION COMPLETE")
        print("=" * 80)
        print(f"\nAll outputs saved to: {self.output_dir}")


def main():
    if len(sys.argv) < 2:
        # Auto-detect latest stream_kalman_ablation directory
        results_dirs = sorted(
            Path('.').glob('results/stream_kalman_ablation_*'),
            key=os.path.getmtime,
            reverse=True
        )

        if results_dirs:
            results_dir = str(results_dirs[0])
            print(f"Auto-detected results directory: {results_dir}")
        else:
            print("Usage: python aggregate_stream_kalman_ablation.py <results_dir>")
            print("\nNo stream_kalman_ablation directories found in ./results/")
            sys.exit(1)
    else:
        results_dir = sys.argv[1]

    if not os.path.exists(results_dir):
        print(f"ERROR: Results directory not found: {results_dir}")
        sys.exit(1)

    aggregator = StreamKalmanAggregator(results_dir)
    aggregator.run()


if __name__ == "__main__":
    main()
