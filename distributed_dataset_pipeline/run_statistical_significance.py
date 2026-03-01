#!/usr/bin/env python3
"""Paired statistical significance tests: Kalman vs Raw across LOSO folds.

Tests: paired t-test, Wilcoxon signed-rank, Nadeau & Bengio (2003) corrected t-test.
Reports p-values, Cohen's d effect size, 95% CI, per-fold comparison.

Usage:
    # Train both configs per dataset and run tests
    python distributed_dataset_pipeline/run_statistical_significance.py --num-gpus 8

    # Load existing results (no training)
    python distributed_dataset_pipeline/run_statistical_significance.py --results-only \\
        --smartfallmm-kalman exps/run1 --smartfallmm-raw exps/run2

    # Single dataset, quick test
    python distributed_dataset_pipeline/run_statistical_significance.py --datasets upfall --quick
"""

import argparse
import json
import pickle
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy import stats

DATASETS = {
    'smartfallmm': {
        'name': 'SmartFallMM',
        'kalman_config': 'config/best_config/smartfallmm/kalman.yaml',
        'raw_config': 'config/best_config/smartfallmm/raw.yaml',
        'num_folds': 22,
    },
    'upfall': {
        'name': 'UP-FALL',
        'kalman_config': 'config/best_config/upfall/kalman.yaml',
        'raw_config': 'config/best_config/upfall/raw.yaml',
        'num_folds': 15,
    },
    'wedafall': {
        'name': 'WEDA-FALL',
        'kalman_config': 'config/best_config/wedafall/kalman.yaml',
        'raw_config': 'config/best_config/wedafall/raw.yaml',
        'num_folds': 12,
    },
}

METRICS = ['f1_score', 'accuracy', 'precision', 'recall', 'auc']
METRIC_LABELS = {'f1_score': 'F1', 'accuracy': 'Acc', 'precision': 'Prec',
                 'recall': 'Rec', 'auc': 'AUC'}


@dataclass
class PairedTestResult:
    dataset: str
    metric: str
    n_pairs: int
    kalman_mean: float
    kalman_std: float
    raw_mean: float
    raw_std: float
    mean_diff: float
    std_diff: float
    t_stat: float
    t_pvalue: float
    w_stat: float
    w_pvalue: float
    nb_t_stat: float
    nb_pvalue: float
    cohens_d: float
    ci_lower: float
    ci_upper: float
    fold_diffs: List[float] = field(default_factory=list)


def load_fold_results(pkl_path: Path) -> Dict[str, Dict[str, float]]:
    """Load fold_results.pkl → {subject: {metric: value}}."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    fold_list = list(data.values()) if isinstance(data, dict) else data
    results = {}

    for fold in fold_list:
        if not isinstance(fold, dict) or fold.get('status') == 'failed':
            continue
        subject = str(fold.get('test_subject', ''))
        if not subject:
            continue

        test = fold.get('test', {})
        if not isinstance(test, dict):
            continue

        metrics = {}
        for m in METRICS:
            val = test.get(m, test.get(m.replace('_score', ''), 0))
            if val and val <= 1:
                val *= 100
            metrics[m] = float(val) if val else 0.0

        # n_test / n_train for Nadeau-Bengio correction
        ta = fold.get('threshold_analysis', {})
        metrics['n_test'] = ta.get('n_samples', len(ta.get('targets', [])))
        metrics['n_train'] = fold.get('fall_windows', 0) + fold.get('adl_windows', 0)

        results[subject] = metrics

    return results


def paired_statistical_tests(
    kalman: Dict[str, Dict], raw: Dict[str, Dict], metric: str, dataset_name: str,
) -> PairedTestResult:
    """Run paired t-test, Wilcoxon, and Nadeau-Bengio corrected t-test."""
    common = sorted(set(kalman.keys()) & set(raw.keys()))
    if len(common) < 3:
        raise ValueError(f"Need >= 3 paired folds, got {len(common)}")

    k_vals = np.array([kalman[s][metric] for s in common])
    r_vals = np.array([raw[s][metric] for s in common])
    diffs = k_vals - r_vals
    k = len(diffs)

    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))

    # 1. Standard paired t-test
    t_stat, t_pval = stats.ttest_rel(k_vals, r_vals)

    # 2. Wilcoxon signed-rank test
    nonzero = diffs[diffs != 0]
    if len(nonzero) >= 6:
        w_stat, w_pval = stats.wilcoxon(k_vals, r_vals)
    else:
        w_stat, w_pval = float('nan'), float('nan')

    # 3. Nadeau & Bengio (2003) corrected t-test
    # Corrects for non-independence of CV folds (overlapping training sets)
    # SE = sqrt((1/k + n_test/n_train) * var_diff)
    n_tests = [kalman[s].get('n_test', 0) for s in common]
    n_trains = [kalman[s].get('n_train', 0) for s in common]

    if sum(n_trains) > 0 and sum(n_tests) > 0:
        ratio = float(np.mean(
            np.array(n_tests) / np.array([max(n, 1) for n in n_trains])
        ))
    else:
        ratio = 1.0 / (k - 1)  # fallback: 1/(k-1) for LOSO

    var_diff = float(np.var(diffs, ddof=1))
    nb_se = np.sqrt((1.0 / k + ratio) * var_diff) if var_diff > 0 else 1e-10
    nb_t_stat = mean_diff / nb_se
    nb_pval = float(2 * stats.t.sf(abs(nb_t_stat), df=k - 1))

    # Cohen's d (paired)
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0

    # 95% CI on mean difference
    se = std_diff / np.sqrt(k) if std_diff > 0 else 1e-10
    t_crit = stats.t.ppf(0.975, df=k - 1)
    ci_lo = mean_diff - t_crit * se
    ci_hi = mean_diff + t_crit * se

    return PairedTestResult(
        dataset=dataset_name, metric=metric, n_pairs=k,
        kalman_mean=float(np.mean(k_vals)), kalman_std=float(np.std(k_vals, ddof=1)),
        raw_mean=float(np.mean(r_vals)), raw_std=float(np.std(r_vals, ddof=1)),
        mean_diff=mean_diff, std_diff=std_diff,
        t_stat=float(t_stat), t_pvalue=float(t_pval),
        w_stat=float(w_stat), w_pvalue=float(w_pval),
        nb_t_stat=float(nb_t_stat), nb_pvalue=float(nb_pval),
        cohens_d=float(cohens_d),
        ci_lower=float(ci_lo), ci_upper=float(ci_hi),
        fold_diffs=diffs.tolist(),
    )


def sig_marker(p: float) -> str:
    if np.isnan(p): return ''
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return ''


def fmt_p(p: float) -> str:
    if np.isnan(p): return 'N/A'
    if p < 0.001: return f'{p:.1e}'
    return f'{p:.4f}'


def effect_label(d: float) -> str:
    d = abs(d)
    if d >= 0.8: return 'large'
    if d >= 0.5: return 'medium'
    if d >= 0.2: return 'small'
    return 'negligible'


def run_training(config: str, work_dir: Path, num_gpus: int,
                 max_folds: Optional[int] = None, timeout: int = 10800) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, 'ray_train.py', '--config', config,
           '--work-dir', str(work_dir), '--num-gpus', str(num_gpus)]
    if max_folds:
        cmd.extend(['--max-folds', str(max_folds)])

    print(f'  Training: {" ".join(cmd)}')
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout, cwd=str(Path(__file__).resolve().parent.parent))
    if proc.returncode != 0:
        raise RuntimeError(f'Training failed:\n{proc.stderr[-1000:]}')

    pkl = work_dir / 'fold_results.pkl'
    if not pkl.exists():
        raise FileNotFoundError(f'No fold_results.pkl at {pkl}')
    return pkl


def generate_report(all_results: Dict[str, List[PairedTestResult]],
                    per_fold: Dict[str, Dict],
                    output_dir: Path) -> str:
    lines = [
        '# Statistical Significance: Kalman vs Raw Input',
        f'\nGenerated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
        '',
        '## Summary (F1 Score)',
        '',
        '| Dataset | N | Kalman F1 | Raw F1 | ΔF1 | p (t-test) | p (Wilcoxon) | p (NB-corrected) | Cohen\'s d | 95% CI |',
        '|---------|---|-----------|--------|-----|------------|--------------|-------------------|-----------|--------|',
    ]

    for ds_key, results in all_results.items():
        f1 = next((r for r in results if r.metric == 'f1_score'), None)
        if not f1:
            continue
        lines.append(
            f'| {f1.dataset} | {f1.n_pairs} '
            f'| {f1.kalman_mean:.2f} ± {f1.kalman_std:.2f} '
            f'| {f1.raw_mean:.2f} ± {f1.raw_std:.2f} '
            f'| +{f1.mean_diff:.2f} '
            f'| {fmt_p(f1.t_pvalue)}{sig_marker(f1.t_pvalue)} '
            f'| {fmt_p(f1.w_pvalue)}{sig_marker(f1.w_pvalue)} '
            f'| {fmt_p(f1.nb_pvalue)}{sig_marker(f1.nb_pvalue)} '
            f'| {f1.cohens_d:.3f} ({effect_label(f1.cohens_d)}) '
            f'| [{f1.ci_lower:.2f}, {f1.ci_upper:.2f}] |'
        )

    # Per-dataset detail
    for ds_key, results in all_results.items():
        ds_name = results[0].dataset if results else ds_key
        lines.extend(['', f'---', '', f'## {ds_name}', ''])

        # All metrics table
        lines.extend([
            '### All Metrics',
            '',
            '| Metric | Kalman | Raw | Δ | p (t-test) | p (Wilcoxon) | p (NB) | Cohen\'s d |',
            '|--------|--------|-----|---|------------|--------------|--------|-----------|',
        ])
        for r in results:
            label = METRIC_LABELS.get(r.metric, r.metric)
            lines.append(
                f'| {label} '
                f'| {r.kalman_mean:.2f} ± {r.kalman_std:.2f} '
                f'| {r.raw_mean:.2f} ± {r.raw_std:.2f} '
                f'| {r.mean_diff:+.2f} '
                f'| {fmt_p(r.t_pvalue)}{sig_marker(r.t_pvalue)} '
                f'| {fmt_p(r.w_pvalue)}{sig_marker(r.w_pvalue)} '
                f'| {fmt_p(r.nb_pvalue)}{sig_marker(r.nb_pvalue)} '
                f'| {r.cohens_d:.3f} |'
            )

        # Per-fold F1 table
        if ds_key in per_fold:
            pf = per_fold[ds_key]
            lines.extend(['', '### Per-Fold F1 Comparison', '',
                          '| Subject | Kalman F1 | Raw F1 | Δ |',
                          '|---------|-----------|--------|---|'])
            for subj in sorted(pf.keys()):
                kf = pf[subj]['kalman']
                rf = pf[subj]['raw']
                lines.append(f'| {subj} | {kf:.2f} | {rf:.2f} | {kf - rf:+.2f} |')

    # Methodology
    lines.extend([
        '', '---', '',
        '## Methodology',
        '',
        '- **Paired t-test**: Tests mean difference of per-fold metrics. Assumes normality of differences.',
        '- **Wilcoxon signed-rank**: Non-parametric alternative. No normality assumption. '
        'Low power at n < 20; minimum p ≈ 0.002 at n=12.',
        '- **Nadeau-Bengio corrected t-test** (2003): Corrects for non-independence of LOSO folds '
        '(training sets overlap ~95%). Uses SE = sqrt((1/k + n_test/n_train) × var_diff). '
        'More conservative than standard paired t-test.',
        '- **Cohen\'s d**: Effect size. 0.2 = small, 0.5 = medium, 0.8 = large.',
        '- **95% CI**: Confidence interval on the mean difference (t-distribution).',
        '- Significance: \\* p<0.05, \\*\\* p<0.01, \\*\\*\\* p<0.001',
    ])

    report = '\n'.join(lines)
    report_path = output_dir / 'significance_report.md'
    report_path.write_text(report)
    print(f'\nReport: {report_path}')
    return report


def main():
    parser = argparse.ArgumentParser(
        description='Paired statistical tests: Kalman vs Raw across LOSO folds')
    parser.add_argument('--datasets', nargs='+', default=['smartfallmm', 'upfall', 'wedafall'],
                        choices=['smartfallmm', 'upfall', 'wedafall'])
    parser.add_argument('--num-gpus', type=int, default=8)
    parser.add_argument('--max-folds', type=int, default=None)
    parser.add_argument('--quick', action='store_true', help='2 folds only')
    parser.add_argument('--work-dir', type=Path, default=None)
    parser.add_argument('--results-only', action='store_true',
                        help='Load existing fold_results.pkl, skip training')

    for ds in ['smartfallmm', 'upfall', 'wedafall']:
        parser.add_argument(f'--{ds}-kalman', type=Path, default=None,
                            help=f'Existing {ds} Kalman results dir (with fold_results.pkl)')
        parser.add_argument(f'--{ds}-raw', type=Path, default=None,
                            help=f'Existing {ds} Raw results dir (with fold_results.pkl)')

    args = parser.parse_args()
    if args.quick:
        args.max_folds = 2
    if args.work_dir is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.work_dir = Path(f'exps/statistical_significance_{ts}')
    args.work_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    per_fold_data = {}

    for ds_key in args.datasets:
        ds = DATASETS[ds_key]
        print(f'\n{"=" * 60}')
        print(f'  {ds["name"]}')
        print(f'{"=" * 60}')

        # Get fold_results.pkl paths
        kalman_dir = getattr(args, f'{ds_key}_kalman', None)
        raw_dir = getattr(args, f'{ds_key}_raw', None)

        if args.results_only:
            if not kalman_dir or not raw_dir:
                print(f'  Skipping {ds_key}: --{ds_key}-kalman and --{ds_key}-raw required')
                continue
            kalman_pkl = kalman_dir / 'fold_results.pkl'
            raw_pkl = raw_dir / 'fold_results.pkl'
        else:
            kalman_pkl = (kalman_dir / 'fold_results.pkl') if kalman_dir else None
            raw_pkl = (raw_dir / 'fold_results.pkl') if raw_dir else None

            if not kalman_pkl or not kalman_pkl.exists():
                print(f'  Training Kalman ({ds["kalman_config"]})...')
                run_dir = args.work_dir / 'runs' / f'{ds_key}_kalman'
                kalman_pkl = run_training(
                    ds['kalman_config'], run_dir, args.num_gpus, args.max_folds)

            if not raw_pkl or not raw_pkl.exists():
                print(f'  Training Raw ({ds["raw_config"]})...')
                run_dir = args.work_dir / 'runs' / f'{ds_key}_raw'
                raw_pkl = run_training(
                    ds['raw_config'], run_dir, args.num_gpus, args.max_folds)

        if not kalman_pkl.exists():
            print(f'  ERROR: {kalman_pkl} not found')
            continue
        if not raw_pkl.exists():
            print(f'  ERROR: {raw_pkl} not found')
            continue

        # Load and pair by subject
        kalman_folds = load_fold_results(kalman_pkl)
        raw_folds = load_fold_results(raw_pkl)
        common = set(kalman_folds.keys()) & set(raw_folds.keys())
        print(f'  Loaded: {len(kalman_folds)} Kalman folds, {len(raw_folds)} Raw folds, '
              f'{len(common)} paired')

        if len(common) < 3:
            print(f'  Skipping: need >= 3 paired folds')
            continue

        # Store per-fold F1 for detail table
        per_fold_data[ds_key] = {
            s: {'kalman': kalman_folds[s]['f1_score'], 'raw': raw_folds[s]['f1_score']}
            for s in sorted(common)
        }

        # Run tests for each metric
        results = []
        for metric in METRICS:
            try:
                r = paired_statistical_tests(kalman_folds, raw_folds, metric, ds['name'])
                results.append(r)
                label = METRIC_LABELS.get(metric, metric)
                sig = sig_marker(r.t_pvalue)
                print(f'  {label:>5}: Kalman={r.kalman_mean:.2f} Raw={r.raw_mean:.2f} '
                      f'Δ={r.mean_diff:+.2f}  p(t)={fmt_p(r.t_pvalue)}{sig}  '
                      f'p(W)={fmt_p(r.w_pvalue)}{sig_marker(r.w_pvalue)}  '
                      f'd={r.cohens_d:.3f}')
            except Exception as e:
                print(f'  {metric}: ERROR - {e}')

        all_results[ds_key] = results

    if not all_results:
        print('\nNo results to report.')
        return

    # Generate outputs
    generate_report(all_results, per_fold_data, args.work_dir)

    # JSON summary
    json_data = {}
    for ds_key, results in all_results.items():
        json_data[ds_key] = [asdict(r) for r in results]
    json_path = args.work_dir / 'significance_results.json'
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f'JSON:   {json_path}')


if __name__ == '__main__':
    main()
