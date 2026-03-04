#!/usr/bin/env python3
"""Statistical significance: dual-stream (proposed) vs single-stream (baseline).

Paired comparison across 22 LOSO-CV folds on SmartFallMM.
Seven tests + effect sizes + bootstrap CI + Bayesian ROPE analysis.

Usage:
    python tools/significance_table2.py
    python tools/significance_table2.py --proposed PATH --baseline PATH
    python tools/significance_table2.py --latex
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
from scipy import stats

# Default paths (matching paper-reported values: 91.87% and 92.86%)
DEFAULT_PROPOSED = (
    "exps/capacity_ablation_nonorm_20260127_065530/runs/"
    "stream_dual_kalman_stride_8_64_nonorm/fold_results.pkl"
)
DEFAULT_BASELINE = (
    "exps/capacity_ablation_nonorm_20260127_065530/runs/"
    "stream_single_raw_stride_8_64_nonorm/fold_results.pkl"
)

METRICS = ['f1_score', 'accuracy', 'precision', 'recall']
N_BOOTSTRAP = 10000
N_PERMUTATIONS = 10000
ROPE = 1.0  # region of practical equivalence: ±1 pp
RNG_SEED = 42


def load_folds(path):
    with open(path, 'rb') as f:
        folds = pickle.load(f)
    out = {}
    for fold in folds:
        if fold.get('status') != 'success':
            continue
        subj = str(fold['test_subject'])
        test = fold['test']
        out[subj] = {
            m: float(test.get(m, test.get(m.replace('_score', ''), 0)))
            for m in METRICS
        }
        ta = fold.get('threshold_analysis', {})
        out[subj]['n_test'] = ta.get('n_samples', len(ta.get('targets', [])))
        out[subj]['n_train'] = fold.get('fall_windows', 0) + fold.get('adl_windows', 0)
    return out


def bootstrap_ci(diffs, n_boot=N_BOOTSTRAP, alpha=0.05):
    """BCa-like percentile bootstrap 95% CI on mean difference."""
    rng = np.random.RandomState(RNG_SEED)
    k = len(diffs)
    boot_means = np.array([
        np.mean(rng.choice(diffs, size=k, replace=True))
        for _ in range(n_boot)
    ])
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(lo), float(hi), boot_means


def permutation_test(p_vals, b_vals, n_perm=N_PERMUTATIONS):
    """Two-sided permutation test: randomly swap labels within each pair."""
    rng = np.random.RandomState(RNG_SEED)
    k = len(p_vals)
    obs_diff = np.mean(p_vals - b_vals)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=k)
        diffs = p_vals - b_vals
        perm_diff = np.mean(signs * diffs)
        if abs(perm_diff) >= abs(obs_diff):
            count += 1
    return count / n_perm


def sign_test(diffs):
    """Sign test: binomial test on number of positive vs negative differences."""
    pos = np.sum(diffs > 0)
    neg = np.sum(diffs < 0)
    n = pos + neg
    if n == 0:
        return float('nan')
    return float(stats.binom_test(min(pos, neg), n, 0.5))


def bayesian_rope(diffs, rope=ROPE):
    """Bayesian ROPE analysis using posterior from t-distribution.

    Estimates P(proposed > baseline), P(equivalent), P(baseline > proposed)
    using a conjugate Bayesian model (Kruschke, 2013).
    ROPE = region of practical equivalence (±rope pp).
    """
    k = len(diffs)
    mean_d = np.mean(diffs)
    std_d = np.std(diffs, ddof=1)
    se = std_d / np.sqrt(k)

    # Posterior is approximately t-distributed: t(df=k-1, loc=mean_d, scale=se)
    df = k - 1
    if se < 1e-10:
        return {'p_proposed': 0.5, 'p_rope': 1.0, 'p_baseline': 0.0}

    dist = stats.t(df=df, loc=mean_d, scale=se)

    p_baseline = float(dist.cdf(-rope))  # P(diff < -rope)
    p_rope = float(dist.cdf(rope) - dist.cdf(-rope))  # P(-rope <= diff <= rope)
    p_proposed = float(1 - dist.cdf(rope))  # P(diff > rope)

    return {'p_proposed': p_proposed, 'p_rope': p_rope, 'p_baseline': p_baseline}


def hedges_g(diffs, k):
    """Hedge's g: bias-corrected Cohen's d for small samples."""
    d = np.mean(diffs) / np.std(diffs, ddof=1) if np.std(diffs, ddof=1) > 0 else 0.0
    correction = 1 - 3 / (4 * (k - 1) - 1)
    return float(d * correction)


def run_tests(proposed, baseline, metric, rope_width=ROPE):
    """Run all seven statistical tests on a single metric."""
    common = sorted(set(proposed) & set(baseline))
    k = len(common)
    assert k >= 6, f"Need >= 6 paired folds, got {k}"

    p_vals = np.array([proposed[s][metric] for s in common])
    b_vals = np.array([baseline[s][metric] for s in common])
    diffs = p_vals - b_vals

    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)

    # 1. Paired t-test
    _, t_p = stats.ttest_rel(p_vals, b_vals)

    # 2. Wilcoxon signed-rank
    nonzero = diffs[diffs != 0]
    if len(nonzero) >= 6:
        _, w_p = stats.wilcoxon(p_vals, b_vals)
    else:
        w_p = float('nan')

    # 3. Nadeau & Bengio (2003) corrected t-test
    n_tests = np.array([proposed[s].get('n_test', 0) for s in common], dtype=float)
    n_trains = np.array([proposed[s].get('n_train', 1) for s in common], dtype=float)
    n_trains = np.maximum(n_trains, 1)
    if n_tests.sum() > 0 and n_trains.sum() > 0:
        ratio = float(np.mean(n_tests / n_trains))
    else:
        ratio = 1.0 / (k - 1)
    var_diff = np.var(diffs, ddof=1)
    nb_se = np.sqrt((1.0 / k + ratio) * var_diff) if var_diff > 0 else 1e-10
    nb_t = mean_diff / nb_se
    nb_p = float(2 * stats.t.sf(abs(nb_t), df=k - 1))

    # 4. Permutation test
    perm_p = permutation_test(p_vals, b_vals)

    # 5. Sign test
    sign_p = sign_test(diffs)

    # 6. Bootstrap CI
    boot_lo, boot_hi, boot_means = bootstrap_ci(diffs)

    # 7. Bayesian ROPE
    rope = bayesian_rope(diffs, rope=rope_width)

    # Effect sizes
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0
    hg = hedges_g(diffs, k)

    # Parametric 95% CI
    se = std_diff / np.sqrt(k) if std_diff > 0 else 1e-10
    t_crit = stats.t.ppf(0.975, df=k - 1)

    return {
        'metric': metric,
        'n_folds': k,
        'proposed_mean': float(np.mean(p_vals)),
        'proposed_std': float(np.std(p_vals, ddof=1)),
        'baseline_mean': float(np.mean(b_vals)),
        'baseline_std': float(np.std(b_vals, ddof=1)),
        'mean_diff': float(mean_diff),
        'std_diff': float(std_diff),
        # p-values
        't_pval': float(t_p),
        'w_pval': float(w_p),
        'nb_pval': float(nb_p),
        'perm_pval': float(perm_p),
        'sign_pval': float(sign_p),
        # effect sizes
        'cohens_d': float(cohens_d),
        'hedges_g': hg,
        # CIs
        'ci_lo': float(mean_diff - t_crit * se),
        'ci_hi': float(mean_diff + t_crit * se),
        'boot_ci_lo': boot_lo,
        'boot_ci_hi': boot_hi,
        # Bayesian ROPE
        'rope': rope_width,
        'p_proposed': rope['p_proposed'],
        'p_rope': rope['p_rope'],
        'p_baseline': rope['p_baseline'],
        # metadata
        'nb_ratio': ratio,
        'fold_diffs': diffs.tolist(),
        'n_proposed_wins': int(np.sum(diffs > 0)),
        'n_baseline_wins': int(np.sum(diffs < 0)),
    }


def fmt_p(p):
    if np.isnan(p):
        return 'N/A'
    if p < 0.001:
        return f'{p:.1e}'
    return f'{p:.2f}'


def effect_label(d):
    d = abs(d)
    if d >= 0.8: return 'large'
    if d >= 0.5: return 'medium'
    if d >= 0.2: return 'small'
    return 'negligible'


def print_report(results, proposed_label, baseline_label):
    print(f"\n{'='*78}")
    print(f"Statistical Significance: {proposed_label} vs {baseline_label}")
    print(f"{'='*78}\n")

    for r in results:
        m = r['metric'].replace('_score', '').replace('_', ' ').title()
        print(f"--- {m} ---")
        print(f"  {proposed_label:25s}: {r['proposed_mean']:.2f} +/- {r['proposed_std']:.2f}")
        print(f"  {baseline_label:25s}: {r['baseline_mean']:.2f} +/- {r['baseline_std']:.2f}")
        print(f"  Mean difference:          {r['mean_diff']:+.2f}")
        print(f"  Parametric 95% CI:        [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]")
        print(f"  Bootstrap 95% CI:         [{r['boot_ci_lo']:.2f}, {r['boot_ci_hi']:.2f}]  (n={N_BOOTSTRAP})")
        print()
        print(f"  --- Frequentist Tests ---")
        print(f"  Paired t-test:            p = {fmt_p(r['t_pval'])}")
        print(f"  Wilcoxon signed-rank:     p = {fmt_p(r['w_pval'])}")
        print(f"  Nadeau-Bengio corrected:  p = {fmt_p(r['nb_pval'])}  (n_test/n_train={r['nb_ratio']:.4f})")
        print(f"  Permutation test:         p = {fmt_p(r['perm_pval'])}  (n={N_PERMUTATIONS})")
        print(f"  Sign test:                p = {fmt_p(r['sign_pval'])}")
        print()
        print(f"  --- Effect Sizes ---")
        print(f"  Cohen's d:                {r['cohens_d']:.3f} ({effect_label(r['cohens_d'])})")
        print(f"  Hedge's g (corrected):    {r['hedges_g']:.3f} ({effect_label(r['hedges_g'])})")
        print()
        print(f"  --- Bayesian ROPE (+/-{r['rope']} pp) ---")
        print(f"  P(proposed better):       {r['p_proposed']:.3f}")
        print(f"  P(practically equivalent):{r['p_rope']:.3f}")
        print(f"  P(baseline better):       {r['p_baseline']:.3f}")
        print()
        print(f"  Fold wins: proposed={r['n_proposed_wins']}, "
              f"baseline={r['n_baseline_wins']}, "
              f"tied={r['n_folds'] - r['n_proposed_wins'] - r['n_baseline_wins']}")
        print()

    # Per-fold detail
    f1 = next(r for r in results if r['metric'] == 'f1_score')
    diffs = sorted(f1['fold_diffs'])
    print("Per-fold F1 differences (proposed - baseline):")
    print(f"  Range: [{min(diffs):.2f}, {max(diffs):.2f}]")
    print(f"  Median: {np.median(diffs):.2f}")
    print(f"  IQR: [{np.percentile(diffs, 25):.2f}, {np.percentile(diffs, 75):.2f}]")
    print()


def print_latex(results):
    # Table 1: Compact (advisor format)
    print(r"\begin{table}[!htb]")
    print(r"\caption{Statistical significance analysis for the proposed dual-stream "
          r"architecture ($\mathcal{M}_{\tau}$) vs.\ the best single-stream baseline "
          r"($M_{SE+TAP}$) across 22 LOSO-CV folds.}")
    print(r"\label{tab:significance_stream}")
    print(r"  \centering")
    print(r"  \begin{tabular}{lccc}")
    print(r"  \toprule")
    print(r"  \textbf{Metric} & \textbf{Paired $t$-test} & \textbf{Wilcoxon} "
          r"& \textbf{Nadeau--Bengio} \\")
    print(r"  \midrule")
    for r in results:
        if r['metric'] not in ('f1_score', 'accuracy'):
            continue
        m = 'F1 Score' if r['metric'] == 'f1_score' else 'Accuracy'
        print(f"  {m} & $p = {fmt_p(r['t_pval'])}$ "
              f"& $p = {fmt_p(r['w_pval'])}$ "
              f"& $p = {fmt_p(r['nb_pval'])}$ \\\\")
    print(r"  \bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()

    # Table 2: Comprehensive (all 5 frequentist tests + effect sizes)
    print(r"% Comprehensive table with all tests:")
    print(r"\begin{table*}[!htb]")
    print(r"\caption{Comprehensive statistical comparison of $\mathcal{M}_{\tau}$ "
          r"(dual-stream Kalman, F1~=~91.87\%) vs.\ $M_{SE+TAP}$ "
          r"(single-stream raw, F1~=~92.86\%) across 22 LOSO-CV folds. "
          r"$\Delta$ = proposed $-$ baseline. "
          r"No test reaches significance at $\alpha = 0.05$.}")
    print(r"\label{tab:significance_comprehensive}")
    print(r"  \centering")
    print(r"  \small")
    print(r"  \begin{tabular}{lcccccccc}")
    print(r"  \toprule")
    print(r"  \textbf{Metric} & \textbf{$\Delta$} & \textbf{95\% CI}")
    print(r"  & \textbf{Paired $t$} & \textbf{Wilcoxon} & \textbf{N--B}")
    print(r"  & \textbf{Permutation} & \textbf{Sign}")
    print(r"  & \textbf{Hedge's $g$} \\")
    print(r"  \midrule")
    for r in results:
        m = r['metric'].replace('_score', '').replace('_', ' ').title()
        delta = f"{r['mean_diff']:+.2f}"
        ci = f"[{r['boot_ci_lo']:.1f}, {r['boot_ci_hi']:.1f}]"
        hg = f"{r['hedges_g']:.2f}"
        print(f"  {m} & {delta} & {ci} "
              f"& {fmt_p(r['t_pval'])} & {fmt_p(r['w_pval'])} & {fmt_p(r['nb_pval'])} "
              f"& {fmt_p(r['perm_pval'])} & {fmt_p(r['sign_pval'])} "
              f"& {hg} \\\\")
    print(r"  \bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table*}")
    print()

    # Table 3: Bayesian ROPE
    print(r"% Bayesian ROPE analysis:")
    print(r"\begin{table}[!htb]")
    print(r"\caption{Bayesian ROPE analysis ($\pm$" f"{ROPE:.0f}" r"~pp region of "
          r"practical equivalence). Posterior probabilities that the mean per-fold "
          r"difference falls in each region.}")
    print(r"\label{tab:bayesian_rope}")
    print(r"  \centering")
    print(r"  \begin{tabular}{lccc}")
    print(r"  \toprule")
    print(r"  \textbf{Metric} & $P(\mathcal{M}_{\tau} \succ)$ "
          r"& $P(\text{equiv.})$ & $P(M_{SE+TAP} \succ)$ \\")
    print(r"  \midrule")
    for r in results:
        if r['metric'] not in ('f1_score', 'accuracy'):
            continue
        m = 'F1 Score' if r['metric'] == 'f1_score' else 'Accuracy'
        print(f"  {m} & {r['p_proposed']:.3f} & {r['p_rope']:.3f} "
              f"& {r['p_baseline']:.3f} \\\\")
    print(r"  \bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def main():
    parser = argparse.ArgumentParser(description="Table 2 significance tests")
    parser.add_argument('--proposed', type=str, default=DEFAULT_PROPOSED,
                        help='Path to proposed model fold_results.pkl')
    parser.add_argument('--baseline', type=str, default=DEFAULT_BASELINE,
                        help='Path to baseline model fold_results.pkl')
    parser.add_argument('--proposed-label', default='Dual Kalman (M_τ)')
    parser.add_argument('--baseline-label', default='Single Raw (M_SE+TAP)')
    parser.add_argument('--latex', action='store_true', help='Output LaTeX tables')
    parser.add_argument('--metrics', nargs='+', default=['f1_score', 'accuracy'],
                        choices=METRICS)
    parser.add_argument('--rope', type=float, default=ROPE,
                        help='ROPE width in pp (default: 1.0)')
    args = parser.parse_args()

    proposed = load_folds(args.proposed)
    baseline = load_folds(args.baseline)

    print(f"Proposed: {args.proposed} ({len(proposed)} folds)")
    print(f"Baseline: {args.baseline} ({len(baseline)} folds)")

    results = [run_tests(proposed, baseline, m, rope_width=args.rope) for m in args.metrics]

    print_report(results, args.proposed_label, args.baseline_label)

    print("\n--- LaTeX Output ---\n")
    print_latex(results)


if __name__ == '__main__':
    main()
