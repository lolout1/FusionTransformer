#!/usr/bin/env python3
"""
Run PhD-level analysis with dynamic paths.
Works on both cluster and local desktop.

Usage:
    python run_analysis.py                    # Run all analyses
    python run_analysis.py --check            # Check data availability
    python run_analysis.py --trials           # Generate trial plots only
    python run_analysis.py --figures          # Generate summary figures only
"""

import sys
import argparse
from pathlib import Path

# Setup paths dynamically
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Add to path
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

# Import config
from analysis.config import (
    PROJECT_ROOT, DATA_DIR, RESULTS_DIR,
    FIGURES_DIR, TABLES_DIR, TRIAL_PLOTS_DIR,
    FALL_ACTIVITIES, ALL_ACTIVITIES,
    setup_directories, check_data_availability,
    get_transformer_results, get_cnn_results
)

def find_scores(model_type):
    """Find scores.csv with fallback paths."""
    LOCAL_RESULTS = SCRIPT_DIR / 'model_results'

    if model_type == 'transformer':
        paths = [
            get_transformer_results() / 'scores.csv',
            LOCAL_RESULTS / 'transformer' / 'scores.csv',
        ]
    else:
        paths = [
            get_cnn_results() / 'scores.csv',
            LOCAL_RESULTS / 'cnn' / 'scores.csv',
        ]

    for p in paths:
        if p.exists():
            return p

    raise FileNotFoundError(f"Could not find {model_type} scores.csv")


def load_comparison():
    """Load and prepare model comparison data."""
    trans_scores = pd.read_csv(find_scores('transformer'))
    cnn_scores = pd.read_csv(find_scores('cnn'))

    trans_scores = trans_scores[trans_scores['test_subject'] != 'Average'].copy()
    cnn_scores = cnn_scores[cnn_scores['test_subject'] != 'Average'].copy()
    trans_scores['test_subject'] = trans_scores['test_subject'].astype(int)
    cnn_scores['test_subject'] = cnn_scores['test_subject'].astype(int)

    comparison = trans_scores[['test_subject', 'test_f1_score', 'test_precision', 'test_recall']].copy()
    comparison.columns = ['subject', 'trans_f1', 'trans_prec', 'trans_recall']
    cnn_subset = cnn_scores[['test_subject', 'test_f1_score', 'test_precision', 'test_recall']].copy()
    cnn_subset.columns = ['subject', 'cnn_f1', 'cnn_prec', 'cnn_recall']
    comparison = comparison.merge(cnn_subset, on='subject')
    comparison['delta_f1'] = comparison['trans_f1'] - comparison['cnn_f1']
    comparison['winner'] = comparison['delta_f1'].apply(lambda x: 'Transformer' if x > 0 else 'CNN')
    comparison = comparison.sort_values('delta_f1', ascending=False).reset_index(drop=True)

    return comparison


def load_trial_data(subject_id, activity_id, trial_id, sensor='watch'):
    """Load accelerometer and gyroscope data."""
    result = {}
    for modality in ['accelerometer', 'gyroscope']:
        for age_group in ['young', 'old']:
            path = DATA_DIR / age_group / modality / sensor / f'S{subject_id}A{activity_id}T{trial_id:02d}.csv'
            if path.exists():
                df = pd.read_csv(path, header=None)
                if len(df.columns) >= 4:
                    df.columns = ['time', 'x', 'y', 'z'][:len(df.columns)]
                    for col in ['x', 'y', 'z']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    df = df.dropna()
                if len(df) > 0:
                    result[modality] = df
                break
    return result if 'accelerometer' in result else None


def get_subject_trials(subject_id, activity_id, sensor='watch'):
    """Get available trials."""
    trials = []
    for trial in range(1, 10):
        for age_group in ['young', 'old']:
            path = DATA_DIR / age_group / 'accelerometer' / sensor / f'S{subject_id}A{activity_id}T{trial:02d}.csv'
            if path.exists():
                trials.append(trial)
                break
    return trials


def compute_features(acc, gyro=None, fs=30.0):
    """Extract signal features."""
    features = {}
    ax = pd.to_numeric(acc['x'], errors='coerce').values
    ay = pd.to_numeric(acc['y'], errors='coerce').values
    az = pd.to_numeric(acc['z'], errors='coerce').values
    valid = ~(np.isnan(ax) | np.isnan(ay) | np.isnan(az))
    ax, ay, az = ax[valid], ay[valid], az[valid]
    if len(ax) == 0:
        return features

    smv = np.sqrt(ax**2 + ay**2 + az**2)
    features['smv_max'] = smv.max()
    features['smv_std'] = smv.std()
    features['smv_range'] = smv.max() - smv.min()
    features['duration_s'] = len(smv) / fs

    peaks, _ = find_peaks(smv, height=15, prominence=5)
    features['n_impact_peaks'] = len(peaks)
    features['peak_indices'] = peaks
    features['peak_heights'] = smv[peaks] if len(peaks) > 0 else np.array([])

    if gyro is not None and len(gyro) > 0:
        try:
            gx = pd.to_numeric(gyro['x'], errors='coerce').values
            gy = pd.to_numeric(gyro['y'], errors='coerce').values
            gz = pd.to_numeric(gyro['z'], errors='coerce').values
            valid_g = ~(np.isnan(gx) | np.isnan(gy) | np.isnan(gz))
            if valid_g.sum() > 0:
                gx, gy, gz = gx[valid_g], gy[valid_g], gz[valid_g]
                gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
                features['gyro_max'] = gyro_mag.max()
                features['gyro_std'] = gyro_mag.std()
                features['total_rotation_deg'] = np.sum(np.abs(gyro_mag)) / fs
        except:
            pass

    if len(ax) > 1:
        jerk = np.sqrt(np.diff(ax)**2 + np.diff(ay)**2 + np.diff(az)**2) * fs
        features['jerk_max'] = jerk.max()

    return features


def generate_summary_figures(comparison):
    """Generate main summary figures."""
    print("\nGenerating summary figures...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. F1 Comparison
    ax = axes[0, 0]
    x = np.arange(len(comparison))
    width = 0.35
    ax.bar(x - width/2, comparison['trans_f1'], width, label='Transformer', color='#2E7D32', alpha=0.8)
    ax.bar(x + width/2, comparison['cnn_f1'], width, label='CNN', color='#1565C0', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{int(s)}" for s in comparison['subject']], rotation=45, fontsize=8)
    ax.set_ylabel('Test F1 (%)')
    ax.set_title('F1 by Subject', fontweight='bold')
    ax.legend()
    ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylim(60, 105)

    # 2. Delta
    ax = axes[0, 1]
    colors = ['#2E7D32' if d > 0 else '#C62828' for d in comparison['delta_f1']]
    ax.barh(range(len(comparison)), comparison['delta_f1'], color=colors, alpha=0.8)
    ax.set_yticks(range(len(comparison)))
    ax.set_yticklabels([f"S{int(s)}" for s in comparison['subject']], fontsize=9)
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_xlabel('F1 Delta (%)')
    ax.set_title('Performance Delta', fontweight='bold')

    # 3. Scatter
    ax = axes[0, 2]
    scatter = ax.scatter(comparison['cnn_f1'], comparison['trans_f1'],
                         c=comparison['delta_f1'], cmap='RdYlGn', s=100,
                         alpha=0.8, vmin=-15, vmax=15)
    ax.plot([60, 100], [60, 100], 'k--', alpha=0.5)
    ax.set_xlabel('CNN F1 (%)')
    ax.set_ylabel('Transformer F1 (%)')
    ax.set_title('Trans vs CNN', fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Delta')

    # 4. Boxplot
    ax = axes[1, 0]
    bp = ax.boxplot([comparison['trans_f1'], comparison['cnn_f1']],
                    labels=['Transformer', 'CNN'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#2E7D32')
    bp['boxes'][1].set_facecolor('#1565C0')
    ax.set_ylabel('Test F1 (%)')
    ax.set_title('F1 Distribution', fontweight='bold')

    # 5. Pie
    ax = axes[1, 1]
    trans_wins = (comparison['winner'] == 'Transformer').sum()
    cnn_wins = (comparison['winner'] == 'CNN').sum()
    ax.pie([trans_wins, cnn_wins], labels=['Trans', 'CNN'],
           colors=['#2E7D32', '#1565C0'], autopct='%1.0f%%')
    ax.set_title(f'Wins: {trans_wins} vs {cnn_wins}', fontweight='bold')

    # 6. Stats
    ax = axes[1, 2]
    ax.axis('off')
    t_stat, p_val = stats.ttest_rel(comparison['trans_f1'], comparison['cnn_f1'])
    diff = comparison['trans_f1'] - comparison['cnn_f1']
    cohens_d = diff.mean() / diff.std()
    summary = f"""STATISTICS
{'='*35}
Trans: {comparison['trans_f1'].mean():.2f}% ± {comparison['trans_f1'].std():.2f}%
CNN:   {comparison['cnn_f1'].mean():.2f}% ± {comparison['cnn_f1'].std():.2f}%

Mean Δ: {diff.mean():+.2f}%
p-value: {p_val:.6f}
Cohen's d: {cohens_d:.3f}
"""
    ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat'))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'summary_overview.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'summary_overview.png'}")


def generate_trial_plots(comparison):
    """Generate individual trial visualizations."""
    if not DATA_DIR.exists():
        print("\nRaw data not available - skipping trial plots")
        return

    print("\nGenerating trial plots...")

    # Top 4 each
    trans_best = comparison.head(4)['subject'].tolist()
    cnn_best = comparison.tail(4)['subject'].tolist()
    all_subjects = trans_best + cnn_best

    total = 0
    for subj in all_subjects:
        subj = int(subj)
        subj_info = comparison[comparison['subject'] == subj].iloc[0].to_dict()
        subj_dir = TRIAL_PLOTS_DIR / f'S{subj}'
        subj_dir.mkdir(exist_ok=True)

        for act_id, act_name in FALL_ACTIVITIES.items():
            trials = get_subject_trials(subj, act_id)
            for trial in trials:
                data = load_trial_data(subj, act_id, trial)
                if data:
                    save_path = subj_dir / f'S{subj}_{act_name}_T{trial:02d}.png'
                    plot_trial(data, subj, act_id, trial, subj_info, save_path)
                    total += 1

    print(f"  Generated {total} trial plots")


def plot_trial(data, subj, act_id, trial, subj_info, save_path):
    """Plot a single trial."""
    acc = data['accelerometer']
    gyro = data.get('gyroscope')
    features = compute_features(acc, gyro)

    activity_name = FALL_ACTIVITIES.get(act_id, f'A{act_id}')
    fs = 30.0
    time = np.arange(len(acc)) / fs

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    winner_color = '#2E7D32' if subj_info['winner'] == 'Transformer' else '#C62828'

    # Accelerometer
    ax = axes[0, 0]
    ax.plot(time, acc['x'], label='ax', alpha=0.8)
    ax.plot(time, acc['y'], label='ay', alpha=0.8)
    ax.plot(time, acc['z'], label='az', alpha=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acc (m/s²)')
    ax.set_title('Accelerometer')
    ax.legend()

    # SMV
    ax = axes[0, 1]
    smv = np.sqrt(acc['x']**2 + acc['y']**2 + acc['z']**2)
    ax.plot(time, smv, color='purple')
    ax.axhline(y=9.8, color='gray', linestyle='--', alpha=0.5)
    if len(features.get('peak_indices', [])) > 0:
        peak_times = features['peak_indices'] / fs
        ax.scatter(peak_times, features['peak_heights'], color='red', s=50, zorder=5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('SMV (m/s²)')
    ax.set_title(f'SMV (max: {features.get("smv_max", 0):.1f})')

    # Jerk
    ax = axes[0, 2]
    if len(acc) > 1:
        jerk = np.sqrt(np.diff(acc['x'].values)**2 + np.diff(acc['y'].values)**2 + np.diff(acc['z'].values)**2) * fs
        ax.plot(time[:-1], jerk, color='green', alpha=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Jerk (m/s³)')
    ax.set_title(f'Jerk (max: {features.get("jerk_max", 0):.0f})')

    # Gyroscope
    if gyro is not None:
        ax = axes[1, 0]
        time_g = np.arange(len(gyro)) / fs
        ax.plot(time_g, gyro['x'], label='ωx', alpha=0.8)
        ax.plot(time_g, gyro['y'], label='ωy', alpha=0.8)
        ax.plot(time_g, gyro['z'], label='ωz', alpha=0.8)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Gyro (°/s)')
        ax.set_title('Gyroscope')
        ax.legend()

        ax = axes[1, 1]
        gyro_mag = np.sqrt(gyro['x']**2 + gyro['y']**2 + gyro['z']**2)
        ax.plot(time_g, gyro_mag, color='orange')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Gyro Mag (°/s)')
        ax.set_title(f'Gyro Mag (max: {features.get("gyro_max", 0):.0f})')

    # Summary
    ax = axes[1, 2]
    ax.axis('off')
    summary = f"""TRIAL: S{subj} {activity_name} T{trial:02d}
{'='*30}
Duration: {features.get('duration_s', 0):.2f}s
Peaks: {features.get('n_impact_peaks', 0)}

SMV Max: {features.get('smv_max', 0):.1f} m/s²
Gyro Max: {features.get('gyro_max', 0):.0f} °/s
Rotation: {features.get('total_rotation_deg', 0):.1f}°

MODEL PERFORMANCE
{'='*30}
Trans: {subj_info['trans_f1']:.1f}%
CNN: {subj_info['cnn_f1']:.1f}%
Δ: {subj_info['delta_f1']:+.1f}%
Winner: {subj_info['winner']}
"""
    bbox_color = '#e8f5e9' if subj_info['winner'] == 'Transformer' else '#ffebee'
    ax.text(0.1, 0.95, summary, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor=bbox_color, edgecolor=winner_color))

    fig.suptitle(f'S{subj} | {activity_name} | T{trial:02d} | {subj_info["winner"]}',
                fontsize=12, fontweight='bold', color=winner_color)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='PhD-level fall detection analysis')
    parser.add_argument('--check', action='store_true', help='Check data availability')
    parser.add_argument('--trials', action='store_true', help='Generate trial plots only')
    parser.add_argument('--figures', action='store_true', help='Generate summary figures only')
    args = parser.parse_args()

    print("=" * 60)
    print("PhD-LEVEL ANALYSIS: TRANSFORMER vs CNN")
    print("=" * 60)

    # Setup
    setup_directories()

    if args.check:
        check_data_availability()
        return

    # Load data
    print("\nLoading model comparison data...")
    comparison = load_comparison()
    print(f"Loaded {len(comparison)} subjects")
    print(f"Trans wins: {(comparison['winner'] == 'Transformer').sum()}")
    print(f"CNN wins: {(comparison['winner'] == 'CNN').sum()}")

    # Save comparison
    comparison.to_csv(TABLES_DIR / 'subject_comparison.csv', index=False)

    if args.trials:
        generate_trial_plots(comparison)
    elif args.figures:
        generate_summary_figures(comparison)
    else:
        generate_summary_figures(comparison)
        generate_trial_plots(comparison)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Figures: {FIGURES_DIR}")
    print(f"Tables: {TABLES_DIR}")
    print(f"Trial plots: {TRIAL_PLOTS_DIR}")


if __name__ == '__main__':
    main()
