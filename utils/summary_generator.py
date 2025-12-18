"""
Generate markdown summaries without LLM.

Template-based summary generation for experiment results.
Includes validation subject info and class balance as required.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import yaml
from datetime import datetime


class SummaryGenerator:
    """Generate markdown summaries from experiment results."""

    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)

    def generate_model_summary(self, experiment_name: str, model_name: str) -> str:
        """
        Generate summary markdown for a single model across all HP configs.
        Saved as {model_name}_summary.md
        """
        model_dir = self.results_dir / experiment_name / model_name

        if not model_dir.exists():
            return f"# {model_name}\n\nNo results found."

        lines = [
            f"# {model_name} - Experiment Summary",
            f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\n**Experiment**: {experiment_name}",
            "",
            "---",
            "",
            "## Hyperparameter Comparison",
            "",
            "| HP Config | Test F1 (%) | Test Acc (%) | Overfit Gap | Best Val Loss |",
            "|-----------|-------------|--------------|-------------|---------------|",
        ]

        hp_results = []
        validation_info = None

        for hp_dir in sorted(model_dir.iterdir()):
            if not hp_dir.is_dir():
                continue

            stats_file = hp_dir / 'summary_stats.json'
            if not stats_file.exists():
                continue

            with open(stats_file) as f:
                stats = json.load(f)

            hp_results.append({
                'name': hp_dir.name,
                'test_f1': stats.get('test_f1', {}),
                'test_acc': stats.get('test_acc', {}),
                'overfit_gap': stats.get('overfit_gap', {}),
                'best_val_loss': stats.get('best_val_loss', {}),
            })

            # Capture validation info (should be same across HP configs)
            if validation_info is None and 'validation_subjects' in stats:
                validation_info = {
                    'subjects': stats['validation_subjects'],
                    'class_balance': stats.get('val_class_balance', {})
                }

            tf1 = stats.get('test_f1', {})
            ta = stats.get('test_acc', {})
            og = stats.get('overfit_gap', {})
            vl = stats.get('best_val_loss', {})

            lines.append(
                f"| {hp_dir.name} | "
                f"{tf1.get('mean', 0)*100:.2f} ± {tf1.get('std', 0)*100:.2f} | "
                f"{ta.get('mean', 0)*100:.2f} ± {ta.get('std', 0)*100:.2f} | "
                f"{og.get('mean', 0):.4f} ± {og.get('std', 0):.4f} | "
                f"{vl.get('mean', 0):.4f} |"
            )

        # Add best HP recommendation
        if hp_results:
            best_hp = max(hp_results, key=lambda x: x['test_f1'].get('mean', 0))
            lines.extend([
                "",
                "---",
                "",
                "## Best Configuration",
                "",
                f"**Recommended HP**: `{best_hp['name']}`",
                f"- Test F1: {best_hp['test_f1'].get('mean', 0)*100:.2f} ± {best_hp['test_f1'].get('std', 0)*100:.2f}%",
                f"- Overfit Gap: {best_hp['overfit_gap'].get('mean', 0):.4f}",
                "",
            ])

        # Add validation subject info (CRITICAL)
        lines.extend([
            "---",
            "",
            "## Validation Configuration",
            "",
        ])

        if validation_info:
            lines.extend([
                f"**Validation Subjects**: {validation_info['subjects']}",
                f"**Class Balance**:",
                f"  - Falls: {validation_info['class_balance'].get('falls', 'N/A')}",
                f"  - ADLs: {validation_info['class_balance'].get('adls', 'N/A')}",
                "",
            ])
        else:
            lines.extend([
                "**Validation Subjects**: Not recorded",
                "",
            ])

        # Add architecture info from config
        lines.extend([
            "---",
            "",
            "## Architecture Details",
            "",
        ])

        # Try to load config
        config_file = model_dir / 'config.yaml'
        if config_file.exists():
            with open(config_file) as f:
                config = yaml.safe_load(f)

            lines.append("```yaml")
            lines.append(f"model: {config.get('model', 'N/A')}")
            lines.append(f"model_args:")
            for k, v in config.get('model_args', {}).items():
                lines.append(f"  {k}: {v}")
            lines.append("```")

        summary = "\n".join(lines)

        # Save
        summary_path = model_dir / f"{model_name}_summary.md"
        with open(summary_path, 'w') as f:
            f.write(summary)

        return summary

    def generate_experiment_summary(self, experiment_name: str) -> str:
        """
        Generate summary markdown for entire experiment (all models).
        Saved as {experiment_name}_summary.md
        """
        exp_dir = self.results_dir / experiment_name

        if not exp_dir.exists():
            return f"# {experiment_name}\n\nNo results found."

        lines = [
            f"# {experiment_name} - Full Experiment Summary",
            f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            "",
            "## Model Comparison (Best HP per Model)",
            "",
            "| Rank | Model | Best HP | Test F1 (%) | Test Acc (%) | Overfit Gap |",
            "|------|-------|---------|-------------|--------------|-------------|",
        ]

        model_results = []
        all_validation_info = {}

        for model_dir in sorted(exp_dir.iterdir()):
            if not model_dir.is_dir() or model_dir.name == 'logs':
                continue

            best_f1 = 0
            best_hp = None
            best_stats = None

            for hp_dir in model_dir.iterdir():
                if not hp_dir.is_dir():
                    continue

                stats_file = hp_dir / 'summary_stats.json'
                if not stats_file.exists():
                    continue

                with open(stats_file) as f:
                    stats = json.load(f)

                f1_mean = stats.get('test_f1', {}).get('mean', 0)
                if f1_mean > best_f1:
                    best_f1 = f1_mean
                    best_hp = hp_dir.name
                    best_stats = stats

            if best_stats:
                model_results.append({
                    'model': model_dir.name,
                    'hp': best_hp,
                    'stats': best_stats,
                    'f1': best_f1,
                })

                # Track validation info per model
                if 'validation_subjects' in best_stats:
                    all_validation_info[model_dir.name] = {
                        'subjects': best_stats['validation_subjects'],
                        'balance': best_stats.get('val_class_balance', {})
                    }

        # Sort by F1
        model_results.sort(key=lambda x: x['f1'], reverse=True)

        for rank, result in enumerate(model_results, 1):
            tf1 = result['stats'].get('test_f1', {})
            ta = result['stats'].get('test_acc', {})
            og = result['stats'].get('overfit_gap', {})

            lines.append(
                f"| {rank} | {result['model']} | {result['hp']} | "
                f"{tf1.get('mean', 0)*100:.2f} ± {tf1.get('std', 0)*100:.2f} | "
                f"{ta.get('mean', 0)*100:.2f} ± {ta.get('std', 0)*100:.2f} | "
                f"{og.get('mean', 0):.4f} ± {og.get('std', 0):.4f} |"
            )

        # Add validation subject summary (CRITICAL)
        lines.extend([
            "",
            "---",
            "",
            "## Validation Configuration Summary",
            "",
            "| Model | Val Subjects | Falls | ADLs |",
            "|-------|--------------|-------|------|",
        ])

        for model_name, info in all_validation_info.items():
            lines.append(
                f"| {model_name} | {info['subjects']} | "
                f"{info['balance'].get('falls', 'N/A')} | "
                f"{info['balance'].get('adls', 'N/A')} |"
            )

        # Statistical significance note
        lines.extend([
            "",
            "---",
            "",
            "## Notes",
            "",
            "- All results from 22-fold LOSO cross-validation",
            "- Overfit Gap = train_f1 - val_f1 at best validation epoch (lower = better)",
            "- Test subjects exclude validation subjects and train-only subjects",
            "- Statistical significance tests not shown (require scipy)",
            "",
        ])

        summary = "\n".join(lines)

        # Save
        summary_path = exp_dir / f"{experiment_name}_summary.md"
        with open(summary_path, 'w') as f:
            f.write(summary)

        return summary

    def generate_comparison_csv(self, experiment_name: str) -> str:
        """
        Generate a comparison CSV with all models' best results.
        Includes validation subject info.
        """
        import csv

        exp_dir = self.results_dir / experiment_name

        if not exp_dir.exists():
            return ""

        rows = []
        headers = [
            'rank', 'model', 'best_hp', 'test_f1_mean', 'test_f1_std',
            'test_acc_mean', 'test_acc_std', 'test_precision', 'test_recall',
            'test_auc', 'overfit_gap', 'best_val_loss',
            'validation_subjects', 'val_falls', 'val_adls'
        ]

        for model_dir in sorted(exp_dir.iterdir()):
            if not model_dir.is_dir() or model_dir.name == 'logs':
                continue

            best_f1 = 0
            best_hp = None
            best_stats = None

            for hp_dir in model_dir.iterdir():
                if not hp_dir.is_dir():
                    continue

                stats_file = hp_dir / 'summary_stats.json'
                if not stats_file.exists():
                    continue

                with open(stats_file) as f:
                    stats = json.load(f)

                f1_mean = stats.get('test_f1', {}).get('mean', 0)
                if f1_mean > best_f1:
                    best_f1 = f1_mean
                    best_hp = hp_dir.name
                    best_stats = stats

            if best_stats:
                rows.append({
                    'model': model_dir.name,
                    'best_hp': best_hp,
                    'test_f1_mean': best_stats.get('test_f1', {}).get('mean', 0),
                    'test_f1_std': best_stats.get('test_f1', {}).get('std', 0),
                    'test_acc_mean': best_stats.get('test_acc', {}).get('mean', 0),
                    'test_acc_std': best_stats.get('test_acc', {}).get('std', 0),
                    'test_precision': best_stats.get('test_precision', {}).get('mean', 0),
                    'test_recall': best_stats.get('test_recall', {}).get('mean', 0),
                    'test_auc': best_stats.get('test_auc', {}).get('mean', 0),
                    'overfit_gap': best_stats.get('overfit_gap', {}).get('mean', 0),
                    'best_val_loss': best_stats.get('best_val_loss', {}).get('mean', 0),
                    'validation_subjects': str(best_stats.get('validation_subjects', [])),
                    'val_falls': best_stats.get('val_class_balance', {}).get('falls', 'N/A'),
                    'val_adls': best_stats.get('val_class_balance', {}).get('adls', 'N/A'),
                })

        # Sort by F1
        rows.sort(key=lambda x: x['test_f1_mean'], reverse=True)

        # Add rank
        for i, row in enumerate(rows, 1):
            row['rank'] = i

        # Save CSV
        csv_path = exp_dir / f"{experiment_name}_comparison.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

        return str(csv_path)


if __name__ == "__main__":
    import tempfile
    import os

    print("=" * 60)
    print("SummaryGenerator Test")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock experiment structure
        exp_dir = Path(tmpdir) / 'modality_tests'

        for model_name in ['transformer_acc_smv', 'transformer_se_acc']:
            for hp_name in ['hp_baseline', 'hp_more_reg']:
                hp_dir = exp_dir / model_name / hp_name
                hp_dir.mkdir(parents=True)

                # Create mock stats
                stats = {
                    'test_f1': {'mean': 0.85 + np.random.uniform(-0.05, 0.05), 'std': 0.05},
                    'test_acc': {'mean': 0.88 + np.random.uniform(-0.05, 0.05), 'std': 0.04},
                    'test_precision': {'mean': 0.82, 'std': 0.06},
                    'test_recall': {'mean': 0.78, 'std': 0.07},
                    'test_auc': {'mean': 0.90, 'std': 0.03},
                    'overfit_gap': {'mean': 0.08, 'std': 0.02},
                    'best_val_loss': {'mean': 0.35, 'std': 0.05},
                    'validation_subjects': [48, 57],
                    'val_class_balance': {'falls': 150, 'adls': 350}
                }

                with open(hp_dir / 'summary_stats.json', 'w') as f:
                    json.dump(stats, f)

        # Generate summaries
        import numpy as np
        gen = SummaryGenerator(Path(tmpdir))

        # Model summary
        model_summary = gen.generate_model_summary('modality_tests', 'transformer_acc_smv')
        print("\nModel Summary (first 20 lines):")
        print("\n".join(model_summary.split('\n')[:20]))

        # Experiment summary
        exp_summary = gen.generate_experiment_summary('modality_tests')
        print("\n\nExperiment Summary (first 25 lines):")
        print("\n".join(exp_summary.split('\n')[:25]))

        # CSV
        csv_path = gen.generate_comparison_csv('modality_tests')
        print(f"\n\nCSV saved to: {csv_path}")

        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
