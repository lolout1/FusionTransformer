"""
Comprehensive metrics logging for all folds, all metrics, all HP configs.

Stores train, val, test metrics per epoch and per fold.
Supports graph generation selection based on performance.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class FoldMetrics:
    """Metrics for a single LOSO fold."""
    fold_id: int
    test_subject: str

    # Per-epoch history (for graph generation)
    train_loss_history: List[float] = field(default_factory=list)
    val_loss_history: List[float] = field(default_factory=list)
    train_f1_history: List[float] = field(default_factory=list)
    val_f1_history: List[float] = field(default_factory=list)
    train_acc_history: List[float] = field(default_factory=list)
    val_acc_history: List[float] = field(default_factory=list)

    # Final metrics (best validation epoch)
    best_epoch: int = 0
    best_val_loss: float = float('inf')
    best_val_f1: float = 0.0
    best_val_acc: float = 0.0

    # Test metrics (evaluated at best_epoch)
    test_loss: float = 0.0
    test_f1: float = 0.0
    test_acc: float = 0.0
    test_precision: float = 0.0
    test_recall: float = 0.0
    test_auc: float = 0.0

    # Train metrics at best epoch (for overfit calculation)
    train_f1_at_best: float = 0.0
    train_loss_at_best: float = 0.0

    @property
    def overfit_gap(self) -> float:
        """train_f1 - val_f1 at best epoch. Lower = better generalization."""
        return self.train_f1_at_best - self.best_val_f1


class MetricsLogger:
    """
    Logs and aggregates metrics across all folds for an experiment.
    """

    def __init__(self,
                 output_dir: Path,
                 experiment_name: str,
                 model_name: str,
                 hp_name: str,
                 validation_subjects: List[int] = None,
                 val_class_balance: Dict[str, int] = None):
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.hp_name = hp_name
        self.validation_subjects = validation_subjects or [48, 57]
        self.val_class_balance = val_class_balance or {'falls': 0, 'adls': 0}
        self.folds: Dict[int, FoldMetrics] = {}

        # Create directory structure
        self.metrics_dir = self.output_dir / experiment_name / model_name / hp_name
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def add_fold(self, fold_id: int, test_subject: str) -> FoldMetrics:
        """Initialize metrics for a new fold."""
        fold = FoldMetrics(fold_id=fold_id, test_subject=test_subject)
        self.folds[fold_id] = fold
        return fold

    def log_epoch(self, fold_id: int, epoch: int,
                  train_loss: float, val_loss: float,
                  train_f1: float, val_f1: float,
                  train_acc: float, val_acc: float):
        """Log metrics for a single epoch."""
        fold = self.folds[fold_id]
        fold.train_loss_history.append(train_loss)
        fold.val_loss_history.append(val_loss)
        fold.train_f1_history.append(train_f1)
        fold.val_f1_history.append(val_f1)
        fold.train_acc_history.append(train_acc)
        fold.val_acc_history.append(val_acc)

        # Update best if validation loss improved
        if val_loss < fold.best_val_loss:
            fold.best_epoch = epoch
            fold.best_val_loss = val_loss
            fold.best_val_f1 = val_f1
            fold.best_val_acc = val_acc
            fold.train_f1_at_best = train_f1
            fold.train_loss_at_best = train_loss

    def log_test_metrics(self, fold_id: int,
                         test_loss: float, test_f1: float, test_acc: float,
                         test_precision: float, test_recall: float, test_auc: float):
        """Log final test metrics for a fold."""
        fold = self.folds[fold_id]
        fold.test_loss = test_loss
        fold.test_f1 = test_f1
        fold.test_acc = test_acc
        fold.test_precision = test_precision
        fold.test_recall = test_recall
        fold.test_auc = test_auc

    def update_val_class_balance(self, falls: int, adls: int):
        """Update validation set class balance."""
        self.val_class_balance = {'falls': falls, 'adls': adls}

    def save_fold_metrics_csv(self) -> pd.DataFrame:
        """Save all fold metrics to CSV."""
        rows = []
        for fold_id, fold in sorted(self.folds.items()):
            rows.append({
                'fold_id': fold_id,
                'test_subject': fold.test_subject,
                'best_epoch': fold.best_epoch,
                'train_f1': round(fold.train_f1_at_best, 4),
                'val_f1': round(fold.best_val_f1, 4),
                'test_f1': round(fold.test_f1, 4),
                'train_loss': round(fold.train_loss_at_best, 6),
                'val_loss': round(fold.best_val_loss, 6),
                'test_loss': round(fold.test_loss, 6),
                'test_acc': round(fold.test_acc, 4),
                'test_precision': round(fold.test_precision, 4),
                'test_recall': round(fold.test_recall, 4),
                'test_auc': round(fold.test_auc, 4),
                'overfit_gap': round(fold.overfit_gap, 4),
            })

        df = pd.DataFrame(rows)

        # Add average row
        if not df.empty:
            avg_row = {'fold_id': 'AVG', 'test_subject': 'Average'}
            for col in df.columns:
                if col not in ['fold_id', 'test_subject']:
                    avg_row[col] = round(df[col].mean(), 4 if 'loss' not in col else 6)
            df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

            # Add std row
            std_row = {'fold_id': 'STD', 'test_subject': 'Std Dev'}
            for col in df.columns:
                if col not in ['fold_id', 'test_subject']:
                    # Exclude the AVG row when computing std
                    numeric_vals = pd.to_numeric(df[col].iloc[:-1], errors='coerce')
                    std_row[col] = round(numeric_vals.std(), 4 if 'loss' not in col else 6)
            df = pd.concat([df, pd.DataFrame([std_row])], ignore_index=True)

        df.to_csv(self.metrics_dir / 'fold_metrics.csv', index=False)
        return df

    def compute_summary_stats(self) -> Dict[str, Any]:
        """Compute mean, std, min, max for all metrics."""
        df = self.save_fold_metrics_csv()

        # Use only numeric rows (exclude AVG, STD rows)
        df_numeric = df[~df['fold_id'].isin(['AVG', 'STD'])].copy()

        stats = {}
        for col in ['train_f1', 'val_f1', 'test_f1', 'test_acc', 'test_precision',
                    'test_recall', 'test_auc', 'overfit_gap', 'best_val_loss']:
            if col in df_numeric.columns:
                values = pd.to_numeric(df_numeric[col], errors='coerce')
                stats[col] = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                }

        # Add validation subject info (CRITICAL per user requirement)
        stats['validation_subjects'] = self.validation_subjects
        stats['val_class_balance'] = self.val_class_balance

        # Save as JSON
        with open(self.metrics_dir / 'summary_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        return stats

    def get_folds_for_graphs(self) -> Dict[str, List[int]]:
        """
        Identify which folds need graphs computed.
        Returns dict with category -> list of fold_ids.
        """
        if not self.folds:
            return {}

        df = pd.DataFrame([
            {'fold_id': f.fold_id, 'test_f1': f.test_f1,
             'overfit_gap': f.overfit_gap, 'best_val_loss': f.best_val_loss}
            for f in self.folds.values()
        ])

        result = {
            'top3_best': df.nlargest(3, 'test_f1')['fold_id'].tolist(),
            'top3_worst': df.nsmallest(3, 'test_f1')['fold_id'].tolist(),
            'top2_least_overfit': df.nsmallest(2, 'overfit_gap')['fold_id'].tolist(),
            'top2_lowest_val_loss': df.nsmallest(2, 'best_val_loss')['fold_id'].tolist(),
        }

        return result

    def get_fold(self, fold_id: int) -> Optional[FoldMetrics]:
        """Get fold metrics by ID."""
        return self.folds.get(fold_id)


if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("MetricsLogger Test")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = MetricsLogger(
            output_dir=Path(tmpdir),
            experiment_name='test_exp',
            model_name='test_model',
            hp_name='hp_baseline'
        )

        # Simulate 5 folds
        for fold_id in range(5):
            fold = logger.add_fold(fold_id, str(30 + fold_id))

            # Simulate 10 epochs
            for epoch in range(10):
                train_loss = 0.5 - epoch * 0.03 + np.random.uniform(-0.02, 0.02)
                val_loss = 0.6 - epoch * 0.02 + np.random.uniform(-0.03, 0.03)
                train_f1 = 0.5 + epoch * 0.04 + np.random.uniform(-0.02, 0.02)
                val_f1 = 0.4 + epoch * 0.03 + np.random.uniform(-0.03, 0.03)
                train_acc = 0.5 + epoch * 0.04 + np.random.uniform(-0.02, 0.02)
                val_acc = 0.4 + epoch * 0.03 + np.random.uniform(-0.03, 0.03)

                logger.log_epoch(fold_id, epoch, train_loss, val_loss,
                                 train_f1, val_f1, train_acc, val_acc)

            # Log test metrics
            logger.log_test_metrics(fold_id,
                                    test_loss=0.3 + fold_id * 0.05,
                                    test_f1=0.75 + fold_id * 0.03,
                                    test_acc=0.80 + fold_id * 0.02,
                                    test_precision=0.78,
                                    test_recall=0.72,
                                    test_auc=0.85)

        # Update validation class balance
        logger.update_val_class_balance(falls=150, adls=350)

        # Compute stats
        stats = logger.compute_summary_stats()

        print(f"\nTest F1: {stats['test_f1']['mean']:.4f} ± {stats['test_f1']['std']:.4f}")
        print(f"Validation subjects: {stats['validation_subjects']}")
        print(f"Val class balance: {stats['val_class_balance']}")

        # Get folds for graphs
        graph_folds = logger.get_folds_for_graphs()
        print(f"\nFolds for graphs: {graph_folds}")

        # Check CSV was created
        csv_path = Path(tmpdir) / 'test_exp' / 'test_model' / 'hp_baseline' / 'fold_metrics.csv'
        assert csv_path.exists(), "CSV not created!"

        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
