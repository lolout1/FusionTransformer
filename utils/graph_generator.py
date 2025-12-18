"""
Generate publication-quality graphs for PhD report.

Handles individual fold graphs and averaged graphs across all folds.
Uses matplotlib with seaborn styling for professional appearance.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING
import warnings

# Import sklearn metrics for ROC curve and confusion matrix
try:
    from sklearn.metrics import roc_curve, auc, confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

if TYPE_CHECKING:
    from utils.metrics_logger import MetricsLogger, FoldMetrics

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'seaborn-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 150

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


class GraphGenerator:
    """Generate graphs for selected folds and averaged metrics."""

    def __init__(self, output_dir: Path, metrics_logger: 'MetricsLogger'):
        self.output_dir = Path(output_dir)
        self.logger = metrics_logger
        self.graphs_dir = output_dir / 'graphs'
        self.graphs_dir.mkdir(parents=True, exist_ok=True)

    def generate_fold_graphs(self, fold_id: int, category: str,
                             y_true: np.ndarray = None, y_pred: np.ndarray = None,
                             y_prob: np.ndarray = None):
        """
        Generate all graphs for a single fold.

        Args:
            fold_id: Fold identifier
            category: One of 'top3_best', 'top3_worst', 'top2_least_overfit', 'top2_lowest_val_loss'
            y_true: Ground truth labels (for ROC/confusion matrix)
            y_pred: Predicted labels
            y_prob: Predicted probabilities
        """
        fold = self.logger.get_fold(fold_id)
        if fold is None:
            return

        category_dir = self.graphs_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)

        subject = fold.test_subject

        # 1. Train/Val Loss Curves
        self._plot_loss_curves(fold, category_dir, subject)

        # 2. Train/Val F1 Curves
        self._plot_f1_curves(fold, category_dir, subject)

        # 3. ROC Curve (if predictions provided)
        if y_true is not None and y_prob is not None and HAS_SKLEARN:
            self._plot_roc_curve(y_true, y_prob, category_dir, subject, fold.test_auc)

        # 4. Confusion Matrix (if predictions provided)
        if y_true is not None and y_pred is not None and HAS_SKLEARN:
            self._plot_confusion_matrix(y_true, y_pred, category_dir, subject)

    def _plot_loss_curves(self, fold: 'FoldMetrics', output_dir: Path, subject: str):
        """Plot training and validation loss curves."""
        fig, ax = plt.subplots(figsize=(10, 6))

        epochs = range(1, len(fold.train_loss_history) + 1)
        ax.plot(epochs, fold.train_loss_history, 'b-', label='Train Loss', linewidth=2)
        ax.plot(epochs, fold.val_loss_history, 'r-', label='Val Loss', linewidth=2)

        # Mark best epoch
        if fold.best_epoch < len(epochs):
            ax.axvline(x=fold.best_epoch + 1, color='g', linestyle='--',
                       label=f'Best Epoch ({fold.best_epoch + 1})', alpha=0.7)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'Training & Validation Loss - Subject {subject}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'fold_S{subject}_train_val_loss.png', dpi=150)
        plt.close()

    def _plot_f1_curves(self, fold: 'FoldMetrics', output_dir: Path, subject: str):
        """Plot training and validation F1 curves."""
        fig, ax = plt.subplots(figsize=(10, 6))

        epochs = range(1, len(fold.train_f1_history) + 1)
        ax.plot(epochs, fold.train_f1_history, 'b-', label='Train F1', linewidth=2)
        ax.plot(epochs, fold.val_f1_history, 'r-', label='Val F1', linewidth=2)

        # Mark best epoch
        if fold.best_epoch < len(epochs):
            ax.axvline(x=fold.best_epoch + 1, color='g', linestyle='--',
                       label=f'Best Epoch ({fold.best_epoch + 1})', alpha=0.7)

        # Add gap annotation
        gap = fold.overfit_gap
        ax.annotate(f'Overfit Gap: {gap:.4f}',
                    xy=(0.02, 0.98), xycoords='axes fraction',
                    fontsize=10, ha='left', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1 Score')
        ax.set_title(f'Training & Validation F1 - Subject {subject}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'fold_S{subject}_train_val_f1.png', dpi=150)
        plt.close()

    def _plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray,
                        output_dir: Path, subject: str, auc_score: float):
        """Plot ROC curve."""
        fig, ax = plt.subplots(figsize=(8, 8))

        fpr, tpr, _ = roc_curve(y_true, y_prob)

        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc_score:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')

        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - Subject {subject}')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(output_dir / f'fold_S{subject}_roc_curve.png', dpi=150)
        plt.close()

    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                               output_dir: Path, subject: str):
        """Plot confusion matrix."""
        fig, ax = plt.subplots(figsize=(8, 6))

        cm = confusion_matrix(y_true, y_pred)

        if HAS_SEABORN:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['ADL', 'Fall'], yticklabels=['ADL', 'Fall'])
        else:
            # Fallback without seaborn
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['ADL', 'Fall'])
            ax.set_yticklabels(['ADL', 'Fall'])
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")

        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix - Subject {subject}')

        plt.tight_layout()
        plt.savefig(output_dir / f'fold_S{subject}_confusion_matrix.png', dpi=150)
        plt.close()

    def generate_averaged_graphs(self):
        """Generate graphs averaged across all folds with std bands."""
        averaged_dir = self.graphs_dir / 'averaged'
        averaged_dir.mkdir(parents=True, exist_ok=True)

        # Collect all histories (aligned by epoch)
        all_train_loss = []
        all_val_loss = []
        all_train_f1 = []
        all_val_f1 = []

        for fold in self.logger.folds.values():
            all_train_loss.append(fold.train_loss_history)
            all_val_loss.append(fold.val_loss_history)
            all_train_f1.append(fold.train_f1_history)
            all_val_f1.append(fold.val_f1_history)

        if not all_train_loss:
            return

        # Convert to numpy arrays (handle variable lengths by padding/truncating)
        max_epochs = max(len(h) for h in all_train_loss)

        def pad_histories(histories, max_len):
            padded = []
            for h in histories:
                if len(h) < max_len:
                    h = list(h) + [h[-1]] * (max_len - len(h))  # Pad with last value
                padded.append(h[:max_len])
            return np.array(padded)

        train_loss_arr = pad_histories(all_train_loss, max_epochs)
        val_loss_arr = pad_histories(all_val_loss, max_epochs)
        train_f1_arr = pad_histories(all_train_f1, max_epochs)
        val_f1_arr = pad_histories(all_val_f1, max_epochs)

        # Plot averaged loss curves
        self._plot_averaged_curves(
            train_loss_arr, val_loss_arr,
            'Loss', 'Mean Training & Validation Loss (±1 std)',
            averaged_dir / 'mean_train_val_loss.png'
        )

        # Plot averaged F1 curves
        self._plot_averaged_curves(
            train_f1_arr, val_f1_arr,
            'F1 Score', 'Mean Training & Validation F1 (±1 std)',
            averaged_dir / 'mean_train_val_f1.png'
        )

    def _plot_averaged_curves(self, train_arr: np.ndarray, val_arr: np.ndarray,
                              ylabel: str, title: str, output_path: Path):
        """Plot averaged curves with std bands."""
        fig, ax = plt.subplots(figsize=(10, 6))

        epochs = range(1, train_arr.shape[1] + 1)

        train_mean = train_arr.mean(axis=0)
        train_std = train_arr.std(axis=0)
        val_mean = val_arr.mean(axis=0)
        val_std = val_arr.std(axis=0)

        ax.plot(epochs, train_mean, 'b-', label='Train (mean)', linewidth=2)
        ax.fill_between(epochs, train_mean - train_std, train_mean + train_std,
                        color='blue', alpha=0.2)

        ax.plot(epochs, val_mean, 'r-', label='Val (mean)', linewidth=2)
        ax.fill_between(epochs, val_mean - val_std, val_mean + val_std,
                        color='red', alpha=0.2)

        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()


if __name__ == "__main__":
    import tempfile
    from utils.metrics_logger import MetricsLogger

    print("=" * 60)
    print("GraphGenerator Test")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create logger with test data
        logger = MetricsLogger(
            output_dir=Path(tmpdir),
            experiment_name='test_exp',
            model_name='test_model',
            hp_name='hp_baseline'
        )

        # Simulate 5 folds
        for fold_id in range(5):
            fold = logger.add_fold(fold_id, str(30 + fold_id))

            # Simulate 20 epochs
            for epoch in range(20):
                train_loss = 0.5 - epoch * 0.02 + np.random.uniform(-0.02, 0.02)
                val_loss = 0.6 - epoch * 0.015 + np.random.uniform(-0.03, 0.03)
                train_f1 = 0.5 + epoch * 0.02 + np.random.uniform(-0.02, 0.02)
                val_f1 = 0.4 + epoch * 0.015 + np.random.uniform(-0.03, 0.03)
                train_acc = 0.5 + epoch * 0.02 + np.random.uniform(-0.02, 0.02)
                val_acc = 0.4 + epoch * 0.015 + np.random.uniform(-0.03, 0.03)

                logger.log_epoch(fold_id, epoch, train_loss, val_loss,
                                 train_f1, val_f1, train_acc, val_acc)

            logger.log_test_metrics(fold_id, 0.3, 0.75 + fold_id * 0.03, 0.80, 0.78, 0.72, 0.85)

        # Create graph generator
        graph_gen = GraphGenerator(
            output_dir=Path(tmpdir) / 'test_exp' / 'test_model' / 'hp_baseline',
            metrics_logger=logger
        )

        # Generate graphs for fold 0
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.rand(100)
        y_pred = (y_prob > 0.5).astype(int)

        graph_gen.generate_fold_graphs(0, 'top3_best', y_true, y_pred, y_prob)
        graph_gen.generate_averaged_graphs()

        # Check graphs were created
        graphs_dir = Path(tmpdir) / 'test_exp' / 'test_model' / 'hp_baseline' / 'graphs'
        assert (graphs_dir / 'top3_best').exists(), "top3_best folder not created!"
        assert (graphs_dir / 'averaged').exists(), "averaged folder not created!"

        print("\nGraphs created successfully!")
        print("=" * 60)
        print("All tests passed!")
        print("=" * 60)
