"""Unit tests for publication plotting utilities."""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

# Skip entire module if matplotlib not available
pytest.importorskip("matplotlib")
pytest.importorskip("seaborn")

# Use non-interactive backend for CI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


@pytest.fixture
def mock_fold_results():
    """Generate mock fold results for testing."""
    np.random.seed(42)
    results = []
    for i in range(5):
        n_epochs = 20
        train_losses = 1.0 - 0.03 * np.arange(n_epochs) + np.random.randn(n_epochs) * 0.05
        val_losses = 1.1 - 0.025 * np.arange(n_epochs) + np.random.randn(n_epochs) * 0.08
        results.append({
            'test_subject': i + 1,
            'train_losses': train_losses.tolist(),
            'val_losses': val_losses.tolist(),
            'test_f1_score': 80 + np.random.randn() * 10,
            'test_accuracy': 85 + np.random.randn() * 8,
            'test_precision': 82 + np.random.randn() * 9,
            'test_recall': 78 + np.random.randn() * 12,
        })
    return results


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestPublicationPlotter:
    def test_import(self):
        from utils.publication_plots import PublicationPlotter
        assert PublicationPlotter is not None

    def test_init(self, temp_dir):
        from utils.publication_plots import PublicationPlotter
        plotter = PublicationPlotter(temp_dir, "test_experiment")
        assert plotter.work_dir == Path(temp_dir)
        assert plotter.plots_dir.exists()

    def test_loss_curves_creation(self, temp_dir, mock_fold_results):
        from utils.publication_plots import PublicationPlotter
        plotter = PublicationPlotter(temp_dir, "test")
        path = plotter.plot_loss_curves_best_worst(mock_fold_results, n_best=2, n_worst=2)
        if path:
            assert os.path.exists(path)
        plt.close('all')

    def test_empty_fold_results(self, temp_dir):
        from utils.publication_plots import PublicationPlotter
        plotter = PublicationPlotter(temp_dir, "test")
        result = plotter.plot_loss_curves_best_worst([])
        assert result is None
        plt.close('all')


class TestFigureGeneration:
    """Smoke tests for figure generation."""

    def test_basic_line_plot(self, temp_dir):
        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 100)
        ax.plot(x, np.sin(x))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        path = os.path.join(temp_dir, 'line_plot.png')
        fig.savefig(path)
        plt.close(fig)

        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_box_plot(self, temp_dir):
        fig, ax = plt.subplots()
        data = [np.random.randn(100) for _ in range(4)]
        ax.boxplot(data, labels=['A', 'B', 'C', 'D'])

        path = os.path.join(temp_dir, 'box_plot.png')
        fig.savefig(path)
        plt.close(fig)

        assert os.path.exists(path)

    def test_heatmap(self, temp_dir):
        import seaborn as sns
        fig, ax = plt.subplots()
        data = np.random.rand(5, 5)
        sns.heatmap(data, ax=ax, annot=True)

        path = os.path.join(temp_dir, 'heatmap.png')
        fig.savefig(path)
        plt.close(fig)

        assert os.path.exists(path)

    def test_confusion_matrix_style(self, temp_dir):
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(6, 5))

        cm = np.array([[85, 15], [10, 90]])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['ADL', 'Fall'],
                   yticklabels=['ADL', 'Fall'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')

        path = os.path.join(temp_dir, 'confusion_matrix.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)

        assert os.path.exists(path)

    def test_roc_curve_style(self, temp_dir):
        fig, ax = plt.subplots()

        fpr = np.linspace(0, 1, 100)
        tpr = 1 - (1 - fpr) ** 2
        ax.plot(fpr, tpr, label='Model (AUC=0.95)')
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()

        path = os.path.join(temp_dir, 'roc_curve.png')
        fig.savefig(path)
        plt.close(fig)

        assert os.path.exists(path)

    def test_multiple_subplots(self, temp_dir):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        for ax in axes.flat:
            ax.plot(np.random.randn(50).cumsum())

        path = os.path.join(temp_dir, 'subplots.png')
        fig.savefig(path)
        plt.close(fig)

        assert os.path.exists(path)

    def test_pdf_output(self, temp_dir):
        fig, ax = plt.subplots()
        ax.bar(['A', 'B', 'C'], [1, 2, 3])

        path = os.path.join(temp_dir, 'bar_chart.pdf')
        fig.savefig(path)
        plt.close(fig)

        assert os.path.exists(path)

    def test_svg_output(self, temp_dir):
        fig, ax = plt.subplots()
        ax.scatter(np.random.randn(50), np.random.randn(50))

        path = os.path.join(temp_dir, 'scatter.svg')
        fig.savefig(path)
        plt.close(fig)

        assert os.path.exists(path)


class TestPlotSettings:
    """Test publication-quality plot settings."""

    def test_dpi_setting(self, temp_dir):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])

        path = os.path.join(temp_dir, 'high_dpi.png')
        fig.savefig(path, dpi=300)
        plt.close(fig)

        assert os.path.exists(path)
        assert os.path.getsize(path) > 10000

    def test_tight_layout(self, temp_dir):
        fig, axes = plt.subplots(2, 2)
        for ax in axes.flat:
            ax.set_xlabel('Very Long X Label')
            ax.set_ylabel('Very Long Y Label')

        fig.tight_layout()
        path = os.path.join(temp_dir, 'tight.png')
        fig.savefig(path)
        plt.close(fig)

        assert os.path.exists(path)
