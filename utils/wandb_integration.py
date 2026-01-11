"""
Weights & Biases Integration Module for SmartFallMM

This module provides comprehensive W&B integration for experiment tracking,
including:
- Run initialization with config tracking
- Per-epoch metric logging
- Per-fold result aggregation
- Model artifact versioning
- Training curve visualization
- Offline mode support for SLURM jobs

Usage:
    from utils.wandb_integration import WandbLogger

    # Initialize
    logger = WandbLogger(config, project="smartfall-mm")

    # Log during training
    logger.log_epoch_metrics(epoch, train_metrics, val_metrics, fold_idx)

    # Log fold results
    logger.log_fold_complete(fold_idx, test_subject, metrics, model_path)

    # Finish
    logger.finish()
"""

import os
import subprocess
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import numpy as np

# Optional W&B import with graceful fallback
try:
    import wandb
    from wandb import Table, Artifact
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# Optional matplotlib for figure logging
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for headless servers
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None


class WandbLogger:
    """
    Comprehensive W&B logger for SmartFallMM experiments.

    Handles initialization, metric logging, artifact management, and
    graceful degradation when W&B is unavailable.
    """

    # Default W&B settings
    DEFAULT_ENTITY = "abheek-texas-state-university"
    DEFAULT_PROJECT = "smartfall-mm"

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        project: str = DEFAULT_PROJECT,
        entity: str = DEFAULT_ENTITY,
        name: Optional[str] = None,
        group: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        mode: str = "online",
        enabled: bool = True,
        resume: Optional[str] = None,
        job_type: str = "train",
    ):
        """
        Initialize W&B logger.

        Args:
            config: Experiment configuration dict (logged as wandb.config)
            project: W&B project name
            entity: W&B entity (team/username)
            name: Run name (auto-generated if None)
            group: Run group for organizing related runs
            tags: List of tags for filtering
            notes: Markdown notes for the run
            mode: "online", "offline", or "disabled"
            enabled: Whether W&B logging is enabled
            resume: Run ID to resume, or "allow"/"must"/"never"
            job_type: Type of job ("train", "sweep", "eval")
        """
        self.enabled = enabled and WANDB_AVAILABLE
        self.config = config or {}
        self.project = project
        self.entity = entity
        self.mode = mode
        self.run = None

        # Fold tracking
        self.fold_results: List[Dict] = []
        self.current_fold_idx: Optional[int] = None
        self.current_test_subject: Optional[Union[int, List[int]]] = None

        # Training history for averaging
        self.all_folds_train_loss: List[List[float]] = []
        self.all_folds_val_loss: List[List[float]] = []

        if not self.enabled:
            if not WANDB_AVAILABLE:
                print("[WandbLogger] wandb not installed. Logging disabled.")
            else:
                print("[WandbLogger] Logging disabled by user.")
            return

        # Initialize W&B run
        try:
            self.run = wandb.init(
                project=project,
                entity=entity,
                config=self._flatten_config(config),
                name=name,
                group=group,
                tags=tags or self._auto_tags(config),
                notes=notes,
                mode=mode,
                resume=resume,
                job_type=job_type,
                reinit=True,
            )

            # Log system info
            self._log_system_info()

            print(f"[WandbLogger] Initialized run: {self.run.name}")
            print(f"[WandbLogger] View at: {self.run.url}")

        except Exception as e:
            print(f"[WandbLogger] Failed to initialize: {e}")
            self.enabled = False

    def _flatten_config(self, config: Optional[Dict]) -> Dict:
        """Flatten nested config dict for W&B."""
        if config is None:
            return {}

        flat = {}

        def _flatten(d: Dict, prefix: str = ""):
            for k, v in d.items():
                key = f"{prefix}{k}" if prefix else k
                if isinstance(v, dict):
                    _flatten(v, f"{key}.")
                else:
                    flat[key] = v

        _flatten(config)
        return flat

    def _auto_tags(self, config: Optional[Dict]) -> List[str]:
        """Generate automatic tags from config."""
        tags = []

        if config is None:
            return tags

        # Model type
        model = config.get('model', '')
        if 'kalman' in model.lower():
            tags.append('kalman')
        if 'transformer' in model.lower():
            tags.append('transformer')
        if 'lstm' in model.lower():
            tags.append('lstm')
        if 'cnn' in model.lower() or 'mamba' in model.lower():
            tags.append('cnn-mamba')

        # Dataset config
        dataset_args = config.get('dataset_args', {})
        if dataset_args.get('enable_kalman_fusion'):
            tags.append('kalman-fusion')

        sensors = dataset_args.get('sensors', [])
        if 'watch' in sensors:
            tags.append('watch')

        return tags

    def _log_system_info(self):
        """Log system and environment information."""
        if not self.enabled or not self.run:
            return

        info = {}

        # Git info
        try:
            info['git_commit'] = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode('utf-8').strip()[:8]
            info['git_branch'] = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode('utf-8').strip()
        except:
            pass

        # SLURM info
        slurm_job_id = os.environ.get('SLURM_JOB_ID')
        if slurm_job_id:
            info['slurm_job_id'] = slurm_job_id
            info['slurm_node'] = os.environ.get('SLURM_NODELIST', 'unknown')

        # Update config with system info
        wandb.config.update(info, allow_val_change=True)

    def log_epoch_metrics(
        self,
        epoch: int,
        train_metrics: Optional[Dict[str, float]] = None,
        val_metrics: Optional[Dict[str, float]] = None,
        fold_idx: Optional[int] = None,
        test_subject: Optional[Union[int, List[int]]] = None,
        commit: bool = True,
    ):
        """
        Log metrics for a single epoch.

        Args:
            epoch: Epoch number (0-indexed)
            train_metrics: Dict with keys like 'loss', 'f1_score', 'accuracy', etc.
            val_metrics: Same structure for validation metrics
            fold_idx: Current LOSO fold index
            test_subject: Current test subject(s)
            commit: Whether to commit the log immediately
        """
        if not self.enabled or not self.run:
            return

        log_dict = {"epoch": epoch + 1}  # 1-indexed for display

        if fold_idx is not None:
            log_dict["fold"] = fold_idx + 1
            self.current_fold_idx = fold_idx

        if test_subject is not None:
            if isinstance(test_subject, list):
                log_dict["test_subject"] = "_".join(map(str, sorted(test_subject)))
            else:
                log_dict["test_subject"] = test_subject
            self.current_test_subject = test_subject

        # Log training metrics
        if train_metrics:
            for key, value in train_metrics.items():
                log_dict[f"train/{key}"] = value

        # Log validation metrics
        if val_metrics:
            for key, value in val_metrics.items():
                log_dict[f"val/{key}"] = value

        wandb.log(log_dict, commit=commit)

    def log_test_metrics(
        self,
        metrics: Dict[str, float],
        fold_idx: Optional[int] = None,
        test_subject: Optional[Union[int, List[int]]] = None,
    ):
        """
        Log test metrics for a completed fold.

        Args:
            metrics: Test metrics dict
            fold_idx: Fold index
            test_subject: Test subject(s)
        """
        if not self.enabled or not self.run:
            return

        log_dict = {}

        if fold_idx is not None:
            log_dict["fold"] = fold_idx + 1

        if test_subject is not None:
            if isinstance(test_subject, list):
                log_dict["test_subject"] = "_".join(map(str, sorted(test_subject)))
            else:
                log_dict["test_subject"] = test_subject

        for key, value in metrics.items():
            log_dict[f"test/{key}"] = value

        wandb.log(log_dict)

    def log_fold_complete(
        self,
        fold_idx: int,
        test_subject: Union[int, List[int]],
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        test_metrics: Dict[str, float],
        best_epoch: int,
        model_path: Optional[str] = None,
        train_loss_history: Optional[List[float]] = None,
        val_loss_history: Optional[List[float]] = None,
    ):
        """
        Log completion of a LOSO fold with full metrics.

        Args:
            fold_idx: Fold index (0-indexed)
            test_subject: Test subject(s)
            train_metrics: Final training metrics
            val_metrics: Best validation metrics
            test_metrics: Test metrics
            best_epoch: Epoch where best model was saved
            model_path: Path to saved model checkpoint
            train_loss_history: List of training losses per epoch
            val_loss_history: List of validation losses per epoch
        """
        if not self.enabled or not self.run:
            return

        # Format test subject string
        if isinstance(test_subject, list):
            test_subject_str = "_".join(map(str, sorted(test_subject)))
        else:
            test_subject_str = str(test_subject)

        # Store fold results for summary table
        fold_result = {
            'fold': fold_idx + 1,
            'test_subject': test_subject_str,
            'best_epoch': best_epoch,
            'train_f1': train_metrics.get('f1_score', 0),
            'val_f1': val_metrics.get('f1_score', 0),
            'test_f1': test_metrics.get('f1_score', 0),
            'test_accuracy': test_metrics.get('accuracy', 0),
            'test_precision': test_metrics.get('precision', 0),
            'test_recall': test_metrics.get('recall', 0),
            'test_auc': test_metrics.get('auc', 0),
        }
        self.fold_results.append(fold_result)

        # Log fold summary metrics
        log_dict = {
            f"fold_{fold_idx+1}/test_f1": test_metrics.get('f1_score', 0),
            f"fold_{fold_idx+1}/test_accuracy": test_metrics.get('accuracy', 0),
            f"fold_{fold_idx+1}/best_epoch": best_epoch,
        }
        wandb.log(log_dict)

        # Store loss histories for averaged plot
        if train_loss_history:
            self.all_folds_train_loss.append(train_loss_history)
        if val_loss_history:
            self.all_folds_val_loss.append(val_loss_history)

        # Log model artifact
        if model_path and os.path.exists(model_path):
            self.log_model_artifact(
                model_path,
                name=f"model-fold-{fold_idx+1}-S{test_subject_str}",
                metadata={
                    'fold': fold_idx + 1,
                    'test_subject': test_subject_str,
                    'test_f1': test_metrics.get('f1_score', 0),
                    'best_epoch': best_epoch,
                }
            )

    def log_model_artifact(
        self,
        model_path: str,
        name: str = "model",
        artifact_type: str = "model",
        metadata: Optional[Dict] = None,
    ):
        """
        Log a model checkpoint as a W&B artifact.

        Args:
            model_path: Path to model file
            name: Artifact name
            artifact_type: Artifact type (default "model")
            metadata: Additional metadata dict
        """
        if not self.enabled or not self.run:
            return

        if not os.path.exists(model_path):
            print(f"[WandbLogger] Model path not found: {model_path}")
            return

        try:
            artifact = wandb.Artifact(
                name=name,
                type=artifact_type,
                metadata=metadata or {},
            )
            artifact.add_file(model_path)
            self.run.log_artifact(artifact)
        except Exception as e:
            print(f"[WandbLogger] Failed to log artifact: {e}")

    def create_fold_summary_table(self) -> Optional['Table']:
        """
        Create a W&B Table with per-fold results summary.

        Returns:
            W&B Table object or None if not enabled
        """
        if not self.enabled or not self.run or not self.fold_results:
            return None

        columns = [
            'fold', 'test_subject', 'best_epoch',
            'train_f1', 'val_f1', 'test_f1',
            'test_accuracy', 'test_precision', 'test_recall', 'test_auc'
        ]

        table = wandb.Table(columns=columns)

        for result in self.fold_results:
            table.add_data(*[result.get(col, '') for col in columns])

        # Log summary statistics separately to run.summary (not in table to avoid type conflicts)
        if len(self.fold_results) > 1:
            summary_stats = {}
            for col in columns[3:]:  # Skip fold, test_subject, best_epoch
                values = [r[col] for r in self.fold_results if isinstance(r.get(col), (int, float))]
                if values:
                    summary_stats[f'avg_{col}'] = round(np.mean(values), 4)
                    summary_stats[f'std_{col}'] = round(np.std(values), 4)
                    summary_stats[f'min_{col}'] = round(min(values), 4)
                    summary_stats[f'max_{col}'] = round(max(values), 4)

            # Log to run summary
            for key, value in summary_stats.items():
                self.run.summary[key] = value

            # Also log number of folds
            self.run.summary['num_folds'] = len(self.fold_results)

        return table

    def log_training_curves(
        self,
        train_losses: List[float],
        val_losses: List[float],
        title: str = "Training Curves",
        fold_idx: Optional[int] = None,
    ):
        """
        Log training/validation loss curves as a figure.

        Args:
            train_losses: List of training losses per epoch
            val_losses: List of validation losses per epoch
            title: Plot title
            fold_idx: Fold index for labeling
        """
        if not self.enabled or not self.run or not MATPLOTLIB_AVAILABLE:
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)

        ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        ax.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Mark best epoch (minimum val loss)
        if val_losses:
            best_epoch = np.argmin(val_losses) + 1
            best_loss = min(val_losses)
            ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best (E{best_epoch})')
            ax.scatter([best_epoch], [best_loss], color='g', s=100, zorder=5)

        plt.tight_layout()

        # Log to W&B
        key = f"fold_{fold_idx+1}/training_curve" if fold_idx is not None else "training_curve"
        wandb.log({key: wandb.Image(fig)})
        plt.close(fig)

    def log_averaged_curves(self):
        """Log training curves averaged across all folds."""
        if not self.enabled or not self.run or not MATPLOTLIB_AVAILABLE:
            return

        if not self.all_folds_train_loss or not self.all_folds_val_loss:
            return

        # Find minimum length across folds
        min_epochs = min(
            min(len(tl) for tl in self.all_folds_train_loss),
            min(len(vl) for vl in self.all_folds_val_loss)
        )

        # Compute mean and std
        train_array = np.array([tl[:min_epochs] for tl in self.all_folds_train_loss])
        val_array = np.array([vl[:min_epochs] for vl in self.all_folds_val_loss])

        train_mean = np.mean(train_array, axis=0)
        train_std = np.std(train_array, axis=0)
        val_mean = np.mean(val_array, axis=0)
        val_std = np.std(val_array, axis=0)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, min_epochs + 1)

        ax.plot(epochs, train_mean, 'b-', label='Train Loss (mean)', linewidth=2)
        ax.fill_between(epochs, train_mean - train_std, train_mean + train_std, alpha=0.2, color='b')

        ax.plot(epochs, val_mean, 'r-', label='Val Loss (mean)', linewidth=2)
        ax.fill_between(epochs, val_mean - val_std, val_mean + val_std, alpha=0.2, color='r')

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'Training Curves (Averaged over {len(self.all_folds_train_loss)} folds)', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        wandb.log({"averaged_training_curve": wandb.Image(fig)})
        plt.close(fig)

    def log_confusion_matrix(
        self,
        y_true: List[int],
        y_pred: List[int],
        class_names: List[str] = None,
        fold_idx: Optional[int] = None,
    ):
        """
        Log a confusion matrix visualization.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names for classes (default ["ADL", "Fall"])
            fold_idx: Fold index for labeling
        """
        if not self.enabled or not self.run:
            return

        if class_names is None:
            class_names = ["ADL", "Fall"]

        # Use W&B's built-in confusion matrix
        key = f"fold_{fold_idx+1}/confusion_matrix" if fold_idx is not None else "confusion_matrix"
        wandb.log({
            key: wandb.plot.confusion_matrix(
                y_true=y_true,
                preds=y_pred,
                class_names=class_names,
            )
        })

    def log_summary(self):
        """Log final summary statistics."""
        if not self.enabled or not self.run or not self.fold_results:
            return

        # Compute summary statistics
        test_f1_scores = [r['test_f1'] for r in self.fold_results]

        summary = {
            'summary/mean_test_f1': np.mean(test_f1_scores),
            'summary/std_test_f1': np.std(test_f1_scores),
            'summary/min_test_f1': np.min(test_f1_scores),
            'summary/max_test_f1': np.max(test_f1_scores),
            'summary/num_folds': len(self.fold_results),
        }

        # Log summary
        wandb.log(summary)

        # Log fold summary table
        table = self.create_fold_summary_table()
        if table:
            wandb.log({"fold_summary": table})

        # Log averaged curves
        self.log_averaged_curves()

        # Update run summary
        wandb.run.summary.update({
            'mean_test_f1': np.mean(test_f1_scores),
            'std_test_f1': np.std(test_f1_scores),
            'best_fold_f1': np.max(test_f1_scores),
            'worst_fold_f1': np.min(test_f1_scores),
        })

    def finish(self):
        """Finish the W&B run."""
        if not self.enabled or not self.run:
            return

        # Log final summary
        self.log_summary()

        # Finish run
        wandb.finish()
        print("[WandbLogger] Run finished.")

    @classmethod
    def sync_offline_runs(cls, run_dir: str = "wandb"):
        """
        Sync offline W&B runs to the server.

        Call this after SLURM jobs complete to upload offline runs.

        Args:
            run_dir: Directory containing offline runs (default "wandb")
        """
        if not WANDB_AVAILABLE:
            print("[WandbLogger] wandb not installed. Cannot sync.")
            return

        try:
            subprocess.run(['wandb', 'sync', '--sync-all', run_dir], check=True)
            print(f"[WandbLogger] Synced offline runs from {run_dir}")
        except subprocess.CalledProcessError as e:
            print(f"[WandbLogger] Sync failed: {e}")
        except FileNotFoundError:
            print("[WandbLogger] wandb CLI not found. Install with: pip install wandb")


def init_wandb_from_args(args, enabled: bool = True) -> WandbLogger:
    """
    Initialize WandbLogger from argparse args.

    Args:
        args: Parsed command-line arguments
        enabled: Whether W&B logging is enabled

    Returns:
        WandbLogger instance
    """
    # Build config dict from args
    config = {
        'model': getattr(args, 'model', None),
        'model_args': getattr(args, 'model_args', {}),
        'dataset_args': getattr(args, 'dataset_args', {}),
        'batch_size': getattr(args, 'batch_size', None),
        'num_epoch': getattr(args, 'num_epoch', None),
        'base_lr': getattr(args, 'base_lr', None),
        'optimizer': getattr(args, 'optimizer', None),
        'weight_decay': getattr(args, 'weight_decay', None),
        'loss_type': getattr(args, 'loss_type', None),
        'seed': getattr(args, 'seed', None),
    }

    # Generate run name from config
    model_name = config.get('model', '').split('.')[-1] if config.get('model') else 'unknown'
    name = f"{model_name}_lr{config.get('base_lr', 0)}_seed{config.get('seed', 0)}"

    # Check for single-fold mode
    single_fold = getattr(args, 'single_fold', None)
    if single_fold is not None:
        name = f"{name}_fold{single_fold}"

    # Determine mode (offline for SLURM)
    mode = "offline" if os.environ.get('SLURM_JOB_ID') else "online"

    return WandbLogger(
        config=config,
        name=name,
        enabled=enabled,
        mode=mode,
    )


# Convenience function for quick setup
def setup_wandb(
    config: Dict,
    project: str = WandbLogger.DEFAULT_PROJECT,
    enabled: bool = True,
) -> WandbLogger:
    """
    Quick setup for W&B logging.

    Args:
        config: Experiment configuration
        project: W&B project name
        enabled: Whether logging is enabled

    Returns:
        WandbLogger instance
    """
    return WandbLogger(config=config, project=project, enabled=enabled)
