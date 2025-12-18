"""
ExperimentRunner: Orchestrates training across models, HP configs, and LOSO folds.

This is the core orchestrator for running experiments. It:
- Iterates over models and HP configurations
- Runs 22-fold LOSO cross-validation
- Logs metrics using MetricsLogger
- Generates graphs for selected folds
- Generates markdown summaries
"""

import os
import sys
import yaml
import json
import torch
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from copy import deepcopy
from collections import Counter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.experiment_config import ExperimentConfig, ModelConfig, HyperparameterConfig
from utils.metrics_logger import MetricsLogger, FoldMetrics
from utils.graph_generator import GraphGenerator
from utils.summary_generator import SummaryGenerator


def init_seed(seed: int):
    """Initialize random seeds for reproducibility."""
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def import_class(import_str: str):
    """Dynamically import a class from a module path string."""
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError(f'Class {class_str} cannot be found in {mod_str}')


class ExperimentRunner:
    """
    Orchestrates experiment execution across models, HP configs, and LOSO folds.

    Workflow:
    1. For each model in experiment config:
       2. For each HP config (sequential):
          3. For each LOSO fold (22 test subjects):
             - Train model
             - Evaluate on val/test
             - Log metrics
          4. Generate graphs for selected folds
          5. Generate model summary
       6. Generate HP comparison for model
    7. Generate experiment summary
    """

    # Subject lists
    ALL_SUBJECTS = [29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 43, 44, 45, 46, 48, 49,
                    50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]

    # Fixed validation subjects (excluded from test folds)
    VALIDATION_SUBJECTS = [48, 57]

    # Train-only subjects (poor data quality, never tested)
    TRAIN_ONLY_SUBJECTS = [29, 30, 32, 35, 39, 59]

    def __init__(
        self,
        experiment_config: ExperimentConfig,
        results_base_dir: str = 'results',
        device: int = 0,
        num_workers: int = 4,
        seed: int = 2,
        save_model_weights: bool = False,
        num_epochs: int = 70,
        batch_size: int = 16,
        patience: int = 15
    ):
        """
        Initialize ExperimentRunner.

        Args:
            experiment_config: Configuration for the experiment
            results_base_dir: Base directory for results
            device: GPU device ID (or 'cpu')
            num_workers: Number of dataloader workers
            seed: Random seed for reproducibility
            save_model_weights: Whether to save model weights (default False)
            num_epochs: Maximum training epochs
            batch_size: Training batch size
            patience: Early stopping patience
        """
        self.config = experiment_config
        self.device = device
        self.num_workers = num_workers
        self.seed = seed
        self.save_model_weights = save_model_weights
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.patience = patience

        # Compute test subjects (exclude validation and train-only)
        self.test_subjects = [
            s for s in self.ALL_SUBJECTS
            if s not in self.VALIDATION_SUBJECTS
            and s not in self.TRAIN_ONLY_SUBJECTS
        ]

        # Create results directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = Path(results_base_dir) / f'{experiment_config.name}_v1_{timestamp}'
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Save experiment config
        self._save_experiment_config()

        print(f'ExperimentRunner initialized:')
        print(f'  Experiment: {experiment_config.name}')
        print(f'  Models: {len(experiment_config.models)}')
        print(f'  HP Configs: {len(experiment_config.hp_configs)}')
        print(f'  Test subjects ({len(self.test_subjects)}): {self.test_subjects}')
        print(f'  Validation subjects: {self.VALIDATION_SUBJECTS}')
        print(f'  Train-only subjects: {self.TRAIN_ONLY_SUBJECTS}')
        print(f'  Results dir: {self.results_dir}')

    def _save_experiment_config(self):
        """Save experiment configuration to file."""
        config_path = self.results_dir / 'experiment_config.yaml'

        config_dict = {
            'name': self.config.name,
            'population': self.config.population,
            'test_subjects': self.test_subjects,
            'validation_subjects': self.VALIDATION_SUBJECTS,
            'train_only_subjects': self.TRAIN_ONLY_SUBJECTS,
            'models': [
                {
                    'name': m.name,
                    'model_class': m.model_class,
                    'description': m.description,
                }
                for m in self.config.models
            ],
            'hp_configs': [
                {
                    'name': hp.name,
                    'learning_rate': hp.lr,
                    'weight_decay': hp.weight_decay,
                    'dropout': hp.dropout,
                }
                for hp in self.config.hp_configs
            ],
            'training': {
                'num_epochs': self.num_epochs,
                'batch_size': self.batch_size,
                'patience': self.patience,
                'seed': self.seed,
            }
        }

        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def _create_trainer_args(
        self,
        model_config: ModelConfig,
        hp_config: HyperparameterConfig,
        work_dir: Path
    ) -> Any:
        """
        Create argument namespace for Trainer.

        Args:
            model_config: Model configuration
            hp_config: Hyperparameter configuration
            work_dir: Working directory for this HP config

        Returns:
            Namespace object compatible with Trainer
        """
        import argparse

        args = argparse.Namespace()

        # Model settings
        args.model = model_config.model_class
        args.model_args = deepcopy(model_config.model_args)

        # Apply HP config dropout if model supports it
        if 'dropout' in args.model_args:
            args.model_args['dropout'] = hp_config.dropout

        # Dataset settings
        args.dataset = 'smartfallmm'
        args.dataset_args = deepcopy(model_config.dataset_args)

        # Training settings
        args.num_epoch = self.num_epochs
        args.batch_size = self.batch_size
        args.test_batch_size = self.batch_size
        args.val_batch_size = self.batch_size
        args.start_epoch = 0

        # Optimizer settings
        args.optimizer = 'AdamW'
        args.base_lr = hp_config.lr
        args.weight_decay = hp_config.weight_decay

        # Device settings
        args.device = [self.device] if isinstance(self.device, int) else [self.device]

        # Directory settings
        args.work_dir = str(work_dir)
        args.model_saved_name = model_config.name

        # Subject settings
        args.subjects = self.ALL_SUBJECTS
        args.validation_subjects = self.VALIDATION_SUBJECTS
        args.train_only_subjects = self.TRAIN_ONLY_SUBJECTS

        # Feeder settings
        args.feeder = 'Feeder.Make_Dataset.UTD_mm'
        args.train_feeder_args = {'batch_size': self.batch_size}
        args.val_feeder_args = {'batch_size': self.batch_size}
        args.test_feeder_args = {'batch_size': self.batch_size}

        # Misc settings
        args.seed = self.seed
        args.num_worker = self.num_workers
        args.phase = 'train'
        args.print_log = True
        args.include_val = True
        args.result_file = None
        args.weights = None
        args.config = None
        args.single_fold = None

        # Loss settings
        args.loss = 'loss.BCE'
        args.loss_args = '{}'
        args.loss_type = 'bce'

        # Grouping disabled for standard LOSO
        args.enable_test_grouping = False

        # Kalman preprocessing (from dataset_args if present)
        args.enable_kalman_preprocessing = model_config.dataset_args.get(
            'enable_kalman_preprocessing', False
        )
        args.kalman_args = model_config.dataset_args.get('kalman_args', {})

        return args

    def _get_training_subjects(self, test_subject: int) -> List[int]:
        """
        Get training subjects for a given test subject.

        Training = all test candidates except current test subject + train-only subjects
        """
        train_subjects = [
            s for s in self.test_subjects
            if s != test_subject
        ] + self.TRAIN_ONLY_SUBJECTS

        return sorted(train_subjects)

    def _compute_val_class_balance(self, builder) -> Dict[str, int]:
        """
        Compute class balance in validation set.

        Args:
            builder: DatasetBuilder with loaded data

        Returns:
            Dict with 'falls' and 'adls' counts
        """
        from utils.dataset import split_by_subjects

        # Check if fuse is needed based on modalities
        modalities = getattr(builder, 'modalities', ['accelerometer'])
        fuse = len([m for m in modalities if m != 'skeleton']) > 1

        val_data = split_by_subjects(builder, self.VALIDATION_SUBJECTS, fuse, print_validation=False)

        if val_data and 'labels' in val_data:
            labels = val_data['labels']
            label_counts = Counter(labels)
            return {
                'falls': int(label_counts.get(1, 0)),
                'adls': int(label_counts.get(0, 0))
            }

        return {'falls': 0, 'adls': 0}

    def run_single_fold(
        self,
        model_config: ModelConfig,
        hp_config: HyperparameterConfig,
        test_subject: int,
        metrics_logger: MetricsLogger,
        work_dir: Path
    ) -> Optional[Dict[str, Any]]:
        """
        Run training for a single LOSO fold.

        Args:
            model_config: Model configuration
            hp_config: Hyperparameter configuration
            test_subject: Subject ID for testing
            metrics_logger: MetricsLogger instance
            work_dir: Working directory

        Returns:
            Dict with test metrics and predictions, or None if fold failed
        """
        from main import Trainer
        from utils.dataset import prepare_smartfallmm, split_by_subjects
        from utils.callbacks import EarlyStopping

        print(f'\n{"="*60}')
        print(f'Fold: Test Subject {test_subject}')
        print(f'{"="*60}')

        # Create trainer args
        args = self._create_trainer_args(model_config, hp_config, work_dir)

        # Initialize seed
        init_seed(self.seed)

        try:
            # Create trainer
            trainer = Trainer(args)

            # Set up subjects for this fold
            train_subjects = self._get_training_subjects(test_subject)
            trainer.train_subjects = train_subjects
            trainer.val_subject = self.VALIDATION_SUBJECTS
            trainer.test_subject = [test_subject]

            # Initialize fold in metrics logger
            fold_id = self.test_subjects.index(test_subject)
            fold = metrics_logger.add_fold(fold_id, str(test_subject))
            trainer._init_fold_tracking(fold_id, test_subject)

            # Load data
            if not trainer.load_data():
                print(f'Failed to load data for subject {test_subject}')
                return None

            # Update validation class balance in logger
            if trainer.builder:
                val_balance = self._compute_val_class_balance(trainer.builder)
                metrics_logger.update_val_class_balance(
                    falls=val_balance['falls'],
                    adls=val_balance['adls']
                )

            # Load optimizer and loss
            trainer.load_optimizer(trainer.model.parameters())
            trainer.load_loss()

            # Training loop
            early_stop = EarlyStopping(patience=self.patience, min_delta=0.001)
            best_epoch = 0

            for epoch in range(self.num_epochs):
                # Train
                trainer.train(epoch)

                # Get metrics from trainer
                train_metrics = trainer.current_fold_metrics.get('train', {})
                val_metrics = trainer.current_fold_metrics.get('val', {})

                # Log to MetricsLogger
                metrics_logger.log_epoch(
                    fold_id=fold_id,
                    epoch=epoch,
                    train_loss=train_metrics.get('loss', 0),
                    val_loss=val_metrics.get('loss', 0),
                    train_f1=train_metrics.get('f1_score', 0) / 100,  # Convert from %
                    val_f1=val_metrics.get('f1_score', 0) / 100,
                    train_acc=train_metrics.get('accuracy', 0) / 100,
                    val_acc=val_metrics.get('accuracy', 0) / 100
                )

                # Check for best epoch
                if val_metrics.get('loss', float('inf')) < trainer.best_loss:
                    best_epoch = epoch

                # Early stopping
                early_stop(val_metrics.get('loss', float('inf')))
                if early_stop.early_stop:
                    print(f'Early stopping at epoch {epoch + 1}')
                    break

            # Load best weights and evaluate on test
            trainer.load_model(args.model, args.model_args)
            trainer.load_weights()
            trainer.model.eval()

            # Evaluate test set
            test_loss = trainer.eval(epoch=0, loader_name='test')
            test_metrics = trainer.current_fold_metrics.get('test', {})

            # Log test metrics
            metrics_logger.log_test_metrics(
                fold_id=fold_id,
                test_loss=test_loss,
                test_f1=test_metrics.get('f1_score', 0) / 100,
                test_acc=test_metrics.get('accuracy', 0) / 100,
                test_precision=test_metrics.get('precision', 0) / 100,
                test_recall=test_metrics.get('recall', 0) / 100,
                test_auc=test_metrics.get('auc', 0) / 100
            )

            # Get predictions for graphs (if needed)
            y_true, y_pred, y_prob = self._get_test_predictions(trainer)

            print(f'Test Subject {test_subject}: F1={test_metrics.get("f1_score", 0):.2f}%, '
                  f'Acc={test_metrics.get("accuracy", 0):.2f}%')

            return {
                'test_subject': test_subject,
                'best_epoch': best_epoch,
                'test_metrics': test_metrics,
                'y_true': y_true,
                'y_pred': y_pred,
                'y_prob': y_prob
            }

        except Exception as e:
            print(f'Error in fold for subject {test_subject}: {e}')
            import traceback
            traceback.print_exc()
            return None

    def _get_test_predictions(self, trainer) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get test predictions from trainer for ROC/confusion matrix graphs.

        Returns:
            Tuple of (y_true, y_pred, y_prob) arrays
        """
        y_true_list = []
        y_pred_list = []
        y_prob_list = []

        device_str = trainer._get_device_str()

        with torch.no_grad():
            for batch_idx, (inputs, targets, idx) in enumerate(trainer.data_loader['test']):
                acc_key = trainer._get_inertial_key(inputs)
                acc_data = inputs[acc_key].to(device_str)
                skl_tensor = None
                if trainer.use_skeleton and 'skeleton' in inputs:
                    skl_tensor = inputs['skeleton'].to(device_str)
                targets = targets.to(device_str)

                logits, _ = trainer.model(
                    acc_data.float(),
                    skl_tensor.float() if skl_tensor is not None else None
                )

                probs = torch.sigmoid(logits).squeeze(1)
                preds = (probs > 0.5).int()

                y_true_list.extend(targets.cpu().numpy())
                y_pred_list.extend(preds.cpu().numpy())
                y_prob_list.extend(probs.cpu().numpy())

        return np.array(y_true_list), np.array(y_pred_list), np.array(y_prob_list)

    def run_model_with_hp(
        self,
        model_config: ModelConfig,
        hp_config: HyperparameterConfig
    ) -> Dict[str, Any]:
        """
        Run all LOSO folds for a model with specific HP config.

        Args:
            model_config: Model configuration
            hp_config: Hyperparameter configuration

        Returns:
            Dict with aggregated results
        """
        print(f'\n{"#"*60}')
        print(f'Model: {model_config.name}')
        print(f'HP Config: {hp_config.name}')
        print(f'  LR: {hp_config.lr}, WD: {hp_config.weight_decay}, '
              f'Dropout: {hp_config.dropout}')
        print(f'{"#"*60}')

        # Create work directory
        work_dir = self.results_dir / model_config.name / hp_config.name
        work_dir.mkdir(parents=True, exist_ok=True)

        # Save config for this run
        config_dict = {
            'model': model_config.model_class,
            'model_args': model_config.model_args,
            'dataset_args': model_config.dataset_args,
            'hp': {
                'name': hp_config.name,
                'learning_rate': hp_config.lr,
                'weight_decay': hp_config.weight_decay,
                'dropout': hp_config.dropout,
            },
            'validation_subjects': self.VALIDATION_SUBJECTS,
            'train_only_subjects': self.TRAIN_ONLY_SUBJECTS,
        }
        with open(work_dir / 'config.yaml', 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        # Initialize metrics logger
        metrics_logger = MetricsLogger(
            output_dir=self.results_dir,
            experiment_name=model_config.name,
            model_name=hp_config.name,
            hp_name='',  # Already in path
            validation_subjects=self.VALIDATION_SUBJECTS
        )

        # Adjust metrics_dir to match our structure
        metrics_logger.metrics_dir = work_dir

        # Store fold results for graph generation
        fold_results = {}

        # Run all LOSO folds
        for test_subject in self.test_subjects:
            result = self.run_single_fold(
                model_config=model_config,
                hp_config=hp_config,
                test_subject=test_subject,
                metrics_logger=metrics_logger,
                work_dir=work_dir
            )

            if result:
                fold_results[test_subject] = result

        # Compute and save summary statistics
        stats = metrics_logger.compute_summary_stats()

        # Generate graphs for selected folds
        self._generate_fold_graphs(metrics_logger, fold_results, work_dir)

        print(f'\n{"="*60}')
        print(f'Completed: {model_config.name} / {hp_config.name}')
        print(f'Test F1: {stats.get("test_f1", {}).get("mean", 0)*100:.2f} '
              f'+/- {stats.get("test_f1", {}).get("std", 0)*100:.2f}%')
        print(f'{"="*60}')

        return {
            'model': model_config.name,
            'hp': hp_config.name,
            'stats': stats,
            'fold_count': len(fold_results)
        }

    def _generate_fold_graphs(
        self,
        metrics_logger: MetricsLogger,
        fold_results: Dict[int, Dict],
        work_dir: Path
    ):
        """
        Generate graphs for selected folds.

        Generates graphs for:
        - Top 3 best F1 folds
        - Top 3 worst F1 folds
        - Top 2 least overfit folds
        - Top 2 lowest validation loss folds
        - Averaged curves across all folds
        """
        # Get fold selections
        fold_selections = metrics_logger.get_folds_for_graphs()

        if not fold_selections:
            print('No folds available for graph generation')
            return

        # Create graph generator
        graph_gen = GraphGenerator(
            output_dir=work_dir,
            metrics_logger=metrics_logger
        )

        # Generate graphs for each category
        for category, fold_ids in fold_selections.items():
            for fold_id in fold_ids:
                # Find corresponding test subject
                test_subject = self.test_subjects[fold_id] if fold_id < len(self.test_subjects) else None

                if test_subject and test_subject in fold_results:
                    result = fold_results[test_subject]
                    graph_gen.generate_fold_graphs(
                        fold_id=fold_id,
                        category=category,
                        y_true=result.get('y_true'),
                        y_pred=result.get('y_pred'),
                        y_prob=result.get('y_prob')
                    )

        # Generate averaged graphs
        graph_gen.generate_averaged_graphs()

    def run_model(self, model_config: ModelConfig) -> List[Dict[str, Any]]:
        """
        Run all HP configs for a single model.

        Args:
            model_config: Model configuration

        Returns:
            List of results for each HP config
        """
        results = []

        for hp_config in self.config.hp_configs:
            result = self.run_model_with_hp(model_config, hp_config)
            results.append(result)

        # Generate model summary
        self._generate_model_summary(model_config)

        return results

    def _generate_model_summary(self, model_config: ModelConfig):
        """Generate summary markdown for a model across all HP configs."""
        summary_gen = SummaryGenerator(self.results_dir)

        # Adjust path structure for summary generator
        # Our structure: results_dir / model_name / hp_name /
        # SummaryGenerator expects: results_dir / experiment_name / model_name / hp_name /

        # Generate model summary
        summary = summary_gen.generate_model_summary(
            experiment_name='',  # Models are directly under results_dir
            model_name=model_config.name
        )

        print(f'Generated summary for {model_config.name}')

    def run(self) -> Dict[str, Any]:
        """
        Run the complete experiment.

        Returns:
            Dict with all experiment results
        """
        print(f'\n{"*"*60}')
        print(f'STARTING EXPERIMENT: {self.config.name}')
        print(f'{"*"*60}')

        all_results = []

        for model_config in self.config.models:
            model_results = self.run_model(model_config)
            all_results.extend(model_results)

        # Generate experiment summary
        summary_gen = SummaryGenerator(self.results_dir)

        # Our results are directly in results_dir, not under experiment_name
        # Create a wrapper structure for summary generator
        experiment_summary = self._generate_experiment_summary(all_results)

        print(f'\n{"*"*60}')
        print(f'EXPERIMENT COMPLETE: {self.config.name}')
        print(f'Results saved to: {self.results_dir}')
        print(f'{"*"*60}')

        return {
            'experiment': self.config.name,
            'results_dir': str(self.results_dir),
            'model_results': all_results
        }

    def _generate_experiment_summary(self, all_results: List[Dict]) -> str:
        """
        Generate experiment-level summary comparing all models.

        Args:
            all_results: List of results from all model/HP combinations

        Returns:
            Summary markdown string
        """
        lines = [
            f'# {self.config.name} - Experiment Summary',
            f'\n**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            f'**Total Models**: {len(self.config.models)}',
            f'**HP Configs per Model**: {len(self.config.hp_configs)}',
            f'**Test Subjects**: {len(self.test_subjects)}',
            '',
            '---',
            '',
            '## Best Results per Model',
            '',
            '| Model | Best HP | Test F1 (%) | Test Acc (%) | Overfit Gap |',
            '|-------|---------|-------------|--------------|-------------|',
        ]

        # Group results by model
        model_results = {}
        for result in all_results:
            model_name = result['model']
            if model_name not in model_results:
                model_results[model_name] = []
            model_results[model_name].append(result)

        # Find best HP for each model
        ranked_models = []
        for model_name, results in model_results.items():
            best_result = max(
                results,
                key=lambda x: x['stats'].get('test_f1', {}).get('mean', 0)
            )
            ranked_models.append((model_name, best_result))

        # Sort by F1
        ranked_models.sort(
            key=lambda x: x[1]['stats'].get('test_f1', {}).get('mean', 0),
            reverse=True
        )

        for model_name, best_result in ranked_models:
            stats = best_result['stats']
            tf1 = stats.get('test_f1', {})
            ta = stats.get('test_acc', {})
            og = stats.get('overfit_gap', {})

            lines.append(
                f"| {model_name} | {best_result['hp']} | "
                f"{tf1.get('mean', 0)*100:.2f} +/- {tf1.get('std', 0)*100:.2f} | "
                f"{ta.get('mean', 0)*100:.2f} +/- {ta.get('std', 0)*100:.2f} | "
                f"{og.get('mean', 0):.4f} |"
            )

        # Add validation info
        lines.extend([
            '',
            '---',
            '',
            '## Validation Configuration',
            '',
            f'**Validation Subjects**: {self.VALIDATION_SUBJECTS}',
            f'**Train-Only Subjects**: {self.TRAIN_ONLY_SUBJECTS}',
            '',
            '---',
            '',
            '## Notes',
            '',
            f'- {len(self.test_subjects)}-fold LOSO cross-validation',
            '- Overfit Gap = train_f1 - val_f1 at best validation epoch (lower = better)',
            '- Test subjects exclude validation subjects and train-only subjects',
            '',
        ])

        summary = '\n'.join(lines)

        # Save summary
        summary_path = self.results_dir / f'{self.config.name}_summary.md'
        with open(summary_path, 'w') as f:
            f.write(summary)

        # Also generate comparison CSV
        self._generate_comparison_csv(all_results)

        return summary

    def _generate_comparison_csv(self, all_results: List[Dict]):
        """Generate comparison CSV with best HP per model."""
        import csv

        rows = []

        # Group by model and find best
        model_results = {}
        for result in all_results:
            model_name = result['model']
            if model_name not in model_results:
                model_results[model_name] = []
            model_results[model_name].append(result)

        for model_name, results in model_results.items():
            best_result = max(
                results,
                key=lambda x: x['stats'].get('test_f1', {}).get('mean', 0)
            )
            stats = best_result['stats']

            rows.append({
                'model': model_name,
                'best_hp': best_result['hp'],
                'test_f1_mean': stats.get('test_f1', {}).get('mean', 0),
                'test_f1_std': stats.get('test_f1', {}).get('std', 0),
                'test_acc_mean': stats.get('test_acc', {}).get('mean', 0),
                'test_acc_std': stats.get('test_acc', {}).get('std', 0),
                'test_precision': stats.get('test_precision', {}).get('mean', 0),
                'test_recall': stats.get('test_recall', {}).get('mean', 0),
                'test_auc': stats.get('test_auc', {}).get('mean', 0),
                'overfit_gap': stats.get('overfit_gap', {}).get('mean', 0),
                'validation_subjects': str(self.VALIDATION_SUBJECTS),
                'val_falls': stats.get('val_class_balance', {}).get('falls', 'N/A'),
                'val_adls': stats.get('val_class_balance', {}).get('adls', 'N/A'),
            })

        # Sort by F1
        rows.sort(key=lambda x: x['test_f1_mean'], reverse=True)

        # Add rank
        for i, row in enumerate(rows, 1):
            row['rank'] = i

        # Reorder columns
        headers = ['rank', 'model', 'best_hp', 'test_f1_mean', 'test_f1_std',
                   'test_acc_mean', 'test_acc_std', 'test_precision', 'test_recall',
                   'test_auc', 'overfit_gap', 'validation_subjects', 'val_falls', 'val_adls']

        csv_path = self.results_dir / f'{self.config.name}_comparison.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

        print(f'Comparison CSV saved to: {csv_path}')


if __name__ == '__main__':
    # Test with EXP1 config
    from experiments import get_exp1_config

    config = get_exp1_config()

    print(f'Testing ExperimentRunner with {config.name}')
    print(f'Models: {[m.name for m in config.models]}')
    print(f'HP Configs: {[hp.name for hp in config.hp_configs]}')

    # Don't actually run - just test initialization
    runner = ExperimentRunner(
        experiment_config=config,
        results_base_dir='results/test',
        device=0
    )

    print('\nExperimentRunner initialized successfully!')
    print(f'Results dir: {runner.results_dir}')
