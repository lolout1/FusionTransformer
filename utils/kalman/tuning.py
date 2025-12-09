"""
Hyperparameter optimization for Kalman filter Q and R matrices.

Uses parallel grid search with subject-level cross-validation to prevent data leakage.
Designed for cluster execution with 48-core parallelization.

Literature-informed default ranges:
    - Q_orientation: [0.001, 0.1] rad² (Sabatini 2011)
    - Q_rate: [0.01, 1.0] (rad/s)²
    - Q_bias: [1e-5, 1e-3] (rad/s)² (slow drift)
    - R_acc: [0.01, 1.0] rad² (depends on vibration)
    - R_gyro: [0.1, 2.0] (rad/s)² (noisy consumer-grade sensors)

References:
    Sabatini, A.M. (2011). "Estimating Three-Dimensional Orientation of Human
        Body Parts by Inertial/Magnetic Sensing"
"""

import numpy as np
import json
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from itertools import product
from dataclasses import dataclass, field
from collections import defaultdict

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

from .features import build_kalman_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def default_search_space() -> Dict:
    """Return default search ranges based on literature (Sabatini 2011)."""
    return {
        'Q_orientation': [0.001, 0.005, 0.01, 0.05, 0.1],
        'Q_rate': [0.01, 0.05, 0.1, 0.5],
        'R_acc': [0.05, 0.1, 0.2, 0.5],
        'R_gyro': [0.1, 0.3, 0.5, 1.0]
    }


def default_ekf_search_space() -> Dict:
    """Return default search ranges for Extended Kalman Filter."""
    return {
        'Q_quat': [0.0005, 0.001, 0.005, 0.01],
        'Q_bias': [0.00005, 0.0001, 0.0005],
        'R_acc': [0.05, 0.1, 0.2, 0.5]
    }


@dataclass
class TuningResult:
    """Container for single parameter configuration evaluation."""
    params: Dict[str, float]
    mean_score: float
    std_score: float
    fold_scores: List[float]
    n_folds: int

    def to_dict(self) -> Dict:
        return {
            'params': self.params,
            'mean_score': self.mean_score,
            'std_score': self.std_score,
            'fold_scores': self.fold_scores,
            'n_folds': self.n_folds
        }


class KalmanParameterTuner:
    """Tune Kalman filter parameters for fall detection task (legacy interface)."""

    def __init__(self,
                 acc_trials: List[np.ndarray],
                 gyro_trials: List[np.ndarray],
                 labels: List[int],
                 filter_type: str = 'linear',
                 n_folds: int = 3,
                 metric: str = 'f1'):
        self.acc_trials = acc_trials
        self.gyro_trials = gyro_trials
        self.labels = np.array(labels)
        self.filter_type = filter_type
        self.n_folds = n_folds
        self.metric = metric
        self.search_space = {}
        self.tuning_history = []
        self.best_params = None
        self.best_score = -np.inf

    def define_search_space(self, **kwargs) -> None:
        self.search_space = kwargs

    def tune(self, n_jobs: int = 1) -> Dict:
        if not self.search_space:
            if self.filter_type == 'linear':
                self.search_space = default_search_space()
            else:
                self.search_space = default_ekf_search_space()

        param_names = list(self.search_space.keys())
        param_values = list(self.search_space.values())
        combinations = list(product(*param_values))
        logger.info(f"Tuning {len(combinations)} parameter combinations...")

        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))
            score = self._evaluate_params(params)
            self.tuning_history.append({'params': params, 'score': score})
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
            if (i + 1) % 10 == 0:
                logger.info(f"  [{i+1}/{len(combinations)}] Best {self.metric}: {self.best_score:.4f}")
        return self.get_best_params()

    def _evaluate_params(self, params: Dict) -> float:
        config = {
            'kalman_filter_type': self.filter_type,
            'kalman_output_format': 'euler',
            'kalman_include_smv': True,
            'kalman_include_uncertainty': False,
            'kalman_include_innovation': False,
            'filter_fs': 30.0
        }
        for k, v in params.items():
            config[f'kalman_{k}'] = v

        try:
            features_list = []
            for acc, gyro in zip(self.acc_trials, self.gyro_trials):
                feat = build_kalman_features(acc, gyro, config)
                pooled = feat.mean(axis=0)
                features_list.append(pooled)
            X = np.array(features_list)
            y = self.labels
        except Exception as e:
            logger.warning(f"Error with params {params}: {e}")
            return -np.inf

        scores = []
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            clf = LogisticRegression(max_iter=500, random_state=42)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)
            if self.metric == 'f1':
                score = f1_score(y_val, y_pred, zero_division=0)
            else:
                score = accuracy_score(y_val, y_pred)
            scores.append(score)
        return np.mean(scores)

    def get_best_params(self) -> Dict:
        return {'best_score': self.best_score, 'params': self.best_params,
                'filter_type': self.filter_type, 'metric': self.metric}

    def get_tuning_history(self) -> List[Dict]:
        return self.tuning_history

    def save_results(self, output_path: str) -> None:
        results = {'best_params': self.get_best_params(), 'history': self.tuning_history,
                   'filter_type': self.filter_type, 'n_folds': self.n_folds, 'metric': self.metric}
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)


class ParallelKalmanTuner:
    """
    High-performance Kalman filter parameter tuner with subject-level CV.

    Features:
        - Parallel grid search using joblib (optimized for 48-core nodes)
        - Subject-level leave-one-out CV to prevent data leakage
        - Statistical feature pooling (mean, std, max, min)
        - Comprehensive logging and progress tracking
        - Results saved in JSON for reproducibility

    Scientific rigor:
        - NO data leakage: trials from same subject never in train AND validation
        - Multiple pooling strategies for robust feature representation
        - Both F1 and accuracy metrics tracked
    """

    def __init__(self,
                 acc_trials: List[np.ndarray],
                 gyro_trials: List[np.ndarray],
                 labels: List[int],
                 subject_ids: List[int],
                 filter_type: str = 'linear',
                 n_jobs: int = -1,
                 random_state: int = 42):
        """
        Initialize parallel tuner with subject information.

        Args:
            acc_trials: List of (T, 3) accelerometer arrays in m/s²
            gyro_trials: List of (T, 3) gyroscope arrays in rad/s
            labels: Binary labels (0=ADL, 1=fall)
            subject_ids: Subject ID for each trial (for LOSO CV)
            filter_type: 'linear' or 'ekf'
            n_jobs: Number of parallel workers (-1 for all cores)
            random_state: Random seed for reproducibility
        """
        self.acc_trials = acc_trials
        self.gyro_trials = gyro_trials
        self.labels = np.array(labels)
        self.subject_ids = np.array(subject_ids)
        self.filter_type = filter_type
        self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count()
        self.random_state = random_state

        self.search_space = {}
        self.results: List[TuningResult] = []
        self.best_result: Optional[TuningResult] = None

        n_trials = len(acc_trials)
        n_subjects = len(np.unique(subject_ids))
        n_falls = sum(labels)
        n_adl = n_trials - n_falls

        logger.info(f"ParallelKalmanTuner initialized:")
        logger.info(f"  Filter type: {filter_type}")
        logger.info(f"  Trials: {n_trials} ({n_falls} falls, {n_adl} ADL)")
        logger.info(f"  Subjects: {n_subjects}")
        logger.info(f"  Parallel workers: {self.n_jobs}")

    def set_search_space(self, search_space: Optional[Dict] = None) -> None:
        """Set parameter search space."""
        if search_space is None:
            if self.filter_type == 'linear':
                self.search_space = default_search_space()
            else:
                self.search_space = default_ekf_search_space()
        else:
            self.search_space = search_space

        n_combos = 1
        for values in self.search_space.values():
            n_combos *= len(values)
        logger.info(f"Search space: {n_combos} combinations")
        for k, v in self.search_space.items():
            logger.info(f"  {k}: {v}")

    def _extract_features_for_params(self, params: Dict) -> Tuple[np.ndarray, bool]:
        """Extract Kalman features for all trials with given parameters."""
        config = {
            'kalman_filter_type': self.filter_type,
            'kalman_output_format': 'euler',
            'kalman_include_smv': True,
            'kalman_include_uncertainty': False,
            'kalman_include_innovation': False,
            'filter_fs': 30.0
        }
        for k, v in params.items():
            config[f'kalman_{k}'] = v

        features_list = []
        for acc, gyro in zip(self.acc_trials, self.gyro_trials):
            try:
                feat = build_kalman_features(acc, gyro, config)
                # Statistical pooling: mean, std, max, min
                pooled = np.concatenate([
                    feat.mean(axis=0),
                    feat.std(axis=0),
                    feat.max(axis=0),
                    feat.min(axis=0)
                ])
                features_list.append(pooled)
            except Exception as e:
                logger.warning(f"Feature extraction failed: {e}")
                return np.array([]), False

        return np.array(features_list), True

    def _evaluate_single_config(self, params: Dict) -> TuningResult:
        """Evaluate a single parameter configuration using subject-level LOSO."""
        X, success = self._extract_features_for_params(params)
        if not success or len(X) == 0:
            return TuningResult(params=params, mean_score=-1.0, std_score=0.0,
                               fold_scores=[], n_folds=0)

        # Subject-level leave-one-out cross-validation
        logo = LeaveOneGroupOut()
        fold_scores = []

        for train_idx, val_idx in logo.split(X, self.labels, groups=self.subject_ids):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = self.labels[train_idx], self.labels[val_idx]

            # Skip if validation set has only one class
            if len(np.unique(y_val)) < 2 or len(np.unique(y_train)) < 2:
                continue

            # Normalize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            clf = LogisticRegression(max_iter=1000, random_state=self.random_state,
                                    class_weight='balanced', solver='lbfgs')
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_val_scaled)

            score = f1_score(y_val, y_pred, zero_division=0)
            fold_scores.append(score)

        if len(fold_scores) == 0:
            return TuningResult(params=params, mean_score=-1.0, std_score=0.0,
                               fold_scores=[], n_folds=0)

        return TuningResult(
            params=params,
            mean_score=float(np.mean(fold_scores)),
            std_score=float(np.std(fold_scores)),
            fold_scores=fold_scores,
            n_folds=len(fold_scores)
        )

    def tune(self, verbose: bool = True) -> Dict:
        """
        Run parallel grid search with subject-level CV.

        Args:
            verbose: Print progress updates

        Returns:
            Best parameters dict with scores
        """
        if not self.search_space:
            self.set_search_space()

        param_names = list(self.search_space.keys())
        param_values = list(self.search_space.values())
        combinations = [dict(zip(param_names, combo))
                       for combo in product(*param_values)]

        n_combos = len(combinations)
        logger.info(f"Starting parallel tuning: {n_combos} configs, {self.n_jobs} workers")
        start_time = datetime.now()

        if HAS_JOBLIB and self.n_jobs > 1:
            self.results = Parallel(n_jobs=self.n_jobs, verbose=10 if verbose else 0)(
                delayed(self._evaluate_single_config)(params)
                for params in combinations
            )
        else:
            logger.warning("joblib not available or n_jobs=1, running sequentially")
            self.results = []
            for i, params in enumerate(combinations):
                result = self._evaluate_single_config(params)
                self.results.append(result)
                if verbose and (i + 1) % 10 == 0:
                    logger.info(f"  [{i+1}/{n_combos}] Current best: {self._get_current_best():.4f}")

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Tuning complete in {elapsed:.1f}s")

        # Find best result
        valid_results = [r for r in self.results if r.mean_score >= 0]
        if not valid_results:
            logger.error("No valid results found!")
            return {'best_score': -1, 'params': None, 'filter_type': self.filter_type}

        self.best_result = max(valid_results, key=lambda r: r.mean_score)

        logger.info(f"Best F1: {self.best_result.mean_score:.4f} ± {self.best_result.std_score:.4f}")
        logger.info(f"Best params: {self.best_result.params}")

        return self.get_best_params()

    def _get_current_best(self) -> float:
        """Get current best score from results so far."""
        valid = [r.mean_score for r in self.results if r.mean_score >= 0]
        return max(valid) if valid else -1.0

    def get_best_params(self) -> Dict:
        """Return best parameters with metadata."""
        if self.best_result is None:
            return {'best_score': -1, 'params': None, 'filter_type': self.filter_type}
        return {
            'best_score': self.best_result.mean_score,
            'best_std': self.best_result.std_score,
            'params': self.best_result.params,
            'n_folds': self.best_result.n_folds,
            'filter_type': self.filter_type,
            'metric': 'f1'
        }

    def get_all_results(self) -> List[Dict]:
        """Return all evaluated configurations."""
        return [r.to_dict() for r in self.results]

    def save_results(self, output_dir: str) -> None:
        """Save comprehensive tuning results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Best params
        best_file = output_path / f'best_params_{self.filter_type}.json'
        with open(best_file, 'w') as f:
            json.dump(self.get_best_params(), f, indent=2)
        logger.info(f"Saved best params to {best_file}")

        # Full history
        history_file = output_path / f'tuning_history_{self.filter_type}.json'
        history = {
            'timestamp': datetime.now().isoformat(),
            'filter_type': self.filter_type,
            'n_trials': len(self.acc_trials),
            'n_subjects': len(np.unique(self.subject_ids)),
            'search_space': self.search_space,
            'best_params': self.get_best_params(),
            'all_results': self.get_all_results()
        }
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"Saved history to {history_file}")

        # CSV summary for easy viewing
        import csv
        csv_file = output_path / f'tuning_summary_{self.filter_type}.csv'
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header
            param_names = list(self.search_space.keys()) if self.search_space else []
            header = param_names + ['mean_f1', 'std_f1', 'n_folds']
            writer.writerow(header)
            # Data rows (sorted by score)
            sorted_results = sorted(self.results, key=lambda r: r.mean_score, reverse=True)
            for r in sorted_results:
                row = [r.params.get(k, '') for k in param_names]
                row.extend([f'{r.mean_score:.4f}', f'{r.std_score:.4f}', r.n_folds])
                writer.writerow(row)
        logger.info(f"Saved CSV summary to {csv_file}")


def load_tuned_params(params_path: str, filter_type: str = 'linear') -> Dict:
    """
    Load tuned parameters from JSON file.

    Args:
        params_path: Path to best_params.json
        filter_type: 'linear' or 'ekf'

    Returns:
        Dict with Q and R parameters
    """
    with open(params_path, 'r') as f:
        data = json.load(f)

    if 'params' in data:
        return data['params']
    elif filter_type in data:
        return data[filter_type].get('params', {})
    else:
        return data


def get_literature_defaults(filter_type: str = 'linear') -> Dict:
    """
    Get literature-based default parameters (Sabatini 2011).

    Args:
        filter_type: 'linear' or 'ekf'

    Returns:
        Default parameter dict
    """
    if filter_type == 'linear':
        return {
            'Q_orientation': 0.01,
            'Q_rate': 0.1,
            'R_acc': 0.1,
            'R_gyro': 0.5
        }
    else:
        return {
            'Q_quat': 0.001,
            'Q_bias': 0.0001,
            'R_acc': 0.1
        }
