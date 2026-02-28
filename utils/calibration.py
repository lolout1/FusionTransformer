"""Post-hoc temperature scaling for probability calibration.

Learns a single scalar T that minimizes NLL on held-out predictions.
Focal loss (gamma=2) produces overconfident probabilities — temperature
scaling softens them so threshold=0.5 becomes meaningful.

Reference: Guo et al., "On Calibration of Modern Neural Networks", ICML 2017.
"""

import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit


class TemperatureScaler:
    """Learn and apply temperature scaling to model probabilities."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        self.fit_nll = None
        self.n_samples = 0

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> 'TemperatureScaler':
        """Learn optimal temperature from probabilities and binary labels.

        Converts probabilities to logits, then finds T that minimizes
        negative log-likelihood: NLL = -mean(y*log(σ(z/T)) + (1-y)*log(1-σ(z/T)))
        """
        probs = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1 - 1e-7)
        y = np.asarray(labels, dtype=np.float64)
        logits = logit(probs)
        self.n_samples = len(y)

        def nll(T):
            scaled = logits / T
            p = expit(scaled)
            p = np.clip(p, 1e-7, 1 - 1e-7)
            return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

        result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
        self.temperature = result.x
        self.fit_nll = result.fun
        return self

    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to probabilities."""
        probs = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1 - 1e-7)
        logits = logit(probs)
        return expit(logits / self.temperature)

    def calibrate_logits(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling directly to logits, return probabilities."""
        return expit(np.asarray(logits, dtype=np.float64) / self.temperature)

    def __repr__(self):
        return f"TemperatureScaler(T={self.temperature:.4f}, n={self.n_samples}, nll={self.fit_nll:.4f})"


def learn_temperature_from_folds(
    fold_results_path: str,
    method: str = 'pooled',
) -> TemperatureScaler:
    """Learn temperature from stored fold_results.pkl.

    Args:
        fold_results_path: Path to fold_results.pkl
        method: 'pooled' — pool all folds' test predictions, learn single T.

    Returns:
        Fitted TemperatureScaler.
    """
    with open(fold_results_path, 'rb') as f:
        results = pickle.load(f)

    all_probs, all_labels = [], []
    for fold in results:
        ta = fold.get('threshold_analysis')
        if ta is None:
            continue
        probs = ta.get('probabilities', [])
        targets = ta.get('targets', [])
        if probs and targets:
            all_probs.extend(probs)
            all_labels.extend(targets)

    if not all_probs:
        raise ValueError(f"No probabilities found in {fold_results_path}")

    scaler = TemperatureScaler()
    scaler.fit(np.array(all_probs), np.array(all_labels))
    return scaler


def learn_per_fold_temperature(
    fold_results_path: str,
) -> dict:
    """Learn leave-one-out temperature per fold.

    For fold i, T_i is learned from all OTHER folds' test data.
    Returns dict mapping test_subject -> TemperatureScaler.
    """
    with open(fold_results_path, 'rb') as f:
        results = pickle.load(f)

    fold_data = []
    for fold in results:
        ta = fold.get('threshold_analysis')
        if ta is None:
            continue
        probs = ta.get('probabilities', [])
        targets = ta.get('targets', [])
        if probs and targets:
            fold_data.append({
                'subject': fold['test_subject'],
                'probs': np.array(probs),
                'labels': np.array(targets),
            })

    scalers = {}
    for i, fold in enumerate(fold_data):
        # Pool all other folds
        other_probs = np.concatenate([fd['probs'] for j, fd in enumerate(fold_data) if j != i])
        other_labels = np.concatenate([fd['labels'] for j, fd in enumerate(fold_data) if j != i])
        scaler = TemperatureScaler()
        scaler.fit(other_probs, other_labels)
        scalers[fold['subject']] = scaler

    return scalers


def calibration_diagnostics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Compute calibration diagnostics: ECE, MCE, reliability diagram data."""
    probs = np.asarray(probabilities)
    y = np.asarray(labels)
    bins = np.linspace(0, 1, n_bins + 1)

    ece = 0.0
    mce = 0.0
    bin_data = []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        bin_probs = probs[mask]
        bin_labels = y[mask]
        avg_conf = bin_probs.mean()
        avg_acc = bin_labels.mean()
        n = mask.sum()
        gap = abs(avg_conf - avg_acc)
        ece += gap * n / len(probs)
        mce = max(mce, gap)
        bin_data.append({
            'bin': f'{lo:.1f}-{hi:.1f}',
            'count': int(n),
            'avg_confidence': float(avg_conf),
            'avg_accuracy': float(avg_acc),
            'gap': float(gap),
        })

    return {
        'ece': float(ece),
        'mce': float(mce),
        'n_bins': n_bins,
        'bins': bin_data,
    }
