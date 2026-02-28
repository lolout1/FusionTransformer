"""Temporal queue evaluation with realistic trial-ordered streams.

Loads raw trial data per LOSO test subject, processes in temporal order,
and runs alpha queue simulation with stride ablation and aggregation
method comparison (average vs majority vote with configurable k).

Includes post-hoc temperature scaling for probability calibration.
"""

import csv
import json
import os
import pickle
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, roc_auc_score, matthews_corrcoef, balanced_accuracy_score,
)
from sklearn.preprocessing import StandardScaler

from utils.calibration import (
    TemperatureScaler, learn_temperature_from_folds,
    learn_per_fold_temperature, calibration_diagnostics,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TrialData:
    features: np.ndarray   # (T, C) preprocessed
    label: int             # 0=ADL, 1=Fall
    activity_id: int
    subject_id: int
    trial_id: str
    n_samples: int


# ---------------------------------------------------------------------------
# Component 1: TrialLoader
# ---------------------------------------------------------------------------

class TrialLoader:
    """Load raw trial data per subject, preprocess (Kalman etc.), no windowing."""

    def __init__(self, config: dict):
        self.config = config
        self.dataset_args = config.get('dataset_args', {})
        self._sm_dataset = None

    def _get_dataset(self):
        if self._sm_dataset is not None:
            return self._sm_dataset
        from utils.dataset import SmartFallMM
        sm = SmartFallMM(root_dir=os.path.join(os.getcwd(), 'data'))
        age_group = self.dataset_args.get('age_group', ['young', 'old'])
        modalities = self.dataset_args.get('modalities', ['accelerometer', 'gyroscope'])
        modalities = [m for m in modalities if m != 'skeleton']
        sensors = self.dataset_args.get('sensors', ['watch'])
        sm.pipe_line(age_group=age_group, modalities=modalities, sensors=sensors)
        self._sm_dataset = sm
        return sm

    def load_subject_trials(self, subject_id: int) -> List[TrialData]:
        from utils.loader import csvloader, convert_gyro_to_radians
        from utils.kalman.preprocessing import kalman_fusion_for_loader, assemble_kalman_features

        sm = self._get_dataset()
        trials = [t for t in sm.matched_trials if t.subject_id == subject_id]

        enable_kalman = self.dataset_args.get('enable_kalman_fusion', False)
        convert_gyro = self.dataset_args.get('convert_gyro_to_rad', False)
        enable_truncation = self.dataset_args.get('enable_simple_truncation', False)
        max_trunc_diff = self.dataset_args.get('max_truncation_diff', 50)
        kalman_config = {k: v for k, v in self.dataset_args.items() if k.startswith('kalman_') or k == 'filter_fs'}

        result = []
        for trial in trials:
            label = int(trial.action_id > 9)
            trial_id = "S{:02d}A{:02d}T{:02d}".format(trial.subject_id, trial.action_id, trial.sequence_number)

            try:
                trial_data = {}
                for modality, path in trial.files.items():
                    if modality == 'skeleton':
                        continue
                    data = csvloader(path)
                    if modality == 'gyroscope' and convert_gyro:
                        data = convert_gyro_to_radians(data)
                    trial_data[modality] = data

                if 'accelerometer' not in trial_data:
                    continue

                if enable_truncation and 'gyroscope' in trial_data:
                    acc_len = trial_data['accelerometer'].shape[0]
                    gyro_len = trial_data['gyroscope'].shape[0]
                    diff = abs(acc_len - gyro_len)
                    if diff > max_trunc_diff:
                        continue
                    if diff > 0:
                        min_len = min(acc_len, gyro_len)
                        trial_data['accelerometer'] = trial_data['accelerometer'][:min_len]
                        trial_data['gyroscope'] = trial_data['gyroscope'][:min_len]

                if enable_kalman and 'gyroscope' in trial_data:
                    raw_gyro = trial_data['gyroscope'].copy() if kalman_config.get('kalman_include_gyro_mag', False) else None
                    trial_data = kalman_fusion_for_loader(trial_data, kalman_config)
                    features = assemble_kalman_features(
                        trial_data,
                        include_smv=kalman_config.get('kalman_include_smv', True),
                        exclude_yaw=kalman_config.get('kalman_exclude_yaw', False),
                        include_gyro_mag=kalman_config.get('kalman_include_gyro_mag', False),
                        gyro_data=raw_gyro,
                    )
                else:
                    acc = trial_data['accelerometer']
                    from utils.kalman.features import compute_smv
                    smv = compute_smv(acc).reshape(-1, 1)
                    parts = [smv, acc]
                    if 'gyroscope' in trial_data:
                        parts.append(trial_data['gyroscope'])
                    features = np.hstack(parts)

                window_size = self.dataset_args.get('max_length', 128)
                if features.shape[0] < window_size:
                    continue

                result.append(TrialData(
                    features=features, label=label, activity_id=trial.action_id,
                    subject_id=subject_id, trial_id=trial_id, n_samples=features.shape[0],
                ))
            except Exception:
                continue

        result.sort(key=lambda t: (t.label, t.activity_id, t.trial_id))
        return result


# ---------------------------------------------------------------------------
# Component 2: TemporalQueueEvaluator
# ---------------------------------------------------------------------------

class TemporalQueueEvaluator:
    """Evaluate queue-based fall detection on temporally-ordered trial streams."""

    DEFAULT_STRIDES = [2, 4, 5, 8, 10, 15, 20]
    DEFAULT_QUEUE_SIZES = [3, 5, 8, 10, 15, 20]
    DEFAULT_THRESHOLDS = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]
    DEFAULT_RETAINS = [0, 1, 2, 3, 5]
    DEFAULT_MAJORITY_KS = [0.3, 0.4, 0.5, 0.6, 0.7]

    def __init__(
        self,
        work_dir: str,
        config: Optional[dict] = None,
        device: str = 'cuda:0',
        eval_stride: int = 32,
        eval_strides: Optional[List[int]] = None,
        queue_sizes: Optional[List[int]] = None,
        queue_thresholds: Optional[List[float]] = None,
        queue_retains: Optional[List[int]] = None,
        majority_ks: Optional[List[float]] = None,
        enable_calibration: bool = True,
    ):
        self.work_dir = Path(work_dir)
        self.device = torch.device(device if torch.cuda.is_available() and 'cuda' in device else 'cpu')
        self.eval_stride = eval_stride
        self.eval_strides = eval_strides or self.DEFAULT_STRIDES
        self.queue_sizes = queue_sizes or self.DEFAULT_QUEUE_SIZES
        self.queue_thresholds = queue_thresholds or self.DEFAULT_THRESHOLDS
        self.queue_retains = queue_retains or self.DEFAULT_RETAINS
        self.majority_ks = majority_ks or self.DEFAULT_MAJORITY_KS
        self.enable_calibration = enable_calibration

        self.config = config or self._load_config()
        self.trial_loader = TrialLoader(self.config)

        self.window_size = self.config.get('dataset_args', {}).get('max_length', 128)
        self.enable_normalization = self.config.get('dataset_args', {}).get('enable_normalization', False)
        self.normalize_modalities = self.config.get('dataset_args', {}).get('normalize_modalities', 'all')

    def _load_config(self) -> dict:
        yaml_files = list(self.work_dir.glob('*.yaml')) + list(self.work_dir.glob('*.yml'))
        if not yaml_files:
            raise FileNotFoundError("No config YAML in {}".format(self.work_dir))
        with open(yaml_files[0]) as f:
            return yaml.safe_load(f)

    def _create_model(self):
        model_path = self.config['model']
        module_name, class_name = model_path.rsplit('.', 1)
        import importlib
        mod = importlib.import_module(module_name)
        model_cls = getattr(mod, class_name)
        return model_cls(**self.config.get('model_args', {}))

    def _load_fold_model(self, test_subject: int):
        model = self._create_model()
        ckpt = self.work_dir / "model_{}.pth".format(test_subject)
        if not ckpt.exists():
            raise FileNotFoundError("Checkpoint not found: {}".format(ckpt))
        state = torch.load(str(ckpt), map_location=self.device)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        return model

    def _window_trial(self, features: np.ndarray, stride: int) -> np.ndarray:
        T, C = features.shape
        if T < self.window_size:
            return np.empty((0, self.window_size, C))
        n_windows = (T - self.window_size) // stride + 1
        idx = np.arange(self.window_size)[None, :] + (np.arange(n_windows) * stride)[:, None]
        return features[idx]

    def _normalize_windows(self, windows: np.ndarray) -> np.ndarray:
        if not self.enable_normalization or self.normalize_modalities == 'none':
            return windows
        if self.normalize_modalities in ['all', 'acc_only']:
            n, t, c = windows.shape
            flat = windows.reshape(n * t, c)
            normalized = StandardScaler().fit_transform(flat)
            return normalized.reshape(n, t, c)
        return windows

    def _run_inference(self, model, windows: np.ndarray) -> tuple:
        """Run model inference. Returns (logits, probabilities) arrays."""
        all_logits, all_probs = [], []
        batch_size = self.config.get('test_batch_size', 64)
        with torch.no_grad():
            for i in range(0, len(windows), batch_size):
                batch = torch.from_numpy(windows[i:i + batch_size]).float().to(self.device)
                logits, _ = model(batch)
                logits_np = logits.squeeze(-1).cpu().numpy()
                probs_np = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
                all_logits.append(logits_np)
                all_probs.append(probs_np)
        return np.concatenate(all_logits), np.concatenate(all_probs)

    def _get_test_subjects(self) -> List[int]:
        subjects = set(self.config.get('subjects', []))
        val = set(self.config.get('validation_subjects', []))
        train_only = set(self.config.get('train_only_subjects', []))
        candidates = sorted(subjects - val - train_only)
        return [s for s in candidates if (self.work_dir / "model_{}.pth".format(s)).exists()]

    # ------------------------------------------------------------------
    # Per-subject inference (stride-parameterized)
    # ------------------------------------------------------------------

    def _evaluate_subject_stride(self, model, trials: List[TrialData], stride: int):
        """Window trials at given stride, normalize, run inference. Returns (logits, probs, labels)."""
        all_windows = []
        all_trial_labels = []
        for trial in trials:
            windows = self._window_trial(trial.features, stride)
            if len(windows) == 0:
                continue
            all_windows.append(windows)
            all_trial_labels.extend([trial.label] * len(windows))

        if not all_windows:
            return None, None, None

        windows = np.concatenate(all_windows, axis=0)
        windows = self._normalize_windows(windows)
        logits, probs = self._run_inference(model, windows)
        return logits, probs, np.array(all_trial_labels)

    # ------------------------------------------------------------------
    # Main ablation: stride x aggregation x queue params
    # ------------------------------------------------------------------

    def _load_calibration(self) -> Optional[dict]:
        """Load temperature scaling from fold_results.pkl if available and enabled."""
        if not self.enable_calibration:
            return None
        fold_pkl = self.work_dir / 'fold_results.pkl'
        if not fold_pkl.exists():
            return None
        try:
            global_scaler = learn_temperature_from_folds(str(fold_pkl))
            per_fold_scalers = learn_per_fold_temperature(str(fold_pkl))
            return {
                'global': global_scaler,
                'per_fold': per_fold_scalers,
            }
        except Exception as e:
            print(f"  [Temporal] Calibration loading failed: {e}")
            return None

    def sweep_and_report(self, print_results: bool = True) -> dict:
        """Full ablation: multi-stride, average + majority-k, all queue param combos.

        Runs both uncalibrated and calibrated (temperature-scaled) sweeps when
        fold_results.pkl is available.
        """
        test_subjects = self._get_test_subjects()
        print("  [Temporal] Evaluating {} test subjects across {} strides...".format(
            len(test_subjects), len(self.eval_strides)))

        # Load calibration
        calibration = self._load_calibration()
        if calibration:
            print("  [Temporal] Temperature scaling: global T={:.4f} (ECE reduction via post-hoc calibration)".format(
                calibration['global'].temperature))
        else:
            print("  [Temporal] No fold_results.pkl — skipping calibration")

        # Load trials once per subject (shared across strides)
        subject_trials = {}
        subject_models = {}
        for subj in test_subjects:
            try:
                trials = self.trial_loader.load_subject_trials(subj)
                if trials:
                    subject_trials[subj] = trials
                    subject_models[subj] = self._load_fold_model(subj)
            except Exception:
                continue

        n_subjects = len(subject_trials)
        if n_subjects == 0:
            print("  [Temporal] No subjects with valid trials")
            return {}

        total_trials = sum(len(t) for t in subject_trials.values())

        # Pass 1: inference per stride (expensive) — store both raw and calibrated
        stride_data = {}
        for stride in self.eval_strides:
            per_subject_raw = {}
            per_subject_cal = {}
            all_probs, all_labels = [], []
            all_cal_probs = []
            n_windows = 0

            for subj, trials in subject_trials.items():
                model = subject_models[subj]
                logits, probs, labels = self._evaluate_subject_stride(model, trials, stride)
                if probs is None:
                    continue

                per_subject_raw[str(subj)] = {
                    'probs': probs.tolist(), 'trial_labels': labels.tolist(),
                }
                all_probs.extend(probs.tolist())
                all_labels.extend(labels.tolist())
                n_windows += len(probs)

                # Calibrated probabilities (per-fold T for this subject)
                if calibration:
                    scaler = calibration['per_fold'].get(subj, calibration['global'])
                    cal_probs = scaler.calibrate_logits(logits)
                    per_subject_cal[str(subj)] = {
                        'probs': cal_probs.tolist(), 'trial_labels': labels.tolist(),
                    }
                    all_cal_probs.extend(cal_probs.tolist())

            if not all_probs:
                continue

            all_p = np.array(all_probs)
            all_l = np.array(all_labels)
            n_fall = int(all_l.sum())
            n_adl = len(all_l) - n_fall
            wm = _compute_metrics(all_l, all_p)
            wm['n_adl'] = n_adl
            wm['n_fall'] = n_fall
            wm['adl_fall_ratio'] = n_adl / n_fall if n_fall > 0 else float('inf')

            sd = {
                'per_subject': per_subject_raw,
                'window_metrics': wm,
                'n_windows': n_windows,
            }

            if calibration and all_cal_probs:
                cal_wm = _compute_metrics(all_l, np.array(all_cal_probs))
                cal_wm['n_adl'] = n_adl
                cal_wm['n_fall'] = n_fall
                cal_wm['adl_fall_ratio'] = wm['adl_fall_ratio']
                sd['per_subject_calibrated'] = per_subject_cal
                sd['window_metrics_calibrated'] = cal_wm

            stride_data[stride] = sd

        if not stride_data:
            print("  [Temporal] No windows produced for any stride")
            return {}

        # Pass 2: sweep queue configs (fast)
        methods = ['average'] + ['majority_k{}'.format(int(k * 100)) for k in self.majority_ks]

        all_results = _sweep_all_configs(stride_data, methods, self.queue_sizes,
                                         self.queue_thresholds, self.queue_retains,
                                         calibrated=False)

        all_results_cal = []
        if calibration:
            all_results_cal = _sweep_all_configs(stride_data, methods, self.queue_sizes,
                                                  self.queue_thresholds, self.queue_retains,
                                                  calibrated=True)

        output = {
            'stride_data': stride_data,
            'all_results': all_results,
            'all_results_calibrated': all_results_cal,
            'n_subjects': n_subjects,
            'total_trials': total_trials,
            'methods': methods,
            'calibration': {
                'temperature': calibration['global'].temperature,
                'n_samples': calibration['global'].n_samples,
                'fit_nll': calibration['global'].fit_nll,
            } if calibration else None,
        }

        if print_results:
            _print_stride_aggregation_results(output, self.eval_strides, self.majority_ks)

        _save_stride_aggregation_results(output, self.eval_strides, str(self.work_dir))

        return output


# ---------------------------------------------------------------------------
# Component 3: Queue simulation
# ---------------------------------------------------------------------------

def _simulate_queue_average(probs, trial_labels, queue_size, threshold, retain):
    """Average aggregation: mean probability > threshold -> FALL."""
    queue = deque(maxlen=queue_size)
    label_buf = deque(maxlen=queue_size)
    batch_labels = []
    preds, gt = [], []

    for prob, label in zip(probs, trial_labels):
        queue.append(prob)
        label_buf.append(label)
        batch_labels.append(label)
        if len(queue) < queue_size:
            continue

        avg = sum(queue) / len(queue)
        decision = 1 if avg > threshold else 0
        ground_truth = 1 if any(l == 1 for l in batch_labels) else 0
        preds.append(decision)
        gt.append(ground_truth)

        if avg > threshold:
            queue.clear(); label_buf.clear(); batch_labels = []
        else:
            kp = list(queue)[-retain:] if retain > 0 else []
            kl = list(label_buf)[-retain:] if retain > 0 else []
            queue.clear(); label_buf.clear()
            queue.extend(kp); label_buf.extend(kl)
            batch_labels = list(kl)

    return preds, gt


def _simulate_queue_majority(probs, trial_labels, queue_size, threshold, retain, k_fraction):
    """Majority vote: if >= k_fraction of windows predict fall (prob>threshold), decide FALL."""
    queue = deque(maxlen=queue_size)
    label_buf = deque(maxlen=queue_size)
    batch_labels = []
    preds, gt = [], []

    for prob, label in zip(probs, trial_labels):
        queue.append(prob)
        label_buf.append(label)
        batch_labels.append(label)
        if len(queue) < queue_size:
            continue

        n_fall = sum(1 for p in queue if p > threshold)
        decision = 1 if n_fall >= k_fraction * queue_size else 0
        ground_truth = 1 if any(l == 1 for l in batch_labels) else 0
        preds.append(decision)
        gt.append(ground_truth)

        if decision == 1:
            queue.clear(); label_buf.clear(); batch_labels = []
        else:
            kp = list(queue)[-retain:] if retain > 0 else []
            kl = list(label_buf)[-retain:] if retain > 0 else []
            queue.clear(); label_buf.clear()
            queue.extend(kp); label_buf.extend(kl)
            batch_labels = list(kl)

    return preds, gt


def _sweep_all_configs(stride_data, methods, queue_sizes, thresholds, retains, calibrated=False):
    """Sweep all queue configs across all strides. Returns sorted results list."""
    key = 'per_subject_calibrated' if calibrated else 'per_subject'
    all_results = []
    for stride, sd in stride_data.items():
        psd = sd.get(key)
        if psd is None:
            continue
        for method in methods:
            for size in queue_sizes:
                for thresh in thresholds:
                    for retain in retains:
                        if retain >= size:
                            continue
                        m = _sweep_one_config(psd, method, size, thresh, retain)
                        if m:
                            all_results.append({
                                'stride': stride, 'method': method,
                                'queue_size': size, 'threshold': thresh, 'retain': retain,
                                **m,
                            })
    all_results.sort(key=lambda r: r['f1'], reverse=True)
    return all_results


def _sweep_one_config(per_subject_data, method, queue_size, threshold, retain):
    """Run one queue config across all subjects, return pooled metrics dict or None."""
    all_gt, all_preds = [], []

    # Parse method
    if method == 'average':
        sim_fn = lambda p, l: _simulate_queue_average(p, l, queue_size, threshold, retain)
    elif method.startswith('majority_k'):
        k = int(method.split('majority_k')[1]) / 100.0
        sim_fn = lambda p, l: _simulate_queue_majority(p, l, queue_size, threshold, retain, k)
    else:
        return None

    for data in per_subject_data.values():
        qp, qgt = sim_fn(data['probs'], data['trial_labels'])
        all_gt.extend(qgt)
        all_preds.extend(qp)

    if not all_gt:
        return None

    y_true = np.array(all_gt)
    y_pred = np.array(all_preds)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        'f1': f1_score(y_true, y_pred, zero_division=0) * 100,
        'precision': precision_score(y_true, y_pred, zero_division=0) * 100,
        'recall': recall_score(y_true, y_pred, zero_division=0) * 100,
        'accuracy': accuracy_score(y_true, y_pred) * 100,
        'specificity': spec * 100,
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        'total_decisions': len(y_true),
        'gt_positive': int(y_true.sum()),
        'gt_negative': int((y_true == 0).sum()),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    metrics = {
        'n_samples': len(labels),
        'n_positive': int(labels.sum()),
        'n_negative': int((labels == 0).sum()),
        'threshold': threshold,
        'f1': f1_score(labels, preds, zero_division=0),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'specificity': spec,
        'accuracy': accuracy_score(labels, preds),
        'balanced_accuracy': balanced_accuracy_score(labels, preds),
        'mcc': matthews_corrcoef(labels, preds),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
    }

    if len(set(labels)) > 1:
        metrics['auc'] = roc_auc_score(labels, probs)

    return metrics


# ---------------------------------------------------------------------------
# Component 4: Print and save
# ---------------------------------------------------------------------------

def _print_top_n(results, title, n=20, show_acc=True):
    """Print a top-N table from sorted results list."""
    if not results:
        return
    print("\n\n--- {} ---".format(title))
    if show_acc:
        print("{:>3} {:>6} {:>14} {:>5} {:>6} {:>4} | {:>7} {:>7} {:>7} {:>7} {:>7} | {:>5} {:>4} {:>4}".format(
            '#', 'Stride', 'Method', 'QSize', 'Thresh', 'Ret', 'F1', 'Prec', 'Rec', 'Spec', 'Acc', 'Dec', 'GT+', 'GT-'))
        print("-" * 110)
        for i, r in enumerate(results[:n], 1):
            print("{:>3} {:>6} {:>14} {:>5} {:>6.1f} {:>4} | {:>6.2f}% {:>6.2f}% {:>6.2f}% {:>6.2f}% {:>6.2f}% | {:>5} {:>4} {:>4}".format(
                i, r['stride'], r['method'], r['queue_size'], r['threshold'], r['retain'],
                r['f1'], r['precision'], r['recall'], r['specificity'], r['accuracy'],
                r['total_decisions'], r['gt_positive'], r['gt_negative']))
    else:
        print("{:>3} {:>6} {:>14} {:>5} {:>6} {:>4} | {:>7} {:>7} {:>7} {:>7} | {:>5} {:>4} {:>4}".format(
            '#', 'Stride', 'Method', 'QSize', 'Thresh', 'Ret', 'F1', 'Prec', 'Rec', 'Spec', 'Dec', 'GT+', 'GT-'))
        print("-" * 105)
        for i, r in enumerate(results[:n], 1):
            print("{:>3} {:>6} {:>14} {:>5} {:>6.1f} {:>4} | {:>6.2f}% {:>6.2f}% {:>6.2f}% {:>6.2f}% | {:>5} {:>4} {:>4}".format(
                i, r['stride'], r['method'], r['queue_size'], r['threshold'], r['retain'],
                r['f1'], r['precision'], r['recall'], r['specificity'],
                r['total_decisions'], r['gt_positive'], r['gt_negative']))


def _print_stride_aggregation_results(output, strides, majority_ks):
    stride_data = output['stride_data']
    all_results = output['all_results']
    all_results_cal = output.get('all_results_calibrated', [])
    cal_info = output.get('calibration')
    n_subjects = output['n_subjects']
    total_trials = output['total_trials']

    print()
    print("=" * 110)
    print("TEMPORAL QUEUE EVALUATION ({} test subjects, {} trials)".format(n_subjects, total_trials))
    if cal_info:
        print("  Temperature Scaling: T={:.4f} (learned from {} samples, NLL={:.4f})".format(
            cal_info['temperature'], cal_info['n_samples'], cal_info['fit_nll']))
    print("=" * 110)

    # Table 1: Window-level with ADL:Fall ratio (raw vs calibrated)
    has_cal_wm = any('window_metrics_calibrated' in stride_data[s] for s in strides if s in stride_data)
    print("\n--- WINDOW-LEVEL METRICS BY EVAL STRIDE ---")
    if has_cal_wm:
        print("{:>6} {:>8} {:>9} | {:>7} {:>7} {:>7} {:>7} {:>7} | {:>7} {:>7} {:>7} {:>7} {:>7}".format(
            'Stride', 'Windows', 'ADL:Fall',
            'F1', 'Prec', 'Rec', 'Spec', 'AUC',
            'F1(cal)', 'Prec(c)', 'Rec(c)', 'Spec(c)', 'AUC(c)'))
        print("-" * 120)
    else:
        print("{:>6} {:>8} {:>7} {:>6} {:>9} {:>6} | {:>7} {:>7} {:>7} {:>7} {:>7} {:>7}".format(
            'Stride', 'Windows', 'ADL', 'Fall', 'ADL:Fall', 'Fall%', 'F1', 'Prec', 'Rec', 'Spec', 'AUC', 'MCC'))
        print("-" * 105)

    for s in strides:
        if s not in stride_data:
            continue
        wm = stride_data[s]['window_metrics']
        n_w = wm['n_samples']
        ratio = wm['adl_fall_ratio']
        auc = wm.get('auc', 0)

        if has_cal_wm:
            cwm = stride_data[s].get('window_metrics_calibrated', wm)
            cauc = cwm.get('auc', 0)
            print("{:>6} {:>8} {:>7.2f}:1 | {:>6.2f}% {:>6.2f}% {:>6.2f}% {:>6.2f}% {:>6.2f}% | {:>6.2f}% {:>6.2f}% {:>6.2f}% {:>6.2f}% {:>6.2f}%".format(
                s, n_w, ratio,
                wm['f1']*100, wm['precision']*100, wm['recall']*100, wm['specificity']*100, auc*100,
                cwm['f1']*100, cwm['precision']*100, cwm['recall']*100, cwm['specificity']*100, cauc*100))
        else:
            n_fall = wm['n_fall']
            n_adl = wm['n_adl']
            fall_pct = n_fall / n_w * 100 if n_w > 0 else 0
            print("{:>6} {:>8} {:>7} {:>6} {:>7.2f}:1 {:>5.1f}% | {:>6.2f}% {:>6.2f}% {:>6.2f}% {:>6.2f}% {:>6.2f}% {:>6.4f}".format(
                s, n_w, n_adl, n_fall, ratio, fall_pct,
                wm['f1']*100, wm['precision']*100, wm['recall']*100, wm['specificity']*100, auc*100, wm['mcc']))

    if not all_results:
        print("=" * 110)
        return

    # Table 2: Best F1 per stride x method
    method_labels = ['average'] + ['maj_k{}'.format(int(k*100)) for k in majority_ks]
    method_keys = ['average'] + ['majority_k{}'.format(int(k*100)) for k in majority_ks]
    queue_sizes = sorted(set(r['queue_size'] for r in all_results))

    print("\n\n--- BEST QUEUE F1 PER STRIDE x QUEUE SIZE x METHOD ---")
    header = "{:>6} {:>5} | ".format('Stride', 'QSize')
    for ml in method_labels:
        header += "{:>10}".format(ml)
    print(header)
    print("-" * (14 + 10 * len(method_labels)))

    for s in strides:
        if s not in stride_data:
            continue
        for qs in queue_sizes:
            line = "{:>6} {:>5} | ".format(s, qs)
            for mk in method_keys:
                subset = [r for r in all_results if r['stride'] == s and r['queue_size'] == qs and r['method'] == mk]
                if subset:
                    best = max(subset, key=lambda r: r['f1'])
                    line += "{:>9.2f}%".format(best['f1'])
                else:
                    line += "{:>10}".format('--')
            print(line)
        if s != strides[-1]:
            print()

    # Table 3: Average vs Majority-k head-to-head
    print("\n\n--- AVERAGE vs MAJORITY-K HEAD-TO-HEAD (best F1 per stride) ---")
    header = "{:>6} | {:>9} | ".format('Stride', 'Average')
    for k in majority_ks:
        header += "{:>9} ".format('k={}%'.format(int(k*100)))
    header += "| {:>14} {:>8}".format('Best Method', 'Best F1')
    print(header)
    print("-" * (30 + 10 * len(majority_ks) + 25))

    for s in strides:
        if s not in stride_data:
            continue
        avg_sub = [r for r in all_results if r['stride'] == s and r['method'] == 'average']
        if not avg_sub:
            continue
        avg_best = max(avg_sub, key=lambda r: r['f1'])
        candidates = [('average', avg_best['f1'])]

        line = "{:>6} | {:>8.2f}% | ".format(s, avg_best['f1'])
        for k in majority_ks:
            mk = 'majority_k{}'.format(int(k*100))
            subset = [r for r in all_results if r['stride'] == s and r['method'] == mk]
            if subset:
                best = max(subset, key=lambda r: r['f1'])
                line += "{:>8.2f}% ".format(best['f1'])
                candidates.append((mk, best['f1']))
            else:
                line += "{:>9} ".format('--')
        overall = max(candidates, key=lambda x: x[1])
        line += "| {:>14} {:>7.2f}%".format(overall[0], overall[1])
        print(line)

    # Table 4: Top 20 overall (uncalibrated)
    _print_top_n(all_results, 'TOP 20 CONFIGS (UNCALIBRATED)', 20)

    # Table 5: Top 20 calibrated (if available)
    if all_results_cal:
        _print_top_n(all_results_cal, 'TOP 20 CONFIGS (CALIBRATED, T={:.4f})'.format(
            cal_info['temperature'] if cal_info else 1.0), 20)

        # Table 6: Calibration impact — compare best uncal vs best cal per stride
        print("\n\n--- CALIBRATION IMPACT: BEST F1 PER STRIDE (Uncalibrated vs Calibrated) ---")
        print("{:>6} | {:>9} {:>7} {:>7} {:>7} | {:>9} {:>7} {:>7} {:>7} | {:>7}".format(
            'Stride', 'F1(raw)', 'Prec', 'Rec', 'Spec', 'F1(cal)', 'Prec', 'Rec', 'Spec', 'Delta'))
        print("-" * 100)
        for s in strides:
            if s not in stride_data:
                continue
            raw_sub = [r for r in all_results if r['stride'] == s]
            cal_sub = [r for r in all_results_cal if r['stride'] == s]
            if not raw_sub or not cal_sub:
                continue
            raw_best = max(raw_sub, key=lambda r: r['f1'])
            cal_best = max(cal_sub, key=lambda r: r['f1'])
            delta = cal_best['f1'] - raw_best['f1']
            print("{:>6} | {:>8.2f}% {:>6.2f}% {:>6.2f}% {:>6.2f}% | {:>8.2f}% {:>6.2f}% {:>6.2f}% {:>6.2f}% | {:>+6.2f}%".format(
                s,
                raw_best['f1'], raw_best['precision'], raw_best['recall'], raw_best['specificity'],
                cal_best['f1'], cal_best['precision'], cal_best['recall'], cal_best['specificity'],
                delta))

        # Table 7: Deployment stride=10 — calibrated vs uncalibrated
        raw_s10 = [r for r in all_results if r['stride'] == 10]
        cal_s10 = [r for r in all_results_cal if r['stride'] == 10]
        if raw_s10 and cal_s10:
            print("\n\n--- DEPLOYMENT (Stride=10) TOP 10: CALIBRATED vs UNCALIBRATED ---")
            print("{:>3} {:>5} | {:>14} {:>5} {:>6} {:>4} {:>7} {:>7} {:>7} {:>7} | {:>14} {:>5} {:>6} {:>4} {:>7} {:>7} {:>7} {:>7}".format(
                '#', '',
                'Method(raw)', 'QSize', 'Thr', 'Ret', 'F1', 'Prec', 'Rec', 'Spec',
                'Method(cal)', 'QSize', 'Thr', 'Ret', 'F1', 'Prec', 'Rec', 'Spec'))
            print("-" * 160)
            for i in range(min(10, len(raw_s10), len(cal_s10))):
                rr = raw_s10[i]
                rc = cal_s10[i]
                print("{:>3} {:>5} | {:>14} {:>5} {:>6.1f} {:>4} {:>6.2f}% {:>6.2f}% {:>6.2f}% {:>6.2f}% | {:>14} {:>5} {:>6.1f} {:>4} {:>6.2f}% {:>6.2f}% {:>6.2f}% {:>6.2f}%".format(
                    i+1, '',
                    rr['method'], rr['queue_size'], rr['threshold'], rr['retain'],
                    rr['f1'], rr['precision'], rr['recall'], rr['specificity'],
                    rc['method'], rc['queue_size'], rc['threshold'], rc['retain'],
                    rc['f1'], rc['precision'], rc['recall'], rc['specificity']))

    # High-specificity configs
    high_spec = [r for r in all_results if r['specificity'] >= 90]
    if high_spec:
        high_spec.sort(key=lambda r: r['f1'], reverse=True)
        _print_top_n(high_spec, 'TOP 15 HIGH-SPECIFICITY CONFIGS (Spec >= 90%)', 15, show_acc=False)

    if all_results_cal:
        high_spec_cal = [r for r in all_results_cal if r['specificity'] >= 90]
        if high_spec_cal:
            high_spec_cal.sort(key=lambda r: r['f1'], reverse=True)
            _print_top_n(high_spec_cal, 'TOP 15 HIGH-SPECIFICITY CALIBRATED (Spec >= 90%)', 15, show_acc=False)

    # Decision timing
    print("\n\n--- QUEUE DECISION TIMING (at 30Hz) ---")
    print("{:>6} {:>5} | {:>9} {:>13} {:>14}".format('Stride', 'QSize', 'Window/s', 'Sec/Decision', 'Decisions/Min'))
    print("-" * 55)
    for s in strides:
        for qs in queue_sizes:
            sec_per_window = s / 30.0
            sec_per_decision = qs * sec_per_window
            dec_per_min = 60.0 / sec_per_decision
            print("{:>6} {:>5} | {:>8.1f}/s {:>12.1f}s {:>13.1f}".format(
                s, qs, 1/sec_per_window, sec_per_decision, dec_per_min))

    # Window vs best queue comparison
    if all_results:
        best = all_results[0]
        best_stride = best['stride']
        wm = stride_data[best_stride]['window_metrics']
        print("\n\n--- WINDOW vs BEST QUEUE COMPARISON ---")
        print("{:>25} {:>8} {:>8} {:>8} {:>8} {:>8}".format('', 'F1', 'Prec', 'Recall', 'Acc', 'Spec'))
        print("{:>25} {:>7.2f}% {:>7.2f}% {:>7.2f}% {:>7.2f}% {:>7.2f}%".format(
            'Window (s={})'.format(best_stride),
            wm['f1']*100, wm['precision']*100, wm['recall']*100, wm['accuracy']*100, wm['specificity']*100))
        print("{:>25} {:>7.2f}% {:>7.2f}% {:>7.2f}% {:>7.2f}% {:>7.2f}%".format(
            'Queue (uncalibrated)', best['f1'], best['precision'], best['recall'], best['accuracy'], best['specificity']))

        if all_results_cal:
            best_cal = all_results_cal[0]
            print("{:>25} {:>7.2f}% {:>7.2f}% {:>7.2f}% {:>7.2f}% {:>7.2f}%".format(
                'Queue (calibrated)', best_cal['f1'], best_cal['precision'], best_cal['recall'],
                best_cal['accuracy'], best_cal['specificity']))
            d_f1 = best_cal['f1'] - best['f1']
            print("{:>25} {:>+7.2f}%".format('Calibration gain', d_f1))

        print("  Best uncal: stride={}, {}, size={}, thresh={}, retain={}".format(
            best['stride'], best['method'], best['queue_size'], best['threshold'], best['retain']))
        if all_results_cal:
            best_cal = all_results_cal[0]
            print("  Best calib: stride={}, {}, size={}, thresh={}, retain={}".format(
                best_cal['stride'], best_cal['method'], best_cal['queue_size'], best_cal['threshold'], best_cal['retain']))

    print("=" * 110)


def _save_stride_aggregation_results(output, strides, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    stride_data = output['stride_data']
    all_results = output['all_results']
    all_results_cal = output.get('all_results_calibrated', [])
    cal_info = output.get('calibration')

    # Window metrics per stride
    wm_dict = {}
    for s in strides:
        if s in stride_data:
            wm_dict[str(s)] = stride_data[s]['window_metrics']
    with open(out / 'temporal_stride_window_metrics.json', 'w') as f:
        json.dump(wm_dict, f, indent=2)
    print("  [Temporal] Saved: {}".format(out / 'temporal_stride_window_metrics.json'))

    ablation_fields = ['stride', 'method', 'queue_size', 'threshold', 'retain',
                       'f1', 'precision', 'recall', 'accuracy', 'specificity',
                       'tp', 'fp', 'tn', 'fn', 'total_decisions', 'gt_positive', 'gt_negative']

    # Full ablation CSV (uncalibrated)
    if all_results:
        with open(out / 'temporal_stride_ablation.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=ablation_fields)
            writer.writeheader()
            for r in all_results:
                writer.writerow({k: r.get(k) for k in ablation_fields})
        print("  [Temporal] Saved: {}".format(out / 'temporal_stride_ablation.csv'))

    # Calibrated ablation CSV
    if all_results_cal:
        with open(out / 'temporal_stride_ablation_calibrated.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=ablation_fields)
            writer.writeheader()
            for r in all_results_cal:
                writer.writerow({k: r.get(k) for k in ablation_fields})
        print("  [Temporal] Saved: {}".format(out / 'temporal_stride_ablation_calibrated.csv'))

    # Best config JSON (includes calibrated if better)
    if all_results:
        best = dict(all_results[0])
        best['calibrated'] = False
        if all_results_cal and all_results_cal[0]['f1'] > best['f1']:
            best = dict(all_results_cal[0])
            best['calibrated'] = True
        if cal_info:
            best['temperature'] = cal_info['temperature']
        with open(out / 'temporal_queue_best.json', 'w') as f:
            json.dump(best, f, indent=2)
        print("  [Temporal] Saved: {}".format(out / 'temporal_queue_best.json'))

    # Calibration metadata
    if cal_info:
        with open(out / 'temporal_calibration.json', 'w') as f:
            json.dump(cal_info, f, indent=2)
        print("  [Temporal] Saved: {}".format(out / 'temporal_calibration.json'))

    # Backward compat: stride=32 single-stride files
    if 32 in stride_data:
        wm32 = stride_data[32]['window_metrics']
        wm32_save = dict(wm32)
        wm32_save['eval_stride'] = 32
        wm32_save['n_subjects'] = output['n_subjects']
        wm32_save['total_trials'] = output['total_trials']
        with open(out / 'temporal_queue_metrics.json', 'w') as f:
            json.dump(wm32_save, f, indent=2)

        avg32 = [r for r in all_results if r['stride'] == 32 and r['method'] == 'average']
        if avg32:
            fields = ['queue_size', 'threshold', 'retain', 'f1', 'precision', 'recall',
                      'accuracy', 'specificity', 'tp', 'fp', 'tn', 'fn',
                      'total_decisions', 'gt_positive', 'gt_negative']
            with open(out / 'temporal_queue_sweep.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fields + ['n_subjects'])
                writer.writeheader()
                for r in sorted(avg32, key=lambda x: -x['f1']):
                    row = {k: r.get(k) for k in fields}
                    row['n_subjects'] = output['n_subjects']
                    writer.writerow(row)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_temporal_queue_evaluation(
    work_dir: str,
    config: Optional[dict] = None,
    device: str = 'cuda:0',
    print_results: bool = True,
    eval_strides: Optional[List[int]] = None,
    queue_sizes: Optional[List[int]] = None,
    queue_thresholds: Optional[List[float]] = None,
    queue_retains: Optional[List[int]] = None,
    majority_ks: Optional[List[float]] = None,
    enable_calibration: Optional[bool] = None,
    # Backward compat
    eval_stride: int = 32,
) -> dict:
    """Top-level entry point for temporal queue evaluation with stride ablation.

    Calibration (temperature scaling) is controlled by:
    1. enable_calibration parameter (explicit override)
    2. dataset_args.enable_calibration in config YAML (default: true)
    3. Defaults to true if fold_results.pkl exists
    """
    print("\n[Temporal Queue Eval] Starting temporal queue evaluation...")

    # Read calibration setting from config if not explicitly passed
    if enable_calibration is None and config:
        enable_calibration = config.get('dataset_args', {}).get('enable_calibration', True)
    if enable_calibration is None:
        enable_calibration = True

    evaluator = TemporalQueueEvaluator(
        work_dir=work_dir,
        config=config,
        device=device,
        eval_stride=eval_stride,
        eval_strides=eval_strides,
        queue_sizes=queue_sizes,
        queue_thresholds=queue_thresholds,
        queue_retains=queue_retains,
        majority_ks=majority_ks,
        enable_calibration=enable_calibration,
    )

    return evaluator.sweep_and_report(print_results=print_results)
