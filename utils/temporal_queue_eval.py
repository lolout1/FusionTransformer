"""Temporal queue evaluation with realistic trial-ordered streams.

Loads raw trial data per LOSO test subject, processes in temporal order,
and runs alpha queue simulation to produce meaningful queue-level metrics
at deployment-realistic queue sizes (10, 15, 20).
"""

import json
import os
import traceback
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, roc_auc_score, matthews_corrcoef, balanced_accuracy_score,
)
from sklearn.preprocessing import StandardScaler


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
            trial_id = f"S{trial.subject_id:02d}A{trial.action_id:02d}T{trial.sequence_number:02d}"

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

                # Simple truncation
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

                # Kalman fusion
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
                    # Raw mode: [smv, ax, ay, az, gx, gy, gz]
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
                    features=features,
                    label=label,
                    activity_id=trial.action_id,
                    subject_id=subject_id,
                    trial_id=trial_id,
                    n_samples=features.shape[0],
                ))
            except Exception:
                continue

        # Sort: ADLs first (action_id <= 9), then falls (action_id > 9)
        # Within each group, sort by action_id then trial_id for deterministic order
        result.sort(key=lambda t: (t.label, t.activity_id, t.trial_id))
        return result


# ---------------------------------------------------------------------------
# Component 2: TemporalQueueEvaluator
# ---------------------------------------------------------------------------

class TemporalQueueEvaluator:
    """Evaluate queue-based fall detection on temporally-ordered trial streams."""

    def __init__(
        self,
        work_dir: str,
        config: Optional[dict] = None,
        device: str = 'cuda:0',
        eval_stride: int = 32,
        queue_sizes: List[int] = None,
        queue_thresholds: List[float] = None,
        queue_retains: List[int] = None,
    ):
        self.work_dir = Path(work_dir)
        self.device = torch.device(device if torch.cuda.is_available() and 'cuda' in device else 'cpu')
        self.eval_stride = eval_stride
        self.queue_sizes = queue_sizes or [10, 15, 20]
        self.queue_thresholds = queue_thresholds or [0.3, 0.4, 0.5, 0.6, 0.7]
        self.queue_retains = queue_retains or [0, 2, 5]

        self.config = config or self._load_config()
        self.trial_loader = TrialLoader(self.config)

        self.window_size = self.config.get('dataset_args', {}).get('max_length', 128)
        self.enable_normalization = self.config.get('dataset_args', {}).get('enable_normalization', False)
        self.normalize_modalities = self.config.get('dataset_args', {}).get('normalize_modalities', 'all')

    def _load_config(self) -> dict:
        yaml_files = list(self.work_dir.glob('*.yaml')) + list(self.work_dir.glob('*.yml'))
        if not yaml_files:
            raise FileNotFoundError(f"No config YAML in {self.work_dir}")
        with open(yaml_files[0]) as f:
            return yaml.safe_load(f)

    def _create_model(self):
        model_path = self.config['model']
        module_name, class_name = model_path.rsplit('.', 1)
        import importlib
        mod = importlib.import_module(module_name)
        model_cls = getattr(mod, class_name)
        model_args = self.config.get('model_args', {})
        return model_cls(**model_args)

    def _load_fold_model(self, test_subject: int):
        model = self._create_model()
        ckpt = self.work_dir / f"model_{test_subject}.pth"
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        state = torch.load(str(ckpt), map_location=self.device)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        return model

    def _window_trial(self, features: np.ndarray) -> np.ndarray:
        """Sliding window with fixed eval_stride (no class-aware)."""
        T, C = features.shape
        if T < self.window_size:
            return np.empty((0, self.window_size, C))
        n_windows = (T - self.window_size) // self.eval_stride + 1
        idx = np.arange(self.window_size)[None, :] + (np.arange(n_windows) * self.eval_stride)[:, None]
        return features[idx]

    def _normalize_windows(self, windows: np.ndarray) -> np.ndarray:
        """Z-score normalization matching training pipeline."""
        if not self.enable_normalization or self.normalize_modalities == 'none':
            return windows
        # Only normalize accelerometer key (which is the only modality after Kalman)
        if self.normalize_modalities in ['all', 'acc_only']:
            n, t, c = windows.shape
            flat = windows.reshape(n * t, c)
            normalized = StandardScaler().fit_transform(flat)
            return normalized.reshape(n, t, c)
        return windows

    def _run_inference(self, model, windows: np.ndarray) -> np.ndarray:
        """Batch inference, returns probability array."""
        probs = []
        batch_size = self.config.get('test_batch_size', 64)
        with torch.no_grad():
            for i in range(0, len(windows), batch_size):
                batch = torch.from_numpy(windows[i:i + batch_size]).float().to(self.device)
                logits, _ = model(batch)
                prob = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
                probs.append(prob)
        return np.concatenate(probs)

    def _get_test_subjects(self) -> List[int]:
        """Determine testable subjects from config."""
        subjects = set(self.config.get('subjects', []))
        val = set(self.config.get('validation_subjects', []))
        train_only = set(self.config.get('train_only_subjects', []))
        candidates = sorted(subjects - val - train_only)
        # Filter to subjects that have model checkpoints
        return [s for s in candidates if (self.work_dir / f"model_{s}.pth").exists()]

    def evaluate_subject(self, test_subject: int) -> dict:
        """Full pipeline for one LOSO fold subject."""
        model = self._load_fold_model(test_subject)
        trials = self.trial_loader.load_subject_trials(test_subject)

        if not trials:
            return {'subject': test_subject, 'skipped': True, 'reason': 'no_trials'}

        # Window each trial, track per-window trial labels
        all_windows = []
        all_trial_labels = []

        for trial in trials:
            windows = self._window_trial(trial.features)
            if len(windows) == 0:
                continue
            all_windows.append(windows)
            all_trial_labels.extend([trial.label] * len(windows))

        if not all_windows:
            return {'subject': test_subject, 'skipped': True, 'reason': 'no_windows'}

        windows = np.concatenate(all_windows, axis=0)
        trial_labels = np.array(all_trial_labels)

        # Normalize
        windows = self._normalize_windows(windows)

        # Inference
        probs = self._run_inference(model, windows)

        # Window-level metrics
        preds = (probs >= 0.5).astype(int)
        n_fall = int(trial_labels.sum())
        n_adl = len(trial_labels) - n_fall

        return {
            'subject': test_subject,
            'skipped': False,
            'n_trials': len(trials),
            'n_fall_trials': sum(1 for t in trials if t.label == 1),
            'n_adl_trials': sum(1 for t in trials if t.label == 0),
            'n_windows': len(probs),
            'n_fall_windows': n_fall,
            'n_adl_windows': n_adl,
            'probs': probs.tolist(),
            'trial_labels': trial_labels.tolist(),
        }

    def evaluate_all(self, test_subjects: Optional[List[int]] = None) -> dict:
        """Evaluate across all LOSO folds."""
        if test_subjects is None:
            test_subjects = self._get_test_subjects()

        print(f"  [Temporal] Evaluating {len(test_subjects)} subjects (eval_stride={self.eval_stride})...")

        per_subject = {}
        n_skipped = 0
        total_trials = 0
        total_windows = 0

        for subj in test_subjects:
            result = self.evaluate_subject(subj)
            per_subject[subj] = result
            if result.get('skipped'):
                n_skipped += 1
            else:
                total_trials += result['n_trials']
                total_windows += result['n_windows']

        return {
            'per_subject': per_subject,
            'n_subjects': len(test_subjects),
            'n_skipped': n_skipped,
            'total_trials': total_trials,
            'total_windows': total_windows,
            'eval_stride': self.eval_stride,
        }

    def sweep_and_report(self, print_results: bool = True) -> dict:
        """Full pipeline: evaluate all folds, sweep queue params, report."""
        eval_results = self.evaluate_all()

        if eval_results['total_windows'] == 0:
            print("  [Temporal] No windows to evaluate")
            return {}

        # Pool window-level predictions
        all_probs = []
        all_labels = []
        per_subject_data = {}

        for subj, result in eval_results['per_subject'].items():
            if result.get('skipped'):
                continue
            all_probs.extend(result['probs'])
            all_labels.extend(result['trial_labels'])
            per_subject_data[str(subj)] = {
                'probs': result['probs'],
                'trial_labels': result['trial_labels'],
            }

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        # Window-level pooled metrics
        window_metrics = _compute_metrics(all_labels, all_probs)

        # Queue parameter sweep
        sweep_results = []
        for size in self.queue_sizes:
            for thresh in self.queue_thresholds:
                for retain in self.queue_retains:
                    if retain >= size:
                        continue
                    qr = _simulate_queue_all_subjects(
                        per_subject_data, size, thresh, retain,
                    )
                    if qr['total_decisions'] > 0:
                        sweep_results.append({
                            'params': {'queue_size': size, 'threshold': thresh, 'retain': retain},
                            'pooled_metrics': qr['metrics'],
                            'total_decisions': qr['total_decisions'],
                            'gt_positive': qr['gt_positive'],
                            'gt_negative': qr['gt_negative'],
                            'n_subjects': qr['n_subjects'],
                        })

        sweep_results.sort(key=lambda r: r['pooled_metrics'].get('f1', 0), reverse=True)

        output = {
            'eval_results': eval_results,
            'window_metrics': window_metrics,
            'sweep_results': sweep_results,
            'per_subject_data': per_subject_data,
        }

        if print_results:
            _print_temporal_results(
                window_metrics, sweep_results, eval_results,
            )

        _save_temporal_results(
            window_metrics, sweep_results, per_subject_data,
            eval_results, str(self.work_dir),
        )

        return output


# ---------------------------------------------------------------------------
# Component 3: Queue simulation with trial-aware GT
# ---------------------------------------------------------------------------

def _simulate_temporal_queue(
    probs: List[float],
    trial_labels: List[int],
    queue_size: int,
    threshold: float,
    retain: int,
) -> dict:
    """Queue simulation on temporally-ordered stream with trial-label GT."""
    queue = deque(maxlen=queue_size)
    label_buf = deque(maxlen=queue_size)
    batch_labels = []

    queue_preds = []
    queue_gt = []

    for prob, label in zip(probs, trial_labels):
        queue.append(prob)
        label_buf.append(label)
        batch_labels.append(label)

        if len(queue) < queue_size:
            continue

        avg = sum(queue) / len(queue)
        gt = 1 if any(l == 1 for l in batch_labels) else 0
        decision = 1 if avg > threshold else 0

        queue_preds.append(decision)
        queue_gt.append(gt)

        if avg > threshold:
            queue.clear()
            label_buf.clear()
            batch_labels = []
        else:
            kept_probs = list(queue)[-retain:] if retain > 0 else []
            kept_labels = list(label_buf)[-retain:] if retain > 0 else []
            queue.clear()
            label_buf.clear()
            queue.extend(kept_probs)
            label_buf.extend(kept_labels)
            batch_labels = list(kept_labels)

    if not queue_preds:
        return {'n_decisions': 0, 'skipped': True}

    y_true = np.array(queue_gt)
    y_pred = np.array(queue_preds)

    labels_present = set(y_true)
    if len(labels_present) < 2:
        # Only one class in GT
        cm_labels = [0, 1]
        cm = confusion_matrix(y_true, y_pred, labels=cm_labels)
        tn, fp, fn, tp = cm.ravel()
        return {
            'n_decisions': len(queue_preds),
            'skipped': False,
            'queue_preds': queue_preds,
            'queue_gt': queue_gt,
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'accuracy': accuracy_score(y_true, y_pred),
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        'n_decisions': len(queue_preds),
        'skipped': False,
        'queue_preds': queue_preds,
        'queue_gt': queue_gt,
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred),
        'specificity': spec,
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
    }


def _simulate_queue_all_subjects(
    per_subject_data: Dict[str, dict],
    queue_size: int,
    threshold: float,
    retain: int,
) -> dict:
    """Run queue per subject, pool all decisions for micro-averaged metrics."""
    all_gt = []
    all_preds = []
    n_subjects = 0

    for subj, data in per_subject_data.items():
        result = _simulate_temporal_queue(
            data['probs'], data['trial_labels'],
            queue_size, threshold, retain,
        )
        if result.get('skipped'):
            continue
        all_gt.extend(result['queue_gt'])
        all_preds.extend(result['queue_preds'])
        n_subjects += 1

    if not all_gt:
        return {'total_decisions': 0, 'metrics': {}, 'gt_positive': 0, 'gt_negative': 0, 'n_subjects': 0}

    y_true = np.array(all_gt)
    y_pred = np.array(all_preds)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    metrics = {
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred),
        'specificity': spec,
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
    }

    return {
        'total_decisions': len(all_gt),
        'metrics': metrics,
        'gt_positive': int(y_true.sum()),
        'gt_negative': int((y_true == 0).sum()),
        'n_subjects': n_subjects,
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

def _print_temporal_results(
    window_metrics: dict,
    sweep_results: List[dict],
    eval_results: dict,
    n_top: int = 10,
):
    n_subj = eval_results['n_subjects'] - eval_results['n_skipped']
    n_trials = eval_results['total_trials']
    n_windows = eval_results['total_windows']
    stride = eval_results['eval_stride']

    print()
    print("=" * 90)
    print(f"TEMPORAL QUEUE EVALUATION ({n_subj} subjects, stride={stride})")
    print("=" * 90)
    print(f"Subjects: {n_subj} | Trials: {n_trials} | Windows: {n_windows} (eval_stride={stride})")

    wm = window_metrics
    auc_str = f"  AUC: {wm.get('auc', 0)*100:>7.2f}%" if 'auc' in wm else ""
    print(f"\nWindow-Level (temporal, t=0.50):")
    print(f"  F1: {wm['f1']*100:>7.2f}%    Precision: {wm['precision']*100:>7.2f}%    Recall: {wm['recall']*100:>7.2f}%")
    print(f"  Accuracy: {wm['accuracy']*100:>7.2f}%    Specificity: {wm['specificity']*100:>7.2f}%{auc_str}")
    print(f"  MCC: {wm['mcc']:.4f}    BalAcc: {wm['balanced_accuracy']*100:.2f}%")
    print(f"  Confusion:   TP={wm['tp']}  FP={wm['fp']}  TN={wm['tn']}  FN={wm['fn']}")
    print(f"  Class balance: {wm['n_positive']} fall ({wm['n_positive']/wm['n_samples']*100:.1f}%) / {wm['n_negative']} ADL ({wm['n_negative']/wm['n_samples']*100:.1f}%)")

    if sweep_results:
        print(f"\n{'='*90}")
        print(f"QUEUE PARAMETER SWEEP ({len(sweep_results)} configs, {n_subj} subjects)")
        print(f"{'='*90}")
        print(f"  Size  Thresh  Retain |      F1     Prec   Recall      Acc     Spec |   GT+   GT-   Dec")
        print(f"  {'-'*82}")

        for r in sweep_results[:n_top]:
            p = r['params']
            m = r['pooled_metrics']
            print(f"  {p['queue_size']:>4}  {p['threshold']:>6.2f}  {p['retain']:>6} | "
                  f"{m['f1']*100:>6.2f}% {m['precision']*100:>6.2f}% {m['recall']*100:>6.2f}% "
                  f"{m['accuracy']*100:>6.2f}% {m['specificity']*100:>6.2f}% | "
                  f"{r['gt_positive']:>4} {r['gt_negative']:>4} {r['total_decisions']:>5}")

        # Window vs best queue comparison
        best = sweep_results[0]
        bm = best['pooled_metrics']
        bp = best['params']

        print(f"\n{'-'*90}")
        print(f"WINDOW vs QUEUE COMPARISON (temporal)")
        print(f"{'-'*90}")
        header = f"  {'':>18} {'F1':>8}  {'Prec':>8}  {'Recall':>8}  {'Acc':>8}  {'Spec':>8}"
        print(header)

        win_label = 'Window (t=0.5)'
        queue_label = 'Queue (best)'
        print(f"  {win_label:>18} {wm['f1']*100:>7.2f}% {wm['precision']*100:>7.2f}% "
              f"{wm['recall']*100:>7.2f}% {wm['accuracy']*100:>7.2f}% {wm['specificity']*100:>7.2f}%")
        print(f"  {queue_label:>18} {bm['f1']*100:>7.2f}% {bm['precision']*100:>7.2f}% "
              f"{bm['recall']*100:>7.2f}% {bm['accuracy']*100:>7.2f}% {bm['specificity']*100:>7.2f}%")

        d_f1 = (bm['f1'] - wm['f1']) * 100
        d_prec = (bm['precision'] - wm['precision']) * 100
        d_rec = (bm['recall'] - wm['recall']) * 100
        d_acc = (bm['accuracy'] - wm['accuracy']) * 100
        d_spec = (bm['specificity'] - wm['specificity']) * 100
        print(f"  {'Delta':>18} {d_f1:>+7.2f}% {d_prec:>+7.2f}% {d_rec:>+7.2f}% "
              f"{d_acc:>+7.2f}% {d_spec:>+7.2f}%")
        print(f"  Best queue config: size={bp['queue_size']}, threshold={bp['threshold']}, retain={bp['retain']}")

    print("=" * 90)


def _save_temporal_results(
    window_metrics: dict,
    sweep_results: List[dict],
    per_subject_data: Dict[str, dict],
    eval_results: dict,
    output_dir: str,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Window-level metrics
    wm_save = {k: v for k, v in window_metrics.items()}
    wm_save['eval_stride'] = eval_results['eval_stride']
    wm_save['n_subjects'] = eval_results['n_subjects'] - eval_results['n_skipped']
    wm_save['total_trials'] = eval_results['total_trials']
    with open(out / 'temporal_queue_metrics.json', 'w') as f:
        json.dump(wm_save, f, indent=2)
    print(f"  [Temporal] Saved metrics: {out / 'temporal_queue_metrics.json'}")

    # Sweep CSV
    if sweep_results:
        import csv
        fields = ['queue_size', 'threshold', 'retain', 'f1', 'precision', 'recall',
                  'accuracy', 'specificity', 'tp', 'fp', 'tn', 'fn',
                  'total_decisions', 'gt_positive', 'gt_negative', 'n_subjects']
        with open(out / 'temporal_queue_sweep.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in sweep_results:
                row = {**r['params']}
                for k, v in r['pooled_metrics'].items():
                    row[k] = round(v * 100, 2) if isinstance(v, float) and v <= 1 else v
                row['total_decisions'] = r['total_decisions']
                row['gt_positive'] = r['gt_positive']
                row['gt_negative'] = r['gt_negative']
                row['n_subjects'] = r['n_subjects']
                writer.writerow(row)
        print(f"  [Temporal] Saved sweep: {out / 'temporal_queue_sweep.csv'}")

    # Best config JSON
    if sweep_results:
        best = sweep_results[0]
        best_save = {
            'params': best['params'],
            'pooled_metrics': best['pooled_metrics'],
            'total_decisions': best['total_decisions'],
            'gt_positive': best['gt_positive'],
            'gt_negative': best['gt_negative'],
            'eval_stride': eval_results['eval_stride'],
        }
        with open(out / 'temporal_queue_best.json', 'w') as f:
            json.dump(best_save, f, indent=2)
        print(f"  [Temporal] Saved best config: {out / 'temporal_queue_best.json'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_temporal_queue_evaluation(
    work_dir: str,
    config: Optional[dict] = None,
    device: str = 'cuda:0',
    eval_stride: int = 32,
    print_results: bool = True,
    queue_sizes: Optional[List[int]] = None,
    queue_thresholds: Optional[List[float]] = None,
    queue_retains: Optional[List[int]] = None,
) -> dict:
    """Top-level entry point for temporal queue evaluation."""
    print("\n[Temporal Queue Eval] Starting temporal queue evaluation...")

    evaluator = TemporalQueueEvaluator(
        work_dir=work_dir,
        config=config,
        device=device,
        eval_stride=eval_stride,
        queue_sizes=queue_sizes,
        queue_thresholds=queue_thresholds,
        queue_retains=queue_retains,
    )

    return evaluator.sweep_and_report(print_results=print_results)
