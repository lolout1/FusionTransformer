#!/usr/bin/env python3
"""
EventTokenResampler Unit Tests and Stress Tests.

Tests the TimestampAwareStudent model's ability to handle irregular timestamps.
Compares against a fixed-rate baseline to validate the benefit of learned time queries.

Usage:
    python kd/test_resampler.py --unit-tests
    python kd/test_resampler.py --stress-tests --data-root /path/to/data
    python kd/test_resampler.py --all --data-root /path/to/data --num-gpus 2
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Ensure kd module is importable
_script_dir = Path(__file__).parent.parent.resolve()
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))


# =============================================================================
# UNIT TESTS
# =============================================================================

class UnitTestResult:
    def __init__(self, name: str, passed: bool, message: str = '', duration: float = 0.0):
        self.name = name
        self.passed = passed
        self.message = message
        self.duration = duration

    def __repr__(self):
        status = 'PASS' if self.passed else 'FAIL'
        return f'[{status}] {self.name}: {self.message} ({self.duration:.3f}s)'


def run_unit_tests(device: str = 'cuda') -> List[UnitTestResult]:
    """Run unit tests for EventTokenResampler."""
    from kd.resampler import EventTokenResampler, TimestampAwareStudent

    results = []
    device = device if torch.cuda.is_available() else 'cpu'

    # Test 1: Basic forward pass
    def test_basic_forward():
        model = EventTokenResampler(input_dim=6, embed_dim=48, num_tokens=64).to(device)
        events = torch.randn(4, 100, 6, device=device)
        timestamps = torch.linspace(0, 3, 100, device=device).unsqueeze(0).expand(4, -1)
        tokens, attn = model(events, timestamps, return_attn=True)

        assert tokens.shape == (4, 64, 48), f'Expected (4, 64, 48), got {tokens.shape}'
        assert attn.shape[0] == 4 and attn.shape[1] == 64, f'Attention shape mismatch: {attn.shape}'
        return 'Output shapes correct'

    # Test 2: Variable sequence lengths
    def test_variable_lengths():
        model = EventTokenResampler(input_dim=6, embed_dim=48, num_tokens=64).to(device)

        for seq_len in [16, 64, 128, 256, 512]:
            events = torch.randn(2, seq_len, 6, device=device)
            timestamps = torch.linspace(0, 4, seq_len, device=device).unsqueeze(0).expand(2, -1)
            tokens, _ = model(events, timestamps)
            assert tokens.shape == (2, 64, 48), f'Failed for seq_len={seq_len}'

        return 'Handles variable lengths 16-512'

    # Test 3: Empty sequence handling
    def test_empty_sequence():
        model = EventTokenResampler(input_dim=6, embed_dim=48, num_tokens=64).to(device)
        events = torch.zeros(2, 0, 6, device=device)
        timestamps = torch.zeros(2, 0, device=device)
        tokens, _ = model(events, timestamps)

        assert tokens.shape == (2, 64, 48), f'Empty seq failed: {tokens.shape}'
        assert torch.isfinite(tokens).all(), 'Contains NaN/Inf'
        return 'Empty sequences handled gracefully'

    # Test 4: Mask handling
    def test_mask_handling():
        model = EventTokenResampler(input_dim=6, embed_dim=48, num_tokens=64).to(device)
        events = torch.randn(4, 100, 6, device=device)
        timestamps = torch.linspace(0, 3, 100, device=device).unsqueeze(0).expand(4, -1)

        # Create mask with varying valid lengths
        mask = torch.zeros(4, 100, dtype=torch.bool, device=device)
        mask[0, :80] = True
        mask[1, :50] = True
        mask[2, :100] = True
        mask[3, :20] = True

        tokens, _ = model(events, timestamps, mask=mask)
        assert tokens.shape == (4, 64, 48), f'Mask handling failed: {tokens.shape}'
        assert torch.isfinite(tokens).all(), 'Contains NaN/Inf with mask'
        return 'Mask handling correct'

    # Test 5: Irregular timestamps
    def test_irregular_timestamps():
        model = EventTokenResampler(input_dim=6, embed_dim=48, num_tokens=64).to(device)
        events = torch.randn(4, 100, 6, device=device)

        # Create irregular timestamps with gaps and bursts
        timestamps = torch.zeros(4, 100, device=device)
        for b in range(4):
            t = 0.0
            for i in range(100):
                # Random delta: mostly small, occasionally large gap
                if np.random.rand() < 0.1:
                    delta = np.random.uniform(0.5, 2.0)  # Large gap
                else:
                    delta = np.random.uniform(0.01, 0.05)  # Normal
                t += delta
                timestamps[b, i] = t

        tokens, _ = model(events, timestamps)
        assert tokens.shape == (4, 64, 48), f'Irregular timestamps failed'
        assert torch.isfinite(tokens).all(), 'NaN/Inf with irregular timestamps'
        return 'Irregular timestamps handled'

    # Test 6: Extreme timestamp values
    def test_extreme_timestamps():
        model = EventTokenResampler(input_dim=6, embed_dim=48, num_tokens=64).to(device)
        events = torch.randn(2, 100, 6, device=device)

        # Very large timestamps
        timestamps = torch.linspace(1e6, 1e6 + 10, 100, device=device).unsqueeze(0).expand(2, -1)
        tokens, _ = model(events, timestamps)
        assert torch.isfinite(tokens).all(), 'Failed with large timestamps'

        # Very small timestamps
        timestamps = torch.linspace(0, 0.001, 100, device=device).unsqueeze(0).expand(2, -1)
        tokens, _ = model(events, timestamps)
        assert torch.isfinite(tokens).all(), 'Failed with small timestamps'

        return 'Extreme timestamp values handled'

    # Test 7: Duplicate timestamps
    def test_duplicate_timestamps():
        model = EventTokenResampler(input_dim=6, embed_dim=48, num_tokens=64).to(device)
        events = torch.randn(2, 100, 6, device=device)

        # Many duplicates (simulating sensor bug)
        timestamps = torch.zeros(2, 100, device=device)
        timestamps[:, :50] = 1.0
        timestamps[:, 50:] = 2.0

        tokens, _ = model(events, timestamps)
        assert torch.isfinite(tokens).all(), 'Failed with duplicate timestamps'
        return 'Duplicate timestamps handled'

    # Test 8: Gradient flow
    def test_gradient_flow():
        model = EventTokenResampler(input_dim=6, embed_dim=48, num_tokens=64).to(device)
        events = torch.randn(4, 100, 6, device=device, requires_grad=True)
        timestamps = torch.linspace(0, 3, 100, device=device).unsqueeze(0).expand(4, -1)

        tokens, _ = model(events, timestamps)
        loss = tokens.mean()
        loss.backward()

        assert events.grad is not None, 'No gradient to events'
        assert torch.isfinite(events.grad).all(), 'Gradient contains NaN/Inf'
        return 'Gradients flow correctly'

    # Test 9: Full student model
    def test_full_student():
        model = TimestampAwareStudent(
            input_dim=6, embed_dim=48, num_tokens=64,
            num_heads=4, num_layers=2, dropout=0.1
        ).to(device)

        events = torch.randn(4, 100, 6, device=device)
        timestamps = torch.linspace(0, 3, 100, device=device).unsqueeze(0).expand(4, -1)

        logits, features = model(events, timestamps)
        assert logits.shape == (4, 1), f'Logits shape: {logits.shape}'
        assert features.shape == (4, 48), f'Features shape: {features.shape}'
        assert torch.isfinite(logits).all() and torch.isfinite(features).all()
        return 'Full student model works'

    # Test 10: Determinism with same input
    def test_determinism():
        torch.manual_seed(42)
        model = EventTokenResampler(input_dim=6, embed_dim=48, num_tokens=64).to(device)
        model.eval()

        events = torch.randn(2, 100, 6, device=device)
        timestamps = torch.linspace(0, 3, 100, device=device).unsqueeze(0).expand(2, -1)

        with torch.no_grad():
            tokens1, _ = model(events, timestamps)
            tokens2, _ = model(events, timestamps)

        assert torch.allclose(tokens1, tokens2, atol=1e-6), 'Non-deterministic in eval mode'
        return 'Deterministic in eval mode'

    # Run all tests
    tests = [
        ('basic_forward', test_basic_forward),
        ('variable_lengths', test_variable_lengths),
        ('empty_sequence', test_empty_sequence),
        ('mask_handling', test_mask_handling),
        ('irregular_timestamps', test_irregular_timestamps),
        ('extreme_timestamps', test_extreme_timestamps),
        ('duplicate_timestamps', test_duplicate_timestamps),
        ('gradient_flow', test_gradient_flow),
        ('full_student', test_full_student),
        ('determinism', test_determinism),
    ]

    for name, test_fn in tests:
        start = time.time()
        try:
            msg = test_fn()
            results.append(UnitTestResult(name, True, msg, time.time() - start))
        except Exception as e:
            results.append(UnitTestResult(name, False, str(e)[:100], time.time() - start))

    return results


# =============================================================================
# FIXED-RATE BASELINE MODEL
# =============================================================================

class FixedRateStudent(nn.Module):
    """
    Baseline student that assumes fixed-rate sampling.
    Uses simple interpolation to resample to fixed length.
    """

    def __init__(
        self,
        input_dim: int = 6,
        embed_dim: int = 48,
        num_tokens: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.5,
        se_reduction: int = 4,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim

        # Simple linear projection (no time features)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Squeeze-Excitation
        self.se = SqueezeExcitation(embed_dim, reduction=se_reduction)

        # Temporal Attention Pooling
        self.pool = TemporalAttentionPooling(embed_dim, hidden_dim=embed_dim // 2)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(
        self,
        events: torch.Tensor,
        timestamps: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            events: (B, N, input_dim)
            timestamps: ignored (assumes fixed rate)
            mask: (B, N) optional

        Returns:
            logits: (B, 1)
            features: (B, embed_dim)
        """
        B, N, D = events.shape

        # Simple interpolation to fixed length (ignores timestamps)
        if N != self.num_tokens:
            events = events.transpose(1, 2)  # (B, D, N)
            events = F.interpolate(events, size=self.num_tokens, mode='linear', align_corners=False)
            events = events.transpose(1, 2)  # (B, num_tokens, D)

        # Project
        x = self.proj(events)  # (B, L, embed_dim)

        # Transformer
        x = self.transformer(x)

        # Channel attention
        x = self.se(x)

        # Pool
        features = self.pool(x)

        # Classify
        logits = self.classifier(features)

        return logits, features


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = x.mean(dim=1)
        excitation = self.fc(squeeze)
        return x * excitation.unsqueeze(1)


class TemporalAttentionPooling(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_scores = self.attention(x)
        attn_weights = F.softmax(attn_scores, dim=1)
        return (x * attn_weights).sum(dim=1)


# =============================================================================
# TIMESTAMP RELIABILITY ANALYSIS
# =============================================================================

def analyze_timestamp_reliability(data_root: str, subjects: List[int], max_files: int = 100) -> Dict:
    """
    Analyze actual timestamp characteristics in SmartFallMM IMU data.

    Checks for:
    - Drift: Does delta_t deviate from expected rate?
    - Duplicates: Same timestamp for multiple samples
    - Gaps: Large jumps in timestamps
    - Monotonicity: Are timestamps always increasing?
    """
    from kd.data_loader import TrialMatcher, load_imu_with_timestamps

    matcher = TrialMatcher(data_root, age_group='young', device='watch')
    trials = matcher.find_matched_trials(subjects=subjects, require_skeleton=False, require_gyro=False)

    stats = {
        'files_analyzed': 0,
        'total_samples': 0,
        'expected_rate_hz': 32,  # SmartFallMM nominal rate
        'delta_t': {
            'mean_ms': [],
            'std_ms': [],
            'min_ms': [],
            'max_ms': [],
        },
        'duplicates': {
            'count': 0,
            'fraction': [],
        },
        'gaps': {
            'over_100ms': 0,
            'over_500ms': 0,
            'over_1s': 0,
            'max_gap_s': 0.0,
        },
        'monotonicity': {
            'violations': 0,
            'files_with_violations': 0,
        },
        'drift': {
            'expected_duration_s': [],
            'actual_duration_s': [],
            'drift_ratio': [],  # actual / expected
        },
        'reliability_score': 0.0,  # 0-100
    }

    expected_delta_ms = 1000.0 / stats['expected_rate_hz']  # ~31.25ms

    for trial in trials[:max_files]:
        acc_path = trial['files'].get('accelerometer')
        if not acc_path:
            continue

        try:
            timestamps, values = load_imu_with_timestamps(acc_path)
            if len(timestamps) < 10:
                continue

            stats['files_analyzed'] += 1
            stats['total_samples'] += len(timestamps)

            # Delta_t analysis
            delta_t = np.diff(timestamps) * 1000  # Convert to ms
            stats['delta_t']['mean_ms'].append(np.mean(delta_t))
            stats['delta_t']['std_ms'].append(np.std(delta_t))
            stats['delta_t']['min_ms'].append(np.min(delta_t))
            stats['delta_t']['max_ms'].append(np.max(delta_t))

            # Duplicate detection (delta_t == 0)
            n_duplicates = np.sum(delta_t == 0)
            stats['duplicates']['count'] += n_duplicates
            stats['duplicates']['fraction'].append(n_duplicates / len(delta_t))

            # Gap detection
            stats['gaps']['over_100ms'] += np.sum(delta_t > 100)
            stats['gaps']['over_500ms'] += np.sum(delta_t > 500)
            stats['gaps']['over_1s'] += np.sum(delta_t > 1000)
            stats['gaps']['max_gap_s'] = max(stats['gaps']['max_gap_s'], np.max(delta_t) / 1000)

            # Monotonicity check
            violations = np.sum(delta_t < 0)
            stats['monotonicity']['violations'] += violations
            if violations > 0:
                stats['monotonicity']['files_with_violations'] += 1

            # Drift analysis
            n_samples = len(timestamps)
            expected_duration = (n_samples - 1) * expected_delta_ms / 1000  # seconds
            actual_duration = timestamps[-1] - timestamps[0]
            stats['drift']['expected_duration_s'].append(expected_duration)
            stats['drift']['actual_duration_s'].append(actual_duration)
            if expected_duration > 0:
                stats['drift']['drift_ratio'].append(actual_duration / expected_duration)

        except Exception:
            continue

    # Compute aggregated statistics
    if stats['files_analyzed'] > 0:
        summary = {
            'files_analyzed': stats['files_analyzed'],
            'total_samples': stats['total_samples'],
            'delta_t_mean_ms': np.mean(stats['delta_t']['mean_ms']),
            'delta_t_std_ms': np.mean(stats['delta_t']['std_ms']),
            'delta_t_expected_ms': expected_delta_ms,
            'duplicate_fraction': np.mean(stats['duplicates']['fraction']) * 100,
            'gaps_over_100ms': stats['gaps']['over_100ms'],
            'gaps_over_1s': stats['gaps']['over_1s'],
            'max_gap_s': stats['gaps']['max_gap_s'],
            'monotonicity_violations': stats['monotonicity']['violations'],
            'drift_ratio_mean': np.mean(stats['drift']['drift_ratio']),
            'drift_ratio_std': np.std(stats['drift']['drift_ratio']),
        }

        # Compute reliability score (0-100)
        # Penalize: high std, duplicates, large gaps, violations, drift
        score = 100.0

        # Penalize high std (expect ~5ms, penalize >20ms)
        if summary['delta_t_std_ms'] > 20:
            score -= min(30, (summary['delta_t_std_ms'] - 20) * 0.5)

        # Penalize duplicates (expect 0%, penalize >5%)
        if summary['duplicate_fraction'] > 5:
            score -= min(20, (summary['duplicate_fraction'] - 5) * 2)

        # Penalize large gaps
        if summary['max_gap_s'] > 1:
            score -= min(20, summary['max_gap_s'] * 2)

        # Penalize monotonicity violations
        if summary['monotonicity_violations'] > 0:
            score -= min(20, summary['monotonicity_violations'] * 0.1)

        # Penalize drift (expect ~1.0, penalize deviation)
        drift_dev = abs(summary['drift_ratio_mean'] - 1.0)
        if drift_dev > 0.1:
            score -= min(10, drift_dev * 50)

        summary['reliability_score'] = max(0, score)

        return summary

    return {'error': 'No files analyzed'}


def print_timestamp_analysis(analysis: Dict):
    """Print formatted timestamp analysis."""
    print('\n' + '=' * 60)
    print('TIMESTAMP RELIABILITY ANALYSIS')
    print('=' * 60)

    if 'error' in analysis:
        print(f'Error: {analysis["error"]}')
        return

    print(f'\nFiles analyzed: {analysis["files_analyzed"]}')
    print(f'Total samples: {analysis["total_samples"]:,}')

    print(f'\nDelta_t Statistics:')
    print(f'  Expected (32 Hz): {analysis["delta_t_expected_ms"]:.1f} ms')
    print(f'  Actual mean:      {analysis["delta_t_mean_ms"]:.1f} ms')
    print(f'  Actual std:       {analysis["delta_t_std_ms"]:.1f} ms')

    print(f'\nTimestamp Issues:')
    print(f'  Duplicate fraction: {analysis["duplicate_fraction"]:.1f}%')
    print(f'  Gaps > 100ms:       {analysis["gaps_over_100ms"]}')
    print(f'  Gaps > 1s:          {analysis["gaps_over_1s"]}')
    print(f'  Max gap:            {analysis["max_gap_s"]:.1f}s')
    print(f'  Monotonicity violations: {analysis["monotonicity_violations"]}')

    print(f'\nDrift Analysis:')
    print(f'  Drift ratio (actual/expected): {analysis["drift_ratio_mean"]:.3f} ± {analysis["drift_ratio_std"]:.3f}')
    if analysis["drift_ratio_mean"] > 1.5:
        print(f'  WARNING: Timestamps appear stretched (slow clock)')
    elif analysis["drift_ratio_mean"] < 0.5:
        print(f'  WARNING: Timestamps appear compressed (fast clock)')

    print(f'\nRELIABILITY SCORE: {analysis["reliability_score"]:.0f}/100')

    if analysis["reliability_score"] >= 80:
        print('  Status: RELIABLE - timestamps can be trusted')
    elif analysis["reliability_score"] >= 50:
        print('  Status: MODERATE - use with caution, EventTokenResampler helps')
    else:
        print('  Status: UNRELIABLE - timestamps heavily corrupted')


# =============================================================================
# STRESS TEST DATASET
# =============================================================================

class IMUStressTestDataset(Dataset):
    """
    Dataset for stress testing with real SmartFallMM IMU data.
    Loads data with actual irregular timestamps.
    """

    def __init__(
        self,
        data_root: str,
        subjects: List[int],
        window_size: int = 128,
        max_samples: int = 1000,
    ):
        self.window_size = window_size
        self.samples = []

        from kd.data_loader import TrialMatcher, load_imu_with_timestamps

        matcher = TrialMatcher(data_root, age_group='young', device='watch')
        trials = matcher.find_matched_trials(subjects=subjects, require_skeleton=False, require_gyro=True)

        for trial in trials[:max_samples]:
            try:
                # Load accelerometer
                acc_path = trial['files'].get('accelerometer')
                gyro_path = trial['files'].get('gyroscope')

                if not acc_path or not gyro_path:
                    continue

                acc_ts, acc_vals = load_imu_with_timestamps(acc_path)
                gyro_ts, gyro_vals = load_imu_with_timestamps(gyro_path)

                if len(acc_vals) < window_size // 2 or len(gyro_vals) < window_size // 2:
                    continue

                # Align gyro to acc length
                if len(gyro_vals) != len(acc_vals):
                    target_len = min(len(acc_vals), len(gyro_vals))
                    acc_vals = acc_vals[:target_len]
                    acc_ts = acc_ts[:target_len]
                    gyro_vals = gyro_vals[:target_len]

                # Concatenate acc + gyro
                imu = np.concatenate([acc_vals, gyro_vals], axis=-1)  # (N, 6)

                # Create windows
                n_frames = len(imu)
                stride = window_size // 2
                for start in range(0, max(1, n_frames - window_size), stride):
                    end = min(start + window_size, n_frames)
                    if end - start < window_size // 2:
                        continue

                    self.samples.append({
                        'imu': imu[start:end].astype(np.float32),
                        'timestamps': acc_ts[start:end].astype(np.float32),
                        'label': trial['label'],
                    })

            except Exception:
                continue

        print(f'Loaded {len(self.samples)} stress test samples')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        imu = sample['imu']
        ts = sample['timestamps']
        label = sample['label']

        # Pad to window_size if needed
        if len(imu) < self.window_size:
            pad_len = self.window_size - len(imu)
            imu = np.pad(imu, ((0, pad_len), (0, 0)), mode='constant')
            ts = np.pad(ts, (0, pad_len), mode='edge')

        return {
            'events': torch.from_numpy(imu[:self.window_size]),
            'timestamps': torch.from_numpy(ts[:self.window_size]),
            'label': torch.tensor(label, dtype=torch.long),
        }


def collate_stress_test(batch: List[Dict]) -> Dict:
    return {
        'events': torch.stack([b['events'] for b in batch]),
        'timestamps': torch.stack([b['timestamps'] for b in batch]),
        'labels': torch.stack([b['label'] for b in batch]),
    }


# =============================================================================
# STRESS TEST EVALUATION
# =============================================================================

@dataclass
class StressTestConfig:
    """Configuration for a stress test run."""
    name: str
    perturbation_type: str  # 'jitter', 'dropout', 'burst', 'warp', 'clean'
    perturbation_value: float
    description: str = ''


def apply_perturbation(
    events: torch.Tensor,
    timestamps: torch.Tensor,
    config: StressTestConfig,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Apply perturbation based on config."""
    from kd.stress_test import (
        add_timestamp_jitter,
        add_event_dropout,
        add_burst_sampling,
        add_warp,
    )

    mask = None

    if config.perturbation_type == 'clean':
        pass
    elif config.perturbation_type == 'jitter':
        timestamps = add_timestamp_jitter(timestamps, jitter_std_ms=config.perturbation_value)
    elif config.perturbation_type == 'dropout':
        events, timestamps, mask = add_event_dropout(events, timestamps, dropout_rate=config.perturbation_value)
    elif config.perturbation_type == 'burst':
        timestamps = add_burst_sampling(events, timestamps, burst_fraction=config.perturbation_value)
    elif config.perturbation_type == 'warp':
        timestamps = add_warp(timestamps, warp_strength=config.perturbation_value)

    return events, timestamps, mask


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    config: StressTestConfig,
    device: str = 'cuda',
) -> Dict:
    """Evaluate model under a specific perturbation."""
    model.eval()
    all_preds = []
    all_labels = []

    for batch in dataloader:
        events = batch['events'].to(device)
        timestamps = batch['timestamps'].to(device)
        labels = batch['labels'].to(device)

        # Apply perturbation
        events, timestamps, mask = apply_perturbation(events, timestamps, config)

        # Forward pass
        try:
            if mask is not None:
                logits, _ = model(events, timestamps, mask)
            else:
                logits, _ = model(events, timestamps)
        except Exception as e:
            print(f'  Error in forward: {e}')
            continue

        preds = (torch.sigmoid(logits.squeeze(-1)) > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    if len(all_preds) == 0:
        return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'accuracy': 0.0}

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()
    tn = ((all_preds == 0) & (all_labels == 0)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / len(all_labels)

    return {
        'f1': float(f1) * 100,
        'precision': float(precision) * 100,
        'recall': float(recall) * 100,
        'accuracy': float(accuracy) * 100,
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 30,
    lr: float = 0.001,
    patience: int = 10,
    device: str = 'cuda',
) -> Tuple[nn.Module, Dict]:
    """Train model and return best checkpoint."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_val_f1 = 0.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            events = batch['events'].to(device)
            timestamps = batch['timestamps'].to(device)
            labels = batch['labels'].to(device).float()

            # Normalize
            events = (events - events.mean()) / (events.std() + 1e-8)

            optimizer.zero_grad()
            logits, _ = model(events, timestamps)
            loss = criterion(logits.squeeze(-1), labels)

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # Validate
        clean_config = StressTestConfig('clean', 'clean', 0.0)
        val_metrics = evaluate_model(model, val_loader, clean_config, device)

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 5 == 0:
            print(f'  Epoch {epoch+1}: loss={train_loss/len(train_loader):.3f} val_f1={val_metrics["f1"]:.1f}%')

        if epochs_without_improvement >= patience:
            print(f'  Early stop at epoch {epoch+1}')
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {'best_val_f1': best_val_f1, 'final_epoch': epoch + 1}


def run_stress_tests(
    data_root: str,
    train_subjects: List[int],
    test_subjects: List[int],
    epochs: int = 30,
    device: str = 'cuda',
) -> Dict:
    """Run full stress test comparison between time modes and baseline."""
    from kd.resampler import TimestampAwareStudent

    print('=' * 60)
    print('STRESS TEST: Time Modes Comparison')
    print('=' * 60)
    print('Modes:')
    print('  - position: Ignore timestamps, use sequence position')
    print('  - timestamps: Use raw timestamps (may be noisy)')
    print('  - cleaned: Clip gaps, handle duplicates')
    print('  - baseline: Fixed-rate interpolation (no resampler)')
    print('=' * 60)

    # Create datasets
    print('\nLoading data...')
    train_dataset = IMUStressTestDataset(data_root, train_subjects, max_samples=500)
    test_dataset = IMUStressTestDataset(data_root, test_subjects, max_samples=200)

    if len(train_dataset) == 0 or len(test_dataset) == 0:
        print('ERROR: No data loaded')
        return {}

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_stress_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_stress_test)

    print(f'Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}')

    # Define stress test configurations
    stress_configs = [
        StressTestConfig('clean', 'clean', 0.0, 'No perturbation'),
        StressTestConfig('jitter_5ms', 'jitter', 5.0, '5ms timestamp jitter'),
        StressTestConfig('jitter_10ms', 'jitter', 10.0, '10ms timestamp jitter'),
        StressTestConfig('jitter_20ms', 'jitter', 20.0, '20ms timestamp jitter'),
        StressTestConfig('jitter_50ms', 'jitter', 50.0, '50ms timestamp jitter'),
        StressTestConfig('dropout_5pct', 'dropout', 0.05, '5% event dropout'),
        StressTestConfig('dropout_10pct', 'dropout', 0.10, '10% event dropout'),
        StressTestConfig('dropout_20pct', 'dropout', 0.20, '20% event dropout'),
        StressTestConfig('dropout_30pct', 'dropout', 0.30, '30% event dropout'),
        StressTestConfig('burst_25pct', 'burst', 0.25, '25% burst sampling'),
        StressTestConfig('burst_50pct', 'burst', 0.50, '50% burst sampling'),
        StressTestConfig('warp_10pct', 'warp', 0.1, '10% time warp'),
        StressTestConfig('warp_20pct', 'warp', 0.2, '20% time warp'),
    ]

    # Test configurations: different time modes + baseline
    model_configs = [
        ('position', 'Resampler (position-only)', {'time_mode': 'position'}),
        ('timestamps', 'Resampler (raw timestamps)', {'time_mode': 'timestamps'}),
        ('cleaned', 'Resampler (cleaned timestamps)', {'time_mode': 'cleaned'}),
        ('baseline', 'Fixed-Rate Baseline', {}),
    ]

    results = {}

    for model_key, model_name, model_kwargs in model_configs:
        print('\n' + '-' * 40)
        print(f'Training {model_name}...')
        print('-' * 40)

        if model_key == 'baseline':
            model = FixedRateStudent(
                input_dim=6, embed_dim=48, num_tokens=64,
                num_heads=4, num_layers=2, dropout=0.3
            )
        else:
            model = TimestampAwareStudent(
                input_dim=6, embed_dim=48, num_tokens=64,
                num_heads=4, num_layers=2, dropout=0.3,
                **model_kwargs
            )

        model, train_info = train_model(
            model, train_loader, test_loader,
            epochs=epochs, device=device
        )
        print(f'Training complete. Best val F1: {train_info["best_val_f1"]:.1f}%')

        print('\nEvaluating under stress conditions...')
        results[model_key] = {}
        for config in stress_configs:
            metrics = evaluate_model(model, test_loader, config, device)
            results[model_key][config.name] = metrics
            print(f'  {config.name}: F1={metrics["f1"]:.1f}%')

    return results


def print_comparison_table(results: Dict):
    """Print formatted comparison table."""
    models = list(results.keys())
    conditions = list(results[models[0]].keys())

    print('\n' + '=' * 90)
    print('COMPARISON: Time Mode Variants')
    print('=' * 90)

    # Header
    header = f'{"Condition":<18}'
    for m in models:
        header += f' {m:>12}'
    print(header)
    print('-' * 90)

    # Results per condition
    for cond in conditions:
        row = f'{cond:<18}'
        for m in models:
            f1 = results[m][cond]['f1']
            row += f' {f1:>11.1f}%'
        print(row)

    # Summary
    print('-' * 90)
    row = f'{"Mean":<18}'
    means = {}
    for m in models:
        mean_f1 = np.mean([results[m][c]['f1'] for c in conditions])
        means[m] = mean_f1
        row += f' {mean_f1:>11.1f}%'
    print(row)

    # Best model
    print('=' * 90)
    best = max(means, key=means.get)
    print(f'Best model: {best} (mean F1: {means[best]:.1f}%)')

    # Recommendation based on clean performance
    if 'clean' in conditions:
        clean_scores = {m: results[m]['clean']['f1'] for m in models}
        best_clean = max(clean_scores, key=clean_scores.get)
        print(f'Best on clean data: {best_clean} (F1: {clean_scores[best_clean]:.1f}%)')


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='EventTokenResampler Unit and Stress Tests',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--unit-tests', action='store_true', help='Run unit tests')
    parser.add_argument('--stress-tests', action='store_true', help='Run stress tests')
    parser.add_argument('--analyze-timestamps', action='store_true', help='Analyze timestamp reliability')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--data-root', type=str, default='data', help='Data directory')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
    args = parser.parse_args()

    if args.all:
        args.unit_tests = True
        args.stress_tests = True
        args.analyze_timestamps = True

    if not args.unit_tests and not args.stress_tests and not args.analyze_timestamps:
        args.analyze_timestamps = True  # Default to timestamp analysis first

    results = {}

    # Timestamp reliability analysis (should run first)
    if args.analyze_timestamps:
        subjects = list(range(29, 64))
        analysis = analyze_timestamp_reliability(args.data_root, subjects, max_files=100)
        print_timestamp_analysis(analysis)
        results['timestamp_analysis'] = analysis

        if analysis.get('reliability_score', 0) < 50:
            print('\nWARNING: Timestamps are unreliable.')
            print('Consider using sequence position instead of raw timestamps.')
            print('EventTokenResampler may still help via learned attention, but')
            print('the time-based features (delta_t, tau) may be noisy.')
        print()

    # Unit tests
    if args.unit_tests:
        print('\n' + '=' * 60)
        print('UNIT TESTS')
        print('=' * 60)
        unit_results = run_unit_tests(device=args.device)

        passed = sum(1 for r in unit_results if r.passed)
        total = len(unit_results)

        for r in unit_results:
            print(r)

        print(f'\nUnit Tests: {passed}/{total} passed')
        results['unit_tests'] = {
            'passed': passed,
            'total': total,
            'results': [{'name': r.name, 'passed': r.passed, 'message': r.message} for r in unit_results]
        }

    # Stress tests
    if args.stress_tests:
        # Use fixed train/test split
        train_subjects = [34, 35, 36, 37, 38, 39, 43, 44, 45, 46, 48, 49, 50, 51]
        test_subjects = [52, 53, 54, 55]

        stress_results = run_stress_tests(
            args.data_root, train_subjects, test_subjects,
            epochs=args.epochs, device=args.device
        )

        if stress_results:
            print_comparison_table(stress_results)
            results['stress_tests'] = stress_results

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\nResults saved to: {output_path}')


if __name__ == '__main__':
    main()
