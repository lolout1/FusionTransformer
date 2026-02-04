"""
Stress test utilities for timestamp robustness evaluation.

Simulates real-world timestamp issues:
- Jitter: Random perturbation of timestamps
- Dropout: Random removal of events
- Burst: Concentration of events in short intervals
"""

import argparse
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def add_timestamp_jitter(
    timestamps: torch.Tensor,
    jitter_std_ms: float = 10.0,
    preserve_order: bool = True,
) -> torch.Tensor:
    """
    Add random Gaussian jitter to timestamps.

    Args:
        timestamps: (B, N) or (N,) timestamps in seconds
        jitter_std_ms: Standard deviation of jitter in milliseconds
        preserve_order: If True, sort timestamps after jitter

    Returns:
        Perturbed timestamps with same shape
    """
    jitter = torch.randn_like(timestamps) * (jitter_std_ms / 1000.0)
    perturbed = timestamps + jitter

    if preserve_order:
        perturbed, _ = torch.sort(perturbed, dim=-1)

    return perturbed


def add_event_dropout(
    events: torch.Tensor,
    timestamps: torch.Tensor,
    dropout_rate: float = 0.1,
    min_events: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Randomly drop events from sequence.

    Args:
        events: (B, N, D) event features
        timestamps: (B, N) timestamps
        dropout_rate: Fraction of events to drop
        min_events: Minimum events to keep

    Returns:
        events: (B, N', D) with N' < N
        timestamps: (B, N')
        mask: (B, N) original mask showing which events were kept
    """
    B, N, D = events.shape
    device = events.device

    # Generate dropout mask
    keep_mask = torch.rand(B, N, device=device) > dropout_rate

    # Ensure minimum events per sample
    for b in range(B):
        if keep_mask[b].sum() < min_events:
            # Randomly select min_events to keep
            indices = torch.randperm(N, device=device)[:min_events]
            keep_mask[b] = False
            keep_mask[b, indices] = True

    # Apply mask and pad to max kept length
    max_kept = keep_mask.sum(dim=1).max().item()

    new_events = torch.zeros(B, max_kept, D, device=device)
    new_timestamps = torch.zeros(B, max_kept, device=device)
    new_mask = torch.zeros(B, max_kept, dtype=torch.bool, device=device)

    for b in range(B):
        kept_indices = keep_mask[b].nonzero(as_tuple=True)[0]
        n_kept = len(kept_indices)
        new_events[b, :n_kept] = events[b, kept_indices]
        new_timestamps[b, :n_kept] = timestamps[b, kept_indices]
        new_mask[b, :n_kept] = True

    return new_events, new_timestamps, new_mask


def add_burst_sampling(
    events: torch.Tensor,
    timestamps: torch.Tensor,
    burst_fraction: float = 0.5,
    burst_duration_fraction: float = 0.1,
) -> torch.Tensor:
    """
    Simulate burst sampling by concentrating timestamps in short intervals.

    Args:
        events: (B, N, D) event features (unchanged)
        timestamps: (B, N) timestamps
        burst_fraction: Fraction of events to move into burst
        burst_duration_fraction: Duration of burst as fraction of total

    Returns:
        Modified timestamps with bursts
    """
    B, N = timestamps.shape
    device = timestamps.device

    new_timestamps = timestamps.clone()

    for b in range(B):
        # Get time range
        t_min = timestamps[b].min()
        t_max = timestamps[b].max()
        duration = t_max - t_min

        # Select burst center
        burst_center = t_min + torch.rand(1, device=device).item() * duration
        burst_radius = duration * burst_duration_fraction / 2

        # Select events to move into burst
        n_burst = int(N * burst_fraction)
        burst_indices = torch.randperm(N, device=device)[:n_burst]

        # Move selected events into burst
        new_times = burst_center + (torch.rand(n_burst, device=device) - 0.5) * burst_radius * 2
        new_timestamps[b, burst_indices] = new_times

    # Sort timestamps
    new_timestamps, sort_indices = torch.sort(new_timestamps, dim=-1)

    return new_timestamps


def add_warp(
    timestamps: torch.Tensor,
    warp_strength: float = 0.2,
) -> torch.Tensor:
    """
    Apply non-linear time warping to timestamps.

    Simulates clock drift/acceleration.

    Args:
        timestamps: (B, N) timestamps
        warp_strength: Strength of warping (0 = none, 1 = extreme)

    Returns:
        Warped timestamps
    """
    B, N = timestamps.shape
    device = timestamps.device

    # Normalize timestamps to [0, 1]
    t_min = timestamps.min(dim=-1, keepdim=True).values
    t_max = timestamps.max(dim=-1, keepdim=True).values
    t_norm = (timestamps - t_min) / (t_max - t_min + 1e-8)

    # Apply random polynomial warp
    # warp(t) = t + strength * sin(freq * t)
    freq = 2 + torch.rand(B, 1, device=device) * 4  # 2-6 cycles
    warped_norm = t_norm + warp_strength * torch.sin(freq * t_norm * 3.14159 * 2)

    # Ensure monotonicity by sorting
    warped_norm, _ = torch.sort(warped_norm, dim=-1)

    # Denormalize
    warped = warped_norm * (t_max - t_min) + t_min

    return warped


class StressTestEvaluator:
    """
    Evaluates model robustness under various timestamp perturbations.

    Usage:
        evaluator = StressTestEvaluator(model)
        results = evaluator.run_suite(test_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def evaluate_with_perturbation(
        self,
        test_loader,
        perturbation_fn: Optional[Callable] = None,
        perturbation_name: str = 'clean',
    ) -> Dict[str, float]:
        """
        Evaluate model with optional timestamp perturbation.

        Args:
            test_loader: Test data loader
            perturbation_fn: Function to perturb (events, timestamps) -> (events, timestamps, mask)
            perturbation_name: Name for logging

        Returns:
            Metrics dict
        """
        all_preds = []
        all_labels = []

        for batch in test_loader:
            if len(batch) == 3:
                events, _, labels = batch
            else:
                events, labels = batch

            events = events.to(self.device)
            labels = labels.to(self.device)

            B, T, _ = events.shape
            timestamps = torch.arange(T, device=self.device).float().unsqueeze(0).expand(B, -1)
            timestamps = timestamps / 30.0

            # Apply perturbation
            mask = None
            if perturbation_fn is not None:
                result = perturbation_fn(events, timestamps)
                if len(result) == 3:
                    events, timestamps, mask = result
                else:
                    timestamps = result

            # Forward
            if mask is not None:
                logits, _ = self.model(events, timestamps, mask)
            else:
                logits, _ = self.model(events, timestamps)

            preds = (torch.sigmoid(logits.squeeze(-1)) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Compute metrics
        import numpy as np
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        tp = ((all_preds == 1) & (all_labels == 1)).sum()
        fp = ((all_preds == 1) & (all_labels == 0)).sum()
        fn = ((all_preds == 0) & (all_labels == 1)).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return {
            'name': perturbation_name,
            'f1': float(f1),
            'precision': float(precision),
            'recall': float(recall),
        }

    def run_suite(
        self,
        test_loader,
        jitter_levels: List[float] = [0, 5, 10, 20, 50],
        dropout_levels: List[float] = [0, 0.05, 0.1, 0.2, 0.3],
        burst_levels: List[float] = [0, 0.25, 0.5, 0.75],
    ) -> Dict[str, List[Dict]]:
        """
        Run full stress test suite.

        Args:
            test_loader: Test data loader
            jitter_levels: Jitter std in ms
            dropout_levels: Dropout rates
            burst_levels: Burst fractions

        Returns:
            Results dict with 'jitter', 'dropout', 'burst' keys
        """
        results = {'jitter': [], 'dropout': [], 'burst': []}

        # Jitter tests
        for jitter in jitter_levels:
            if jitter == 0:
                fn = None
                name = 'clean'
            else:
                fn = lambda e, t, j=jitter: (e, add_timestamp_jitter(t, j), None)
                name = f'jitter_{jitter}ms'

            metrics = self.evaluate_with_perturbation(test_loader, fn, name)
            metrics['jitter_ms'] = jitter
            results['jitter'].append(metrics)
            print(f"Jitter {jitter}ms: F1={metrics['f1']:.4f}")

        # Dropout tests
        for rate in dropout_levels:
            if rate == 0:
                fn = None
                name = 'clean'
            else:
                fn = lambda e, t, r=rate: add_event_dropout(e, t, r)
                name = f'dropout_{int(rate*100)}pct'

            metrics = self.evaluate_with_perturbation(test_loader, fn, name)
            metrics['dropout_rate'] = rate
            results['dropout'].append(metrics)
            print(f"Dropout {rate*100}%: F1={metrics['f1']:.4f}")

        # Burst tests
        for burst in burst_levels:
            if burst == 0:
                fn = None
                name = 'clean'
            else:
                fn = lambda e, t, b=burst: (e, add_burst_sampling(e, t, b), None)
                name = f'burst_{int(burst*100)}pct'

            metrics = self.evaluate_with_perturbation(test_loader, fn, name)
            metrics['burst_fraction'] = burst
            results['burst'].append(metrics)
            print(f"Burst {burst*100}%: F1={metrics['f1']:.4f}")

        return results

    def save_results(self, results: Dict, output_path: Path):
        """Save results to JSON."""
        import json

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Stress test model robustness')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Path to data directory')
    parser.add_argument('--output', type=str, default='stress_test_results.json',
                        help='Output path for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    # Load model
    from kd.resampler import TimestampAwareStudent
    model = TimestampAwareStudent()
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))

    # Create evaluator
    evaluator = StressTestEvaluator(model, device=args.device)

    # Note: Would need to create test_loader from data
    # This is a placeholder - actual implementation depends on data loading setup
    print("Stress test framework initialized.")
    print("To run tests, create a test_loader and call evaluator.run_suite(test_loader)")


if __name__ == '__main__':
    main()
