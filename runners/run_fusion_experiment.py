#!/usr/bin/env python3
"""
Run EXP3: Kalman Fusion Tests

Compares different Kalman fusion architectures:
- KalmanTransformerBaseline (65/35 asymmetric)
- KalmanSingleStream (unified 7ch)
- KalmanGatedFusion (learned gating)
- KalmanBalancedRatio (50/50 balanced)

Each with/without SE+TAP attention (8 variants total).

Usage:
    python -m runners.run_fusion_experiment [--device 0] [--results-dir results]
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments import get_exp3_config
from runners.experiment_runner import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(description='Run EXP3: Kalman Fusion Tests')

    parser.add_argument('--device', type=int, default=0,
                        help='GPU device ID (default: 0)')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Base directory for results (default: results)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of dataloader workers (default: 4)')
    parser.add_argument('--num-epochs', type=int, default=70,
                        help='Maximum training epochs (default: 70)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Training batch size (default: 16)')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience (default: 15)')
    parser.add_argument('--seed', type=int, default=2,
                        help='Random seed (default: 2)')
    parser.add_argument('--save-weights', action='store_true',
                        help='Save model weights (default: False)')

    args = parser.parse_args()

    # Get experiment config
    config = get_exp3_config()

    print(f'\n{"="*60}')
    print(f'EXP3: Kalman Fusion Tests')
    print(f'{"="*60}')
    print(f'Models ({len(config.models)}):')
    for m in config.models:
        print(f'  - {m.name}: {m.description}')
    print(f'\nHP Configs: {[hp.name for hp in config.hp_configs]}')
    print(f'Device: {args.device}')
    print(f'Results dir: {args.results_dir}')
    print(f'{"="*60}\n')

    # Create and run experiment
    runner = ExperimentRunner(
        experiment_config=config,
        results_base_dir=args.results_dir,
        device=args.device,
        num_workers=args.num_workers,
        seed=args.seed,
        save_model_weights=args.save_weights,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        patience=args.patience
    )

    results = runner.run()

    print(f'\n{"="*60}')
    print(f'EXP3 Complete!')
    print(f'Results saved to: {results["results_dir"]}')
    print(f'{"="*60}')

    return results


if __name__ == '__main__':
    main()
