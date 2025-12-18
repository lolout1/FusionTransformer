#!/usr/bin/env python
"""
Run a single model from any experiment.

Usage:
    python -m runners.run_single_model --experiment architecture --model-index 0
    python -m runners.run_single_model --experiment architecture --model-name single_6ch_no_se
"""

import argparse
from pathlib import Path

from experiments import get_exp1_config, get_exp2_config, get_exp3_config
from runners.experiment_runner import ExperimentRunner


EXPERIMENT_CONFIGS = {
    'modality': get_exp1_config,
    'architecture': get_exp2_config,
    'fusion': get_exp3_config,
}


def main():
    parser = argparse.ArgumentParser(description='Run a single model experiment')
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['modality', 'architecture', 'fusion'],
                        help='Experiment type')
    parser.add_argument('--model-index', type=int, default=None,
                        help='Model index (0-based)')
    parser.add_argument('--model-name', type=str, default=None,
                        help='Model name')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Base results directory')
    parser.add_argument('--device', type=int, default=0,
                        help='Device ID (0 for CPU, GPU ID otherwise)')
    parser.add_argument('--num-workers', type=int, default=16,
                        help='Number of data loading workers')
    args = parser.parse_args()

    if args.model_index is None and args.model_name is None:
        parser.error('Must specify either --model-index or --model-name')

    # Get experiment config
    exp_config = EXPERIMENT_CONFIGS[args.experiment]()

    # Find model
    model_config = None
    if args.model_name:
        for m in exp_config.models:
            if m.name == args.model_name:
                model_config = m
                break
        if model_config is None:
            print(f"Error: Model '{args.model_name}' not found")
            print(f"Available models: {[m.name for m in exp_config.models]}")
            return
    else:
        if args.model_index >= len(exp_config.models):
            print(f"Error: Model index {args.model_index} out of range (0-{len(exp_config.models)-1})")
            return
        model_config = exp_config.models[args.model_index]

    print("=" * 60)
    print(f"Running Single Model: {model_config.name}")
    print("=" * 60)
    print(f"Experiment: {exp_config.name}")
    print(f"Model: {model_config.name}")
    print(f"Description: {model_config.description}")
    print(f"HP Configs: {[hp.name for hp in exp_config.hp_configs]}")
    print(f"Device: {args.device}")
    print(f"Workers: {args.num_workers}")
    print("=" * 60)

    # Create runner with single model
    runner = ExperimentRunner(
        experiment_config=exp_config,
        results_base_dir=Path(args.results_dir),
        device=args.device,
        num_workers=args.num_workers,
    )

    # Run only the specified model
    results = runner.run_model(model_config)

    print("\n" + "=" * 60)
    print(f"COMPLETED: {model_config.name}")
    print("=" * 60)

    if results:
        print(f"Results saved to: {runner.results_dir}")
        for hp_name, hp_results in results.items():
            if hp_results:
                print(f"\n{hp_name}:")
                print(f"  Mean F1: {hp_results.get('mean_f1', 'N/A'):.4f}")
                print(f"  Std F1: {hp_results.get('std_f1', 'N/A'):.4f}")


if __name__ == '__main__':
    main()
