#!/usr/bin/env python3
"""
Ray Distributed LOSO Training for FusionTransformer.

This script distributes LOSO (Leave-One-Subject-Out) cross-validation folds
across multiple GPUs using Ray for parallel execution.

Usage:
    # Basic usage (3 GPUs, default)
    python ray_train.py --config config/smartfallmm/lkf_euler_baseline.yaml

    # Custom GPU count
    python ray_train.py --config config/smartfallmm/lkf_euler_baseline.yaml --num-gpus 6

    # With W&B logging
    python ray_train.py --config config/smartfallmm/lkf_euler_baseline.yaml --num-gpus 3 --enable-wandb

    # Test run (2 folds only)
    python ray_train.py --config config/smartfallmm/lkf_euler_baseline.yaml --num-gpus 1 --max-folds 2

    # Connect to Ray cluster
    python ray_train.py --config config/smartfallmm/lkf_euler_baseline.yaml --ray-address "ray://head:10001"
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))


def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Ray Distributed LOSO Training for FusionTransformer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic 3-GPU training
  python ray_train.py --config config/smartfallmm/lkf_euler_baseline.yaml

  # Scale to 6 GPUs
  python ray_train.py --config config/smartfallmm/lkf_euler_baseline.yaml --num-gpus 6

  # Quick test with 2 folds
  python ray_train.py --config config/smartfallmm/lkf_euler_baseline.yaml --num-gpus 1 --max-folds 2
        """
    )

    # Required arguments
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to YAML config file'
    )

    # GPU configuration
    parser.add_argument(
        '--num-gpus', '-g',
        type=int,
        default=3,
        help='Number of GPUs to use for distributed training (default: 3)'
    )

    # Output configuration
    parser.add_argument(
        '--work-dir', '-o',
        type=str,
        default=None,
        help='Output directory for results (default: auto-generated)'
    )

    # W&B configuration
    parser.add_argument(
        '--enable-wandb',
        action='store_true',
        help='Enable Weights & Biases logging'
    )
    parser.add_argument(
        '--wandb-project',
        type=str,
        default='smartfall-mm',
        help='W&B project name (default: smartfall-mm)'
    )
    parser.add_argument(
        '--wandb-entity',
        type=str,
        default='abheek-texas-state-university',
        help='W&B entity/team name'
    )

    # Ray configuration
    parser.add_argument(
        '--ray-address',
        type=str,
        default=None,
        help='Ray cluster address (default: local cluster)'
    )

    # Training configuration
    parser.add_argument(
        '--seed',
        type=int,
        default=2,
        help='Random seed (default: 2)'
    )
    parser.add_argument(
        '--max-folds',
        type=int,
        default=None,
        help='Maximum number of folds to run (for testing, default: all)'
    )

    # Model override (for ablation studies)
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='Model class override (e.g., Models.encoder_ablation.KalmanConv1dLinear)'
    )
    parser.add_argument(
        '--model-args',
        type=str,
        default=None,
        help='Model args override as dict string (e.g., "{\'acc_encoder\': \'linear\'}")'
    )

    # Training overrides
    parser.add_argument(
        '--loss-type',
        type=str,
        default=None,
        choices=['bce', 'focal', 'cb_focal'],
        help='Loss function: bce, focal, or cb_focal'
    )
    parser.add_argument(
        '--embed-dim',
        type=int,
        default=None,
        help='Embedding dimension override (default: from config)'
    )
    parser.add_argument(
        '--adl-stride',
        type=int,
        default=None,
        help='ADL stride override (default: from config)'
    )
    parser.add_argument(
        '--fall-stride',
        type=int,
        default=None,
        help='Fall stride override (default: from config)'
    )

    # Preprocessing overrides (for ablation studies)
    parser.add_argument(
        '--remove-gravity',
        type=str,
        default=None,
        choices=['true', 'false'],
        help='Enable/disable gravity removal from accelerometer'
    )
    parser.add_argument(
        '--gravity-cutoff',
        type=float,
        default=None,
        help='Gravity removal filter cutoff frequency (Hz)'
    )
    parser.add_argument(
        '--include-smv',
        type=str,
        default=None,
        choices=['true', 'false'],
        help='Enable/disable SMV (Signal Magnitude Vector) in features'
    )

    # Encoder architecture overrides
    parser.add_argument(
        '--acc-encoder',
        type=str,
        default=None,
        choices=['conv1d', 'linear', 'multikernel'],
        help='Accelerometer encoder type'
    )
    parser.add_argument(
        '--ori-encoder',
        type=str,
        default=None,
        choices=['conv1d', 'linear'],
        help='Orientation encoder type'
    )
    parser.add_argument(
        '--acc-kernel',
        type=int,
        default=None,
        help='Accelerometer encoder kernel size (for conv1d)'
    )
    parser.add_argument(
        '--ori-kernel',
        type=int,
        default=None,
        help='Orientation encoder kernel size (for conv1d)'
    )

    # Verbosity
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    return parser.parse_args()


def main():
    """Main entry point for Ray distributed training."""
    args = get_args()

    # Validate config path
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {args.config}")
        sys.exit(1)

    # Import here to avoid slow startup for --help
    from utils.ray_distributed import RayDistributedTrainer

    # Parse string boolean arguments
    remove_gravity = None
    if args.remove_gravity is not None:
        remove_gravity = args.remove_gravity.lower() == 'true'

    include_smv = None
    if args.include_smv is not None:
        include_smv = args.include_smv.lower() == 'true'

    # Create and run distributed trainer
    trainer = RayDistributedTrainer(
        config_path=str(config_path),
        num_gpus=args.num_gpus,
        work_dir=args.work_dir,
        model_override=args.model,
        model_args_override=args.model_args,
        loss_type_override=args.loss_type,
        embed_dim_override=args.embed_dim,
        adl_stride_override=args.adl_stride,
        fall_stride_override=args.fall_stride,
        # Preprocessing overrides
        remove_gravity_override=remove_gravity,
        gravity_cutoff_override=args.gravity_cutoff,
        include_smv_override=include_smv,
        # Encoder overrides
        acc_encoder_override=args.acc_encoder,
        ori_encoder_override=args.ori_encoder,
        acc_kernel_override=args.acc_kernel,
        ori_kernel_override=args.ori_kernel,
        # Other
        enable_wandb=args.enable_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        ray_address=args.ray_address,
        max_folds=args.max_folds,
        seed=args.seed,
    )

    # Run training
    results_df, fold_results = trainer.run()

    # Print final summary
    if not results_df.empty:
        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)

        # Get average row
        avg_row = results_df[results_df['test_subject'] == 'Average']
        if not avg_row.empty:
            avg = avg_row.iloc[0]
            print(f"Mean Test F1:       {avg.get('test_f1_score', 0):.2f}%")
            print(f"Mean Test Accuracy: {avg.get('test_accuracy', 0):.2f}%")
            print(f"Mean Val F1:        {avg.get('val_f1_score', 0):.2f}%")

        print(f"\nResults directory: {trainer.work_dir}")
        print("=" * 70)

        # Print detailed per-fold table at the very end
        if fold_results:
            from utils.ray_distributed import print_fold_results_table
            print_fold_results_table(fold_results)

    return results_df


if __name__ == '__main__':
    main()
