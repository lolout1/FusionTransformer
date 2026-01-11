#!/usr/bin/env python3
"""Ray Tune sweep runner for SmartFallMM hyperparameter optimization."""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

try:
    from ray.tune.integration.wandb import WandbLoggerCallback
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# Search spaces
ARCH_SPACE = {
    "model": tune.grid_search([
        "Models.kalman_best.KalmanBest",
        "Models.kalman_gated_hierarchical.KalmanGatedHierarchicalFusion",
        "Models.cnn_mamba.CNNMambaDualStream",
        "Models.dual_stream_lstm.DualStreamLSTM",
    ]),
    "use_se": tune.grid_search([True, False]),
    "use_tap": tune.grid_search([True, False]),
}

HP_SPACE = {
    "lr": tune.loguniform(1e-4, 1e-2),
    "dropout": tune.uniform(0.3, 0.7),
    "embed_dim": tune.choice([32, 48, 64]),
    "weight_decay": tune.loguniform(1e-5, 1e-2),
    "batch_size": tune.choice([32, 64, 128]),
}

KALMAN_SPACE = {
    "filter_type": tune.grid_search(["linear", "ekf", "ukf"]),
    "Q_orientation": tune.loguniform(0.001, 0.1),
    "Q_rate": tune.loguniform(0.005, 0.1),
    "R_acc": tune.loguniform(0.01, 0.5),
    "R_gyro": tune.loguniform(0.05, 0.5),
}


def build_config(config: dict) -> 'argparse.Namespace':
    """Build training args from Ray Tune config."""
    from argparse import Namespace
    import yaml

    # Load base config
    base_path = Path(__file__).parent.parent / "config/smartfallmm/reproduce_91_val_f1.yaml"
    with open(base_path) as f:
        base = yaml.safe_load(f)

    args = Namespace()
    args.model = config.get("model", base["model"])
    args.model_args = base.get("model_args", {}).copy()
    args.dataset_args = base.get("dataset_args", {}).copy()

    # Apply config overrides
    if "embed_dim" in config:
        args.model_args["embed_dim"] = config["embed_dim"]
    if "dropout" in config:
        args.model_args["dropout"] = config["dropout"]
    if "use_se" in config:
        args.model_args["use_se"] = config["use_se"]
    if "use_tap" in config:
        args.model_args["use_tap"] = config["use_tap"]

    args.batch_size = config.get("batch_size", base.get("batch_size", 64))
    args.num_epoch = config.get("num_epoch", 80)
    args.base_lr = config.get("lr", base.get("base_lr", 1e-3))
    args.weight_decay = config.get("weight_decay", base.get("weight_decay", 5e-4))
    args.optimizer = base.get("optimizer", "adamw")
    args.seed = config.get("seed", 2)

    args.loss_type = base.get("loss_type", "focal")
    args.loss_args = base.get("loss_args", {"alpha": 0.75, "gamma": 2.0})

    # Kalman overrides
    if "filter_type" in config:
        args.dataset_args["kalman_filter_type"] = config["filter_type"]
    if "Q_orientation" in config:
        args.dataset_args["kalman_Q_orientation"] = config["Q_orientation"]
    if "Q_rate" in config:
        args.dataset_args["kalman_Q_rate"] = config["Q_rate"]
    if "R_acc" in config:
        args.dataset_args["kalman_R_acc"] = config["R_acc"]
    if "R_gyro" in config:
        args.dataset_args["kalman_R_gyro"] = config["R_gyro"]

    # Fixed params
    args.subjects = base.get("subjects", [])
    args.validation_subjects = base.get("validation_subjects", [48, 57])
    args.train_only_subjects = base.get("train_only_subjects", [])
    args.feeder = base.get("feeder", "Feeder.Make_Dataset.UTD_mm")
    args.dataset = "smartfallmm"
    args.work_dir = "ray_results/trial"
    args.config = None
    args.phase = "train"
    args.print_log = False
    args.device = [0]
    args.num_worker = 0
    args.include_val = True
    args.single_fold = None
    args.enable_test_grouping = False
    args.model_saved_name = "model"
    args.weights = None
    args.result_file = None
    args.start_epoch = 0
    args.lr_scheduler = "none"
    args.warmup_epochs = 10
    args.min_lr = 1e-6
    args.train_feeder_args = {"batch_size": args.batch_size}
    args.val_feeder_args = {"batch_size": args.batch_size}
    args.test_feeder_args = {"batch_size": args.batch_size}
    args.enable_kalman_preprocessing = False
    args.kalman_args = {}
    args.enable_wandb = False

    return args


def train_smartfall(config: dict):
    """Ray Tune trainable for SmartFallMM."""
    import numpy as np
    import torch

    # Seed
    seed = config.get("seed", 2)
    torch.manual_seed(seed)
    np.random.seed(seed)

    args = build_config(config)

    from main import Trainer
    trainer = Trainer(args)

    # Run subset of folds for faster iteration
    max_folds = config.get("max_folds", 5)
    test_candidates = [s for s in args.subjects
                       if s not in args.validation_subjects
                       and s not in args.train_only_subjects][:max_folds]

    fold_f1s = []

    for fold_idx, test_subject in enumerate(test_candidates):
        trainer.train_subjects = [s for s in test_candidates if s != test_subject]
        trainer.test_subject = [test_subject]
        trainer.val_subject = args.validation_subjects

        trainer._init_fold_tracking(fold_idx, test_subject)
        trainer.model = trainer.load_model(args.model, args.model_args)

        if not trainer.load_data():
            continue

        trainer.load_optimizer(trainer.model.parameters())
        trainer.load_scheduler(args.num_epoch)
        trainer.load_loss()

        for epoch in range(args.num_epoch):
            trainer.train(epoch)

            val_f1 = trainer.current_fold_metrics.get('val', {}).get('f1_score', 0)

            # Report to Ray Tune
            tune.report(
                val_f1=val_f1,
                val_loss=trainer.val_loss_summary[-1] if trainer.val_loss_summary else 0,
                fold=fold_idx,
                epoch=epoch,
            )

            if trainer.early_stop.early_stop:
                break

        # Test
        trainer.load_weights()
        trainer.eval(epoch=0, loader_name='test')
        fold_f1s.append(trainer.test_f1)

    mean_f1 = np.mean(fold_f1s) if fold_f1s else 0
    tune.report(mean_test_f1=mean_f1)


def run_sweep(sweep_type: str, num_samples: int, max_concurrent: int, ray_address: str = None):
    """Run a Ray Tune sweep."""

    # Initialize Ray
    if ray_address:
        ray.init(address=ray_address)
    else:
        ray.init()

    # Select search space
    if sweep_type == "architecture":
        search_space = ARCH_SPACE
        search_alg = None
    elif sweep_type == "hyperparameter":
        search_space = HP_SPACE
        search_alg = OptunaSearch(metric="val_f1", mode="max")
    elif sweep_type == "kalman":
        search_space = KALMAN_SPACE
        search_alg = OptunaSearch(metric="val_f1", mode="max")
    else:
        raise ValueError(f"Unknown sweep type: {sweep_type}")

    # ASHA scheduler
    scheduler = ASHAScheduler(
        metric="val_f1",
        mode="max",
        max_t=80,
        grace_period=10,
        reduction_factor=3,
    )

    # Callbacks
    callbacks = []
    if WANDB_AVAILABLE:
        callbacks.append(WandbLoggerCallback(
            project="smartfall-mm",
            entity="abheek-texas-state-university",
            group=f"ray-tune-{sweep_type}",
            log_config=True,
        ))

    # Run sweep
    analysis = tune.run(
        train_smartfall,
        config=search_space,
        scheduler=scheduler,
        search_alg=search_alg,
        num_samples=num_samples,
        resources_per_trial={"cpu": 12},
        storage_path="ray_results/",
        callbacks=callbacks,
        max_concurrent_trials=max_concurrent,
        verbose=1,
    )

    # Print results
    print("\n=== Best Trial ===")
    best_trial = analysis.best_trial
    print(f"Config: {best_trial.config}")
    print(f"Val F1: {best_trial.last_result.get('val_f1', 'N/A')}")
    print(f"Mean Test F1: {best_trial.last_result.get('mean_test_f1', 'N/A')}")

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Ray Tune sweep runner")
    parser.add_argument("--sweep", choices=["architecture", "hyperparameter", "kalman"],
                        default="hyperparameter")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--concurrent", type=int, default=4)
    parser.add_argument("--ray-address", type=str, default=None)
    args = parser.parse_args()

    run_sweep(args.sweep, args.samples, args.concurrent, args.ray_address)


if __name__ == "__main__":
    main()
