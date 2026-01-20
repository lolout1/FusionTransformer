"""Weights & Biases integration utilities."""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class WandbConfig:
    """Configuration for W&B logging."""
    enabled: bool = False
    project: str = "smartfall-mm"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    tags: list = field(default_factory=list)
    notes: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    mode: str = "online"  # online, offline, disabled

    def __post_init__(self):
        if self.mode not in ["online", "offline", "disabled"]:
            raise ValueError(f"Invalid mode: {self.mode}")


class WandbLogger:
    """W&B logger with graceful fallback."""

    def __init__(self, config: WandbConfig):
        self.config = config
        self.run = None
        self._initialized = False

    @property
    def available(self) -> bool:
        return WANDB_AVAILABLE and self.config.enabled

    def init(self, reinit: bool = False) -> bool:
        if not self.available:
            return False

        if self._initialized and not reinit:
            return True

        try:
            self.run = wandb.init(
                project=self.config.project,
                entity=self.config.entity,
                name=self.config.run_name,
                tags=self.config.tags,
                notes=self.config.notes,
                config=self.config.config,
                mode=self.config.mode,
                reinit=reinit
            )
            self._initialized = True
            return True
        except Exception as e:
            print(f"W&B init failed: {e}")
            return False

    def log(self, data: Dict[str, Any], step: Optional[int] = None):
        if not self._initialized:
            return
        try:
            wandb.log(data, step=step)
        except Exception:
            pass

    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        self.log({name: value}, step=step)

    def log_fold_results(self, fold_idx: int, results: Dict[str, Any]):
        if not self._initialized:
            return

        metrics = {
            f"fold_{fold_idx}/f1": results.get('test_f1_score', 0),
            f"fold_{fold_idx}/accuracy": results.get('test_accuracy', 0),
            f"fold_{fold_idx}/precision": results.get('test_precision', 0),
            f"fold_{fold_idx}/recall": results.get('test_recall', 0),
        }
        self.log(metrics)

    def log_summary(self, summary: Dict[str, Any]):
        if not self._initialized:
            return

        for key, value in summary.items():
            try:
                wandb.run.summary[key] = value
            except Exception:
                pass

    def log_artifact(self, name: str, path: str, type: str = "model"):
        if not self._initialized:
            return

        try:
            artifact = wandb.Artifact(name, type=type)
            artifact.add_file(path)
            wandb.log_artifact(artifact)
        except Exception:
            pass

    def finish(self):
        if self._initialized and self.run:
            try:
                wandb.finish()
            except Exception:
                pass
            self._initialized = False
            self.run = None


def create_wandb_logger(
    enabled: bool = False,
    project: str = "smartfall-mm",
    entity: Optional[str] = None,
    run_name: Optional[str] = None,
    config: Optional[Dict] = None,
    **kwargs
) -> WandbLogger:
    """Factory function to create W&B logger."""
    wandb_config = WandbConfig(
        enabled=enabled,
        project=project,
        entity=entity,
        run_name=run_name,
        config=config or {},
        **kwargs
    )
    return WandbLogger(wandb_config)


def setup_wandb_from_args(args) -> WandbLogger:
    """Setup W&B logger from argparse namespace."""
    config = {}
    if hasattr(args, 'model_args'):
        config['model'] = args.model_args
    if hasattr(args, 'dataset_args'):
        config['dataset'] = args.dataset_args

    return create_wandb_logger(
        enabled=getattr(args, 'enable_wandb', False),
        project=getattr(args, 'wandb_project', 'smartfall-mm'),
        entity=getattr(args, 'wandb_entity', None),
        run_name=getattr(args, 'experiment_name', None),
        config=config
    )
