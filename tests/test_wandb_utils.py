"""Unit tests for W&B utilities (mocked)."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from utils.wandb_utils import WandbConfig, WandbLogger, create_wandb_logger


class TestWandbConfig:
    def test_default_config(self):
        config = WandbConfig()
        assert config.enabled == False
        assert config.project == "smartfall-mm"
        assert config.mode == "online"

    def test_custom_config(self):
        config = WandbConfig(
            enabled=True,
            project="my-project",
            entity="my-team",
            tags=["test", "ci"]
        )
        assert config.enabled == True
        assert config.project == "my-project"
        assert config.entity == "my-team"
        assert config.tags == ["test", "ci"]

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            WandbConfig(mode="invalid")

    def test_valid_modes(self):
        for mode in ["online", "offline", "disabled"]:
            config = WandbConfig(mode=mode)
            assert config.mode == mode


class TestWandbLoggerDisabled:
    def test_disabled_logger(self):
        config = WandbConfig(enabled=False)
        logger = WandbLogger(config)
        assert not logger.available

    def test_disabled_operations_safe(self):
        config = WandbConfig(enabled=False)
        logger = WandbLogger(config)

        logger.log({"metric": 1.0})
        logger.log_metric("test", 0.5)
        logger.log_fold_results(0, {"test_f1_score": 90})
        logger.log_summary({"final": 95})
        logger.finish()


class TestWandbLoggerInit:
    def test_init_without_wandb_available(self):
        config = WandbConfig(enabled=True)
        logger = WandbLogger(config)
        assert logger._initialized == False

    def test_double_finish_safe(self):
        config = WandbConfig(enabled=False)
        logger = WandbLogger(config)
        logger.finish()
        logger.finish()


class TestFactoryFunctions:
    def test_create_wandb_logger(self):
        logger = create_wandb_logger(
            enabled=False,
            project="test-project"
        )
        assert isinstance(logger, WandbLogger)
        assert logger.config.project == "test-project"

    def test_create_with_config(self):
        logger = create_wandb_logger(
            enabled=False,
            project="my-project",
            config={"lr": 0.001, "batch_size": 32}
        )
        assert logger.config.config == {"lr": 0.001, "batch_size": 32}

    def test_setup_from_args(self):
        from utils.wandb_utils import setup_wandb_from_args
        from argparse import Namespace

        args = Namespace(
            enable_wandb=False,
            wandb_project="test",
            wandb_entity="team",
            experiment_name="exp1"
        )

        logger = setup_wandb_from_args(args)
        assert logger.config.project == "test"
        assert logger.config.entity == "team"
        assert logger.config.run_name == "exp1"


class TestEdgeCases:
    def test_log_without_init(self):
        config = WandbConfig(enabled=True)
        logger = WandbLogger(config)
        logger.log({"test": 1})

    def test_finish_without_init(self):
        config = WandbConfig(enabled=True)
        logger = WandbLogger(config)
        logger.finish()

    def test_log_fold_results_without_init(self):
        config = WandbConfig(enabled=True)
        logger = WandbLogger(config)
        logger.log_fold_results(0, {"test_f1_score": 90})

    def test_log_summary_without_init(self):
        config = WandbConfig(enabled=True)
        logger = WandbLogger(config)
        logger.log_summary({"accuracy": 0.95})
