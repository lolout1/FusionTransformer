.PHONY: help test test-unit test-integration train sweep ray-up ray-down clean wandb-sync ablation ablation-ray ablation-single ablation-aggregate

PYTHON := python
CONFIG := config/smartfallmm/reproduce_91_val_f1.yaml

help:
	@echo "SmartFallMM Commands:"
	@echo ""
	@echo "Testing:"
	@echo "  make test              - Run all tests"
	@echo "  make test-unit         - Run unit tests only"
	@echo "  make test-integration  - Run integration tests only"
	@echo ""
	@echo "Training:"
	@echo "  make train             - Train with default config"
	@echo "  make train-wandb       - Train with W&B logging"
	@echo ""
	@echo "Ablation Study:"
	@echo "  make ablation          - Run 8 configs on 8 GPUs in parallel"
	@echo "  make ablation-seq      - Run sequentially (1 GPU)"
	@echo "  make ablation-single V=balanced_lkf_euler  - Run single variant"
	@echo "  make ablation-aggregate  - Aggregate results + LaTeX table"
	@echo ""
	@echo "Sweeps:"
	@echo "  make sweep             - Run Ray Tune hyperparameter sweep"
	@echo "  make ray-up            - Start Ray cluster"
	@echo "  make ray-down          - Stop Ray cluster"
	@echo ""
	@echo "Misc:"
	@echo "  make wandb-sync        - Sync offline W&B runs"
	@echo "  make clean             - Clean temp files"

test:
	pytest tests/ -v --tb=short || exit 1

test-unit:
	pytest tests/unit/ -v --tb=short || exit 1

test-integration:
	pytest tests/integration/ -v --tb=short || exit 1

train:
	$(PYTHON) main.py --config $(CONFIG)

train-wandb:
	$(PYTHON) main.py --config $(CONFIG) --enable-wandb

train-fold:
	$(PYTHON) main.py --config $(CONFIG) --single-fold $(FOLD)

sweep:
	$(PYTHON) runners/ray_tune_sweep.py --sweep hyperparameter --samples 50

sweep-arch:
	$(PYTHON) runners/ray_tune_sweep.py --sweep architecture --samples 20

# Kalman Filter Ablation Study
ablation:
	$(PYTHON) scripts/ablation/run_kalman_ablation.py --mode parallel --gpus 8 --wandb

ablation-seq:
	$(PYTHON) scripts/ablation/run_kalman_ablation.py --mode sequential --wandb

ablation-single:
	$(PYTHON) scripts/ablation/run_kalman_ablation.py --mode single --config $(V) --wandb

ablation-aggregate:
	$(PYTHON) scripts/ablation/aggregate_results.py --output-dir results/kalman_ablation/report

ray-up:
	@bash scripts/ray/cluster.sh start 3

ray-down:
	@bash scripts/ray/cluster.sh stop

ray-status:
	@bash scripts/ray/cluster.sh status

wandb-sync:
	wandb sync --sync-all wandb/

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache tests/.pytest_cache
	rm -rf ray_results/
	rm -rf outputs/ multirun/

# SLURM job submission
submit:
	sbatch --parsable -n 12 --mem=48G --time=3-00:00:00 \
		--partition=shared --job-name=smartfall \
		--output=logs/train_%j.out --error=logs/train_%j.err \
		--wrap="$(PYTHON) main.py --config $(CONFIG)"

submit-wandb:
	sbatch --parsable -n 12 --mem=48G --time=3-00:00:00 \
		--partition=shared --job-name=smartfall-wandb \
		--output=logs/train_%j.out --error=logs/train_%j.err \
		--wrap="$(PYTHON) main.py --config $(CONFIG) --enable-wandb"

submit-ablation:
	@mkdir -p logs
	sbatch --parsable --partition=gpu1 --gres=gpu:1 \
		--cpus-per-task=8 --mem=48G --time=3-00:00:00 \
		--job-name=kalman-ablation \
		--output=logs/ablation_%j.out --error=logs/ablation_%j.err \
		--wrap="$(PYTHON) scripts/ablation/run_kalman_ablation.py --mode sequential --wandb"

submit-ablation-parallel:
	@mkdir -p logs
	sbatch --parsable --partition=gpu1 --gres=gpu:8 \
		--cpus-per-task=32 --mem=128G --time=12:00:00 \
		--job-name=kalman-ablation-8gpu \
		--output=logs/ablation_%j.out --error=logs/ablation_%j.err \
		--wrap="$(PYTHON) scripts/ablation/run_kalman_ablation.py --mode parallel --gpus 8 --wandb"

submit-ablation-single:
	@mkdir -p logs
	sbatch --parsable --partition=gpu1 --gres=gpu:1 \
		--cpus-per-task=8 --mem=32G --time=6:00:00 \
		--job-name=ablation-$(V) \
		--output=logs/ablation_$(V)_%j.out --error=logs/ablation_$(V)_%j.err \
		--wrap="$(PYTHON) scripts/ablation/run_kalman_ablation.py --mode single --config $(V) --wandb"

submit-baseline-ablation:
	@mkdir -p logs
	sbatch --parsable --partition=gpu1 --gres=gpu:1 \
		--cpus-per-task=8 --mem=48G --time=2-00:00:00 \
		--job-name=baseline-ablation \
		--output=logs/baseline_%j.out --error=logs/baseline_%j.err \
		--wrap="$(PYTHON) scripts/ablation/run_baseline_ablation.py --mode sequential --wandb"

baseline-ablation:
	$(PYTHON) scripts/ablation/run_baseline_ablation.py --mode sequential --wandb
