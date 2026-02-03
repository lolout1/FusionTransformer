.PHONY: train train-all train-quick train-resume ablation ablation-quick ablation-kalman \
        install install-dev test test-cov lint format validate-configs clean

CONFIG ?= config/best_config/smartfallmm/kalman_baseline.yaml
NUM_GPUS ?= 8
PARALLEL ?= 4

train:
	python ray_train.py --config $(CONFIG) --num-gpus $(NUM_GPUS)

train-all:
	python ray_train.py --config config/best_config/smartfallmm/kalman_baseline.yaml --num-gpus $(NUM_GPUS) --work-dir exps/train_all_smartfallmm
	python ray_train.py --config config/best_config/upfall/kalman.yaml --num-gpus $(NUM_GPUS) --work-dir exps/train_all_upfall
	python ray_train.py --config config/best_config/wedafall/kalman.yaml --num-gpus $(NUM_GPUS) --work-dir exps/train_all_wedafall
	@echo ""
	@echo "=== Results Summary ==="
	@python3 tools/summarize_results.py exps/train_all_smartfallmm exps/train_all_upfall exps/train_all_wedafall

train-quick:
	python ray_train.py --config $(CONFIG) --num-gpus $(NUM_GPUS) --max-folds 2

train-resume:
	python ray_train.py --config $(CONFIG) --num-gpus $(NUM_GPUS) --resume

ablation:
	python distributed_dataset_pipeline/run_capacity_ablation.py --num-gpus $(NUM_GPUS) --parallel $(PARALLEL)

ablation-quick:
	python distributed_dataset_pipeline/run_capacity_ablation.py --num-gpus $(NUM_GPUS) --parallel $(PARALLEL) --quick

ablation-kalman:
	python distributed_dataset_pipeline/run_capacity_ablation.py --num-gpus $(NUM_GPUS) --parallel $(PARALLEL) --stream-only

install:
	pip install -r requirements.txt

install-dev: install
	pip install pytest pytest-cov ruff pre-commit && pre-commit install

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=. --cov-report=html

lint:
	ruff check .

format:
	ruff format .

validate-configs:
	python -c "import yaml, glob; [yaml.safe_load(open(f)) for f in glob.glob('config/**/*.yaml', recursive=True)]"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; rm -rf .pytest_cache .ruff_cache htmlcov .coverage
