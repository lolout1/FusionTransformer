# FusionTransformer Makefile
# Production-ready targets for training, evaluation, and threshold analysis

.PHONY: help train train-ray analyze-thresholds evaluate-fixed clean

# Default config
CONFIG ?= config/smartfallmm/kalman_linear_tuned.yaml
NUM_GPUS ?= 3
RESULTS_DIR ?= results
THRESHOLDS ?= 0.5,0.55,0.6,0.7,0.9

# Help
help:
	@echo "FusionTransformer - Fall Detection Training Pipeline"
	@echo ""
	@echo "Training:"
	@echo "  make train                 - Single-GPU training (LOSO)"
	@echo "  make train-ray             - Multi-GPU Ray distributed training"
	@echo "  make train-ray NUM_GPUS=4  - Specify number of GPUs"
	@echo ""
	@echo "Threshold Analysis:"
	@echo "  make analyze-thresholds FOLD_RESULTS=results/ray_*/fold_results.pkl"
	@echo "  make evaluate-fixed FOLD_RESULTS=results/ray_*/fold_results.pkl THRESHOLDS=0.5,0.7,0.9"
	@echo ""
	@echo "Variables:"
	@echo "  CONFIG=$(CONFIG)"
	@echo "  NUM_GPUS=$(NUM_GPUS)"
	@echo "  RESULTS_DIR=$(RESULTS_DIR)"
	@echo "  THRESHOLDS=$(THRESHOLDS)"

# Single-GPU LOSO training
train:
	python main.py --config $(CONFIG)

# Multi-GPU Ray distributed training
train-ray:
	@echo "Starting Ray distributed training with $(NUM_GPUS) GPUs..."
	@echo "Config: $(CONFIG)"
	python -c "\
from utils.ray_distributed import RayDistributedTrainer; \
trainer = RayDistributedTrainer('$(CONFIG)', num_gpus=$(NUM_GPUS)); \
trainer.run(); \
print('Training complete'); \
from utils.ray_distributed import print_fold_results_table; \
"

# Analyze thresholds from saved fold results
analyze-thresholds:
ifndef FOLD_RESULTS
	@echo "ERROR: FOLD_RESULTS not specified"
	@echo "Usage: make analyze-thresholds FOLD_RESULTS=results/ray_*/fold_results.pkl"
	@exit 1
endif
	python scripts/analyze_thresholds.py --fold-results $(FOLD_RESULTS) -o $(RESULTS_DIR)/threshold_analysis

# Evaluate specific fixed thresholds
evaluate-fixed:
ifndef FOLD_RESULTS
	@echo "ERROR: FOLD_RESULTS not specified"
	@echo "Usage: make evaluate-fixed FOLD_RESULTS=results/ray_*/fold_results.pkl THRESHOLDS=0.5,0.7,0.9"
	@exit 1
endif
	python scripts/analyze_thresholds.py --fold-results $(FOLD_RESULTS) --thresholds $(THRESHOLDS) -o $(RESULTS_DIR)/threshold_analysis

# Quick threshold check at 0.5, 0.7, 0.9
quick-threshold-check:
ifndef FOLD_RESULTS
	@echo "ERROR: FOLD_RESULTS not specified"
	@echo "Usage: make quick-threshold-check FOLD_RESULTS=results/ray_*/fold_results.pkl"
	@exit 1
endif
	@echo "Evaluating fixed thresholds: 0.5, 0.7, 0.9"
	python scripts/analyze_thresholds.py --fold-results $(FOLD_RESULTS) --thresholds 0.5,0.7,0.9 --no-plot -o $(RESULTS_DIR)/quick_threshold

# Full pipeline: train + analyze
full-pipeline:
	@echo "Running full pipeline: train + threshold analysis"
	$(MAKE) train-ray
	@LATEST_RESULTS=$$(ls -td $(RESULTS_DIR)/ray_* | head -1); \
	echo "Analyzing results from: $$LATEST_RESULTS"; \
	$(MAKE) analyze-thresholds FOLD_RESULTS=$$LATEST_RESULTS/fold_results.pkl

# Generate experiment summary
MIN_F1 ?= 89.0
summary:
	@echo "Generating experiment summary (min F1: $(MIN_F1)%)..."
	python3 scripts/generate_experiment_summary.py --results-dir $(RESULTS_DIR) --min-f1 $(MIN_F1) -o $(RESULTS_DIR)/summary_experiments.md
	@echo "Summary saved to $(RESULTS_DIR)/summary_experiments.md"

# Clean results
clean:
	@echo "Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Deep clean (includes results - use with caution)
clean-all: clean
	@echo "WARNING: This will delete all results!"
	@read -p "Continue? [y/N] " confirm && [ "$$confirm" = "y" ] && rm -rf $(RESULTS_DIR)/* || echo "Aborted"
