.PHONY: help test train train-best train-wandb sweep clean

PYTHON := python
CONFIG_BEST := config/smartfallmm/lkf_euler_baseline.yaml
CONFIG := $(CONFIG_BEST)

help:
	@echo "Kalman-Fused Dual-Stream Transformer"
	@echo ""
	@echo "Training (91.10% F1 target):"
	@echo "  make train             - Train best model (LKF Euler, 21-fold LOSO)"
	@echo "  make train-wandb       - Train with W&B logging"
	@echo "  make train-fold FOLD=0 - Train single fold (debugging)"
	@echo ""
	@echo "SLURM Submission:"
	@echo "  make submit            - Submit training job"
	@echo "  make submit-wandb      - Submit with W&B logging"
	@echo ""
	@echo "Filter Comparison:"
	@echo "  make compare-filters   - Run LKF vs EKF vs UKF comparison"
	@echo ""
	@echo "Misc:"
	@echo "  make clean             - Remove temp files"
	@echo "  make wandb-sync        - Sync offline W&B runs"

test:
	pytest tests/ -v --tb=short

train:
	$(PYTHON) main.py --config $(CONFIG)

train-wandb:
	$(PYTHON) main.py --config $(CONFIG) --enable-wandb

train-fold:
	$(PYTHON) main.py --config $(CONFIG) --single-fold $(FOLD)

# Filter comparison (LKF vs EKF vs UKF)
compare-filters:
	@echo "Running filter comparison..."
	@mkdir -p results/filter_comparison
	$(PYTHON) main.py --config config/smartfallmm/lkf_euler_baseline.yaml --work-dir results/filter_comparison/lkf_euler
	$(PYTHON) main.py --config config/smartfallmm/ekf_quaternion_realtime.yaml --work-dir results/filter_comparison/ekf_quat
	$(PYTHON) main.py --config config/smartfallmm/kalman_gravity_vector.yaml --work-dir results/filter_comparison/lkf_gravity

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

# SLURM job submission (shared partition - CPU)
submit:
	@mkdir -p logs
	sbatch --partition=shared --ntasks=4 --mem=16G --time=2-00:00:00 \
		--job-name=reproduce_91 \
		--output=logs/train_%j.out \
		--wrap="source /mmfs1/home/sww35/miniforge3/etc/profile.d/conda.sh && conda activate py39 && $(PYTHON) main.py --config $(CONFIG) --device cpu"
	@echo "Submitted to shared. Monitor: squeue -u $$USER"

# SLURM job submission (gpu1 partition)
submit-gpu:
	@mkdir -p logs
	sbatch --partition=gpu1 --gres=gpu:1 --mem=32G --time=12:00:00 \
		--job-name=kalman_train \
		--output=logs/train_%j.out \
		--wrap="source /mmfs1/home/sww35/miniforge3/etc/profile.d/conda.sh && conda activate py39 && $(PYTHON) main.py --config $(CONFIG)"
	@echo "Submitted to gpu1. Monitor: squeue -u $$USER"

submit-wandb:
	@mkdir -p logs
	sbatch --partition=gpu1 --gres=gpu:1 --mem=32G --time=12:00:00 \
		--job-name=kalman_wandb \
		--output=logs/train_%j.out \
		--wrap="source /mmfs1/home/sww35/miniforge3/etc/profile.d/conda.sh && conda activate py39 && $(PYTHON) main.py --config $(CONFIG) --enable-wandb"
	@echo "Submitted with W&B. Monitor: squeue -u $$USER"

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

# ============================================================
# Kalman Filter Ablation on Shared Partition (CPU-only)
# ============================================================

KALMAN_CONFIGS := balanced_lkf_euler balanced_ekf_euler balanced_ukf_euler \
                  balanced_lkf_gravity balanced_ekf_quat balanced_ukf_quat \
                  kghf_lkf_euler kghf_ekf_euler kghf_ukf_euler \
                  kghf_lkf_gravity kghf_ekf_quat kghf_ukf_quat

ABLATION_TIMESTAMP := $(shell date +%Y%m%d_%H%M%S)
ABLATION_RESULTS := results/kalman_ablation_$(ABLATION_TIMESTAMP)

# Run all 12 configs sequentially (run manually with env active)
ablation-run:
	@mkdir -p $(ABLATION_RESULTS)
	@echo "============================================================"
	@echo "KALMAN ABLATION - SEQUENTIAL (12 configs)"
	@echo "Results: $(ABLATION_RESULTS)"
	@echo "============================================================"
	@for cfg in $(KALMAN_CONFIGS); do \
		echo ""; \
		echo ">>> Running: $$cfg"; \
		python main.py \
			--config config/smartfallmm/kalman_ablation/$${cfg}.yaml \
			--work_dir $(ABLATION_RESULTS)/$${cfg} \
			--enable_wandb True \
			--device 0 || true; \
	done
	@echo "============================================================"
	@echo "COMPLETE. Results: $(ABLATION_RESULTS)"
	@echo "============================================================"

# Submit all 12 configs in parallel on shared partition (5 days each)
ablation-shared:
	@mkdir -p logs $(ABLATION_RESULTS)
	@echo "============================================================"
	@echo "KALMAN ABLATION - PARALLEL SUBMISSION (12 configs)"
	@echo "Results: $(ABLATION_RESULTS)"
	@echo "============================================================"
	@for cfg in $(KALMAN_CONFIGS); do \
		sbatch --job-name="kf-$${cfg:0:12}" \
		       --partition=shared \
		       --nodes=1 \
		       --ntasks=12 \
		       --mem=48G \
		       --time=5-00:00:00 \
		       --output="logs/ablation_$${cfg}_%j.out" \
		       --error="logs/ablation_$${cfg}_%j.out" \
		       --wrap="cd $(PWD) && $(PYTHON) main.py \
		              --config config/smartfallmm/kalman_ablation/$${cfg}.yaml \
		              --work_dir $(ABLATION_RESULTS)/$${cfg} \
		              --enable_wandb True \
		              --device 0" && \
		echo "Submitted: $$cfg"; \
	done
	@echo "============================================================"
	@echo "Monitor: squeue -u $$USER"
	@echo "Results: $(ABLATION_RESULTS)"
	@echo "============================================================"

# Check job status
ablation-status:
	@squeue -u $$USER --format="%.10i %.20j %.12P %.8T %.12M %.4D %R"

# Cancel all kalman ablation jobs
ablation-cancel:
	@scancel -u $$USER --name="kf-*"
	@echo "Cancelled all kf-* jobs"

# Aggregate results from completed ablation
ablation-results:
	@echo "============================================================"
	@echo "KALMAN ABLATION RESULTS"
	@echo "============================================================"
	@for dir in $(ABLATION_RESULTS)/*/; do \
		cfg=$$(basename $$dir); \
		if [ -f "$$dir/scores.csv" ]; then \
			f1=$$(tail -n +2 "$$dir/scores.csv" | cut -d',' -f15 | \
			      awk '{s+=$$1;n++}END{if(n>0)printf "%.2f",s/n}'); \
			printf "%-25s Test F1: %s%%\n" "$$cfg" "$$f1"; \
		else \
			printf "%-25s PENDING\n" "$$cfg"; \
		fi; \
	done
