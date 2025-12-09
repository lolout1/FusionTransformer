#!/bin/bash
# =============================================================================
# Stream-Kalman Ablation Study - Full 6-Model Comparison
# =============================================================================
# Compares single-stream vs dual-stream architectures with/without Kalman smoothing
#
# Models:
#   A: Single-stream, No Kalman, No SE/TAP (baseline)
#   B: Dual-stream, No Kalman, No SE/TAP
#   C: Single-stream + Kalman smoothing, No SE/TAP
#   D: Dual-stream + Kalman smoothing, No SE/TAP
#   E: Single-stream + Kalman + SE + TAP (full)
#   F: Dual-stream + Kalman + SE + TAP (full reference)
#
# Each model runs 22-fold LOSO cross-validation
# Uses shared partition with 16 CPUs for reliable scheduling
# =============================================================================

set -eo pipefail

TIMESTAMP=$(date +%m%d_%H%M)
RESULTS_DIR="results/stream_kalman_ablation_${TIMESTAMP}"
CPUS_PER_TASK=16
MEMORY="32G"
TIME_LIMIT="7-00:00:00"  # 7 days for 22-fold LOSO

echo "============================================================"
echo "STREAM-KALMAN ABLATION STUDY"
echo "============================================================"
echo "Timestamp:   ${TIMESTAMP}"
echo "Results Dir: ${RESULTS_DIR}"
echo "CPUs/Task:   ${CPUS_PER_TASK}"
echo "Memory:      ${MEMORY}"
echo "Time Limit:  ${TIME_LIMIT}"
echo "Partition:   shared"
echo ""

# Create directory structure
mkdir -p "${RESULTS_DIR}"/{configs,logs,metrics,aggregated}

# Save experiment metadata
cat > "${RESULTS_DIR}/experiment_info.txt" << METAEOF
Stream-Kalman Ablation Study
========================================================
Start Time:        $(date)
Results Directory: ${RESULTS_DIR}
Partition:         shared
CPUs:              ${CPUS_PER_TASK}
Memory:            ${MEMORY}
Time Limit:        ${TIME_LIMIT}

6-Model Ablation Design:
--------------------------------------------------------
  A: Single-stream, No Kalman, No SE/TAP (baseline)
  B: Dual-stream, No Kalman, No SE/TAP
  C: Single-stream + Kalman smoothing, No SE/TAP
  D: Dual-stream + Kalman smoothing, No SE/TAP
  E: Single-stream + Kalman + SE + TAP
  F: Dual-stream + Kalman + SE + TAP (reference)

Scientific Questions:
--------------------------------------------------------
1. Single vs Dual-stream: Compare A vs B (no Kalman)
2. Effect of Kalman: Compare A vs C, B vs D
3. Effect of SE+TAP: Compare C vs E, D vs F
4. Full comparison: A < B < C < D < E < F?

Cross-validation: 22-fold LOSO
Metrics: F1-score (primary), Accuracy, Precision, Recall
METAEOF

# Copy all config files
CONFIG_DIR="config/smartfallmm/ablation_stream_kalman"
cp "${CONFIG_DIR}"/*.yaml "${RESULTS_DIR}/configs/" 2>/dev/null || true

# Define experiments: name|config|description
declare -a EXPERIMENTS=(
    "model_a|${CONFIG_DIR}/model_a_single_raw.yaml|Single-stream Raw"
    "model_b|${CONFIG_DIR}/model_b_dual_raw.yaml|Dual-stream Raw"
    "model_c|${CONFIG_DIR}/model_c_single_kalman.yaml|Single-stream + Kalman"
    "model_d|${CONFIG_DIR}/model_d_dual_kalman.yaml|Dual-stream + Kalman"
    "model_e|${CONFIG_DIR}/model_e_single_kalman_se.yaml|Single-stream + Kalman + SE"
    "model_f|${CONFIG_DIR}/model_f_dual_kalman_se.yaml|Dual-stream + Kalman + SE"
)

echo "Submitting 6 experiments to shared partition..."
echo ""

JOB_IDS=""
for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r name config desc <<< "$exp"

    if [[ ! -f "$config" ]]; then
        echo "WARNING: Config not found: $config"
        continue
    fi

    echo "Submitting: ${name}"
    echo "  Description: ${desc}"
    echo "  Config:      ${config}"

    job_id=$(sbatch \
        --job-name="sk_${name}" \
        --partition="shared" \
        --time="${TIME_LIMIT}" \
        --cpus-per-task="${CPUS_PER_TASK}" \
        --mem="${MEMORY}" \
        --output="${RESULTS_DIR}/logs/${name}.out" \
        --error="${RESULTS_DIR}/logs/${name}.err" \
        --parsable \
        ./scripts/run_stream_kalman_single.sh "${name}" "${config}" "${RESULTS_DIR}")

    echo "  Job ID:      ${job_id}"
    echo ""

    JOB_IDS="${JOB_IDS}${job_id},"
done

# Remove trailing comma
JOB_IDS="${JOB_IDS%,}"

echo "============================================================"
echo "ALL JOBS SUBMITTED"
echo "============================================================"
echo ""
echo "Job IDs: ${JOB_IDS}"
echo ""
echo "Monitor Commands:"
echo "  squeue -u \$USER | grep sk_"
echo "  tail -f ${RESULTS_DIR}/logs/*.out"
echo ""
echo "Aggregation (run after experiments complete):"
echo "  python aggregate_stream_kalman_ablation.py ${RESULTS_DIR}"
echo ""
echo "============================================================"

# Save job IDs for tracking
echo "${JOB_IDS}" > "${RESULTS_DIR}/job_ids.txt"
