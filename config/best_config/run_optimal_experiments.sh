#!/bin/bash
# =============================================================================
# OPTIMAL CONFIGURATION EXPERIMENTS
# =============================================================================
# Runs Kalman fusion vs Raw IMU comparison at optimal hyperparameters
# for both UP-FALL and WEDA-FALL datasets.
#
# Final Results:
#   UP-FALL Kalman:   95.11% ± 3.47% F1
#   UP-FALL Raw:      94.77% ± 3.41% F1
#   WEDA-FALL Kalman: 91.53% ± 4.52% F1
#   WEDA-FALL Raw:    90.43% ± 2.75% F1
#
# Usage:
#   ./config/best_config/run_optimal_experiments.sh [--gpus N] [--dataset DATASET] [--model MODEL]
#
# Examples:
#   ./config/best_config/run_optimal_experiments.sh --gpus 3
#   ./config/best_config/run_optimal_experiments.sh --gpus 3 --dataset upfall
#   ./config/best_config/run_optimal_experiments.sh --gpus 3 --model raw
# =============================================================================

set -e

NUM_GPUS=3
DATASET="all"
MODEL="all"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="results/optimal_comparison_${TIMESTAMP}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus) NUM_GPUS="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --output-dir) OUTPUT_BASE="$2"; shift 2 ;;
        -h|--help) echo "Usage: $0 [--gpus N] [--dataset upfall|wedafall|all] [--model kalman|raw|all]"; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "========================================================================"
echo "OPTIMAL CONFIGURATION EXPERIMENTS"
echo "========================================================================"
echo "Output: ${OUTPUT_BASE}"
echo "GPUs: ${NUM_GPUS}"
echo "Dataset: ${DATASET}"
echo "Model: ${MODEL}"
echo "========================================================================"

mkdir -p "${OUTPUT_BASE}"

run_experiment() {
    local config=$1
    local name=$2
    local work_dir="${OUTPUT_BASE}/${name}"

    echo ""
    echo "========================================================================"
    echo "Running: ${name}"
    echo "Config: ${config}"
    echo "========================================================================"

    python ray_train.py \
        --config "${config}" \
        --num-gpus "${NUM_GPUS}" \
        --work-dir "${work_dir}"

    echo ""
    echo "Results for ${name}:"
    echo "----------------------------------------"
    grep -E "(Test F1|Test Accuracy|Test Macro|Test Precision|Test Recall)" "${work_dir}/summary_report.txt" 2>/dev/null || echo "No results file found"
    echo "----------------------------------------"
}

# UP-FALL experiments
if [[ "$DATASET" == "upfall" || "$DATASET" == "all" ]]; then
    [[ "$MODEL" == "kalman" || "$MODEL" == "all" ]] && run_experiment "config/best_config/upfall/kalman.yaml" "upfall_kalman"
    [[ "$MODEL" == "raw" || "$MODEL" == "all" ]] && run_experiment "config/best_config/upfall/raw.yaml" "upfall_raw"
fi

# WEDA-FALL experiments
if [[ "$DATASET" == "wedafall" || "$DATASET" == "all" ]]; then
    [[ "$MODEL" == "kalman" || "$MODEL" == "all" ]] && run_experiment "config/best_config/wedafall/kalman.yaml" "wedafall_kalman"
    [[ "$MODEL" == "raw" || "$MODEL" == "all" ]] && run_experiment "config/best_config/wedafall/raw.yaml" "wedafall_raw"
fi

# Generate comparison summary
cat > "${OUTPUT_BASE}/comparison_summary.txt" << EOF
========================================================================
OPTIMAL CONFIGURATION COMPARISON
========================================================================
Generated: $(date)

CONFIGURATIONS:
---------------
UP-FALL (18Hz):
  Window: 160 samples (8.9s)
  Stride: fall=8, adl=32
  Embed: 64
  Model: KalmanConv1dConv1d / DualStreamBaseline

WEDA-FALL (50Hz):
  Window: 192 samples (3.84s)
  Stride: fall=8, adl=32
  Embed: 48
  Model: KalmanConv1dConv1d / DualStreamBaseline

========================================================================
RESULTS
========================================================================

EOF

for exp in upfall_kalman upfall_raw wedafall_kalman wedafall_raw; do
    report="${OUTPUT_BASE}/${exp}/summary_report.txt"
    if [[ -f "$report" ]]; then
        echo "${exp}:" >> "${OUTPUT_BASE}/comparison_summary.txt"
        grep -E "(Test F1|Test Accuracy)" "$report" | sed 's/^/  /' >> "${OUTPUT_BASE}/comparison_summary.txt"
        echo "" >> "${OUTPUT_BASE}/comparison_summary.txt"
    fi
done

cat "${OUTPUT_BASE}/comparison_summary.txt"

echo ""
echo "========================================================================"
echo "DONE - All results saved to: ${OUTPUT_BASE}"
echo "========================================================================"
