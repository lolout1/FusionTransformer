#!/bin/bash
# ==============================================================================
# Asymmetric Dual Transformer Experiment
# ==============================================================================
# Tests whether separate transformer encoders per stream with cross-attention
# can improve upon the baseline 91.10% F1 score.
#
# Architecture:
#   - Accelerometer stream: 3 transformer layers (deeper, reliable)
#   - Orientation stream: 1 transformer layer (shallower, noisy)
#   - Cross-attention between streams
#   - Heavy regularization (dropout + weight decay)
#
# Experiments:
#   A: With cross-attention (full model)
#   B: Without cross-attention (ablation)
#
# Usage:
#   ./scripts/run_asymmetric_dual_transformer.sh
#
# ==============================================================================

set -e

# Experiment settings
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/asymmetric_dual_transformer_${TIMESTAMP}"
CONFIG_DIR="config/smartfallmm/asymmetric_dual_transformer"
LOG_DIR="${RESULTS_DIR}/logs"

# Create directories
mkdir -p "${RESULTS_DIR}"
mkdir -p "${LOG_DIR}"

echo "=============================================================="
echo "Asymmetric Dual Transformer Experiment"
echo "=============================================================="
echo "Results directory: ${RESULTS_DIR}"
echo "Timestamp: ${TIMESTAMP}"
echo ""

# ==============================================================================
# Experiment A: With Cross-Attention (Full Model)
# ==============================================================================
echo "[A] Running Asymmetric Dual Transformer WITH Cross-Attention..."
echo "    Config: ${CONFIG_DIR}/asymmetric_cross_attn.yaml"
echo "    Output: ${RESULTS_DIR}/A_with_cross_attn"
echo ""

python main.py \
    --config "${CONFIG_DIR}/asymmetric_cross_attn.yaml" \
    --work-dir "${RESULTS_DIR}/A_with_cross_attn" \
    2>&1 | tee "${LOG_DIR}/A_with_cross_attn.log"

echo ""
echo "[A] Completed. Checking results..."
if [ -f "${RESULTS_DIR}/A_with_cross_attn/scores.csv" ]; then
    echo "=== Experiment A Results ==="
    cat "${RESULTS_DIR}/A_with_cross_attn/scores.csv"
else
    echo "WARNING: scores.csv not found for Experiment A"
fi
echo ""

# ==============================================================================
# Experiment B: Without Cross-Attention (Ablation)
# ==============================================================================
echo "[B] Running Asymmetric Dual Transformer WITHOUT Cross-Attention..."
echo "    Config: ${CONFIG_DIR}/asymmetric_no_cross_attn.yaml"
echo "    Output: ${RESULTS_DIR}/B_no_cross_attn"
echo ""

python main.py \
    --config "${CONFIG_DIR}/asymmetric_no_cross_attn.yaml" \
    --work-dir "${RESULTS_DIR}/B_no_cross_attn" \
    2>&1 | tee "${LOG_DIR}/B_no_cross_attn.log"

echo ""
echo "[B] Completed. Checking results..."
if [ -f "${RESULTS_DIR}/B_no_cross_attn/scores.csv" ]; then
    echo "=== Experiment B Results ==="
    cat "${RESULTS_DIR}/B_no_cross_attn/scores.csv"
else
    echo "WARNING: scores.csv not found for Experiment B"
fi
echo ""

# ==============================================================================
# Summary
# ==============================================================================
echo "=============================================================="
echo "EXPERIMENT SUMMARY"
echo "=============================================================="
echo ""
echo "Baseline (KalmanBalancedFlexible): 91.10% F1"
echo ""

echo "Experiment A (With Cross-Attention):"
if [ -f "${RESULTS_DIR}/A_with_cross_attn/scores.csv" ]; then
    grep "Average" "${RESULTS_DIR}/A_with_cross_attn/scores.csv" || echo "  No average found"
else
    echo "  Results pending"
fi

echo ""
echo "Experiment B (No Cross-Attention):"
if [ -f "${RESULTS_DIR}/B_no_cross_attn/scores.csv" ]; then
    grep "Average" "${RESULTS_DIR}/B_no_cross_attn/scores.csv" || echo "  No average found"
else
    echo "  Results pending"
fi

echo ""
echo "=============================================================="
echo "All experiments completed!"
echo "Results saved to: ${RESULTS_DIR}"
echo "=============================================================="
