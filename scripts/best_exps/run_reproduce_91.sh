#!/bin/bash
# =============================================================================
# Reproduce 91.10% - Model Selection Comparison
# =============================================================================
# Uses EXACT 91.10% config:
#   - Model: KalmanBalancedFlexible (SE + TAP)
#   - Data: old+young with train_only_subjects
#   - Params: lr=1e-3, wd=5e-4, 80 epochs
#   - activation: relu, no positional encoding
#
# Compares:
#   A: Save by Best Val F1 (new)
#   B: Save by Lowest Val Loss (original)
# =============================================================================

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_DIR="results/reproduce_91_${TIMESTAMP}"
CONFIG_DIR="config/smartfallmm"
PYTHON="/mmfs1/home/sww35/miniforge3/envs/py310/bin/python"

echo "=============================================="
echo "Reproduce 91.10% - Model Selection Comparison"
echo "=============================================="
echo "Timestamp: ${TIMESTAMP}"
echo "Hostname: $(hostname)"
echo "Model: KalmanBalancedFlexible (SE + TAP)"
echo "Data: old+young (train_only_subjects)"
echo "=============================================="

mkdir -p "${BASE_DIR}"

# Experiment A: Save by Best Val F1
echo ""
echo "=== [1/2] Save by Best Val F1 ==="
${PYTHON} main.py \
    --config "${CONFIG_DIR}/reproduce_91_val_f1.yaml" \
    --work-dir "${BASE_DIR}/A_val_f1"

# Experiment B: Save by Lowest Val Loss
echo ""
echo "=== [2/2] Save by Lowest Val Loss ==="
${PYTHON} main.py \
    --config "${CONFIG_DIR}/reproduce_91_val_loss.yaml" \
    --work-dir "${BASE_DIR}/B_val_loss"

# Summary
echo ""
echo "=============================================="
echo "All Experiments Complete!"
echo "=============================================="
echo ""
echo "Results Summary:"
echo "----------------"
for exp in A_val_f1 B_val_loss; do
    if [ -f "${BASE_DIR}/${exp}/scores.csv" ]; then
        avg=$(grep "Average" "${BASE_DIR}/${exp}/scores.csv" | awk -F',' '{printf "F1=%.2f%% Acc=%.2f%%", $16, $15}')
        echo "  ${exp}: ${avg}"
    fi
done

echo ""
echo "Target: 91.10%"
echo ""
echo "Config used:"
echo "  - Model: KalmanBalancedFlexible"
echo "  - SE: True, TAP: True"
echo "  - activation: relu"
echo "  - lr: 1e-3, wd: 5e-4"
echo "  - epochs: 80"
echo "  - Data: old+young with train_only_subjects"
