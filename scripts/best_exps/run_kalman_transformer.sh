#!/bin/bash
# Train Dual-Stream Kalman Transformer (91.10% F1)
# Usage: ./scripts/best_exps/run_kalman_transformer.sh

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
WORK_DIR="results/kalman_transformer_${TIMESTAMP}"
CONFIG="config/smartfallmm/reproduce_91.yaml"

echo "=============================================="
echo "Dual-Stream Kalman Transformer"
echo "=============================================="
echo "Config:    ${CONFIG}"
echo "Output:    ${WORK_DIR}"
echo "Timestamp: ${TIMESTAMP}"
echo "Host:      $(hostname)"
echo "=============================================="

mkdir -p ${WORK_DIR}

python main.py \
    --config ${CONFIG} \
    --work-dir ${WORK_DIR} \
    2>&1 | tee ${WORK_DIR}/train.log

echo ""
echo "=============================================="
echo "Training Complete"
echo "=============================================="

if [ -f "${WORK_DIR}/scores.csv" ]; then
    echo ""
    echo "Results:"
    grep "Average" ${WORK_DIR}/scores.csv
    echo ""
    echo "Expected: Test F1 ~91.10%"
fi

echo "Output saved to: ${WORK_DIR}"
