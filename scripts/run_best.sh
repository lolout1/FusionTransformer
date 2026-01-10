#!/bin/bash
# Train Dual-Stream Kalman Transformer (91.10% F1)
# Usage: ./scripts/run_best.sh [work_dir]

set -e

WORK_DIR=${1:-"results/reproduce_$(date +%Y%m%d_%H%M%S)"}
CONFIG="config/smartfallmm/reproduce_91.yaml"

echo "==========================================="
echo "Dual-Stream Kalman Transformer Training"
echo "==========================================="
echo "Config: ${CONFIG}"
echo "Output: ${WORK_DIR}"
echo "==========================================="

python main.py --config ${CONFIG} --work-dir ${WORK_DIR}

echo ""
echo "==========================================="
echo "Training Complete"
echo "==========================================="
if [ -f "${WORK_DIR}/scores.csv" ]; then
    echo "Results:"
    grep "Average" "${WORK_DIR}/scores.csv"
fi
