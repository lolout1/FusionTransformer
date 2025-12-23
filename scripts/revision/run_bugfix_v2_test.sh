#!/bin/bash
# BUGFIX V2 TEST - Matching 90.85% Baseline
# Run with: bash scripts/revision/run_bugfix_v2_test.sh

cd /mmfs1/home/sww35/FeatureKD

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/revision/bugfix_v2_${TIMESTAMP}"
CONFIG_DIR="config/smartfallmm/bugfix_v2"

mkdir -p "$RESULTS_DIR/logs"

echo "=============================================="
echo "BUGFIX V2 TEST - Matching 90.85% Baseline"
echo "=============================================="
echo "Key changes from bugfix_v1:"
echo "  - weight_decay: 1.0e-3 (was 5.0e-4)"
echo "  - adl_stride: 64 (was 32)"
echo "  - Class-aware truncation: falls=50, ADLs=128"
echo "  - Keeps: _len_check fix (>=1 windows)"
echo ""
echo "Results: $RESULTS_DIR"
echo "Start: $(date)"
echo "=============================================="

# A: Dual Kalman (Euler) - should match ~90.85% baseline
echo ""
echo "=== A_dual_kalman (Target: ~90.85%) ==="
python main.py --config $CONFIG_DIR/A_dual_kalman.yaml --work-dir $RESULTS_DIR/A_dual_kalman 2>&1 | tee $RESULTS_DIR/logs/A.log

# B: Dual Raw - should match ~89.08% baseline
echo ""
echo "=== B_dual_raw (Target: ~89.08%) ==="
python main.py --config $CONFIG_DIR/B_dual_raw.yaml --work-dir $RESULTS_DIR/B_dual_raw 2>&1 | tee $RESULTS_DIR/logs/B.log

# F: EKF Quaternion
echo ""
echo "=== F_ekf_quaternion ==="
python main.py --config $CONFIG_DIR/F_ekf_quaternion.yaml --work-dir $RESULTS_DIR/F_ekf_quaternion 2>&1 | tee $RESULTS_DIR/logs/F.log

# G: Madgwick Quaternion
echo ""
echo "=== G_madgwick_quaternion ==="
python main.py --config $CONFIG_DIR/G_madgwick_quaternion.yaml --work-dir $RESULTS_DIR/G_madgwick_quaternion 2>&1 | tee $RESULTS_DIR/logs/G.log

echo ""
echo "=============================================="
echo "SUMMARY"
echo "=============================================="
echo ""

for exp in A_dual_kalman B_dual_raw F_ekf_quaternion G_madgwick_quaternion; do
    if [ -f "$RESULTS_DIR/$exp/scores.csv" ]; then
        echo "=== $exp ==="
        grep "Average" "$RESULTS_DIR/$exp/scores.csv"
        echo ""
    fi
done

echo "Complete: $(date)"
echo "Results: $RESULTS_DIR"
