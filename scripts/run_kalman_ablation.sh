#!/bin/bash
# Kalman Filter Ablation Study - Sequential

cd /mmfs1/home/sww35/FeatureKD

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS="results/kalman_ablation_${TIMESTAMP}"
mkdir -p ${RESULTS}

CONFIGS=(
    balanced_lkf_euler
    balanced_ekf_euler
    balanced_ukf_euler
    balanced_lkf_gravity
    balanced_ekf_quat
    balanced_ukf_quat
    kghf_lkf_euler
    kghf_ekf_euler
    kghf_ukf_euler
    kghf_lkf_gravity
    kghf_ekf_quat
    kghf_ukf_quat
)

echo "============================================================"
echo "KALMAN ABLATION STUDY"
echo "Start: $(date)"
echo "Results: ${RESULTS}"
echo "============================================================"

for cfg in "${CONFIGS[@]}"; do
    echo ""
    echo ">>> Running: ${cfg}"
    python main.py \
        --config config/smartfallmm/kalman_ablation/${cfg}.yaml \
        --work-dir ${RESULTS}/${cfg} \
        --enable-wandb \
        --device 0
done

echo ""
echo "============================================================"
echo "COMPLETE: $(date)"
echo "============================================================"
echo "SUMMARY:"
for cfg in "${CONFIGS[@]}"; do
    if [ -f "${RESULTS}/${cfg}/scores.csv" ]; then
        f1=$(tail -n +2 "${RESULTS}/${cfg}/scores.csv" | cut -d',' -f15 | awk '{s+=$1;n++}END{if(n>0)printf "%.2f",s/n}')
        printf "%-25s F1: %s%%\n" "${cfg}" "${f1}"
    fi
done
