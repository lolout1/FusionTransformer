#!/bin/bash
# EKF Quick Tuning - 3 folds only (worst performers: 31, 38, 52)
# Estimated time: ~3-4 hours for 24 configs × 3 folds
# Run with: bash scripts/revision/run_ekf_quick_tune.sh

cd /mmfs1/home/sww35/FeatureKD

RESULTS_DIR="results/revision/ekf_quick_tune"
BASE_CONFIG="config/smartfallmm/critique_ablation/ekf_quaternion.yaml"
CONFIG_DIR="$RESULTS_DIR/configs"

mkdir -p $RESULTS_DIR/logs $RESULTS_DIR/results $CONFIG_DIR

echo "EKF Quick Tuning (3 folds)"
echo "Start: $(date)"

# Quick subset of parameters (12 instead of 24)
Q_QUAT_VALUES="0.001 0.01 0.05"
Q_BIAS_VALUES="0.0001 0.001"
R_ACC_VALUES="0.05 0.1"

# Test subjects (worst performers)
TEST_SUBJECTS="31 38 52"

echo "exp_name,q_quat,q_bias,r_acc,s31_f1,s38_f1,s52_f1,avg_f1" > $RESULTS_DIR/tuning_results.csv

BEST_F1=0
BEST_CONFIG=""
exp_id=0

for q_quat in $Q_QUAT_VALUES; do
    for q_bias in $Q_BIAS_VALUES; do
        for r_acc in $R_ACC_VALUES; do
            exp_id=$((exp_id + 1))
            exp_name="ekf_q${q_quat}_b${q_bias}_r${r_acc}"

            echo ""
            echo "=== Experiment $exp_id/12: $exp_name ==="

            # Create config
            config_file="$CONFIG_DIR/${exp_name}.yaml"
            cp $BASE_CONFIG $config_file
            sed -i "s/kalman_Q_quat:.*/kalman_Q_quat: $q_quat/" $config_file
            sed -i "s/kalman_Q_bias:.*/kalman_Q_bias: $q_bias/" $config_file
            sed -i "s/kalman_R_acc:.*/kalman_R_acc: $r_acc/" $config_file

            # Modify subjects to only test on our 3 validation subjects
            # This runs full training but only tests on 3 specific subjects
            sed -i "s/^subjects:.*/subjects: [31, 38, 52]/" $config_file

            work_dir="$RESULTS_DIR/results/${exp_name}"

            python main.py \
                --config $config_file \
                --work-dir $work_dir \
                2>&1 | tee $RESULTS_DIR/logs/${exp_name}.log | tail -5

            # Extract F1 for each subject
            f1_31="" f1_38="" f1_52=""
            for subj in 31 38 52; do
                scores_file="$work_dir/fold_s${subj}/scores_s${subj}.csv"
                if [ -f "$scores_file" ]; then
                    f1=$(grep "^$subj," "$scores_file" 2>/dev/null | cut -d',' -f16)
                    eval "f1_${subj}=$f1"
                fi
            done

            # Calculate average
            if [ -n "$f1_31" ] && [ -n "$f1_38" ] && [ -n "$f1_52" ]; then
                avg_f1=$(python3 -c "print(f'{($f1_31 + $f1_38 + $f1_52) / 3:.2f}')")
                echo "  S31: $f1_31%, S38: $f1_38%, S52: $f1_52%"
                echo "  Average: $avg_f1%"

                is_better=$(python3 -c "print(1 if $avg_f1 > $BEST_F1 else 0)")
                if [ "$is_better" -eq 1 ]; then
                    BEST_F1=$avg_f1
                    BEST_CONFIG="Q_quat=$q_quat, Q_bias=$q_bias, R_acc=$r_acc"
                    echo "  ** NEW BEST **"
                fi

                echo "$exp_name,$q_quat,$q_bias,$r_acc,$f1_31,$f1_38,$f1_52,$avg_f1" >> $RESULTS_DIR/tuning_results.csv
            else
                echo "  Missing results"
            fi
        done
    done
done

echo ""
echo "=== QUICK TUNING COMPLETE ==="
echo "End: $(date)"
echo ""
echo "Best config: $BEST_CONFIG"
echo "Best avg F1: $BEST_F1%"
echo ""
echo "Results: $RESULTS_DIR/tuning_results.csv"
echo ""
echo "Next: Run full 21-fold LOSO with best params"
