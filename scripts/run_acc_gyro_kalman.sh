#!/bin/bash
#===============================================================================
# AccGyroKalman Training Script
#
# Runs AccGyroKalmanTransformer on WEDA-FALL and UP-FALL datasets.
#
# Model: Dual-stream Transformer
#   - Stream 1: Acc + Orientation (6ch) [smv, ax, ay, az, roll, pitch]
#   - Stream 2: Raw Gyroscope (3ch) [gx, gy, gz]
#
# Usage:
#   ./scripts/run_acc_gyro_kalman.sh              # Full run (both datasets)
#   ./scripts/run_acc_gyro_kalman.sh --upfall     # UP-FALL only
#   ./scripts/run_acc_gyro_kalman.sh --wedafall   # WEDA-FALL only
#   ./scripts/run_acc_gyro_kalman.sh --quick      # Quick test (2 folds)
#===============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Detect GPUs
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo "1")
echo "Detected $NUM_GPUS GPUs"

# Default settings
DATASETS="upfall wedafall"
MAX_FOLDS=""
OUTPUT_BASE="exps/acc_gyro_kalman"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --upfall)
            DATASETS="upfall"
            shift
            ;;
        --wedafall)
            DATASETS="wedafall"
            shift
            ;;
        --quick)
            MAX_FOLDS="--max-folds 2"
            shift
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --upfall      Run only on UP-FALL dataset"
            echo "  --wedafall    Run only on WEDA-FALL dataset"
            echo "  --quick       Quick test (2 folds per dataset)"
            echo "  --num-gpus N  Number of GPUs to use"
            echo "  --help        Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Activate venv
if [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
fi

# Timestamp for output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_BASE}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "AccGyroKalman Training"
echo "============================================================"
echo "Datasets:   $DATASETS"
echo "GPUs:       $NUM_GPUS"
echo "Output:     $OUTPUT_DIR"
echo "============================================================"
echo ""

# Run training for each dataset
for DATASET in $DATASETS; do
    CONFIG="config/best_config/${DATASET}/acc_gyro_kalman.yaml"
    WORK_DIR="${OUTPUT_DIR}/${DATASET}"

    if [[ ! -f "$CONFIG" ]]; then
        echo "ERROR: Config not found: $CONFIG"
        continue
    fi

    echo ""
    echo "============================================================"
    echo "Training on $DATASET"
    echo "Config: $CONFIG"
    echo "============================================================"

    python ray_train.py \
        --config "$CONFIG" \
        --num-gpus "$NUM_GPUS" \
        --work-dir "$WORK_DIR" \
        $MAX_FOLDS \
        2>&1 | tee "${WORK_DIR}/train.log"

    echo "Completed $DATASET"
done

echo ""
echo "============================================================"
echo "All training completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "============================================================"

# Show summary
for DATASET in $DATASETS; do
    SUMMARY="${OUTPUT_DIR}/${DATASET}/summary_report.txt"
    if [[ -f "$SUMMARY" ]]; then
        echo ""
        echo "=== $DATASET Results ==="
        cat "$SUMMARY"
    fi
done
