#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

NUM_GPUS=3
MAX_FOLDS=""
OUTPUT_DIR=""
RUNNER="ray"
ENABLE_WANDB=0
GENERATE_PLOTS=1
DATASETS="smartfallmm,upfall,wedafall"

SMARTFALL_CONFIG="${ROOT_DIR}/config/best_config/smartfallmm/kalman.yaml"
UPFALL_CONFIG="${ROOT_DIR}/config/best_config/upfall/kalman.yaml"
WEDA_CONFIG="${ROOT_DIR}/config/best_config/wedafall/kalman.yaml"

usage() {
  cat <<USAGE
Usage: ./train.sh [options]

Options:
  --num-gpus N        GPUs per run (default: 3)
  --max-folds N       Limit folds per dataset
  --output-dir PATH   Base output directory
  --datasets LIST     Comma list: smartfallmm,upfall,wedafall (default: all)
  --runner MODE       ray|pipeline (default: ray)
  --wandb             Enable W&B logging
  --no-plots          Skip plot generation
  --upfall-config P   Override UP-FALL config path
  --smartfall-config P  Override SmartFallMM config path
  --wedafall-config P   Override WEDA-FALL config path
  -h, --help          Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-gpus) NUM_GPUS="$2"; shift 2 ;;
    --max-folds) MAX_FOLDS="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --datasets) DATASETS="$2"; shift 2 ;;
    --runner) RUNNER="$2"; shift 2 ;;
    --wandb) ENABLE_WANDB=1; shift 1 ;;
    --no-plots) GENERATE_PLOTS=0; shift 1 ;;
    --upfall-config) UPFALL_CONFIG="$2"; shift 2 ;;
    --smartfall-config) SMARTFALL_CONFIG="$2"; shift 2 ;;
    --wedafall-config) WEDA_CONFIG="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "${OUTPUT_DIR}" ]]; then
  TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  OUTPUT_DIR="${ROOT_DIR}/results/best_kalman_${TIMESTAMP}"
fi

mkdir -p "${OUTPUT_DIR}"

log_section() {
  echo ""
  echo "================================================================"
  echo "$1"
  echo "================================================================"
}

run_ray() {
  local dataset="$1"
  local config="$2"
  local work_dir="${OUTPUT_DIR}/${dataset}"

  if [[ ! -f "$config" ]]; then
    echo "Config not found: $config"
    exit 1
  fi

  mkdir -p "$work_dir"

  log_section "Running ${dataset} (ray_train)"
  echo "Config:  $config"
  echo "Output:  $work_dir"
  echo "GPUs:    $NUM_GPUS"
  echo "Folds:   ${MAX_FOLDS:-all}"

  cmd=("$PYTHON_BIN" "${ROOT_DIR}/ray_train.py" --config "$config" --num-gpus "$NUM_GPUS" --work-dir "$work_dir")
  if [[ -n "${MAX_FOLDS}" ]]; then
    cmd+=(--max-folds "$MAX_FOLDS")
  fi
  if [[ "$ENABLE_WANDB" -eq 1 ]]; then
    cmd+=(--enable-wandb)
  fi

  "${cmd[@]}"

  if [[ -f "$work_dir/summary_report.txt" ]]; then
    echo ""
    echo "Summary (test metrics):"
    grep -E "(Test F1|Test Accuracy|Test Macro|Test Precision|Test Recall)" "$work_dir/summary_report.txt" || true
  fi

  if [[ "$GENERATE_PLOTS" -eq 1 ]] && [[ -f "${ROOT_DIR}/tools/visualize.py" ]]; then
    echo "Plot generation available via: python tools/visualize.py --help"
  fi

  WORK_DIR="$work_dir" "$PYTHON_BIN" - <<'PY'
import csv
import os
import pickle
from pathlib import Path

work_dir = Path(os.environ["WORK_DIR"])
fold_path = work_dir / "fold_results.pkl"
if not fold_path.exists():
    print(f"No fold results at {fold_path}")
    raise SystemExit(0)

with open(fold_path, "rb") as f:
    fold_results = pickle.load(f)

rows = []
for fold in fold_results:
    val_losses = fold.get("val_losses") or []
    train_losses = fold.get("train_losses") or []
    best_epoch = None
    best_val_loss = None
    best_train_loss = None
    if val_losses:
        best_idx = min(range(len(val_losses)), key=lambda i: val_losses[i])
        best_epoch = best_idx + 1
        best_val_loss = val_losses[best_idx]
        if train_losses and best_idx < len(train_losses):
            best_train_loss = train_losses[best_idx]

    rows.append({
        "test_subject": fold.get("test_subject"),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_train_loss": best_train_loss,
        "test_f1_score": fold.get("test_f1_score"),
        "val_f1_score": fold.get("val_f1_score"),
        "train_f1_score": fold.get("train_f1_score"),
    })

out_path = work_dir / "best_epoch_summary.csv"
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

best_epochs = [r["best_epoch"] for r in rows if r["best_epoch"]]
if best_epochs:
    avg_best_epoch = sum(best_epochs) / len(best_epochs)
    print(f"Best-epoch summary: {out_path}")
    print(f"Average best epoch: {avg_best_epoch:.2f}")
PY
}

run_pipeline_upfall() {
  log_section "Running UP-FALL embed ablation (pipeline)"
  "$PYTHON_BIN" "${ROOT_DIR}/distributed_dataset_pipeline/run_upfall_embed_ablation.py" \
    --num-gpus "$NUM_GPUS" --parallel 1
}

run_pipeline_wedafall() {
  log_section "Running WEDA-FALL ablation (pipeline)"
  "$PYTHON_BIN" "${ROOT_DIR}/distributed_dataset_pipeline/run_wedafall_ablation.py" \
    --num-gpus "$NUM_GPUS" --parallel 1
}

IFS=',' read -r -a DATASET_LIST <<< "$DATASETS"

log_section "Train run"
echo "Runner:   $RUNNER"
echo "Datasets: ${DATASETS}"
echo "Output:   $OUTPUT_DIR"

declare -A CONFIGS
CONFIGS[smartfallmm]="$SMARTFALL_CONFIG"
CONFIGS[upfall]="$UPFALL_CONFIG"
CONFIGS[wedafall]="$WEDA_CONFIG"

for ds in "${DATASET_LIST[@]}"; do
  if [[ "$RUNNER" == "pipeline" ]]; then
    case "$ds" in
      upfall) run_pipeline_upfall ;;
      wedafall) run_pipeline_wedafall ;;
      smartfallmm) run_ray "$ds" "${CONFIGS[$ds]}" ;;
      *) echo "Unknown dataset: $ds"; exit 1 ;;
    esac
  else
    run_ray "$ds" "${CONFIGS[$ds]}"
  fi
done

log_section "Done"
echo "Results: $OUTPUT_DIR"
