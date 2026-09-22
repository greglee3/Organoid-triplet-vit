#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_ROOT="${PROJECT_ROOT}/05_final_tuning/PROFESSOR_FEEDBACK/07_locked_triplet_final_test"
OUTPUT_ROOT="${PROJECT_ROOT}/runs/05_final_tuning/PROFESSOR_FEEDBACK/07_locked_triplet_final_test"
TRAIN_ROOT="${OUTPUT_ROOT}/paired_seed_42_five_fold"
TEST_ROOT="${OUTPUT_ROOT}/final_public_test"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
TRAIN="${PROJECT_ROOT}/05_final_tuning/02_ordinal_aware_triplet/train_pilot.py"

mkdir -p "${OUTPUT_ROOT}"
if [[ -f "${OUTPUT_ROOT}/LOCKED_CONFIG.json" ]]; then
  if ! cmp -s "${SCRIPT_ROOT}/LOCKED_CONFIG.json" "${OUTPUT_ROOT}/LOCKED_CONFIG.json"; then
    echo "ERROR: persisted lock config differs from the code copy" >&2
    exit 1
  fi
else
  cp "${SCRIPT_ROOT}/LOCKED_CONFIG.json" "${OUTPUT_ROOT}/LOCKED_CONFIG.json"
fi
sha256sum "${OUTPUT_ROOT}/LOCKED_CONFIG.json" > "${OUTPUT_ROOT}/LOCKED_CONFIG.sha256"

CE_METRICS="${TRAIN_ROOT}/lambda_0/oof_metrics.json"
TRIPLET_METRICS="${TRAIN_ROOT}/lambda_0p0025/oof_metrics.json"
if [[ ! -f "${CE_METRICS}" || ! -f "${TRIPLET_METRICS}" ]]; then
  "${PYTHON}" -u "${TRAIN}" \
    --data "${PROJECT_ROOT}/data/Asano_Full_Data.zip" \
    --output "${TRAIN_ROOT}" \
    --ordinal-lambda 0 --lambdas 0,0.0025 \
    --mining-mode semi-hard --adjacent-margin 0.1 --far-margin 0.1 \
    --embedding-head shared --projection-dim 128 \
    --epochs 20 --folds 5 \
    --classes-per-batch 3 --samples-per-class 8 --workers 8 \
    --lr 3e-5 --head-lr-multiplier 10 \
    --optimizer adamw --weight-decay 0.01 \
    --warmup-epochs 2 --scheduler cosine \
    --scheduler-milestones 10,15 --scheduler-gamma 0.1 \
    --triplet-ce-only-epochs 0 --triplet-ramp-epochs 0 \
    --augmentation paper --seed 42 \
    --min-c-specificity 0.90 --max-c-recall-drop 0.02 \
    --max-c-precision-drop 0.03 --device cuda
else
  echo "paired five-fold training already complete; skip"
fi

"${PYTHON}" "${SCRIPT_ROOT}/evaluate_locked_test_once.py" \
  --data "${PROJECT_ROOT}/data/Asano_Full_Data.zip" \
  --run-root "${TRAIN_ROOT}" \
  --lock-config "${OUTPUT_ROOT}/LOCKED_CONFIG.json" \
  --output "${TEST_ROOT}" \
  --batch-size 25 --workers 8 --confirm-final-test

echo "Locked Triplet-ViT final test workflow complete"
