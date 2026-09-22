#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
TRAINER="${PROJECT_ROOT}/05_final_tuning/02_ordinal_aware_triplet/train_pilot.py"
SUMMARY_SCRIPT="${PROJECT_ROOT}/05_final_tuning/PROFESSOR_FEEDBACK/14_ce_vs_triplet_confirmatory/summarize_confirmation.py"
OUTPUT_ROOT="${PROJECT_ROOT}/runs/05_final_tuning/PROFESSOR_FEEDBACK/14_ce_vs_triplet_confirmatory"
RUN_ROOT="${OUTPUT_ROOT}/runs"
SUMMARY_ROOT="${OUTPUT_ROOT}/summary"
SEEDS=(70384 12731 76313 98831 80324)

mkdir -p "${RUN_ROOT}" "${SUMMARY_ROOT}"

for SEED in "${SEEDS[@]}"; do
  SEED_ROOT="${RUN_ROOT}/seed_${SEED}"
  CE_RESULT="${SEED_ROOT}/lambda_0/oof_metrics.json"
  TRIPLET_RESULT="${SEED_ROOT}/lambda_0p0025/oof_metrics.json"

  if [[ -f "${CE_RESULT}" && -f "${TRIPLET_RESULT}" ]]; then
    echo "complete; skip seed=${SEED}"
    continue
  fi

  echo "start seed=${SEED}: paired CE vs Triplet, 5-fold"
  "${PYTHON}" -u "${TRAINER}" \
    --data "${PROJECT_ROOT}/data/Asano_Full_Data.zip" \
    --output "${SEED_ROOT}" \
    --ordinal-lambda 0 \
    --lambdas 0,0.0025 \
    --mining-mode semi-hard \
    --adjacent-margin 0.10 \
    --far-margin 0.10 \
    --embedding-head shared \
    --projection-dim 384 \
    --distance-metric cosine \
    --embedding-normalization l2 \
    --epochs 20 \
    --folds 5 \
    --classes-per-batch 3 \
    --samples-per-class 8 \
    --workers 8 \
    --lr 3e-5 \
    --head-lr-multiplier 10 \
    --optimizer adamw \
    --weight-decay 0.01 \
    --warmup-epochs 2 \
    --scheduler cosine \
    --scheduler-milestones 10,15 \
    --scheduler-gamma 0.1 \
    --triplet-ce-only-epochs 0 \
    --triplet-ramp-epochs 0 \
    --augmentation paper \
    --seed "${SEED}" \
    --min-c-specificity 0.90 \
    --max-c-recall-drop 0.02 \
    --max-c-precision-drop 0.03 \
    --device cuda
done

"${PYTHON}" "${SUMMARY_SCRIPT}" \
  --run-root "${RUN_ROOT}" \
  --output "${SUMMARY_ROOT}" \
  --seeds "70384,12731,76313,98831,80324"

touch "${OUTPUT_ROOT}/COMPLETE.flag"
echo "Final paired CE-ViT vs Triplet-ViT confirmation complete"

