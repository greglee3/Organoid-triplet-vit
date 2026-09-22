#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_ROOT}"

CODE=05_final_tuning/PROFESSOR_FEEDBACK/11_locked_triplet_notebook_eff_final_test
RUN=runs/05_final_tuning/PROFESSOR_FEEDBACK/11_locked_triplet_notebook_eff_final_test
TRIPLET_RUN=runs/05_final_tuning/PROFESSOR_FEEDBACK/07_locked_triplet_final_test/paired_seed_42_five_fold
TRIPLET_TEST=runs/05_final_tuning/PROFESSOR_FEEDBACK/07_locked_triplet_final_test/final_public_test
FOLDS="$TRIPLET_RUN/folds.json"
TRIPLET_OOF="$TRIPLET_RUN/lambda_0p0025"
TRIPLET_LOCK=05_final_tuning/PROFESSOR_FEEDBACK/07_locked_triplet_final_test/LOCKED_CONFIG.json
EFFICIENT_OOF="$RUN/seed_42_notebook_efficientnetv2_s"
LOCK="$RUN/ENSEMBLE_LOCK.json"
FINAL="$RUN/final_public_test"

mkdir -p "$RUN"

if [[ ! -f "$EFFICIENT_OOF/oof_metrics.json" ]]; then
  .venv/bin/python -u \
    05_final_tuning/PROFESSOR_FEEDBACK/03_vit_triplet_efficientnet/train_efficientnet_oof.py \
    --data data/Asano_Full_Data.zip \
    --folds-json "$FOLDS" \
    --output "$EFFICIENT_OOF" \
    --seed 42 \
    --epochs 50 \
    --batch-size 35 \
    --workers 16 \
    --lr 0.00015 \
    --backbone-lr 0.00015 \
    --weight-decay 0.01 \
    --trainable-mode all \
    --scheduler step \
    --step-size 3 \
    --gamma 0.97 \
    --gridmask-probability 0.3 \
    --gridmask-implementation notebook \
    --horizontal-flip-probability 0.5 \
    --vertical-flip-probability 0.5 \
    --device cuda
fi

if [[ ! -f "$LOCK" ]]; then
  .venv/bin/python -u "$CODE/lock_ensemble.py" \
    --triplet-oof-root "$TRIPLET_OOF" \
    --efficient-oof-root "$EFFICIENT_OOF" \
    --folds-json "$FOLDS" \
    --triplet-lock-config "$TRIPLET_LOCK" \
    --output "$LOCK"
fi

.venv/bin/python -u "$CODE/evaluate_ensemble_test_once.py" \
  --data data/Asano_Full_Data.zip \
  --ensemble-lock "$LOCK" \
  --triplet-test-root "$TRIPLET_TEST" \
  --output "$FINAL" \
  --batch-size 35 \
  --workers 16 \
  --confirm-final-test
