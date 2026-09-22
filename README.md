# Organoid Triplet ViT

Code for bright-field organoid classification with Triplet-ViT,
EfficientNetV2-S, and an equal-weight probability ensemble. A matched CE-only
ViT control is included. This repository packages the final manuscript
workflows, not the complete development history.

## Status and scope

- Prepared for private review; public release and licensing require author approval.
- Dataset images and checkpoints are not included.
- Historical result JSON files are included in `reference_results/`.
- Model training logic is preserved from the server snapshot. No full retraining
  has been performed as part of packaging; see `VALIDATION.md`.
- The original relative directory structure is retained because evaluators
  dynamically import the trainers. Do not rename these directories independently.
- The trainer's legacy directory name contains `ordinal`. The manuscript
  launchers disable that term, use the same 0.10 margin for every class pair,
  and use shared features without a projection head. Legacy optional branches
  remain in the source solely to avoid changing the original implementation.
  No separate ordinal experiment is included or required.

## Environment

The research server was observed on 2026-09-22 with Python 3.10.12,
PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, and timm 0.6.13.
This is a current environment snapshot, not proof that every package version
was identical at the historical training date. The requirements pin direct
dependencies; they are not a complete transitive environment lock.

The launchers target Linux/Bash and an NVIDIA CUDA GPU. From the repository root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cuda124.txt
python -m pip install -r requirements.txt
python scripts/check_package.py
python scripts/smoke_test.py
```

Use a PyTorch installation suitable for your platform if CUDA 12.4 is unavailable.
The manuscript launchers explicitly request CUDA. Pretrained model initialization
may require network access to download weights. Only load checkpoints from trusted sources.

## Dataset

See [data/README.md](data/README.md). Place the authorized original archive at
`data/Asano_Full_Data.zip`. Do not commit images or weights. The ZIP reader
supports a leading `data/` directory and preserves the original train/test split.

## Reproduce the final workflows

These commands perform substantial training. They are not installation tests.
Run them from the repository root in the order shown for the public-test results.

### Matched CE-only ViT and Triplet-ViT public test

```bash
bash 05_final_tuning/PROFESSOR_FEEDBACK/07_locked_triplet_final_test/run_locked_training_and_test.sh
```

Seed 42; shuffled image-level KFold (not stratified or grouped); five folds;
20 epochs; 24-image balanced batches (8 per class); GridMask and AugMix each
with probability 0.7. AdamW uses encoder LR 3e-5 and head LR 3e-4, weight decay
0.01, two warm-up epochs, and cosine scheduling. Triplet weight is 0.0025;
the CE control uses 0. Both conditions share fold assignments and initial weights.
The best validation macro-F1 checkpoint is selected, breaking ties with lower loss.

### EfficientNetV2-S and 50:50 ensemble public test

```bash
bash 05_final_tuning/PROFESSOR_FEEDBACK/11_locked_triplet_notebook_eff_final_test/run_locked_training_and_test.sh
```

This requires the preceding ViT workflow and reuses its fold assignments.
EfficientNet trains all parameters for 50 epochs with batch size 35, AdamW
LR 1.5e-4, weight decay 0.01, StepLR(step_size=3, gamma=0.97), notebook-style
GridMask probability 0.3, and horizontal/vertical flips each with probability 0.5.
Its head is 1280 → 512 → SiLU → Dropout(0.3) → 3. Checkpoints minimize validation CE.
Within each architecture, five fold probabilities are averaged. The two resulting
vectors are averaged with weights 0.5 and 0.5. There is no joint model training.

### Five-seed train-only OOF comparison

```bash
bash 05_final_tuning/PROFESSOR_FEEDBACK/14_ce_vs_triplet_confirmatory/run_confirmation.sh
```

Seeds: 70384, 12731, 76313, 98831, 80324. This trains 50 fold-specific models:
5 seeds × 5 folds × 2 conditions. Metrics are computed from pooled OOF
predictions within each seed and then summarized across seeds. It does not
use the public test set. This is separate from the seed-42 public-test evaluation.

### Results and figures

```bash
python scripts/report_results.py
```

This checks the supplied historical aggregate results and writes summary JSON
and public-test confusion matrices to `generated/`. It does not run inference.
After reproducing the public-test workflows, generate ROC curves from their
saved predictions with:

```bash
python scripts/plot_roc.py
```

Raw per-image probability arrays are not included in this review package;
ROC curves require the arrays produced by the evaluation commands.

## Evaluation safeguards

The public-test evaluators require explicit confirmation and refuse to overwrite
completed evaluations. The launchers supply that confirmation, so running them
includes public-test evaluation. Do not run them during hyperparameter selection.
Completed training stages may be skipped based on existing output files; do not
mix changed settings with an old `runs/` directory. Preserve prior runs separately.

Historical metadata explicitly records that the Triplet-ViT public-test result
was already available before the specific ensemble evaluation. Do not describe
the entire development history as a completely unseen external validation.
The public test contains held-out images from the same public source dataset.

## Expected results

| Model | Accuracy (%) | Macro-F1 (%) | C recall (%) | C specificity (%) |
|---|---:|---:|---:|---:|
| CE-only ViT | 73.00 | 73.02 | 82.00 | 92.50 |
| Triplet-ViT | 71.33 | 71.51 | 79.00 | 94.00 |
| EfficientNetV2-S | 73.67 | 72.60 | 88.00 | 91.00 |
| 50:50 ensemble | 73.00 | 72.70 | 83.00 | 93.00 |

The five-seed OOF mean changes (Triplet minus CE) were +2.70 percentage points
in C recall and +1.02 in C F1. The recall improvement did not carry over to
the public test. Numerical reproduction can vary with hardware and dependencies.

## Provenance and publication

See [docs/RESULT_MAPPING.md](docs/RESULT_MAPPING.md) and
[docs/PACKAGING_NOTES.md](docs/PACKAGING_NOTES.md). `SOURCE_MANIFEST.json` records
SHA-256 values of the selected server source files before packaging changes.
No manuscript draft, credentials, server address, or training images are included.
Third-party code provenance and redistribution rights must be reviewed before
public release. No software license has been selected; do not assume unrestricted reuse.
