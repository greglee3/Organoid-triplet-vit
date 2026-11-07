# Organoid Triplet ViT

Training and analysis code for bright-field classification of hypothalamic–pituitary
organoid differentiation. The current revision compares CE-only ViT, Triplet-ViT,
EfficientNetV2-S, and a 50:50 Triplet–EfficientNet probability ensemble across ten
matched seeds. Historical single-seed results remain available separately.

## Current results

Each model uses the same 300 held-out images. Within each seed, five fold-specific
softmax vectors are averaged before classification. The table reports the mean
and sample standard deviation of the ten per-seed metrics, not metrics from a
prediction ensemble across seeds.

| Model | Accuracy (%) | Category C recall (%) |
|---|---:|---:|
| CE-only ViT | 71.00 ± 0.65 | 80.70 ± 2.36 |
| Triplet-ViT | 70.57 ± 1.09 | 81.50 ± 2.01 |
| EfficientNetV2-S | 74.07 ± 1.11 | 82.50 ± 3.57 |
| 50:50 ensemble | 74.00 ± 1.33 | 83.80 ± 2.20 |

The numerical C-recall increases were accompanied by lower precision and
specificity. The matched ViT recall-difference image-bootstrap 95% interval
included zero. The ensemble comparison does not isolate a triplet-objective
effect because it does not include a matched CE–EfficientNet ensemble control.
These results describe screening trade-offs, not universal superiority.

## Check the results without training

Install the analysis packages and run from the repository root:

```bash
python -m pip install numpy scipy scikit-learn matplotlib pillow
python scripts/check_package.py
python scripts/report_revision.py
python scripts/report_revision_thresholds.py
python -m unittest discover -s tests
```

The reporters read supplied per-image probabilities and reference labels. They
do not load microscopy images, download weights, or run model inference. Results
are written under `generated/`. The main reporter verifies all 40 model–seed
confusion matrices, the ensemble arithmetic, and the stored metrics. It also
computes paired class-stratified image-bootstrap intervals and exact per-seed
McNemar tests. No t-test is used by this current reporter.

## Data and reproducibility records

- `revision/results/test_predictions.csv`: 12,000 records (300 images × 4 models × 10 seeds).
- `revision/folds/`: exact training/validation assignments and ordered image IDs for all ten seeds.
- `revision/oof/`: both ViT conditions' 1,200 OOF predictions for every seed.
- `revision/results/`: baseline, normalization, common-recipe, masking, threshold, uncertainty, and historical comparison evidence.
- `revision/diagnostics/`: training histories, triplet activity and hyperparameter-selection records.
- `revision/environment_observed_20261107.json`: observed package versions, exact pretrained-weight URLs and cached-file SHA-256 hashes.
- `revision/SOURCE_MANIFEST.json`: source and distributed-file hashes for curated records and source snapshots.
- [Experiment and table mapping](docs/RESULT_MAPPING.md): current manuscript/supplementary locations and scope.
- [Reproduction guide](docs/REVISION_REPRODUCIBILITY.md): seeds, settings, weight identifiers, commands and limits.

Exact primary seeds: `70384, 12731, 76313, 98831, 80324, 104729, 130363, 155921,
180749, 205019`. Seed 42 is historical and is not in the ten-seed summary.

Original images and trained weights are not redistributed. Obtain the original
public dataset as described in [data/README.md](data/README.md). The preserved
labels are A: RAX-positive area ≥70%; B: ≥40% and <70%; C: <40%. Per-image
continuous RAX values and verified biological organoid/batch IDs were unavailable.

## Reproduce training and inference

Use the documented research environment (Python 3.10, PyTorch 2.6.0/CUDA 12.4,
torchvision 0.21.0, timm 0.6.13). The direct dependency pins are in
`requirements.txt` and `requirements-cuda124.txt`. Numerical results may vary with
hardware and dependency versions. Linux with an NVIDIA GPU is the training target.

```bash
python scripts/run_revision_training.py --help
python scripts/run_revision_training.py --stage all
```

The default is a **dry run**. Inspect the commands first. Add `--execute` only
when prepared to run substantial GPU training and held-out test inference.
Use `--seeds` for a prespecified subset. The command refuses to overwrite a
completed evaluation and validates fold/image identities before test inference.
Only load trusted checkpoints.

The original trainers are unchanged. `revision/workflows/` additionally preserves
the experiment-specific research scripts behind supplementary analyses. Those
scripts expect their documented historical directory/dependency layout and are
not a unified one-command pipeline; see the reproduction guide before using them.

## Historical evidence and release status

The previous single-seed package is documented in
[Historical reproduction](docs/HISTORICAL_REPRODUCTION.md). Existing
`reference_results/` and training workflows 07/11/14 were retained for provenance.
The legacy `scripts/report_results.py` and `scripts/plot_roc.py` are historical
seed-42 tools; use `report_revision.py` for the current manuscript.

See [validation](VALIDATION.md) for what was actually tested during packaging.
No manuscript drafts, reviewer correspondence, microscopy images, checkpoints,
credentials or machine addresses are part of this release. Repository visibility
and licensing require author approval; no new license or visibility change is
implied by this update. Reviewers must be granted access if the repository is private.
