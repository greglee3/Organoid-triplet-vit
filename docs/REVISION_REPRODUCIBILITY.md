# Revision reproduction guide

## Protocol and aggregation

The primary ten seeds are 70384, 12731, 76313, 98831, 80324, 104729, 130363,
155921, 180749 and 205019. The first five are the original OOF seeds; five were
added during revision after inspecting earlier results. All use the same 1,200
training and 300 held-out images. Fold manifests preserve the original ordering,
labels, shuffled image-level KFold assignments and seed. Biological identifiers
were unavailable; these are not verified organoid- or batch-disjoint folds.

ViT conditions share initialization, fold assignments and the training recipe:
20 epochs, balanced batches of 24 (8/class), AdamW encoder LR 3e-5, head LR 3e-4,
weight decay 0.01, two warm-up epochs then cosine scheduling, GridMask and AugMix
each with probability 0.7. Triplet weight is 0.0025 versus 0 for CE-only; uniform
margin is 0.1, cosine distance on normalized shared 384-dimensional features,
without a projection head or ordinal penalty. Checkpoints maximize validation
macro-F1, with loss tie-breaking. Legacy trainer names include `ordinal` but the
reported commands disable it.

The original EfficientNet recipe uses 50 epochs, batch size 35, AdamW LR 1.5e-4,
weight decay 0.01, StepLR every 3 epochs with gamma 0.97, notebook-style GridMask
probability 0.3 and horizontal/vertical flips each 0.5; all parameters are trained.
The head is 1280–512–SiLU–Dropout(0.3)–3. Checkpoints minimize validation CE.
Table S11 separately evaluates a common recipe on three seeds; do not merge its
results with the original-recipe ten-seed table.

The main pipeline converts images to RGB, resizes directly to 224×224 with
bilinear antialiased interpolation, then uses ToTensor without channel-wise
normalization. Five fold softmax vectors are averaged per model and seed.
The 50:50 ensemble averages the Triplet and EfficientNet vectors. Metrics are
computed separately for each seed before calculating mean and sample SD (ddof=1).

## Pretrained weights and environment

The code calls timm 0.6.13 `vit_small_patch16_224` and `tf_efficientnetv2_s`
with `pretrained=True`. The observed default weight identifiers are:

| Model | Weight artifact | SHA-256 of cached artifact |
|---|---|---|
| ViT-Small | `S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz` | `545815b4e770d2fa6ca4b3ccba7c16b035e474354e52d17ca197ea4efecbf4d3` |
| EfficientNetV2-S | `tf_efficientnetv2_s-eb54923e.pth` | `eb54923e44978b92563d63bb0282b61768b1029a334b845fbeead6ea4e731865` |

Full URLs, timm configurations and observed package versions are recorded in
`revision/environment_observed_20261107.json`. This is a read-only current
environment/cache inspection, not an immutable training-time environment lock.
The original source hashes and later packaging transformations are recorded in
the two SOURCE_MANIFEST files. Cite a specific Git commit (`git rev-parse HEAD`)
rather than only the moving `main` branch for the submitted revision.

## Analysis without model training

`scripts/report_revision.py` reconstructs per-seed metrics, mean/SD, ROC curves,
mean confusion matrices and paired uncertainty from the 12,000 prediction rows.
The bootstrap uses 20,000 class-stratified paired image resamples with RNG seed
20261102; the same resampled image identities are used for both models and every
seed. It averages metric differences, not probabilities across seeds. Exact
McNemar tests are computed per seed, with Holm adjustments across seeds, without
pooling repeated images as independent samples.

`scripts/report_revision_thresholds.py` chooses thresholds using OOF predictions
only at the historical specificity targets of 90% and 92.5%, validates them
against the frozen thresholds, and applies them unchanged to test probabilities.
Report realized test specificity as well as recall; the target is not forced
onto test results. This is an exploratory revision analysis, not prospective
clinical threshold validation.
The threshold reporter also exports Category C mean precision–recall curves;
their shaded bands describe sample SD across fitted seeds, not confidence intervals.

## Training entry point

`scripts/run_revision_training.py` uses the unchanged original trainers, explicit
seed subsets and fresh output directories. It defaults to displaying commands.
Use `--help`, review the dry run, and add `--execute` only when intended. It
checks exact fold and image identities and writes source/checkpoint provenance.
There is no automatic resume or overwrite of historical experiments. The new
entry point was statically checked and dry-run tested, not fully retrained during
packaging. Running it cannot guarantee bitwise reproduction on another machine.

## Supplementary source snapshots

`revision/workflows/` contains selected experiment-specific research scripts.
Private root paths were replaced with a configurable `ORGANOID_PROJECT_ROOT`
where possible. These are source snapshots, not guaranteed standalone commands:
several require historical run layouts, reference-script aliases, checkpoints,
or original source-hash guards. The fresh primary training entry point above is
the supported portable interface. Do not launch archival scripts merely by importing
them. Their corresponding saved numerical evidence is supplied independently.

The three-image masking check can be reconstructed from the original archive
and `masking_three_images_protocol.json`: use the recorded original-pixel polygon,
preserve inside pixels, fill outside black, and apply the original preprocessing.
ROI outlines are approximate manual regions, not segmentation ground truth.
This small perturbation check cannot establish absence of background dependence.

## Access and remaining limitations

Dataset pixels and trained checkpoints are not distributed here. Supplemental
normalization and common-recipe results have archived aggregate records but no
per-image inference exports in this curated package. Reproducing those analyses
from weights requires their experiment-specific workflow and trained checkpoints.
The supplied primary predictions allow the main paired claims to be checked
without those assets.

No software license or repository visibility change is implied. Before submission,
the authors should provide reviewer access to a private repository or approve a
public release after reviewing third-party redistribution rights. Do not claim
unrestricted public access while the repository is private.
