# Current manuscript result mapping

This mapping follows the November 7 revision. Paths are relative to the repository
root. `revision/results/` contains derived numerical records, not microscopy images.

| Manuscript item | Distributed evidence | Implementation or analysis |
|---|---|---|
| Table 1 dataset counts and image independence | `revision/overlap/`; `data/README.md` | `revision/workflows/dataset_leakage_audit.py` |
| Table 2 four-model ten-seed results and Figure 2 | `revision/results/test_predictions.csv`, `ten_seed_reference.json`; `revision/folds/` | `scripts/run_revision_training.py`; `scripts/report_revision.py` |
| Table 2 MLP and small CNN | `revision/results/simple_baselines_seed42.json`, `mlp_test_predictions_seed42.csv`, `cnn_test_predictions_seed42.csv` | `revision/workflows/train_baselines.py` |
| Table 3 five-seed internal OOF | `reference_results/multiseed_oof.json`; first five seeds in `revision/oof/` | Original workflow `05_final_tuning/PROFESSOR_FEEDBACK/14_ce_vs_triplet_confirmatory/` |
| Table 4 and opening supplementary ten-seed uncertainty | `revision/results/uncertainty.json` and current test CSV | `scripts/report_revision.py`: image bootstrap and exact McNemar; no t-test |
| Figure S1 CE and triplet losses and semi-hard activity | `revision/diagnostics/*_lambda_*.csv`, `Figure_S1_source_data.csv` | Original trainer logging; training loss only for panel c |
| Table S1A weight sensitivity | `revision/diagnostics/lambda_extension.json`, `09_lambda_followup*`, `08_lambda_retuning*` | `revision/workflows/lambda_followup.py`, `lambda_extension_20261101.py` |
| Table S1B margin sensitivity and S1C selection history | `revision/diagnostics/10_margin*`, `11_margin*`, `12_margin*`, `13_margin*`, `*LR_WD*` | `revision/workflows/margin_confirmation.py`; historical trainer settings |
| Table S2 B–C semi-hard activity | `revision/diagnostics/bc_activity.csv` | `revision/workflows/bc_activity_diagnostic.py` |
| Table S3 normalization sensitivity | `revision/results/normalization_seed42.json` | `revision/workflows/normalization_seed42_20261101.py` |
| Figure S2 illustrative predictions | `revision/results/illustrative_selection.json`, `illustrative_predictions_seed70384.csv` | Image IDs identify cases in the original dataset; images not redistributed |
| Tables S4–S5 ten-seed paired correctness | Current test CSV | `scripts/report_revision.py` / per-image argmax comparisons |
| Figure S3 and Table S6 masking | `revision/results/masking_three_images_protocol.json`, `masking_three_images_results.json` | Polygon coordinates, seed, image IDs and preprocessing supplied; retain inside pixels and blacken outside |
| Figure S4 ROC/PR and Tables S7–S8 threshold/ranking analysis | `revision/results/threshold_and_curves.json`, `frozen_oof_thresholds.json`; `revision/oof/`; current test CSV | `scripts/report_revision.py`; `scripts/report_revision_thresholds.py` |
| Table S9 supporting five-seed OOF matched-specificity comparison | `revision/results/oof_matched_specificity_five_seeds.json`, `oof_matched_specificity_seedwise.csv` | `revision/workflows/threshold_matched_analysis.py` |
| Figure S5 and Table S10 directional errors | Current test CSV and `revision/results/threshold_and_curves.json` | Mean confusion matrices in `scripts/report_revision.py`; archived direction-specific counts |
| Table S11 common training recipe | `revision/results/common_recipe_three_seeds.json` | `revision/workflows/efficientnet_matched_20261103.py` |
| Table S12 historical single-seed results | `reference_results/public_test_vit.json`, `public_test_ensemble.json` | Original workflows 07 and 11; not the primary ten-seed results |
| Aspect-ratio comparison supplied separately during revision | `revision/results/response_only_aspect_ratio_ten_seeds.json` | `revision/workflows/aspect_ratio_pilot.py`, `aspect_ratio_ten_seeds.py`; not a main/supplementary table |

The current primary test set is the same 300 images across seeds, not 3,000
independent test images. OOF estimates used validation folds for checkpoint
selection as well as scoring. Image-overlap checks do not establish biological
organoid/batch independence. Historical uncertainty JSON retains earlier seed-level
calculations for provenance; current manuscript intervals come from the paired
image bootstrap, not those t-tests.

Exploratory 10/20/300-image masking, crop extensions, alternative Asano recipes,
and unrelated model variants are not presented as additional primary manuscript
experiments. Their absence from this curated mapping is not evidence that they
were never investigated.
