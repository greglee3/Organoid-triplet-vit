# Packaging notes

This folder was assembled from an explicit server allowlist on 2026-09-22.
Existing server files and checkpoints were not modified.

## Preserved

The two trainers, dataset reader, augmentation implementation, helper modules,
evaluators, ensemble-lock code, summary code, and locked ViT settings retain
their server contents. The original relative layout is necessary for imports.
`paper_reproduction.py` and `training_common.py` include legacy functions, but
are required dependencies. Follow README launchers rather than legacy defaults.

## Packaging changes

- Replaced a fixed server working directory in the ensemble shell launcher
  with a path resolved from the launcher's own location.
- Added LF-normalized packaged hashes alongside raw source hashes so integrity
  checks remain valid across Git checkouts with different line endings.
- Added README, dependency specifications, ignore rules, tests, provenance,
  and result-reporting scripts. These additions are packaging utilities, not
  the historical training implementation.
- Excluded the legacy broad component test that covers exploratory methods;
  the new smoke test checks the manuscript's uniform-margin configuration.
- Copied four small historical reference JSON files, including seed-42 fold
  membership. No image pixels, checkpoints, raw predictions, or full logs are included.

## Before public release

- Confirm author/lab approval and choose a software license.
- Review attribution and license obligations for code adapted from third-party
  implementations; dependency installation does not settle source-code rights.
- Confirm dataset citation and permitted redistribution before adding any data.
- Decide whether to release checkpoints separately with checksums and provenance.
- Validate installation in a clean environment and, if needed, perform a full
  independent reproduction. Current checks are not a substitute for retraining.
- Review repository history and files for credentials before changing visibility.

The package is for private repository review, not an approved public release.
