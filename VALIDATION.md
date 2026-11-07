# Revision package validation

The November 7 2026 update was checked without retraining or altering historical
experiments. The dated September validation is preserved in
`docs/VALIDATION_20260922.md`.

| Check | Result |
|---|---|
| Forty model–seed probability arrays | 300×3, finite, normalized, shared 300 image IDs and labels |
| Stored test metrics | All forty confusion matrices and metric sets reproduced |
| Ensemble construction | Equal-weight Triplet/EfficientNet probabilities verified |
| Exact primary folds | All fifty folds match shuffled KFold with the recorded seeds; ordered training IDs agree |
| Reporting tests | Eleven synthetic tests passed |
| Paired bootstrap | All six aggregate and sixty per-seed intervals match archived results exactly |
| Exact McNemar | All ten discordant counts, exact p-values and Holm adjustments match |
| Validation-selected thresholds | All forty thresholds and held-out confusion counts match archived analysis |
| New training wrapper | Syntax, help, dry runs, invalid-seed rejection and trainer CLI compatibility checked |
| Training during packaging | Not performed |

The package is not a fresh-environment or independent full GPU reproduction.
The current environment snapshot is not proof of the exact historical transitive
environment. Source snapshots for supplementary workflows require their historical
dependencies. Targeted credential/path checks do not constitute a security audit.
Microscopy images, weights, manuscript drafts and reviewer correspondence are
excluded. Review access and licensing remain author responsibilities.
