# Historical package validation on September 22 2026

Checks completed on 2026-09-22 using a temporary copy in the existing research
server Python environment. The temporary directory was removed automatically.
Original research files, checkpoints, and outputs were not modified.

| Check | Result |
|---|---|
| Python syntax and JSON parsing | Passed |
| Exported source hashes and packaged-file hashes | Passed |
| Seed-42 reference sample inventory | 1,200 images |
| Balanced sampling | 8 images per class, 24 per batch; deterministic for a fixed seed |
| Uniform-margin semi-hard triplet calculation | Finite loss and gradients; invariant to class-name permutation |
| No eligible triplet | Zero loss |
| ViT synthetic CPU forward pass | 3 logits, 384-dimensional L2-normalized embedding; no projection head |
| EfficientNet synthetic CPU forward pass | 3 logits; all model parameters trainable |
| Trainer, evaluator, locking and summary CLI imports | Passed using `--help` |
| Three shell launchers | Passed `bash -n` syntax checks |
| Four public-test metric sets | Recomputed from confusion matrices and matched reference JSON |
| ROC plotting on historical prediction arrays | Passed; nine AUC values matched manuscript data |
| Targeted credential and absolute-server-path scan | No matching credentials or server-specific absolute paths found |

## Limits

This was not a fresh dependency installation or full experimental rerun.
The model tests used synthetic images and `pretrained=False`, so they did not
download pretrained weights or load trained checkpoints. The ROC check read
saved prediction arrays only. GPU training, end-to-end dataset training,
checkpoint inference, and reproducibility on another machine remain untested
for this packaged copy. The credential scan is a targeted check, not a security
certification. License and public-release approval are still pending.
