# Dataset preparation

Images and model weights are not included. Obtain the dataset through the
Data availability section of Asano et al., Communications Biology (2024),
https://doi.org/10.1038/s42003-024-07109-1 and follow its access and reuse terms.
No redistribution permission is implied by this repository.

The supplied workflow expects `data/Asano_Full_Data.zip`. Preserve the original
image filenames and archive paths. The ZIP must contain paths ending in:

```
train/A/<image>.tif
train/B/<image>.tif
train/C/<image>.tif
test/A/<image>.tif
test/B/<image>.tif
test/C/<image>.tif
```

A leading directory such as `data/` is supported. There are 1,200 training
images (400 per class) and 300 test images (100 per class). The reader sorts
the full image paths; renaming files or changing their relative prefixes can
affect fold membership. Compare the generated seed-42 `folds.json` with
`reference_results/seed42_folds.json` before interpreting reproduction results.

The models consume RGB-converted bright-field images and categorical labels.
They do not consume fluorescence images or continuous RAX measurements.
Training augmentation is not applied to validation or test images.
