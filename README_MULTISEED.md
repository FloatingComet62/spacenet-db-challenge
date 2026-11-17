# SpaceNet 8 — multi-seed and augmentation diagnostics (v2)

This update is a **drop-in replacement** for the previous experiment-suite ZIP.
Extract into the **same repository root** as `exp.py`, `src/`, and `data/`.
Only suite files are replaced. It does **not** include or modify any
`experiments_output/`, source TIFFs, source annotations, existing checkpoints,
or legacy code in `src/`.

## Commands

```bash
# Extract into your existing project root, overwriting previous suite source files:
unzip -o spacenet8_experiments_v2.zip

# In your existing CUDA/PyTorch environment:
python -m pytest -q tests/test_experiments.py tests/test_multiseed.py
python exp.py study --phase initial --dry-run
bash experiments_multiseed.sh

# Inspect model mistakes on validation images (not test):
python exp.py inspect-errors --run experiments_output/runs/clahe_seed111 --count 6
python exp.py inspect-errors --run experiments_output/runs/augmentation_seed111 --count 6
```

`experiments_multiseed.sh` runs the following `study --phase initial` tasks:

| Experiment | Seed 111 | Seed 222 | Seed 333 |
|---|---|---|---|
| raw_bce_dice | reuse | train | train |
| clahe | reuse | train | train |
| augmentation (combined) | reuse | — | — |
| augmentation_geometry | train | — | — |
| augmentation_photometric | train | — | — |

`reuse` assumes your seed-111 results were trained with **16 epochs, batch 2,
LR 1e-4, workers 2, stride 512** and the same frozen dataset split. The command
*checks* these values and refuses mismatches rather than combining unlike runs.
Nothing is overwritten automatically. If your seed-111 runs used different
settings, pass the matching parameters to `python exp.py study` instead of using
the convenience shell script. Do not run `python exp.py prepare --force` for this
study: you must retain the frozen data/split underlying the previous scores.

The suite deliberately trains only the **missing** runs. An existing incomplete
run blocks execution until you inspect, remove, or rename that run folder.

When initial experiments complete, you can extend the augmentation comparison
to **three matched seeds** (this adds six more runs):

```bash
python exp.py study --phase full --seeds 111 222 333 --epochs 16 --batch-size 2 --lr 0.0001 --workers 2 --stride 512
python exp.py validation-report --seeds 111 222 333
```

`--phase confirm` trains only CLAHE and raw+BCE/Dice for the chosen seeds;
`--phase diagnose` trains only the three augmentation policies for the **first**
provided seed. Both phases respect existing runs and check run compatibility.

## What this tests

- Raw + BCE/Dice versus CLAHE with the same architecture, spatial split,
  training budget, preprocessing timing, seed list, and evaluation.
- Original `augmentation` (geometry + brightness/contrast jitter) versus
  `augmentation_geometry` (flip / 90-degree rotations only) and
  `augmentation_photometric` (brightness/contrast only).
- The legacy combined mode is implemented exactly as in the first ZIP, so
  existing seed-111 `augmentation` checkpoints remain compatible.
- Validation threshold tuning is reused; all group summaries use tuned
  **validation**, not test, IoU. Standard deviations are **sample SD** across
  seeds (not confidence intervals).

## Outputs

```text
experiments_output/
  augmentation_preview.png            # source patch and three augmentations, GT outlined red
  validation_by_seed.csv               # every available seed, tuned val metrics
  validation_summary.csv               # mean and sample standard deviation
  validation_paired_deltas.csv         # within-seed differences versus raw_bce_dice
  runs/
    clahe_seed222/{best.pt,history.csv,config.json,threshold.json,...}
    ...
    clahe_seed111/
      validation_error_report.csv      # ranked validation tiles by FP fraction
      validation_errors.png            # original / confidence / TP-FP-FN overlay
```

Validation error overlays use **green = true positive**, **red = false
positive**, **blue = false negative**. Probability maps use the selected
validation threshold. Errors are sorted by false-positive area as a fraction
of image size; the CSV also includes false discovery fraction, tile IoU, and
other metrics.

## Why the test set is intentionally not run

You already evaluated the original seed-111 runs on test data and selected
CLAHE / raw+BCE-Dice after seeing those test scores. Therefore this follow-up
is **exploratory**; a newly independent held-out geographic area is required
for a fully unbiased final assessment. This package does not silently reuse
those known test scores for tuning or selecting the new models. The existing
`exp.py finalize` command is still available but should not be part of
experiment selection.

**Official SpaceNet 8 one-to-one comparison:** the earlier
`README_EXPERIMENTS.md` still documents how to run the actual official baseline
and score its `*_buildingpred.tif` predictions on the same held-out tiles.
The included `matched_bce` model is *not* the official SpaceNet 8 model.

This package was tested on a small synthetic raster dataset (13 tests, CPU).
No training on the actual Germany tiles or RTX 4050 was performed by this
packaging process.
