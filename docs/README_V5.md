# SpaceNet 8 Germany: v5 confidence-aware postprocessing

**Scope:** Validation-only study of *existing* trained models. No retraining and no test evaluation. The v5 ZIP contains the complete prior v4 suite plus the v5 extension, so it can be extracted over your existing project root. It overwrites the experiment suite source (`exp.py`, `experiments/`, tests and READMEs), but not your `src/`, data, checkpoints, `experiments_output/`, or existing v3/v4 results.

## Why v5?

v4 improved validation instance F1 by filtering predicted components under 64 pixels, but this often removed correctly matched small buildings. v5 keeps large components and selectively retains small, high-confidence components. It chooses **one common rule per model family across all three random seeds**, rather than tuning independent postprocessing rules for individual seeds.

The three model seeds share the same validation image tiles. They are *not* independent geographic test areas. Reported improvements are development-set improvements, not unbiased held-out generalization estimates.

## Installation

Unzip at the root of your existing project (where `exp.py`, `src/`, and `experiments_output/` are):

```bash
unzip -o spacenet8_experiments_v5.zip
python -m pytest -q
```

Use your existing environment. No new requirements beyond the v4 requirements are added. The project must already have completed v4 `postprocess-study` for each of these six directories:

```text
experiments_output/runs/raw_bce_dice_seed{111,222,333}
experiments_output/runs/augmentation_photometric_seed{111,222,333}
```

Those directories must contain `best.pt`, `config.json`, `threshold.json`, the v3 prediction cache (otherwise it will be generated from the checkpoint), and `diagnostics_v4/validation_selection.json`. Missing runs are errors; the study never silently drops a seed.

## Run the full default study

```bash
bash experiments_v5.sh
```

This runs the validation-only grid on both model families with seeds 111, 222, 333, and then creates separate small-building before/after galleries for seed 111 of each family. **It never reads test masks or runs `finalize`.**

To run only one model or to change the grid:

```bash
python exp.py v5-study --models augmentation_photometric --seeds 111 222 333
python exp.py v5-gallery --model augmentation_photometric --seed 111 --count 9
```

You can also call `bash experiments_v5.sh --models augmentation_photometric`, which runs the study only; run the gallery command separately in this case.

## Candidate rules

v5 considers:

1. **Unchanged v3 baseline:** same per-checkpoint validation-tuned threshold and the original 16-pixel *instance extraction* minimum; the pixel mask is unfiltered.
2. **v4-style filtering:** keep only 4-connected components with area >= 64 pixels.
3. **Confidence filtering:** keep components >= 64 pixels; for components >= 8 or 16 pixels, keep only if their **mean**, **90th percentile**, or **maximum** pixel probability exceeds a cutoff. Smaller components are discarded.

The default candidate grid tests `saved`, `0.25`, `0.35` as probability thresholds, small minimum area 8 and 16, and confidence cutoffs 0.60 or 0.80. `saved` uses **each model checkpoint's already-tuned validation threshold**; the filtering rule remains identical across seeds, but that threshold can vary by seed. For a literally identical numeric threshold, choose one of the fixed-threshold candidates in the output.

The unchanged baseline is always a candidate, so the selector does not force a harmful configuration when no confidence-aware policy satisfies the constraints.

## Selection rule

Select the candidate with the **highest mean validation instance F1** across all three seeds, breaking ties with worst-seed F1 and mean pixel IoU, subject to *both* restrictions for **every single seed**:

- Small-building true positives (GT polygon area <100 px², IoU>=0.50 one-to-one matching) cannot decrease relative to that seed's unchanged v3 baseline.
- Global tile-level pixel IoU cannot fall by more than 0.005 absolute versus that seed's unchanged v3 baseline.

These are intentionally stricter than selecting solely on F1. The default `0.005` is a **0.5 percentage-point** IoU tolerance; set `--max-iou-drop 0` to forbid any IoU decrease. Use `--small-recall-slack 0` to enforce no lost small matches (the default).

The same dataset split, polygon instance evaluator, and small/medium/large cutoffs from v3/v4 are used. Selection is done only on validation, so it does **not** establish independent generalization. Consider v5 a hypothesis test for whether a practical confidence-aware policy outperforms the v4 rule without losing small objects.

Custom larger sweep:

```bash
python exp.py v5-study \
  --thresholds saved 0.15 0.20 0.25 0.30 0.35 0.40 \
  --small-areas 8 16 24 32 \
  --confidence-cutoffs 0.50 0.60 0.70 0.80 0.90 \
  --confidence-metrics mean p90 max \
  --max-iou-drop 0.005
```

The larger sweep may take substantially longer than the default. Cached validation probabilities avoid rerunning GPU inference in a healthy existing installation. The grid still performs component and polygon matching for each candidate, which uses CPU time.

## Saved outputs

```text
experiments_output/diagnostics_v5/
  validation_suite_summary.csv
  raw_bce_dice/
    validation_shared_policy.json
    validation_shared_policy_grid.csv
    validation_seed_comparison.csv
    validation_building_changes.csv
    validation_shared_tradeoff.png
    validation_small_gallery_seed111.png
  augmentation_photometric/
    ...same files...
```

- **`validation_shared_policy.json`**: selected common policy, seed-level constraint definitions, checkpoint and split SHA-256 provenance, and the number of eligible candidates.
- **`validation_shared_policy_grid.csv`**: candidates, eligibility, mean and worst-seed instance F1, mean small recall and per-seed matched counts.
- **`validation_seed_comparison.csv`**: v3 vs v4 vs v5 F1, pixel IoU, and small-building matched counts for every seed.
- **`validation_building_changes.csv`**: each ground-truth building on every seed, with v3/v4/v5 match flags and v4-to-v5 `recovered`, `lost`, `both_matched`, or `both_missed` class. These are *matches to annotated GT*, not independently adjudicated physical buildings.
- **`validation_shared_tradeoff.png`**: mean pixel IoU vs mean instance F1, and mean small-building recall vs F1. Blue candidates meet *all seed-level* constraints; gray candidates do not.
- **`validation_small_gallery_seed111.png`**: same small building with original, v4 and v5 postprocessing. GT object outline yellow; true positives green, false positives red, false negatives blue. Gallery selects candidate objects for manual inspection and labels per-object match/miss.

You can view the key files with:

```bash
xdg-open experiments_output/diagnostics_v5/augmentation_photometric/validation_shared_tradeoff.png
xdg-open experiments_output/diagnostics_v5/augmentation_photometric/validation_small_gallery_seed111.png
cat experiments_output/diagnostics_v5/augmentation_photometric/validation_seed_comparison.csv
```

## Fair-comparison limitations

- The 3 seeds do **not** resolve the known 32 cross-split neighboring tile pairs. A spatially buffered, independent holdout should be used for a final scientific claim.
- The same validation images are used to select policies and report results. Do not treat the v5 score as a clean independent test estimate; do not keep retuning on existing test results.
- A retained small component may still be a false-positive object. A high mean or peak probability does not establish a correct building classification.
- v5 does not change model weights, improve boundary separation explicitly, or automatically reproduce the official SpaceNet model. Official comparison still requires genuine external predictions on the same evaluation tiles.

If the best eligible shared rule is `baseline_saved`, that's an informative outcome: the current probabilities do not support a safe confidence-filter improvement under the chosen constraints. It is *not* a software error.
