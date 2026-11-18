# SpaceNet 8 Germany — v4 instance-analysis experiments

This ZIP is a **drop-in upgrade** of the existing experiment suite. Extract from your project root (where `exp.py`, `data/`, and `src/` live). It contains the v1–v3 experiment files plus the new v4 logic. It **does not include** or modify your original imagery, project `src/`, `experiments_output/`, existing model checkpoints, or frozen splits.

**Primary goal:** improve building-instance F1 by tuning postprocessing while preventing regression in *global pixel IoU*. **No model retraining is performed.** Every v4 search and visualization is *validation-only*. Your previously evaluated test set is not used for selection.

## Start here

```bash
unzip -o spacenet8_experiments_v4.zip
python -m pip install -r experiment_requirements.txt
python -m pytest -q
python exp.py postprocess-study --run experiments_output/runs/raw_bce_dice_seed111
python exp.py instance-audit --run experiments_output/runs/raw_bce_dice_seed111 --selection baseline
python exp.py instance-audit --run experiments_output/runs/raw_bce_dice_seed111 --selection best
```

Or run the same three commands using:

```bash
bash experiments_v4.sh
```

There is also an optional second run for your photometric checkpoint:

```bash
bash experiments_v4.sh experiments_output/runs/augmentation_photometric_seed111
```

If the `diagnostics_v3/validation_probs` cache exists from the v3 study, v4 reuses it after checking the checkpoint SHA-256 and frozen-split hash. Otherwise, v4 performs a one-time inference of **validation tiles only** and caches the probabilities. Keep your existing Python/Torch CUDA setup; the new watershed stage uses `scipy` and `scikit-image`. If you do not want these additional dependencies, set `--watershed-top 0`.

## What is compared?

**All configurations use your saved network outputs and the same validation GT polygons.** Each tile is scored once in original pixel coordinates (no patch double-counting).

1. **Unmodified v3 baseline:** the *saved* threshold (`0.25` in your current raw run), binary mask unchanged for pixel metrics, predicted polygons extracted with a **16-pixel minimum area**. This is designed to exactly reproduce v3's `building-sizes` instance score (your raw run: F1 around `0.4835`).
2. **Instance threshold sweep:** change only the probability cutoff, with the same 16-pixel predicted-polygon minimum as the baseline.
3. **Connected-component area filtering:** remove complete 4-connected foreground components smaller than 0, 8, 16, 32, or 64 pixels before polygon extraction. Unlike the baseline area rule, this also affects pixel IoU because components are removed from the binary mask.
4. **Distance-transform watershed:** try splitting connected predicted regions into distinct building instances. It does not by itself change retained foreground pixels; 2 strong binary configurations are tried with 3 minimum-peak distances each by default. Very aggressive watershed may over-segment, so compare the measured scores.

Instance matching: one-to-one greedy polygon matching at IoU ≥ 0.50, using the same `match_objects()` routine and clipped WKT polygons as v3. This is a **building-only diagnostic**, not the complete official SpaceNet 8 leaderboard score.

### Selection criterion

By default, v4 selects the highest **validation instance F1** among configurations satisfying:

```
selected global pixel IoU >= unchanged v3 baseline global pixel IoU
```

The equality check includes a tiny numerical tolerance; pixel IoU is globally micro-averaged across complete tiles. `validation_postprocess_grid.csv` records all proposals, even those that fail the pixel-IoU constraint. For exploratory precision/recall tradeoffs, you can permit an *absolute* pixel IoU drop:

```bash
python exp.py postprocess-study \
  --run experiments_output/runs/raw_bce_dice_seed111 \
  --max-iou-drop 0.005
```

This is exploratory **validation-only** model selection. It does not guarantee improvement on an independent location or the existing test set. Your v3 observation of **32 cross-split adjacent tile pairs** is still important; later, use a new spatially buffered holdout to make final generalization claims.

## Available commands

### Search threshold and instance postprocessing

```bash
python exp.py postprocess-study \
  --run experiments_output/runs/raw_bce_dice_seed111 \
  --thresholds 0.15 0.20 0.25 0.30 0.35 0.40 0.50 0.60 \
  --areas 0 8 16 32 64 \
  --peak-distances 6 12 20 \
  --watershed-top 2 \
  --max-iou-drop 0
```

All values shown are defaults. The saved threshold is always included, even if you omit it from `--thresholds`. Set `--watershed-top 0` to skip watershed and perform the threshold/area study alone. Increasing grids increases postprocessing computation but **does not retrain or re-run your network** if cached probabilities exist.

Outputs under `experiments_output/runs/<run>/diagnostics_v4/`:

```
validation_postprocess_grid.csv        # every configuration and eligible flag
validation_selection.json              # unchanged baseline, selected settings, F1/IoU delta, hashes
validation_postprocess_tradeoff.png    # scatter of F1 vs pixel IoU, F1 vs threshold
validation_paired_tiles.csv            # baseline / selected performance on each validation tile
```

To open the plot:

```bash
xdg-open experiments_output/runs/raw_bce_dice_seed111/diagnostics_v4/validation_postprocess_tradeoff.png
```

### Diagnose missed / merged / fragmented objects

```bash
python exp.py instance-audit \
  --run experiments_output/runs/raw_bce_dice_seed111 \
  --selection baseline \
  --count 12

python exp.py instance-audit \
  --run experiments_output/runs/raw_bce_dice_seed111 \
  --selection best \
  --count 12
```

`best` requires that `postprocess-study` has completed; it verifies the saved checkpoint and split hashes. The `--selection baseline` audit does not need a v4 selection. Area categories default to: small `< 100 px²`, medium `[100, 1000) px²`, large `>= 1000 px²`, consistent with v3.

Each selected audit produces:

```
<selection>_instance_audit/
  validation_per_building.csv           # each GT polygon, matched flag, best IoU, coverage
  validation_size_recall.csv            # matched / total per area category
  validation_failure_summary.json       # heuristic categories
  validation_instance_failures.png      # cropped overlays of unmatched buildings
```

Overlay legend: **green** correct prediction, **red** false positive, **blue** false negative, **yellow** boundary of the highlighted reference building. The failure categories (`no_predicted_overlap`, `very_low_coverage`, `possible_merged_buildings`, `possible_fragmentation`, `below_iou_0_50`) are automated hypotheses. Inspect the original image and polygons to confirm a cause; they are not ground-truth diagnoses.

### Side-by-side against exported official baseline masks

If you have building predictions *actually generated* by the original SpaceNet 8 foundation network, store one raster per validation tile under a directory, with names such as `<tile_id>_buildingpred.tif` (binary, nonzero = building). Then run:

```bash
python exp.py compare-v4-external \
  --run experiments_output/runs/raw_bce_dice_seed111 \
  --pred-dir /path/to/official/validation/building_masks \
  --input-type binary \
  --selection best
```

You can also compare with `--selection baseline`. For probability-valued rasters use `--input-type probabilities --threshold 0.5` instead. Outputs:

```
diagnostics_v4/external_comparison/
  validation_comparison.json            # matched global pixel + object metrics
  validation_paired_tiles.csv
```

The adapter **does not download or run the official model** and cannot verify provenance of masks. You must train/run the actual official implementation, and ensure that its outputs correspond to the **same validation tile IDs and pixel grid**. A local ResNet-34 trained by this package is not the official SpaceNet baseline. See `README_EXPERIMENTS.md` and `README_V3.md` for original baseline preparation.

## Study interpretation

- A result such as `instance F1=0.51, pixel IoU=0.5737` is a *validation finding*, not a test score or a SpaceNet leaderboard score.
- A higher threshold can change polygon separation even when pixel IoU remains relatively flat.
- Tiny reference buildings (< 100 px²) can be especially sensitive to 0.50 polygon-matching IoU; inspect coverage and best match IoU in the per-building CSV before assuming the network failed to see them.
- Watershed partitions existing foreground; it cannot recover missing building pixels or solve incomplete labels.
- If no eligible configuration beats the baseline, the best selection should remain the baseline. **No improvement is guaranteed**.
- Do not run `finalize --all` to select settings. After reviewing multiple model seeds, use a new geographically independent, buffered holdout for final claims.

## Verification

The packaged suite has 23 passing synthetic tests, including v4 integration with a cached fake checkpoint, threshold sweep, filtering, watershed, per-building reports, and external mask comparisons. These tests do not require your actual Germany imagery or your NVIDIA GPU. Actual experiment results must be computed on your machine.
