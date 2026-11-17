> **Multi-seed follow-up:** See [README_MULTISEED.md](README_MULTISEED.md) for the seed 111/222/333 reproducibility study, augmentation diagnosis, and validation-only false-positive overlays. Existing checkpoints are reused when compatible.

# SpaceNet 8 Germany: building segmentation experiments

This is an **additive** experiment suite for your current repository. It uses the existing
`data/PRE-event/*.tif` and `data/Germany_Training_Public_reference.csv` and writes ONLY to
`experiments_output/` (plus files inside its own suite). It does **not** overwrite
`src/`, `checkpoints/`, `data/building_dataset/`, your original TIFFs, or the CSV.

## Quick start (Debian / GPU)

Extract the ZIP **into your repository root** — the directory containing `src/` and `data/`.
The ZIP has files at its root: `exp.py`, `experiments/`, `experiments_bootstrap.sh`, etc.

Use your existing CUDA-enabled PyTorch environment. Do **not** reinstall PyTorch if your
current build already sees the RTX 4050. Add missing Python dependencies:

```bash
# Activate the project's existing virtual environment first, if appropriate.
python -m pip install -r experiment_requirements.txt
python exp.py doctor
python -m pytest -q tests/test_experiments.py
bash experiments_bootstrap.sh
```

`experiments_bootstrap.sh` does the following in order:
1. Checks dependency availability and dataset paths.
2. Rebuilds corrected building masks from **all** pre-event TIFFs (including empty scenes).
3. Creates and freezes 70/15/15-ish spatial-block train/val/test splits, based on groups
   of adjacent tile row/column coordinates (not a guaranteed geographic buffer).
4. Runs four preliminary 16-epoch ResNet34 U-Nets (batch 2, mixed precision):
   - `matched_bce`: raw RGB, BCE loss
   - `raw_bce_dice`: raw RGB, BCE + soft Dice
   - `clahe`: CLAHE + BCE/Dice
   - `augmentation`: raw RGB + geometric/photometric augmentations + BCE/Dice
5. Tunes the best probability threshold for each run on **validation only**.

The bootstrap script skips only fully completed runs (`completed.json`). If a run was interrupted, use `--overwrite` to rerun it.

This preliminary run is designed to smoke-test the pipeline; 16 epochs is not an
established convergence point. For publication-quality comparisons, use the full
run below with the same training budget across experiments. Note that the `matched_bce`
model is a **matched single-task baseline**, not the actual SpaceNet 8 official model.

## Explicit commands

```bash
python exp.py prepare                            # first time only
python exp.py inspect                            # mask counts + split summary
python exp.py run --preset quick                  # four preliminary experiments
python exp.py run --preset all --epochs 50 --seed 112  # full controlled study
python exp.py run --names gamma sharpen --epochs 50 --seed 112
python exp.py tune --run experiments_output/runs/raw_bce_dice_seed111
python exp.py finalize --all                      # test set; do this after model selection
python exp.py results                            # experiments_output/results.csv
```

**Do not run** `--preset all` immediately following `--preset quick` with the same
seed; overlapping names will refuse to overwrite existing results. Choose a fresh
output seed for a new full run (e.g. `--seed 112`), or deliberately use
`--overwrite` to replace each existing checkpoint. All variants in a single
study should use the **same seed, epoch count, training budget, data split and
batch size**. Extra seeds estimate variance only after you have a primary run.

If you have limited VRAM, use `--batch-size 1`; expected settings for 6GB are
`--batch-size 2`, `--workers 2`, `--stride 512`. You can increase inference
coverage by `--stride 256`, but this is slower. Data preparation requires TIFF
channels 1–3 to be `uint8`; other bit depths fail loudly instead of silently
normalizing incorrectly.

`prepare --force` rebuilds only generated NPZ files and the split, and refuses
to proceed if experiment runs are already present. Keep the generated split
unchanged across the study. This is essential for meaningful comparisons.

## How to perform a true 1-to-1 comparison with the *official* SpaceNet 8 baseline

The official baseline is not the SMP U-Net in this bundle. It predicts **buildings
AND road speed** using a custom ResNet34 U-Net, different output heads, image
preprocessing, labels and postprocessing. This bundle cannot claim official
leaderboard comparability by itself. It **can** compare its own building output
against actual official foundation-model outputs on exactly the same Germany tiles
using exactly the same full-tile metric implementation.

Official source: https://github.com/SpaceNetChallenge/SpaceNet8/tree/main/baseline

1. Clone the **official** repository separately, prepare your Germany files with
   their `baseline/data_prep/geojson_prep.py` and
   `baseline/data_prep/create_masks.py`. This requires the original road/flood
   annotations in your `data/annotations/` directory; the CSV building masks
   generated by this bundle are **not** enough to train the official multitask model.
   Follow its README; the official tooling expects an AOI-style folder
   `Germany_Training_Public/{PRE-event,POST-event,annotations}`.
2. After its data prep, generate its usual train/val CSVs. Filter these to our
   fixed spatial split **instead** of using its random split. Example:

   ```bash
   python exp.py export-official-csv \
     --input-csv /path/to/sn8_data_train.csv /path/to/sn8_data_val.csv
   ```

   This produces `experiments_output/official_csv/train.csv`, `val.csv`, `test.csv`.
   The script refuses to produce files if any expected pre-event tile is missing.
3. In the separate official repository, train using its original training command,
   substituting the frozen CSV split:

   ```bash
   python baseline/train_foundation_features.py \
     --train_csv /absolute/path/experiments_output/official_csv/train.csv \
     --val_csv /absolute/path/experiments_output/official_csv/val.csv \
     --save_dir /path/to/official_foundation \
     --model_name resnet34 --lr 0.0001 --batch_size 2 --n_epochs 50 --gpu 0
   ```

4. Run its original `foundation_eval.py` using our **test.csv**:

   ```bash
   python baseline/foundation_eval.py \
     --model_path /path/to/official_foundation/best_model.pth \
     --in_csv /absolute/path/experiments_output/official_csv/test.csv \
     --save_preds_dir /absolute/path/official_predictions \
     --model_name resnet34 --gpu 0
   ```

   The destination directory may need to exist in advance.
   Official building predictions are named `<PRE-tile-id>_buildingpred.tif`,
   single-band with pixel values 0/255.

5. Evaluate the official model on our **pixel ground truth and test tile IDs**:

   ```bash
   python exp.py compare-external \
     --pred-dir /absolute/path/official_predictions \
     --name official_spacenet8 --input-type binary
   ```

6. Evaluate your model's **saved test probabilities** with its validation-selected
   threshold (read `threshold.json`, e.g. `0.45`), for *identical* pixel + instance
   metrics:

   ```bash
   python exp.py compare-external \
     --pred-dir experiments_output/runs/raw_bce_dice_seed111/test_predictions \
     --name mine_raw_bce_dice --input-type probabilities --threshold 0.45
   ```

7. Visualize example test scenes side-by-side, with green true positives,
   red false positives and blue false negatives:

   ```bash
   python exp.py visualize-compare \
     --a-dir experiments_output/runs/raw_bce_dice_seed111/test_predictions \
     --a-kind probabilities --a-threshold 0.45 \
     --b-dir /absolute/path/official_predictions \
     --b-kind binary --count 8
   ```

All comparisons refuse to silently resize predictions. If the official inference
uses a different TIFF geometry, coordinate convention, or normalization, investigate
and fix that in the official pipeline rather than forcing arrays to match.

### What exactly gets measured?

- **Pixel IoU/Dice/Precision/Recall:** global TP/FP/FN, counting each full-tile pixel
  exactly once, including at overlaps. A tile with no buildings is still scored.
- **Object F1@0.5:** predicted building blobs raster-to-polygon, discard areas below
  `--min-area` (default 16 px²), greedy one-to-one match to original WKT pixel
  polygons at IoU >= 0.5. This is an auxiliary approximate footprint score,
  **not** the official SpaceNet8 flood/building score. It does not split adjacent
  buildings and must not be equated with the original challenge leaderboard.
- **Performance:** inference time and peak allocated GPU memory for your own
  experiments (the official external comparison has only predictions, not runtime).

For fair scientific claims, train competing models on identical geographical splits,
record differences in preprocessing, input data and auxiliary road supervision, and
keep the held-out test set untouched until model/hyperparameter selection is done.

## Output organization

```text
experiments_output/
  manifest.json         # counts, reference CSV checksum and generated tile checksums
  tiles.csv             # polygon counts and positive pixel coverage per tile
  splits.json           # frozen split IDs and spatial block groups
  dataset/*.npz         # newly generated clean RGB+building masks
  runs/<experiment>_seed<seed>/
    config.json
    best.pt             # selected by validation FULL-tile IoU at threshold 0.5
    completed.json      # prevents skipping interrupted training runs
    history.csv
    threshold.json      # best validation threshold
    test_metrics.json   # only after finalize
    test_predictions/   # per-tile raw probability arrays, .npy
  results.csv
  external/<name>/test_metrics.json
  external/<name>/test_per_tile.csv
  official_csv/{train,val,test}.csv
  visual_comparisons/*.png
```

## Notes / assumptions

- Uses `Germany_Training_Public_reference.csv` `ImageId`, `Object` and `Wkt_Pix`
  columns, with building polygons in **pixel** coordinates. `Object=Building`
  (case insensitive) is used. All building polygons per tile are appended before
  rasterization, fixing your prior indentation bug. Invalid geometry is repaired
  using `make_valid`; non-polygons and malformed WKT are counted separately.
- All pre-event TIFFs, including TIFFs without building annotations, are included
  in the training corpus. A high empty-mask percentage is **flagged**, not silently
  accepted as evidence the source imagery has no buildings. Inspect
  `tiles.csv` and compare against original `data/annotations` where appropriate.
- Assumes the first three TIFF bands are RGB, and TIFF image ID matches the CSV.
  If this assumption is wrong, stop and correct it rather than training anyway.
- All variants use balanced positive/empty patch sampling, ImageNet normalization,
  ResNet34 U-Net, Adam with LR `1e-4`, 512-pixel patches, and the same split.
  `matched_bce` changes only loss vs the otherwise raw BCE+Dice experiment.
  `lr_schedule` changes only whether StepLR halves LR every 20 epochs.
- Processing is applied on the fly, so you can run CLAHE and raw experiments
  without data mutation or large backup folders. Random flips, 90-degree
  rotations and mild brightness/contrast jitter apply to *training* only.
- Time may differ across runs because GPU timing includes full-tile inference
  and CPU work. For strict speed comparison, benchmark with a fixed device and
  no other GPU load.
- This ZIP includes **no source imagery, checkpoints, or trained models**;
  it cannot produce actual Germany result numbers until run on your machine.

## Test without downloading the SpaceNet dataset

```bash
python -m pytest -q tests/test_experiments.py
```

The synthetic tests check mask coverage with multiple polygons, empty tiles,
spatial block splitting, overlapping-patch reconstruction, preprocessing
immutability, official 0/255 TIFF reading and CSV split export.
