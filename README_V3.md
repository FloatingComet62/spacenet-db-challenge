# SpaceNet 8 Germany — Experiment Suite v3

**Extract this ZIP into the SAME project directory as your existing `exp.py`, `data/`, `src/`, and `experiments_output/`.** This is a drop-in replacement for the suite source from v2. It does **not** include or replace the dataset, existing model runs, legacy `src/`, original TIFFs, or checkpoints.

## What changed

This release adds diagnostics for the completed three-seed study:

- **Full threshold curves** for each existing checkpoint (global pixel IoU, Dice, precision, recall, 0.05–0.95 plus the saved threshold). Generated **only on validation tiles**; existing tuned thresholds are NOT overwritten.
- **Building footprint size analysis** (small / medium / large **by ground-truth polygon area in pixels squared**): one-to-one IoU ≥ 0.5 object matching, size-dependent recall, and overall instance F1 / precision / recall. Uses validation tiles only. The default area thresholds, 100 and 1000 px², are *exploratory*, not SpaceNet-defined categories.
- **Annotation audit**: re-rasterize original `Wkt_Pix` building footprints and compare with the prepared validation masks. With `--run`, rank false-positive pixels near versus far from annotated footprints, and save manual-review overlays. **WKT–mask agreement cannot prove that the annotations are complete**.
- **Geometric augmentation audit**: synthetic marker test proves image/mask transform synchronization across random seeds and saves visual examples from *training* patches. Adds isolated `augmentation_flip` and `augmentation_rotation` policies without changing the old geometric or combined policies.
- **Geographic adjacency audit**: find edge-adjacent image tiles assigned to different splits. Existing spatial block splitting may still allow adjacent tiles across split boundaries; this tool documents the limitation and **does not modify the frozen split**.
- **External official-baseline adapter**: compare existing external SpaceNet foundation building predictions to an internal run on exactly the same validation tiles. No silent resize. Also retains v2's `compare-external`, official CSV export, and instance scoring.

## First commands — no retraining

From the project root:

```bash
unzip -o spacenet8_experiments_v3.zip
python -m pytest -q
python exp.py split-neighbor-audit
python exp.py geometry-audit --samples 40 --count 4
```

Then examine **raw BCE+Dice**, your established internal baseline:

```bash
python exp.py threshold-curves \
  --run experiments_output/runs/raw_bce_dice_seed111

python exp.py building-sizes \
  --run experiments_output/runs/raw_bce_dice_seed111

python exp.py annotation-audit \
  --run experiments_output/runs/raw_bce_dice_seed111 \
  --count 6 --boundary-px 3
```

For photometric augmentation (your highest three-seed mean validation IoU by +0.0020):

```bash
python exp.py threshold-curves \
  --run experiments_output/runs/augmentation_photometric_seed111
python exp.py building-sizes \
  --run experiments_output/runs/augmentation_photometric_seed111
```

All probability inference is performed on **validation images only**. The first diagnostic command for a run saves lossless float32 probability arrays to `experiments_output/runs/<run>/diagnostics_v3/validation_probs/`. Subsequent commands reuse them. Caches are verified against the **checkpoint SHA256 and split hash**. If the checkpoint changes and old cache files remain, diagnostics refuse to run. To deliberately replace a cache, pass `--refresh-cache` to the relevant diagnostic command.

### Where to view the generated pictures

```bash
xdg-open experiments_output/runs/raw_bce_dice_seed111/diagnostics_v3/validation_threshold_curves.png
xdg-open experiments_output/runs/raw_bce_dice_seed111/diagnostics_v3/building_size_recall.png
xdg-open experiments_output/runs/raw_bce_dice_seed111/diagnostics_v3/annotation_review_candidates.png
xdg-open experiments_output/diagnostics_v3/geometry_alignment_audit.png
```

Outputs also include CSVs, a JSON summary of instance matching, per-object match results, and a cross-split neighbor CSV. The annotation review picture shows green annotated reference roofs, red false positives, blue false negatives; **visually inspect original imagery** before interpreting apparent missing roof labels.

## Isolate flips versus rotations (new training, optional)

The original geometric/combined augmentation policy is retained **unchanged**, so old results remain comparable. Two new configurations isolate its component transforms:

- `augmentation_flip`: horizontal and vertical flips, each independently with probability 0.5; no 90° rotation, no photometric jitter.
- `augmentation_rotation`: 0°, 90°, 180°, or 270° rotation, uniformly sampled; no flips, no photometric jitter.

The modes are intentionally **not mutually exclusive probabilities matched** to the original combined geometric policy: they are component-isolation diagnostics. If geometry helps alone but fails combined, that is evidence of an interaction to investigate, not a proven software bug.

The existing multi-seed study uses epochs=16, batch=2, LR=1e-4, workers=2, stride=512. It refuses to reuse incompatible run settings or overwrite incomplete runs:

```bash
python exp.py study --phase geometry --seeds 111 222 333 --dry-run
python exp.py study --phase geometry --seeds 111 222 333
python exp.py validation-report \
  --names raw_bce_dice augmentation_geometry augmentation_flip augmentation_rotation \
  --seeds 111 222 333
```

This trains six additional configurations (two modes × three seeds) and tunes thresholds using validation tiles only. It does not run `finalize` or access the test split. Check `validation_paired_deltas.csv` and the by-seed CSV for paired comparison to raw+BCE+Dice.

Alternatively, to test only one seed initially:

```bash
python exp.py run --names augmentation_flip augmentation_rotation \
  --seed 111 --epochs 16 --batch-size 2 --lr 0.0001 --workers 2 --stride 512
python exp.py tune --run experiments_output/runs/augmentation_flip_seed111
python exp.py tune --run experiments_output/runs/augmentation_rotation_seed111
```

Do not mix different training budgets, or infer reproducibility from a single seed.

## Matched comparison with **the actual official SpaceNet 8 baseline**

**The ZIP does not include official foundation weights, official predictions, or an official-network implementation.** The bundled `matched_bce` checkpoint is **not** the official baseline. This update supplies a common evaluation protocol; you still have to train/run the original model externally.

Official source: https://github.com/SpaceNetChallenge/SpaceNet8

1. Follow that repository's `baseline/` preparation instructions using your original Germany dataset, including the official road labels and masks. Export its prepared CSV using the existing suite command:

   ```bash
   python exp.py export-official-csv \
     --input-csv /path/to/official_prepared.csv
   ```

   The CSV tool requires the official `preimg` field and checks coverage of **all** frozen train/val/test tile IDs. If the baseline's preparation uses different image IDs or missing data, fix that in the official pipeline; **do not** silently drop unmatched samples.

2. Train the official foundation model against `experiments_output/official_csv/train.csv` and `val.csv`, and use the official foundation inference on the **validation** CSV. The original README for this suite (`README_EXPERIMENTS.md`) contains example commands. Use the matching official code version and record the checkpoint, hyperparameters and runtime.

3. Put its exported single-band `*_buildingpred.tif` validation predictions in a directory, one file per validation tile. Then run:

   ```bash
   python exp.py compare-official-validation \
     --run experiments_output/runs/raw_bce_dice_seed111 \
     --pred-dir /absolute/path/to/official/validation/predictions \
     --input-type binary
   ```

   The resulting `diagnostics_v3/official_comparison/` directory contains a JSON report with **global binary pixel IoU, Dice, precision, recall**, per-tile paired differences, and a bar chart. The external model ID is *not authenticated*; cite its checkpoint and implementation in your experimental report.

4. For the same external predictions, calculate **individual building-instance F1@0.5**, which this matched pixel report intentionally does not duplicate:

   ```bash
   python exp.py compare-external \
     --pred-dir /absolute/path/to/official/validation/predictions \
     --split val --name official_foundation_validation \
     --input-type binary
   ```

   To make a fair comparison of object F1, evaluate your own validation probability outputs using this same `compare-external` path and the saved validation threshold, or compare the exported predictions using the same polygon-matching settings and image IDs. Instance F1 here is an *auxiliary* measure, **not the complete official SpaceNet 8 challenge metric**. The official model is multitask (buildings and roads), while our model is building-only, so interpret differences accordingly.

**Do not** reuse leaderboard/public-test IoU for selecting architecture or thresholds. Your earlier four models were already evaluated on the original test split, so later results on that same test split are exploratory. For an unbiased final generalization claim, obtain a new held-out geographically separated area, ideally with a buffer between train and evaluation tiles. Do not overwrite `experiments_output/splits.json` mid-study.

## Tested and known limitations

- Synthetic CPU smoke tests exercise all new diagnostics, the new transform policies, checkpoint inference via a tiny stand-in model, cache read/write, TIFF 0/255 comparison, and absence of v3 test evaluation. No actual Germany imagery or CUDA training was run in this environment.
- `annotation-audit` checks WKT-to-NPZ consistency and proximity of prediction errors to labels. It **cannot** independently verify missing building annotations without human inspection / another independent annotation source.
- Size-based recall uses areas in **image pixel squared**, not real-world m². GT buildings partially cut by tile edges are evaluated on their clipped shapes; this differs from whole-building evaluation.
- Geometric transformations in the old code are already synchronized correctly at the array level, based on the synthetic tests. Performance decline could come from training stochasticity, object semantics, altered sampling, or loss/schedule effects. New runs are diagnostics, not claimed improvements.
- `split-neighbor-audit` only checks east/south adjacency within identical scene prefixes, not distance in meters between distinct satellite scenes. Zero edges does **not** guarantee geographic independence.
- The actual SpaceNet foundation model is **not** run by any command in this ZIP. A genuine 1-to-1 comparison only becomes possible after supplying its predictions and matching the data, training and scoring procedures.
