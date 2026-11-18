# SpaceNet 8 Germany — v6 training and sampling ablations

This is a drop-in extension of v5. Unzip into the existing project root next to `exp.py`, `src/`, and `data/`. All earlier runs, checkpoints, ground-truth masks and test results remain untouched. Use the existing frozen `experiments_output/splits.json`; **do not re-run `prepare`**.

## Experiments

**Duration (recommended first):** Train raw BCE+Dice and photometric BCE+Dice U-Net/ResNet-34 models from scratch for 16 and 60 epochs, at seeds 111, 222 and 333. Compare 16 versus 60 epochs with the same initial seed, balanced sampler, batch size, Adam optimizer, learning rate, and full-tile validation schedule. These are **12 new training runs**, not warm starts from old checkpoints; 16 vs 60 are paired at the same random initialization and epoch RNG sequence.

**Sampling (second):** Compare the 60-epoch photometric `balanced` reference against a 60-epoch photometric `small_aware` sampler at the same three seeds. Both draw the *same number of training patches per epoch*, maintaining 50% positive and 50% empty probability when both groups exist. Balanced allocates 50% of draws to arbitrary positive patches; small-aware allocates 35% of total draws to patches overlapping a small ground-truth **individual building polygon** (area <=100 pixel-square units) and 15% to other positive patches. This is a controlled change in patch distribution, not a loss/architecture/augmentation change. Missing categories have redistributed quotas, which are recorded per run in `sampler.json`; missing small-positive training patches raise an error.

**Boundary (optional third):** Train an additional 60-epoch photometric model with a binary-edge-weighted BCE component in the loss. Its reference is the 60-epoch photometric model without this component. This is **not** explicit adjacent-building boundary labeling, only foreground/background edge emphasis.

All v6 runs use ImageNet-pretrained ResNet-34 U-Net, 512px patches, CUDA AMP when available, batch=2, lr=1e-4, and validate every two epochs (including the final epoch). They select the best checkpoint on full-resolution validation pixel IoU at threshold 0.5, then tune a post-training pixel threshold on validation. V6 does not touch the test set. Existing v1–v5 data and runs are preserved.

## Commands

```bash
unzip -o spacenet8_experiments_v6.zip
python -m pip install -r experiment_requirements.txt
python -m pytest -q
python exp.py doctor

# 1. Duration comparison: inspect first, then run
python exp.py v6-study --phase duration --dry-run
python exp.py v6-study --phase duration
python exp.py v6-report --phase duration

# 2. Sampling comparison after duration (reuses photo balanced e60)
python exp.py v6-study --phase sampling --dry-run
python exp.py v6-study --phase sampling
python exp.py v6-report --phase sampling

# 3. Optional boundary loss ablation
python exp.py v6-study --phase boundary --dry-run
python exp.py v6-study --phase boundary
python exp.py v6-report --phase full
```

You can first pilot one seed with `--seeds 111`; later run all three. `bash experiments_v6.sh` is equivalent to the entire duration phase followed by its report.

If you only want the pixel-validation summary without loading instance polygons and cached per-tile probabilities, use:

```bash
python exp.py v6-report --phase full --pixel-only
```

`v6-report` otherwise uses original v3 instance extraction: each predicted component must have polygon area >=16px², and a prediction is matched to one GT building at polygon IoU >=0.50. It calculates small-building recall using the same area <=100px² criterion. It deliberately **does not** apply per-run v4/v5 optimized postprocessing because that would confound training differences.

## Where to find results

```text
experiments_output/runs/
  v6_raw_bce_dice_balanced_e16_seed111/
  v6_raw_bce_dice_balanced_e60_seed111/
  v6_augmentation_photometric_balanced_e16_seed111/
  v6_augmentation_photometric_balanced_e60_seed111/
  v6_augmentation_photometric_small_aware_e60_seed111/
  v6_augmentation_photometric_balanced_e60_boundary_seed111/
  # each run: config.json, sampler.json, history.csv, latest.pt,
  # best.pt, completed.json, threshold.json
experiments_output/diagnostics_v6/
  validation_by_seed.csv
  validation_summary.csv
  validation_paired_deltas.csv
  validation_learning_curves.png
```

View the plot:

```bash
xdg-open experiments_output/diagnostics_v6/validation_learning_curves.png
```

All report values are **validation-only**. The report provides per-seed deltas and averages, best checkpoint epochs, instance F1, small-building recall, and checkpoint/split hashes. Incomplete seed groups are labeled with their actual `n` and must not be mistaken for complete three-seed evidence.

## Resuming and limitations

V6 stores optimizer, GradScaler, and model weights in `latest.pt` **after every completed epoch**. Rerunning an identical command resumes incomplete v6 runs at the next epoch; a changed hyperparameter, dataset manifest or split raises an error. An interrupted epoch is repeated from the last saved complete epoch. For deterministic resumption, sampling and data-worker RNGs are derived separately from the fixed run seed and epoch, with `persistent_workers=False`. Full 60-epoch training may be substantial on a laptop GPU; maintain enough disk space for both `best.pt` and `latest.pt` for every run.

Do **not** use `python exp.py finalize --all` to choose a model, because the held-out test set has already been used for older work. The frozen validation split includes 32 edge-neighbor pairs across splits, so eventual scientific claims need a newly defined spatially buffered holdout. V6 neither downloads official SpaceNet weights nor establishes leaderboard comparability; use official external predictions with the existing comparison commands after model selection.
