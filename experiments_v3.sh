#!/usr/bin/env bash
set -euo pipefail
python exp.py split-neighbor-audit
python exp.py geometry-audit --samples 40 --count 4
for run in raw_bce_dice_seed111 augmentation_photometric_seed111; do
  directory="experiments_output/runs/${run}"
  python exp.py threshold-curves --run "$directory"
  python exp.py building-sizes --run "$directory"
done
python exp.py annotation-audit --run experiments_output/runs/raw_bce_dice_seed111 --count 6
printf '\nDiagnostics complete. Read README_V3.md for output paths and optional geometric study.\n'
