#!/usr/bin/env bash
set -euo pipefail
python exp.py v5-study "$@"
if [[ $# -eq 0 ]]; then
  python exp.py v5-gallery --model raw_bce_dice --seed 111
  python exp.py v5-gallery --model augmentation_photometric --seed 111
fi
