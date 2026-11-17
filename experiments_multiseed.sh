#!/usr/bin/env bash
# Run from the project root. Reuses completed seed111 runs without overwriting.
set -euo pipefail
python exp.py doctor
python exp.py inspect-augmentation --count 4
python exp.py study --phase initial --seeds 111 222 333 --epochs 16 --batch-size 2 --lr 0.0001 --stride 512
python exp.py validation-report --seeds 111 222 333
printf '\nValidation reports saved under experiments_output/. No test metrics were computed.\n'
