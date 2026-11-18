#!/usr/bin/env bash
set -euo pipefail

RUN="${1:-experiments_output/runs/raw_bce_dice_seed111}"
python exp.py postprocess-study --run "$RUN"
python exp.py instance-audit --run "$RUN" --selection baseline --count 12
python exp.py instance-audit --run "$RUN" --selection best --count 12

echo "Results: $RUN/diagnostics_v4/"
echo "Run postprocess-study on photometric or other seeds separately for a cross-seed sanity check."
