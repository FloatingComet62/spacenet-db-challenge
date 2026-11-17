#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
python exp.py doctor
if [[ ! -f experiments_output/splits.json ]]; then
  python exp.py prepare
fi
python exp.py inspect
python exp.py run --preset quick --skip-existing "$@"
for run in experiments_output/runs/*/best.pt; do
  [[ -f "$run" ]] || continue
  if [[ ! -f "$(dirname "$run")/threshold.json" ]]; then
    python exp.py tune --run "$(dirname "$run")"
  fi
done
printf '\nTraining complete. Inspect validation results, then run: python exp.py finalize --all\n'
