#!/usr/bin/env bash
set -euo pipefail
python exp.py v6-study --phase duration "$@"
python exp.py v6-report --phase duration
