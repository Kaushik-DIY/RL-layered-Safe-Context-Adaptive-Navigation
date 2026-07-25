#!/usr/bin/env bash
# One path from raw seeds to the four money plots (plan sec. 8: `make reproduce`).
# Reproducibility is a portfolio feature.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv-navrl/bin/python

echo "TODO(week6): full reproduction pipeline"
# 1. run S1..S5 x scenarios x seeds in the 2D sim  -> experiments/results/*.csv
# 2. run the 20-episode Gazebo battery              -> rosbags + metrics
# 3. compute statistics (Mann-Whitney U, effect sizes)
# 4. $PY scripts/make_plots.py
