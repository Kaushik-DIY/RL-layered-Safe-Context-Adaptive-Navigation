#!/usr/bin/env bash
# Installs the Python core + RL stack into ./.venv-navrl (no sudo required).
# Project-local env with a distinct name so it never collides with other projects'
# .venv dirs. torch is pulled CPU-only (plan D6: CPU training is sufficient) to save disk.
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=.venv-navrl/bin
[ -d .venv-navrl ] || python3 -m venv .venv-navrl

echo "[1/3] Upgrading pip toolchain..."
$VENV/python -m pip install --upgrade pip wheel setuptools

echo "[2/3] Installing CPU-only PyTorch..."
$VENV/pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "[3/3] Installing scientific + optimization + RL stack..."
$VENV/pip install \
    "numpy<2.0" scipy matplotlib pandas \
    casadi proxsuite \
    pyyaml tqdm rich \
    gymnasium "stable-baselines3>=2.3" sb3-contrib tensorboard \
    onnx onnxruntime \
    pytest

echo "DONE. Freezing to requirements.lock.txt"
$VENV/pip freeze > requirements.lock.txt
echo "ALL_INSTALLS_COMPLETE"
