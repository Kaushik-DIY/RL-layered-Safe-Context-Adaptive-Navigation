# Safe Context-Adaptive Navigation -- task runner.
# All Python runs through the project-local, distinctly-named venv (.venv-navrl)
# so nothing leaks into other projects.

VENV := .venv-navrl
PY   := $(VENV)/bin/python
PIP  := $(PY) -m pip
# The shell profile sources ROS 2, putting /opt/ros on PYTHONPATH; that makes pytest
# autoload ROS's launch_testing plugin (crashes on missing 'lark'). The core/ package
# is deliberately ROS-free, so run it with only the repo on the path.
RUN  := PYTHONPATH=$(CURDIR) $(PY)

.PHONY: help venv install dev-install test lint demo-mpc demo-cbf g2 train tensorboard reproduce clean

help:
	@echo "Targets:"
	@echo "  venv         create the project venv (.venv-navrl)"
	@echo "  install      install the full Python stack (torch CPU) into the venv"
	@echo "  dev-install  editable-install the 'core' package (pip install -e .)"
	@echo "  test         run pytest (clean PYTHONPATH, no ROS)"
	@echo "  demo-mpc     Week-1 MPC path-tracking demo (Gate G1 figure + solve stats)"
	@echo "  demo-cbf     Week-2 CBF yield demo (figure; add --animate for a GIF)"
	@echo "  g2           Week-2 CBF adversarial safety battery (Gate G2, 1000 eps)"
	@echo "  g2-sfm       G2 robustness extension: SFM pedestrians vs the CV assumption"
	@echo "  baselines    Week-3 S1/S2 fixed-tuning battery over the five scenarios"
	@echo "  eval         evaluate a trained policy as S4 + Gate G4 (MODEL=...)"
	@echo "  s5           S5 adversarial stress battery through the full stack"
	@echo "  onnx         export MODEL to ONNX + parity check (torch-free deploy)"
	@echo "  train        PPO training (STAGE=A|B|C SEED=n TRAIN_ARGS='--steps N --resume ...')"
	@echo "  train-smoke  tiny training run: verifies the loop + reports steps/sec"
	@echo "  tensorboard  launch TensorBoard on ./runs"
	@echo "  reproduce    (week6+) full pipeline: seeds -> results -> money plots"
	@echo "  clean        remove caches (keeps the venv)"

venv:
	python3 -m venv $(VENV)

install:
	bash scripts/install_python_deps.sh

dev-install:
	$(PIP) install -e .

test:
	$(RUN) -m pytest

demo-mpc:
	$(RUN) scripts/demo_mpc.py

demo-cbf:
	$(RUN) scripts/demo_cbf.py

g2:
	$(RUN) scripts/g2_battery.py

g2-sfm:
	$(RUN) scripts/g2_battery.py 1000 --sfm

baselines:
	$(RUN) scripts/run_baselines.py

MODEL ?= experiments/models/ppo_B_s0_final.zip
eval:
	$(RUN) scripts/eval_policy.py $(MODEL)

s5:
	$(RUN) scripts/run_s5.py

onnx:
	PYTHONPATH=$(CURDIR):$(CURDIR)/scripts $(PY) scripts/export_onnx.py $(MODEL)

STAGE ?= A
SEED  ?= 0
train:
	$(RUN) -m experiments.train --stage $(STAGE) --seed $(SEED) $(TRAIN_ARGS)

train-smoke:
	$(RUN) -m experiments.train --smoke --stage A

tensorboard:
	$(VENV)/bin/tensorboard --logdir runs

reproduce:
	@echo "TODO(week6): one path from raw seeds to the four money plots"
	bash scripts/reproduce.sh

clean:
	find . -type d -name __pycache__ -not -path './.venv-navrl/*' -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
