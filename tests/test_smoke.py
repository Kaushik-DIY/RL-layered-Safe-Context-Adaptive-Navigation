"""Scaffold smoke tests: environment + configs are wired correctly.

Real Week-1/2 unit tests (MPC tracking, the ABSOLUTE G2 CBF zero-violation battery)
land as those layers are implemented.
"""
from __future__ import annotations

import importlib

import pytest


def test_core_stack_imports():
    """The heavy dependencies resolve inside the project venv."""
    for mod in ["numpy", "scipy", "casadi", "proxsuite", "gymnasium",
                "stable_baselines3", "torch", "onnx"]:
        importlib.import_module(mod)


def test_configs_load_and_are_consistent():
    """YAML configs load and the safety-critical invariants hold."""
    from core.common.params import load_yaml, RobotParams

    robot = RobotParams.from_yaml("robot")
    cbf = load_yaml("cbf")
    rl = load_yaml("rl")

    # Latency used by the CBF must match the platform's injected latency (D2/D5).
    assert cbf["tau"] == robot.tau_latency

    # CBF braking must be conservative: <= physical capability (D5).
    assert cbf["a_brake"] <= robot.a_max_physical

    # THE invariant: the RL policy can never request a margin below the hard floor.
    assert rl["action"]["d_margin_cmd"]["low"] == cbf["d_hard"]

    # Protective field sits just outside the hard floor (D5).
    assert cbf["protective_field"]["radius"] > cbf["d_hard"]


@pytest.mark.parametrize("cfg", ["robot", "mpc", "cbf", "rl", "scenarios"])
def test_every_config_parses(cfg):
    from core.common.params import load_yaml
    assert load_yaml(cfg)  # non-empty dict
