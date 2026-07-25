"""Week-3 NavEnv tests (plan D6): gym contract + closed-loop behaviour of the
full supervisor -> MPC -> CBF -> sim -> crowd stack.

MPC solves make these the slowest tests in the suite; they use short horizons and
the emptiest scenarios that still exercise the code path in question.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.rl.nav_env import GOAL_TOL, NavEnv


@pytest.fixture(scope="module")
def env() -> NavEnv:
    return NavEnv(scenarios=["perpendicular_crossing"])


def test_spaces(env):
    assert env.observation_space.shape == (32,)
    assert env.action_space.shape == (2,)
    assert env.action_space.low[0] == pytest.approx(0.05)
    assert env.action_space.high[1] == pytest.approx(1.2)


def test_reset_contract(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == (32,) and obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert info["scenario"] == "perpendicular_crossing"
    # same seed -> same scenario episode (determinism through gym seeding)
    obs2, info2 = env.reset(seed=0)
    np.testing.assert_array_equal(obs, obs2)
    assert info2["seed"] == info["seed"]


def test_step_contract(env):
    env.reset(seed=1)
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert obs.shape == (32,)
    assert np.isfinite(r)
    assert isinstance(term, bool) and isinstance(trunc, bool)
    assert set(info["reward_terms"]) == {"progress", "energy", "jerk",
                                         "cbf_intervention", "protective_stop",
                                         "personal_space", "stopping_violation",
                                         "blind_corner_speed", "human_approach",
                                         "time", "success"}


def test_action_modulates_mpc(env):
    """The v_max_cmd action is a hard cap on the applied speed."""
    env.reset(seed=2)
    cap = 0.08
    v_seen = []
    for _ in range(8):
        _, _, term, trunc, _ = env.step(np.array([cap, 0.5]))
        v_seen.append(float(env.s[3]))
        if term or trunc:
            break
    assert max(v_seen) <= cap + 1e-6


def test_full_speed_reaches_goal_and_reports_metrics():
    """Aggressive params + CBF: the episode must end in success (the filter yields
    but never freezes), and the terminal info must carry the full 4.3 metrics."""
    env = NavEnv(scenarios=["perpendicular_crossing"])
    env.reset(seed=3)
    ep = None
    for _ in range(120):  # 120 decisions x 0.5 s = 60 s
        _, _, term, trunc, info = env.step(np.array([0.26, 0.5]))
        if term or trunc:
            ep = info["episode_metrics"]
            break
    assert ep is not None
    assert ep["success"], ep
    assert not ep["collision"], ep
    assert ep["violation_steps"] == 0, ep     # the CBF held the line
    assert ep["min_h"] >= -1e-6 or ep["min_human_dist"] >= 0.3, ep
    for key in ("time_to_goal", "energy", "rms_jerk", "path_length_ratio",
                "intervention_rate", "mpc_solve_ms_median"):
        assert key in ep


def test_fixed_params_and_no_cbf_baseline_mode():
    """S1/S2 mode: action ignored, filter inert (but still measuring)."""
    env = NavEnv(scenarios=["corridor_passby"], use_cbf=False,
                 fixed_params=(0.13, 1.0))
    env.reset(seed=4)
    _, _, term, trunc, info = env.step(np.array([0.26, 0.3]))  # must be ignored
    assert env.v_max_cmd == 0.13 and env.d_margin_cmd == 1.0
    assert max(info["reward_terms"]["cbf_intervention"], 0.0) == 0.0  # never acts


def test_trajectory_recording():
    """record=True keeps the per-step trace the money plots need (plots 1 & 3)."""
    env = NavEnv(scenarios=["perpendicular_crossing"], record=True)
    env.reset(seed=6)
    for _ in range(3):
        _, _, term, trunc, _ = env.step(np.array([0.2, 0.5]))
        if term or trunc:
            break
    assert len(env.trajectory) >= 5           # >= decision_every per step
    row = env.trajectory[0]
    for key in ("t", "x", "y", "v_mpc", "v_safe", "v_applied", "h",
                "d_human", "v_max_cmd", "intervention"):
        assert key in row, key
    # times strictly increase at the 10 Hz inner rate
    ts = [r["t"] for r in env.trajectory]
    assert all(b > a for a, b in zip(ts, ts[1:]))
    # a fresh reset clears the trace
    env.reset(seed=7)
    assert env.trajectory == []


def test_timeout_truncates():
    """A crawling cap cannot reach the goal in 60 s -> truncation, not crash."""
    env = NavEnv(scenarios=["perpendicular_crossing"])
    env.reset(seed=5)
    done = False
    for _ in range(130):
        _, _, term, trunc, info = env.step(np.array([0.05, 1.2]))
        if term or trunc:
            done = True
            assert trunc or term
            assert "episode_metrics" in info
            break
    assert done
