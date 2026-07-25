"""Week-2 CBF safety-filter unit tests + Gate G2 (plan sec. 5, D5).

G2 is ABSOLUTE: zero stopping-distance violations under an adversarial robot policy.
The scripted-adversary battery is brought forward here so the safety layer is
verified before any learning exists (plan: "verify before RL"). These tests own the
airtight-safety argument, so they check both behaviour and the frozen invariant.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from core.cbf.cbf_filter import CbfFilter, d_stop
from core.common.params import CbfParams, RobotParams
from core.sim2d.kinematic_sim import KinematicSim, wrap_angle
from scripts.g2_battery import run_battery


@pytest.fixture(scope="module")
def robot() -> RobotParams:
    return RobotParams.from_yaml()


@pytest.fixture(scope="module")
def cbf_cfg() -> CbfParams:
    return CbfParams.from_yaml()


@pytest.fixture
def filt(robot, cbf_cfg) -> CbfFilter:
    return CbfFilter(robot, cbf_cfg)


def _drive_at(filt, robot, human, x0=(0.0, 0.0, 0.0), steps=120):
    """Pursuit-drive full speed toward +x with one CV human; return trajectory info."""
    sim = KinematicSim(robot)
    filt.reset()
    s = sim.reset(list(x0))
    h = np.asarray(human, dtype=float).copy()
    min_d, min_h, min_v_cmd, interventions = np.inf, np.inf, np.inf, []
    for _ in range(steps):
        u_mpc = np.array([robot.v_max, 0.0])  # command straight ahead, full speed
        u_safe, info = filt.filter(s, u_mpc, [h])
        s = sim.step(u_safe)
        h[:2] += h[2:] * robot.dt
        min_d = min(min_d, float(np.hypot(h[0] - s[0], h[1] - s[1])))
        min_h = min(min_h, filt.min_barrier(s, [h]))
        min_v_cmd = min(min_v_cmd, u_safe[0])
        interventions.append(info["intervention"])
    return {"min_d": min_d, "min_h": min_h, "min_v_cmd": min_v_cmd,
            "final_v": float(s[3]), "interventions": np.array(interventions)}


# ---------------------------------------------------------------- primitives
def test_d_stop_formula(cbf_cfg):
    assert d_stop(0.0, cbf_cfg.tau, cbf_cfg.a_brake) == 0.0
    v = 0.26
    assert d_stop(v, cbf_cfg.tau, cbf_cfg.a_brake) == pytest.approx(
        v * cbf_cfg.tau + v * v / (2 * cbf_cfg.a_brake))
    assert d_stop(0.3, cbf_cfg.tau, cbf_cfg.a_brake) > d_stop(0.1, cbf_cfg.tau, cbf_cfg.a_brake)


def test_params_are_frozen_and_valid(cbf_cfg):
    with pytest.raises(dataclasses.FrozenInstanceError):
        cbf_cfg.d_hard = 0.0  # type: ignore[misc]
    assert cbf_cfg.protective_radius > cbf_cfg.d_hard  # the ESPE floor sits outside


# ------------------------------------------------------------------ behaviour
def test_passthrough_when_no_humans(filt, robot):
    """With no humans the filter only enforces the accel ramp, then matches u_mpc."""
    filt.reset()
    s = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    last = None
    for _ in range(40):
        last, info = filt.filter(s, np.array([0.2, 0.1]), humans=[])
        s = np.array([0.0, 0.0, 0.0, last[0], last[1]])
    assert last[0] == pytest.approx(0.2, abs=1e-3)  # ramped up to commanded v
    assert info["intervention"] == pytest.approx(0.0, abs=1e-3)


def test_stops_for_static_human_ahead(filt, robot, cbf_cfg):
    """The core guarantee: barrelling at a person dead ahead, the robot stops short."""
    info = _drive_at(filt, robot, human=[3.0, 0.0, 0.0, 0.0])
    assert info["min_h"] >= -1e-6                 # stopping-distance never violated
    assert info["min_d"] >= cbf_cfg.d_hard        # never breaches the hard floor
    assert info["min_v_cmd"] < robot.v_max        # it actually slowed (intervened)
    assert info["final_v"] == pytest.approx(0.0, abs=1e-2)  # came to rest


def test_no_slowdown_when_driving_away(filt, robot):
    """A human directly behind imposes no cap (closing component <= 0)."""
    info = _drive_at(filt, robot, human=[-2.0, 0.0, 0.0, 0.0])
    # only the initial accel ramp shows up; after it, no residual intervention
    assert info["interventions"][-1] == pytest.approx(0.0, abs=1e-3)
    assert info["min_v_cmd"] >= 0.0


def test_occluded_human_appearing_ahead(filt, robot, cbf_cfg):
    """Blind-corner worst case (plan 4.2 #5, Week-2 unit-test list: 'occluded
    appearance'): the robot cruises at FULL speed with the latency pipeline full of
    full-speed commands when a human materializes reveal_distance (1.2 m) dead
    ahead. Anticipation cannot help; the filter alone must stop the robot short."""
    sim = KinematicSim(robot)
    filt.reset()
    s = sim.reset([0.0, 0.0, 0.0])
    for _ in range(30):  # saturate: wheels AND the 0.4 s pipeline at v_max
        u_safe, _ = filt.filter(s, np.array([robot.v_max, 0.0]), [])
        s = sim.step(u_safe)
    assert s[3] == pytest.approx(robot.v_max)

    human = np.array([s[0] + 1.2, 0.0, 0.0, 0.0])  # appears NOW, 1.2 m ahead
    min_h, min_d = np.inf, np.inf
    for _ in range(80):
        u_safe, _ = filt.filter(s, np.array([robot.v_max, 0.0]), [human])
        s = sim.step(u_safe)
        min_h = min(min_h, filt.barrier(s, human))
        min_d = min(min_d, float(np.hypot(human[0] - s[0], human[1] - s[1])))
    assert min_h >= -1e-6                 # stopping-distance never violated
    assert min_d >= cbf_cfg.d_hard        # stopped short of the hard floor
    assert s[3] == pytest.approx(0.0, abs=1e-2)  # at rest before a static blocker


def test_occluded_crosser_appearing_ahead(filt, robot, cbf_cfg):
    """Same surprise, moving variant: a pedestrian steps out 1.2 m ahead already
    crossing at walking speed (the literal blind-corner emergence)."""
    sim = KinematicSim(robot)
    filt.reset()
    s = sim.reset([0.0, 0.0, 0.0])
    for _ in range(30):
        u_safe, _ = filt.filter(s, np.array([robot.v_max, 0.0]), [])
        s = sim.step(u_safe)

    h = np.array([s[0] + 1.05, 0.55, 0.0, -1.2])  # ~1.2 m away, crossing downward
    min_h, min_d = np.inf, np.inf
    for _ in range(80):
        u_safe, _ = filt.filter(s, np.array([robot.v_max, 0.0]), [h])
        s = sim.step(u_safe)
        h[:2] += h[2:] * robot.dt
        min_h = min(min_h, filt.barrier(s, h))
        min_d = min(min_d, float(np.hypot(h[0] - s[0], h[1] - s[1])))
    assert min_h >= -1e-6
    assert min_d >= cbf_cfg.d_hard


def test_no_restart_into_passthrough_pedestrian(filt, robot, cbf_cfg):
    """Regression (S4 eval, corridor collisions): a STOPPED robot must not be
    allowed to accelerate while a pedestrian walks nearly straight through its
    position. Endpoint-only latency prediction placed the human PAST the robot
    at t+tau (closing cosine flipped), dropped the cap, and waved the robot
    forward into the pedestrian mid-window."""
    filt.reset()  # _v_prev = 0: robot has yielded and stands still
    x = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    human = [0.55, 0.03, -1.3, 0.0]   # walking straight through, tiny offset
    u_safe, info = filt.filter(x, np.array([robot.v_max, 0.0]), [human])
    assert u_safe[0] <= 1e-4, u_safe  # must NOT start moving
    assert info["n_active"] == 1      # the human is capped, not ignored


def test_stays_stopped_through_passthrough_episode(filt, robot, cbf_cfg):
    """Closed-loop version: full-speed commands while the pedestrian passes;
    the robot must remain stationary whenever the human is inside d_hard."""
    sim = KinematicSim(robot)
    filt.reset()
    s = sim.reset([0.0, 0.0, 0.0])
    h = np.array([1.2, 0.05, -1.3, 0.0])
    for _ in range(30):
        u_safe, _ = filt.filter(s, np.array([robot.v_max, 0.0]), [h])
        s = sim.step(u_safe)
        h[:2] += h[2:] * robot.dt
        d = float(np.hypot(h[0] - s[0], h[1] - s[1]))
        if d < cbf_cfg.d_hard:        # human inside the floor: robot must be still
            assert s[3] <= 1e-4, (d, s[3])
        assert filt.min_barrier(s, [h]) >= -1e-6 or s[3] <= 1e-4


def test_protective_field_emergency_stop(filt, robot, cbf_cfg):
    """A human inside the protective radius forces an emergency brake override."""
    filt.reset()
    filt._v_prev = robot.v_max  # robot was moving
    x = np.array([0.0, 0.0, 0.0, robot.v_max, 0.0])
    human = [cbf_cfg.protective_radius - 0.05, 0.0, 0.0, 0.0]  # just inside the field
    u_safe, info = filt.filter(x, np.array([robot.v_max, 0.0]), [human])
    assert info["protective_stop"] is True
    assert u_safe[0] < robot.v_max                # braking, not full speed
    assert u_safe[0] == pytest.approx(max(0.0, robot.v_max - cbf_cfg.a_brake * robot.dt))


# -------------------------------------------------------------------- Gate G2
def test_g2_zero_violations(robot, cbf_cfg):
    """Gate G2 (CI subset): zero violations/collisions over a scripted-adversary battery.
    The full 1000-episode battery is scripts/g2_battery.py."""
    r = run_battery(120, robot, cbf_cfg)
    assert r["n_barrier_violations"] == 0, r
    assert r["n_collisions"] == 0, r
    assert r["global_min_h"] >= -1e-6, r
    assert r["passed"]


def test_g2_sfm_battery_subset(robot, cbf_cfg):
    """Robustness extension (CI subset): the filter's one-step CV assumption vs SFM
    pedestrians who accelerate, curve, and react to the robot. Evidence that the
    sigma inflation + conservative a_brake absorb the model mismatch (plan D5).
    Full battery: scripts/g2_battery.py 1000 --sfm."""
    r = run_battery(60, robot, cbf_cfg, mode="sfm")
    assert r["n_barrier_violations"] == 0, r
    assert r["n_collisions"] == 0, r
    assert r["passed"]
