"""Week-4 curriculum + domain-randomization tests (plan D6)."""
from __future__ import annotations

import numpy as np
import pytest

from core.common.params import load_yaml
from core.rl.curriculum import make_sampler
from core.rl.nav_env import NavEnv
from core.sim2d.scenarios import free_roam


def test_stage_a_is_empty_world():
    rng = np.random.default_rng(0)
    sample = make_sampler("A")
    for _ in range(3):
        spec = sample(rng)
        assert spec.crowd.n == 0
        assert spec.crowd.state().shape == (0, 4)   # empty crowd is well-formed


def test_stage_b_small_crossings():
    rng = np.random.default_rng(1)
    names, counts = set(), set()
    for _ in range(20):
        spec = make_sampler("B")(rng)
        names.add(spec.name)
        counts.add(spec.crowd.n)
    assert names <= {"perpendicular_crossing", "free_roam"}
    assert counts <= {1, 2}


def test_stage_c_crowds_and_no_blind_corner():
    rng = np.random.default_rng(2)
    names = set()
    for _ in range(40):
        spec = make_sampler("C")(rng)
        names.add(spec.name)
        assert spec.name != "blind_corner"          # held out for evaluation
        if spec.name == "free_roam":
            assert 4 <= spec.crowd.n <= 8
    assert {"corridor_passby", "doorway_negotiation", "open_hall"} <= names


def test_unknown_stage_raises():
    with pytest.raises(ValueError):
        make_sampler("D")


def test_empty_crowd_steps():
    spec = free_roam(seed=0, n_pedestrians=0)
    out = spec.crowd.step(0.1, robot_xy=[0.0, 0.0])
    assert out.shape == (0, 4)


def test_domain_randomization_applied():
    dr = load_yaml("rl")["domain_randomization"]
    env = NavEnv(scenario_sampler=make_sampler("B"), domain_rand=dr)
    taus, brakes = set(), set()
    for seed in range(4):
        env.reset(seed=seed)
        taus.add(env.filt.cbf.tau)
        brakes.add(env.filt.cbf.a_brake)
        # tau applied to BOTH filter and sim latency (same physical quantity)
        assert env.sim.latency_steps == int(round(env.filt.cbf.tau / 0.1))
        assert abs(env.filt.cbf.tau - 0.4) <= dr["tau_jitter"] + 1e-9
        assert abs(env.filt.cbf.a_brake / 0.3 - 1.0) <= dr["a_brake_frac"] + 1e-9
        assert env._vel_noise == dr["human_vel_obs_noise_std"]
    assert len(taus) > 1 and len(brakes) > 1       # actually randomized


def test_no_dr_without_flag():
    env = NavEnv(scenario_sampler=make_sampler("A"))
    env.reset(seed=0)
    assert env.filt.cbf.tau == 0.4
    assert env._vel_noise == 0.0
