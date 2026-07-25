"""Week-2 SFM pedestrian tests (plan D7).

The SFM is training-data plumbing, not a safety layer -- these tests pin the
behaviours the scenarios rely on: goal-seeking, mutual avoidance, robot avoidance,
wall containment, the speed clamp, and determinism (seeded scenarios depend on it).
"""
from __future__ import annotations

import numpy as np
import pytest

from core.sim2d.pedestrians import SfmParams, SocialForceCrowd

DT = 0.1


@pytest.fixture(scope="module")
def sfm() -> SfmParams:
    return SfmParams.from_yaml()


def _run(crowd, steps, robot_xy=None):
    traj = [crowd.state()]
    for _ in range(steps):
        traj.append(crowd.step(DT, robot_xy=robot_xy).copy())
    return np.array(traj)  # (T+1, n, 4)


def test_params_from_yaml(sfm):
    assert sfm.relaxation_time > 0
    lo, hi = sfm.desired_speed_range
    assert 0 < lo < hi


def test_reaches_goal_open_space(sfm):
    crowd = SocialForceCrowd(sfm, [[0.0, 0.0]], [[3.0, 0.0]], [1.2])
    _run(crowd, 50)  # 5 s at 1.2 m/s covers 3 m comfortably (incl. ramp-up)
    assert np.hypot(*(crowd.pos[0] - [3.0, 0.0])) < crowd.GOAL_RADIUS + 0.1
    # arrived without a goal_sampler -> stands still
    crowd.step(DT)
    assert np.hypot(*crowd.vel[0]) < 0.2


def test_speed_never_exceeds_clamp(sfm):
    crowd = SocialForceCrowd(sfm, [[0.0, 0.0], [0.4, 0.0]],
                             [[4.0, 0.0], [-4.0, 0.0]], [1.0, 1.5])
    traj = _run(crowd, 80)
    speeds = np.hypot(traj[:, :, 2], traj[:, :, 3])
    caps = crowd.SPEED_FACTOR * crowd.v_des
    assert np.all(speeds <= caps[None, :] + 1e-9)


def test_head_on_pair_avoids_each_other(sfm):
    """Two pedestrians walking at each other sidestep instead of passing through.

    Raw SFM with the literature parameters (A=2, B=0.3) lets a dead head-on pair
    closing at 2.4 m/s brush to ~0.24 m centre distance -- that is in-family for
    the model, and ped-ped spacing carries no safety claim (robot-side distances
    are the CBF's job). The invariant pinned here is no pass-through + progress."""
    crowd = SocialForceCrowd(sfm, [[0.0, 0.0], [4.0, 0.06]],
                             [[4.0, 0.0], [0.0, 0.06]], [1.2, 1.2])
    traj = _run(crowd, 90)
    pair_d = np.hypot(traj[:, 0, 0] - traj[:, 1, 0], traj[:, 0, 1] - traj[:, 1, 1])
    assert pair_d.min() > 0.2           # never pass through each other
    assert np.hypot(*(crowd.pos[0] - [4.0, 0.0])) < 0.7   # still make progress
    assert np.hypot(*(crowd.pos[1] - [0.0, 0.06])) < 0.7


def test_robot_is_repulsive_agent(sfm):
    """A pedestrian detours around a robot parked squarely on its route."""
    robot_xy = np.array([2.0, 0.0])
    crowd = SocialForceCrowd(sfm, [[0.0, 0.0]], [[4.0, 0.0]], [1.2])
    traj = _run(crowd, 120, robot_xy=robot_xy)
    d_robot = np.hypot(traj[:, 0, 0] - robot_xy[0], traj[:, 0, 1] - robot_xy[1])
    assert d_robot.min() > 0.3          # kept clear of the robot
    assert np.hypot(*(crowd.pos[0] - [4.0, 0.0])) < 0.7   # and still got there


def test_walls_contain_pedestrian(sfm):
    """Goal placed beyond a wall: the pedestrian presses toward it but never crosses."""
    walls = np.array([[-1.0, 1.0, 6.0, 1.0]])
    crowd = SocialForceCrowd(sfm, [[0.0, 0.0]], [[4.0, 2.5]], [1.2], walls=walls)
    traj = _run(crowd, 100)
    assert traj[:, 0, 1].max() < 1.0    # wall line never crossed


def test_deterministic_under_seed(sfm):
    """Same seed => identical evolution (the seeded-scenario contract)."""
    def build():
        rng = np.random.default_rng(7)
        return SocialForceCrowd(
            sfm, [[0.0, 0.0], [3.0, 1.0]], [[3.0, 0.0], [0.0, -1.0]], [1.0, 1.3],
            rng=rng, goal_sampler=lambda r, i: r.uniform(-2, 2, size=2))
    a, b = build(), build()
    for _ in range(120):
        sa, sb = a.step(DT), b.step(DT)
    np.testing.assert_array_equal(sa, sb)
