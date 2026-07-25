"""Week-2 scenario-generator tests (plan 4.2).

The evaluation battery's validity rests on these being (a) deterministic per seed
(2D and Gazebo must run the identical episode) and (b) actually producing the
interaction each scenario claims to test. Full behavioural stats come from the
week-3 env; here we pin construction, determinism, and the occlusion contract.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.common.params import load_yaml
from core.sim2d.scenarios import SCENARIO_NAMES, free_roam, make_scenario


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_builds_and_is_wellformed(name):
    spec = make_scenario(name, seed=3)
    assert spec.name == name
    assert spec.robot_start.shape == (3,) and spec.goal.shape == (2,)
    assert spec.walls.ndim == 2 and spec.walls.shape[1] == 4
    assert spec.static_obstacles.shape[1] == 3 if len(spec.static_obstacles) else True
    assert len(spec.static_obstacles) <= 6      # fits the MPC's obstacle capacity
    assert spec.crowd.n >= 1
    assert spec.timeout_s == 60.0               # plan D6 episode timeout


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_deterministic_per_seed(name):
    a, b = make_scenario(name, seed=11), make_scenario(name, seed=11)
    np.testing.assert_array_equal(a.crowd.pos, b.crowd.pos)
    np.testing.assert_array_equal(a.crowd.goals, b.crowd.goals)
    np.testing.assert_array_equal(a.crowd.v_des, b.crowd.v_des)
    # and different seeds actually differ
    c = make_scenario(name, seed=12)
    assert not np.array_equal(a.crowd.pos, c.crowd.pos)


def test_corridor_passby_is_oncoming():
    spec = make_scenario("corridor_passby", seed=0)
    half = load_yaml("scenarios")["scenarios"]["corridor_passby"]["corridor_width"] / 2
    assert abs(spec.crowd.pos[0, 1]) < half            # spawned inside the corridor
    assert spec.crowd.goals[0, 0] < spec.robot_start[0]  # walking toward the robot


def test_perpendicular_crossing_crosses_the_path():
    spec = make_scenario("perpendicular_crossing", seed=1)
    y0, gy = spec.crowd.pos[0, 1], spec.crowd.goals[0, 1]
    assert y0 * gy < 0                                  # start and goal straddle y=0


def test_doorway_has_gap_and_posts():
    spec = make_scenario("doorway_negotiation", seed=2)
    gap = load_yaml("scenarios")["scenarios"]["doorway_negotiation"]["gap_width"]
    assert len(spec.static_obstacles) == 2              # the two doorway posts
    post_ys = np.sort(spec.static_obstacles[:, 1])
    assert post_ys[1] - post_ys[0] == pytest.approx(gap)
    assert spec.crowd.pos[0, 0] > 3.0                   # pedestrian on the far side
    assert spec.crowd.goals[0, 0] < 3.0                 # heading through the doorway


def test_open_hall_crowd_size_and_spawns():
    cfg = load_yaml("scenarios")["scenarios"]["open_hall"]["n_pedestrians"]
    for seed in range(6):
        spec = make_scenario("open_hall", seed=seed)
        assert cfg[0] <= spec.crowd.n <= cfg[1]
        d0 = np.hypot(spec.crowd.pos[:, 0] - spec.robot_start[0],
                      spec.crowd.pos[:, 1] - spec.robot_start[1])
        assert d0.min() >= 1.2                          # nobody spawns on the robot
        assert spec.crowd.goal_sampler is not None      # free-roaming (goals resample)


def test_blind_corner_occlusion_contract():
    """Hidden while in the side passage; revealed at reveal_distance or on entering
    the corridor; once seen, never forgotten (tracker latch)."""
    spec = make_scenario("blind_corner", seed=4)
    rd = load_yaml("scenarios")["scenarios"]["blind_corner"]["reveal_distance"]
    assert spec.reveal_distance == rd
    ped = spec.crowd.pos[0]
    # robot at its start: pedestrian is up the passage, farther than rd -> invisible
    assert np.hypot(*(ped - spec.robot_start[:2])) > rd
    assert spec.visible_humans(spec.robot_start[:2]).shape == (0, 4)
    # robot within reveal_distance -> visible
    near = ped - np.array([0.0, rd - 0.1])
    assert spec.visible_humans(near).shape == (1, 4)
    # latch: far away again, still tracked
    assert spec.visible_humans(spec.robot_start[:2]).shape == (1, 4)


def test_blind_corner_pedestrian_emerges():
    """Run the crowd with its scripted event: the pedestrian must actually enter the
    main corridor (the surprise the scenario exists to create)."""
    spec = make_scenario("blind_corner", seed=5)
    dt, t, entered = 0.1, 0.0, False
    for _ in range(600):  # 60 s timeout
        spec.tick(t)
        state = spec.crowd.step(dt)
        t += dt
        if state[0, 1] <= spec.occlusion_y:
            entered = True
            break
    assert entered, "occluded pedestrian never entered the corridor"


def test_free_roam_builder():
    spec = free_roam(seed=9, n_pedestrians=5)
    assert spec.crowd.n == 5
    assert spec.crowd.goal_sampler is not None


# ------------------------- industrial track (2026-07 replan) -------------------------
def test_industrial_geometry_scales():
    """Industrial arenas are MiR-scale versions of the SAME scenario logic."""
    for name in SCENARIO_NAMES:
        tb3 = make_scenario(name, seed=21, platform="tb3")
        ind = make_scenario(name, seed=21, platform="industrial")
        assert ind.goal[0] > tb3.goal[0], name          # longer runs at 1.5 m/s
    corr = make_scenario("corridor_passby", seed=21, platform="industrial")
    width = abs(corr.walls[1, 1] - corr.walls[0, 1])
    assert width == pytest.approx(3.5)                  # warehouse aisle, not 2 m
    bc = make_scenario("blind_corner", seed=21, platform="industrial")
    assert bc.reveal_distance == pytest.approx(1.2)     # reveal < d_stop: the point


def test_industrial_platform_does_not_change_tb3_episodes():
    """The geom refactor must leave historical TB3 episodes bit-identical."""
    for name in SCENARIO_NAMES:
        a = make_scenario(name, seed=1000, platform="tb3")
        b = make_scenario(name, seed=1000)              # historical call form
        np.testing.assert_array_equal(a.crowd.pos, b.crowd.pos)
        np.testing.assert_array_equal(a.walls, b.walls)


def test_interferer_seeker_closes_on_robot():
    """The bystander must actively approach the robot and hover at the standoff --
    that is the behavior that makes 'slow down' the WRONG response."""
    spec = make_scenario("interferer", seed=7, platform="industrial")
    assert spec.crowd.seekers == {0}
    robot = np.array([1.0, 0.0])
    d0 = float(np.hypot(*(spec.crowd.pos[0] - robot)))
    for _ in range(120):                                # 12 s
        spec.crowd.step(0.1, robot_xy=robot)
    d1 = float(np.hypot(*(spec.crowd.pos[0] - robot)))
    assert d1 < min(d0, 1.4)                            # closed to ~the standoff
    assert d1 > 0.4                                     # ...but hovers, no contact


def test_t_junction_holdout_combines_occlusion_and_seeker():
    spec = make_scenario("t_junction_interferer", seed=8, platform="industrial")
    assert spec.reveal_distance is not None             # occluded emergence armed
    assert spec.crowd.seekers == {1}                    # plus a robot-seeker
    assert len(spec.events) == 1                        # scripted emergence event
    # the occluded pedestrian starts invisible from the robot's start
    assert spec.visible_humans(spec.robot_start[:2]).shape[0] == 1  # seeker only
