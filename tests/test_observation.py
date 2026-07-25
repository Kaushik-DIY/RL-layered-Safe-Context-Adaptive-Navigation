"""Week-3 observation-builder tests (plan D6/D8).

This function is single-sourced across both simulators, so its contract IS the
transfer story: layout, padding, sorting, robot-frame rotation, normalization.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.common.observation import (SCALE, TTCA_MAX, build_observation, obs_dim,
                                     time_to_closest_approach)


def test_dim():
    assert obs_dim(5) == 32
    assert obs_dim(3) == 22


def test_shape_dtype_and_padding():
    obs = build_observation([0, 0, 0, 0.1, 0.0], [3, 0], np.zeros((0, 4)),
                            0.2, 0.5, k_nearest=5)
    assert obs.shape == (32,) and obs.dtype == np.float32
    assert np.all(obs[5:30] == 0.0)          # all human slots zero-padded


def test_goal_terms():
    obs = build_observation([1, 1, np.pi / 2, 0.26, 0.5], [1, 4], [], 0.26, 0.5)
    assert obs[0] == pytest.approx(3.0 / SCALE["dist"])       # goal 3 m away
    assert obs[1] == pytest.approx(0.0, abs=1e-6)             # dead ahead
    assert obs[3] == pytest.approx(1.0)                       # v at platform max
    assert obs[30] == pytest.approx(1.0)                      # v_max_in_effect = max
    assert obs[31] == pytest.approx(0.5 / SCALE["margin"])


def test_humans_sorted_and_truncated():
    """6 humans, K=5: the farthest is dropped; slot 0 is the nearest."""
    humans = [[d, 0, 0, 0] for d in (6.0, 1.0, 3.0, 2.0, 5.0, 4.0)]
    obs = build_observation([0, 0, 0, 0, 0], [10, 0], humans, 0.2, 0.5)
    rel_x = obs[5:30].reshape(5, 5)[:, 0] * SCALE["dist"]
    assert np.allclose(rel_x, [1, 2, 3, 4, 5])                # sorted, 6 m dropped


def test_robot_frame_rotation():
    """A human 2 m NORTH of a robot HEADING north is dead ahead in the robot frame."""
    obs = build_observation([0, 0, np.pi / 2, 0, 0], [0, 5], [[0, 2, 0, 0]], 0.2, 0.5)
    slot = obs[5:10]
    assert slot[0] * SCALE["dist"] == pytest.approx(2.0)      # ahead
    assert slot[1] * SCALE["dist"] == pytest.approx(0.0, abs=1e-6)  # not sideways


def test_ttca():
    # head-on closing at 1 m/s from 2 m -> closest approach in 2 s
    assert time_to_closest_approach(np.array([2.0, 0]), np.array([-1.0, 0])) == pytest.approx(2.0)
    # receding -> t* negative -> clipped to 0
    assert time_to_closest_approach(np.array([2.0, 0]), np.array([1.0, 0])) == 0.0
    # no relative motion -> sentinel
    assert time_to_closest_approach(np.array([2.0, 0]), np.zeros(2)) == TTCA_MAX


def test_ttca_in_slot_accounts_for_robot_motion():
    """Static human, moving robot: the relative velocity comes from the robot, so
    ttca = 2.6 m / 0.26 m/s = 10 s... clipped exactly at TTCA_MAX -- use 1.3 m."""
    obs = build_observation([0, 0, 0, 0.26, 0], [5, 0], [[1.3, 0, 0, 0]], 0.26, 0.5)
    assert obs[5:10][4] * SCALE["ttca"] == pytest.approx(1.3 / 0.26, abs=0.05)


def test_values_are_o1():
    """Normalization: a busy but realistic scene keeps every entry in [-3, 3]."""
    humans = [[2, 1, -1.0, 0.3], [1, -1, 0.5, 1.2], [4, 0, -1.5, 0]]
    obs = build_observation([0.5, -0.2, 0.7, 0.2, -1.0], [6, 1], humans, 0.26, 1.2)
    assert np.all(np.abs(obs) <= 3.0)


# ------------------------- observation v2 (industrial track) -------------------------
def test_v2_dim_and_v1_freeze():
    assert obs_dim(5, version=2) == 35
    assert obs_dim(5, version=1) == 32 == obs_dim(5)   # v1 untouched


def test_v1_bytes_identical_regardless_of_new_kwargs():
    """The frozen TB3 contract: v1 output must ignore walls/posts entirely."""
    state, goal = [0.4, -0.2, 0.3, 0.2, 0.1], [5, 1]
    humans = [[2.0, 0.5, -0.3, 0.0]]
    a = build_observation(state, goal, humans, 0.2, 0.5)
    b = build_observation(state, goal, humans, 0.2, 0.5,
                          walls=np.array([[0, -1, 6, -1]]),
                          posts=np.array([[3, 1, 0.12]]))
    np.testing.assert_array_equal(a, b)


def test_v2_geometry_features():
    """Robot at origin heading +x in a 2 m corridor with a post ahead: the three
    appended features must carry exactly that geometry. post_ahead is the LONGITUDINAL
    (along-heading) distance to the ahead post -- a clean, monotonic corner-distance
    signal -- NOT the Euclidean distance to the off-axis post."""
    walls = np.array([[-1, -1, 8, -1], [-1, 1, 8, 1], [6, -1, 6, 1]])  # end wall x=6
    posts = np.array([[3.0, 1.0, 0.12], [-2.0, 0.0, 0.12]])   # one ahead, one behind
    obs = build_observation([0, 0, 0, 0.1, 0], [5, 0], [], 0.2, 0.5,
                            version=2, walls=walls, posts=posts)
    assert obs.shape == (35,)
    wall_clear, forward_free, post_ahead = (float(v) * SCALE["free"]
                                            for v in obs[32:35])
    assert wall_clear == pytest.approx(1.0)                   # side walls at +-1
    assert forward_free == pytest.approx(6.0)                 # end wall dead ahead
    assert post_ahead == pytest.approx(3.0)                  # longitudinal to ahead post
    # the post BEHIND (x=-2) is ignored; a far-lateral ahead post is gated out too
    far = np.array([[3.0, 5.0, 0.12]])                        # ahead but 5 m to the side
    o2 = build_observation([0, 0, 0, 0.1, 0], [5, 0], [], 0.2, 0.5,
                           version=2, walls=None, posts=far)
    assert float(o2[34]) * SCALE["free"] == pytest.approx(10.0)  # gated -> max_range


def test_post_ahead_is_monotonic_approaching_corner():
    """The corner signal must decrease monotonically as the robot nears the post
    (the property the OLD Euclidean-to-offset-post version violated -- it was floored
    at the lateral offset and jumped up at the corner, so no policy could learn from
    it, 2026-07-24)."""
    from core.common.observation import corner_sight_distance
    posts = np.array([[6.0, 1.75, 0.12], [7.5, 1.75, 0.12]])   # industrial corner
    vals = [corner_sight_distance([x, 0.0, 0.0], posts) for x in (0, 1, 2, 3, 4, 5)]
    assert all(b < a for a, b in zip(vals, vals[1:]))         # strictly decreasing
    assert vals[-1] == pytest.approx(1.0)                     # 1 m short of the post


def test_v2_open_space_saturates():
    """No walls/posts: all three features cap at the FREE sentinel (scaled 1.0)."""
    obs = build_observation([0, 0, 0, 0, 0], [5, 0], [], 0.2, 0.5,
                            version=2, walls=None, posts=None)
    assert np.allclose(obs[32:35], 1.0)


def test_v2_platform_scale():
    """Industrial platform: 1.5 m/s at cruise must land at 1.0, not 5.77."""
    from core.common.platform import load_platform
    p = load_platform("industrial")
    obs = build_observation([0, 0, 0, 1.5, 0], [8, 0], [], 1.5, 2.0,
                            version=2, walls=None, posts=None, scale=p.obs_scale)
    assert obs[3] == pytest.approx(1.0)                       # v / v_max
    assert obs[30] == pytest.approx(1.0)                      # v_max_in_effect
    assert obs[31] == pytest.approx(1.0)                      # margin at box top
