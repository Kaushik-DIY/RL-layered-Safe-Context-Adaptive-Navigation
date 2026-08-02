"""Shared observation construction (plan D6, D8).

THE key transfer enabler: this exact function builds the RL observation in BOTH
the 2D training sim and the Gazebo evaluation (and, later, from any real tracker).
No images, no lidar rasters -- only geometric quantities available identically
everywhere. Keeping this single-sourced is what makes the 2D->Gazebo claim honest.

Observation v1 layout (dim = 3 + 2 + K*5 + 2 = 32 for K = 5) -- FROZEN with the
TB3 track (every trained model/battery on disk consumed exactly this):

    [ goal_dist, goal_bearing, path_curvature,           # 3  goal/path
      v, omega,                                          # 2  robot velocity
      (rel_x, rel_y, rel_vx, rel_vy, ttca) x K,          # 25 nearest humans
      v_max_in_effect, d_margin_in_effect ]              # 2  current params

Observation v2 (industrial track, 2026-07 replan) appends 3 MAP-DERIVED features
-- available on any deployed AMR (they all localize against a site map), so the
transfer story stays honest:

    [ ..., wall_clear, forward_free, post_ahead ]        # +3 occlusion/geometry

  * wall_clear   : distance to the nearest wall segment (lateral safety context)
  * forward_free : ray-cast along the heading to the nearest wall -- how much
                   free run the robot actually has
  * post_ahead   : distance to the nearest mapped constriction post (doorway jamb /
                   blind-corner edge) AHEAD of the robot; these posts are exactly
                   where occluded emergences happen, so this is the "corner coming
                   up" signal that lets a policy learn reveal < d_stop anticipation.

Conventions (the normalization spec lives HERE, single-sourced):
  * Human relative positions/velocities are expressed in the ROBOT frame (rotated
    by -theta): what "ahead" and "left" mean is then platform-pose-free, which is
    what transfers between simulators and to a real tracker.
  * Humans are sorted by distance, truncated/zero-padded to K slots. A zero slot
    (all five numbers 0) means "no human" -- distinguishable from a real human at
    the robot's position because ttca of a real entry is never exactly 0 together
    with zero relative velocity.
  * ttca = time to closest approach under constant velocity: t* = -(p.v)/(v.v),
    clipped to [0, TTCA_MAX]; TTCA_MAX when relative velocity ~ 0 (no approach).
  * Every entry is scaled to O(1). v1 uses the module SCALE verbatim (frozen);
    v2 callers pass the platform's scale dict (Platform.obs_scale) so 1.5 m/s and
    10+ m arenas still land O(1). Path curvature is 0.0 for straight-line
    references (kept so an A* path can feed it later without a layout change).
"""
from __future__ import annotations

import numpy as np

TTCA_MAX = 10.0  # s; also the "no approach" sentinel before scaling
FREE_MAX = 10.0  # m; forward_free / wall_clear / post_ahead cap ("plenty of room")

# v1 normalization scales (divide by these). FROZEN -- the TB3 models consumed them.
SCALE = {
    "dist": 5.0,       # m      goal distance, human relative positions
    "angle": np.pi,    # rad    bearings
    "curv": 1.0,       # 1/m    path curvature
    "v_robot": 0.26,   # m/s    robot v (platform v_max)
    "w_robot": 1.82,   # rad/s  robot omega
    "v_human": 2.0,    # m/s    human relative velocities
    "ttca": TTCA_MAX,  # s
    "margin": 1.2,     # m      d_margin_in_effect (action upper bound)
    "free": FREE_MAX,  # m      v2 geometry features
}


def obs_dim(k_nearest: int, version: int = 1) -> int:
    base = 3 + 2 + 5 * k_nearest + 2
    return base + (3 if version >= 2 else 0)


def time_to_closest_approach(rel_p: np.ndarray, rel_v: np.ndarray) -> float:
    """t* minimizing |rel_p + t*rel_v| under constant velocity, clipped >= 0.
    rel_* are human-minus-robot quantities in any common frame."""
    vv = float(np.dot(rel_v, rel_v))
    if vv < 1e-9:
        return TTCA_MAX
    t = -float(np.dot(rel_p, rel_v)) / vv
    return float(np.clip(t, 0.0, TTCA_MAX))


def forward_free_distance(x: float, y: float, theta: float,
                          walls: np.ndarray, max_range: float = FREE_MAX) -> float:
    """Ray-cast from (x, y) along heading theta against wall segments; distance to
    the first hit, capped at max_range (v2 'free run ahead' feature)."""
    if walls is None or len(walls) == 0:
        return max_range
    dx, dy = float(np.cos(theta)), float(np.sin(theta))
    best = max_range
    for w in np.asarray(walls, dtype=float).reshape(-1, 4):
        ax, ay, bx, by = w
        sx, sy = bx - ax, by - ay
        denom = dx * sy - dy * sx            # ray x segment cross product
        if abs(denom) < 1e-12:
            continue                          # parallel
        t = ((ax - x) * sy - (ay - y) * sx) / denom          # along the ray
        u = ((ax - x) * dy - (ay - y) * dx) / denom          # along the segment
        if t > 1e-9 and -1e-9 <= u <= 1.0 + 1e-9:
            best = min(best, t)
    return float(best)


CORNER_LATERAL_GATE = 3.0   # m; a post beyond this lateral offset is off-path


def corner_sight_distance(robot_state, posts, max_range: float = FREE_MAX,
                          lateral_gate: float = CORNER_LATERAL_GATE) -> float:
    """LONGITUDINAL (along-heading) distance to the nearest mapped constriction post
    that is ahead and roughly in the travel corridor -- a clean 'how far until the
    blind corner / doorway ahead' signal.

    This decreases MONOTONICALLY to ~0 as the robot reaches the constriction, unlike
    a Euclidean-distance-to-an-offset-post measure, which is floored by the post's
    lateral offset (never gets small) and JUMPS UP at the corner as the nearest post
    switches -- a broken signal the policy cannot learn corner anticipation from
    (diagnosed 2026-07-24). Both the observation and the w9 anticipatory-speed reward
    read this same value, so they are consistent by construction.
    """
    if posts is None or len(posts) == 0:
        return max_range
    x, y, theta = float(robot_state[0]), float(robot_state[1]), float(robot_state[2])
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    best = max_range
    for p in np.asarray(posts, dtype=float).reshape(-1, 3):
        rx = cos_t * (p[0] - x) + sin_t * (p[1] - y)     # along-heading (longitudinal)
        ry = -sin_t * (p[0] - x) + cos_t * (p[1] - y)    # lateral
        if rx > 0.0 and abs(ry) < lateral_gate:
            best = min(best, rx)
    return float(best)


def geometry_features(robot_state, walls, posts,
                      max_range: float = FREE_MAX) -> tuple[float, float, float]:
    """(wall_clear, forward_free, post_ahead) for observation v2, all in metres,
    capped at max_range. posts: (m, 3) mapped constriction circles [x, y, r].
    post_ahead is the clean longitudinal corner-sight distance (see above)."""
    x, y, theta = float(robot_state[0]), float(robot_state[1]), float(robot_state[2])
    wall_clear = max_range
    if walls is not None and len(walls):
        for w in np.asarray(walls, dtype=float).reshape(-1, 4):
            ab = w[2:] - w[:2]
            t = np.clip(np.dot([x, y] - w[:2], ab) / max(float(np.dot(ab, ab)), 1e-12),
                        0.0, 1.0)
            cp = w[:2] + t * ab
            wall_clear = min(wall_clear, float(np.hypot(x - cp[0], y - cp[1])))
    forward_free = forward_free_distance(x, y, theta, walls, max_range)
    post_ahead = corner_sight_distance(robot_state, posts, max_range)
    return wall_clear, forward_free, post_ahead


def build_observation(robot_state, goal_xy, humans, v_max_in_effect,
                      d_margin_in_effect, k_nearest: int = 5,
                      path_curvature: float = 0.0,
                      robot_vel_xy=None, version: int = 1,
                      walls=None, posts=None, scale=None) -> np.ndarray:
    """Build the normalized observation vector (dim = obs_dim(k_nearest, version)).

    robot_state : [x, y, theta, v, omega]
    goal_xy     : [x, y] of the (current) goal / carrot target
    humans      : (n, 4) [x, y, vx, vy] world frame, as seen by the tracker
                  (occlusions already applied by the caller)
    robot_vel_xy: robot world-frame velocity for relative-velocity computation;
                  derived from (v, theta) if None (unicycle: no lateral slip).
    version     : 1 = frozen TB3 layout; 2 = + (wall_clear, forward_free,
                  post_ahead) map features (walls/posts must then be given).
    scale       : normalization dict; None = the frozen v1 SCALE. v2 callers pass
                  their Platform.obs_scale.
    """
    S = SCALE if scale is None else scale
    x, y, theta, v, omega = (float(robot_state[i]) for i in range(5))
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    gx, gy = float(goal_xy[0]) - x, float(goal_xy[1]) - y
    goal_dist = float(np.hypot(gx, gy))
    goal_bearing = float(np.arctan2(gy, gx) - theta)
    goal_bearing = (goal_bearing + np.pi) % (2 * np.pi) - np.pi

    if robot_vel_xy is None:
        robot_vel_xy = np.array([v * cos_t, v * sin_t])

    humans = np.asarray(humans, dtype=float).reshape(-1, 4)
    slots = np.zeros((k_nearest, 5))
    if len(humans) > 0:
        d = np.hypot(humans[:, 0] - x, humans[:, 1] - y)
        order = np.argsort(d)[:k_nearest]
        for slot, idx in enumerate(order):
            h = humans[idx]
            rel_p = np.array([h[0] - x, h[1] - y])
            rel_v = np.array([h[2], h[3]]) - robot_vel_xy
            ttca = time_to_closest_approach(rel_p, rel_v)
            # rotate into the robot frame
            rx = cos_t * rel_p[0] + sin_t * rel_p[1]
            ry = -sin_t * rel_p[0] + cos_t * rel_p[1]
            rvx = cos_t * rel_v[0] + sin_t * rel_v[1]
            rvy = -sin_t * rel_v[0] + cos_t * rel_v[1]
            slots[slot] = [rx / S["dist"], ry / S["dist"],
                           rvx / S["v_human"], rvy / S["v_human"],
                           ttca / S["ttca"]]

    parts = [
        [goal_dist / S["dist"], goal_bearing / S["angle"],
         path_curvature / S["curv"],
         v / S["v_robot"], omega / S["w_robot"]],
        slots.ravel(),
        [v_max_in_effect / S["v_robot"], d_margin_in_effect / S["margin"]],
    ]
    if version >= 2:
        wall_clear, forward_free, post_ahead = geometry_features(
            robot_state, walls, posts)
        parts.append([wall_clear / S["free"], forward_free / S["free"],
                      post_ahead / S["free"]])
    obs = np.concatenate(parts).astype(np.float32)
    assert obs.shape == (obs_dim(k_nearest, version),)
    return obs
