"""Social-force model pedestrians (Helbing-Molnar), implemented in-house (plan D7).

Full control, no dependency risk. Force terms per pedestrian i:

    drive_i  = (v_des_i * e_goal_i - v_i) / T            relaxation to desired velocity
    ped_ij   = A   * exp((r_ps  - d_ij) / B) * n_ij      pairwise exponential repulsion
    robot_i  = A_r * exp((r_ps  - d_iR) / B) * n_iR      robot treated as repulsive agent
    wall_iw  = A   * exp((r_w   - d_iw) / B) * n_iw      nearest point on each segment
    noise_i  ~ N(0, NOISE_SIGMA^2) per axis              Helbing's fluctuation term

The fluctuation term is not decoration: with perfectly collinear geometry (obstacle
dead ahead on the line to the goal) the repulsion is exactly anti-parallel to the
drive force and the pedestrian stalls in equilibrium instead of stepping around --
the noise breaks that symmetry, as in the original model. It is drawn from the
crowd's own seeded rng, so seeded scenarios stay bit-for-bit deterministic.

Euler-integrated at the caller's dt; speed clamped at 1.3x desired (Helbing's v_max).
Repulsion is isotropic (no anisotropy factor lambda) -- adequate for the small crowds
in the five scenarios and keeps the model at the plan's ~80-line scope; note it as a
stated simplification if pedestrian realism is ever challenged.

The robot couples in as a POSITION-ONLY repulsive agent: pedestrians yield around it
as they would a person, but nothing here reads controller internals.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.common.params import load_yaml


@dataclass(frozen=True)
class SfmParams:
    """Frozen SFM constants (scenarios.yaml `social_force_model`, plan D7)."""

    relaxation_time: float          # s     T in the drive term
    desired_speed_range: tuple      # m/s   (lo, hi) for randomized desired speeds
    repulsion_A: float              # m/s^2 pedestrian/wall interaction strength
    repulsion_B: float              # m     interaction range
    robot_repulsion_A: float        # m/s^2 pedestrians treat the robot as repulsive
    personal_space: float           # m     comfort radius the repulsion defends

    @classmethod
    def from_yaml(cls, name: str = "scenarios") -> "SfmParams":
        d = load_yaml(name)["social_force_model"]
        lo, hi = d["desired_speed"]
        return cls(
            relaxation_time=d["relaxation_time"],
            desired_speed_range=(float(lo), float(hi)),
            repulsion_A=d["repulsion_A"],
            repulsion_B=d["repulsion_B"],
            robot_repulsion_A=d["robot_repulsion_A"],
            personal_space=d["personal_space"],
        )


def closest_point_on_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Closest point to p on segment a-b (used for wall repulsion distances)."""
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(float(np.dot(ab, ab)), 1e-12), 0.0, 1.0))
    return a + t * ab


class SocialForceCrowd:
    """Helbing-Molnar social-force crowd. State per pedestrian: [x, y, vx, vy].

    walls        : (n, 4) segments [x1, y1, x2, y2] (repel pedestrians; robot walls
                   are the scenario/env's concern, not this class's)
    goal_sampler : optional (rng, i) -> [x, y]; when given, a pedestrian reaching its
                   goal draws a new one (free-roam, scenario 4). Without it, arrivals
                   stand still (scripted crossings/pass-bys end their role by design).
    seekers      : optional set of pedestrian indices in INTERFERER mode (2026-07
                   replan): a curious bystander whose goal is continuously re-aimed
                   at the robot -- approaches, hovers at SEEK_STANDOFF, follows when
                   the robot moves off (the documented "people crowd the robot" AMR
                   problem). The robot-repulsion force still applies inside
                   personal_space, so a seeker rings the robot rather than touching
                   it -- but it DOES actively close, unlike every other pedestrian.
    """

    GOAL_RADIUS = 0.3      # m      "arrived" threshold
    SPEED_FACTOR = 1.3     # -      Helbing's v_max = 1.3 * desired speed
    WALL_CLEARANCE = 0.25  # m      body-radius-scale standoff the wall force defends
    NOISE_SIGMA = 0.15     # m/s^2  fluctuation force (breaks collinear deadlocks)
    SEEK_STANDOFF = 0.9    # m      a curious bystander hovers about here

    def __init__(self, sfm: SfmParams, positions, goals, desired_speeds,
                 walls=None, rng=None, goal_sampler=None, seekers=None):
        self.sfm = sfm
        # reshape(-1, 2) (not atleast_2d) so an EMPTY crowd is well-formed (0, 2)
        # -- curriculum stage A trains in an empty world (plan D6).
        self.pos = np.asarray(positions, dtype=float).reshape(-1, 2).copy()
        self.vel = np.zeros_like(self.pos)
        self.goals = np.asarray(goals, dtype=float).reshape(-1, 2).copy()
        self.v_des = np.asarray(desired_speeds, dtype=float).reshape(-1).copy()
        self.walls = (np.zeros((0, 4)) if walls is None
                      else np.asarray(walls, dtype=float).reshape(-1, 4))
        self.rng = np.random.default_rng(0) if rng is None else rng
        self.goal_sampler = goal_sampler
        self.seekers = set() if seekers is None else set(seekers)
        assert len(self.pos) == len(self.goals) == len(self.v_des)

    @property
    def n(self) -> int:
        return len(self.pos)

    def state(self) -> np.ndarray:
        """(n, 4) array [x, y, vx, vy] -- the format every consumer uses."""
        return np.hstack([self.pos, self.vel])

    def step(self, dt: float, robot_xy=None) -> np.ndarray:
        """Advance all pedestrians one dt. robot_xy: robot position or None."""
        sfm = self.sfm
        new_pos, new_vel = self.pos.copy(), self.vel.copy()
        for i in range(self.n):
            # interferer mode: re-aim at the robot each step; hover at standoff
            if i in self.seekers and robot_xy is not None:
                r = np.asarray(robot_xy, float)[:2]
                if float(np.hypot(*(r - self.pos[i]))) > self.SEEK_STANDOFF:
                    self.goals[i] = r
                else:
                    self.goals[i] = self.pos[i].copy()   # arrived: hover
            # drive term (arrivals stand, or redraw a goal in free-roam)
            to_goal = self.goals[i] - self.pos[i]
            d_goal = float(np.hypot(*to_goal))
            if d_goal < self.GOAL_RADIUS and self.goal_sampler is not None:
                self.goals[i] = np.asarray(self.goal_sampler(self.rng, i), dtype=float)
                to_goal = self.goals[i] - self.pos[i]
                d_goal = float(np.hypot(*to_goal))
            v_desired = (self.v_des[i] * to_goal / d_goal
                         if d_goal >= self.GOAL_RADIUS else np.zeros(2))
            f = (v_desired - self.vel[i]) / sfm.relaxation_time

            for j in range(self.n):                      # pedestrian-pedestrian
                if j != i:
                    f += self._repulsion(self.pos[i], self.pos[j],
                                         sfm.repulsion_A, sfm.personal_space)
            if robot_xy is not None:                     # robot as repulsive agent
                f += self._repulsion(self.pos[i], np.asarray(robot_xy, float)[:2],
                                     sfm.robot_repulsion_A, sfm.personal_space)
            for w in self.walls:                         # walls
                cp = closest_point_on_segment(self.pos[i], w[:2], w[2:])
                f += self._repulsion(self.pos[i], cp,
                                     sfm.repulsion_A, self.WALL_CLEARANCE)
            f += self.NOISE_SIGMA * self.rng.standard_normal(2)  # fluctuation

            v = self.vel[i] + f * dt
            speed = float(np.hypot(*v))
            v_cap = self.SPEED_FACTOR * self.v_des[i]
            if speed > v_cap:
                v = v * (v_cap / speed)
            new_vel[i] = v
            new_pos[i] = self.pos[i] + v * dt
        self.pos, self.vel = new_pos, new_vel
        return self.state()

    def _repulsion(self, p, q, A: float, clearance: float) -> np.ndarray:
        """Exponential repulsion pushing p away from q: A*exp((clearance-d)/B)*n."""
        diff = p - q
        d = float(np.hypot(*diff))
        if d < 1e-9:
            return np.zeros(2)
        return A * np.exp((clearance - d) / self.sfm.repulsion_B) * (diff / d)
