"""Gymnasium environment wrapping the 2D sim + MPC + CBF stack (plan D6).

The agent is a SUPERVISOR: at 2 Hz (every `decision_every`=5 MPC steps) it outputs

    a = [v_max_cmd in [0.05, 0.26], d_margin_cmd in [d_hard, 1.2]]

which modulate the MPC (hard v-cap, soft human-potential radius). The inner loop
runs MPC (10 Hz) -> CBF filter -> sim, with the SFM crowd stepping alongside.

The same env also runs the FIXED-TUNING baselines S1/S2 and the no-filter ablation
S3 (plan 4.1): `fixed_params=(v_max, d_margin)` bypasses the action, `use_cbf=False`
removes the filter from the loop. The CBF is ALWAYS instantiated for measurement
(barrier h, the violations metric) even when it does not act -- that is how the
S2-crosses-the-line / S4-never-does money plot is produced from one code path.

Reference path: straight line start->goal with a moving carrot. All five scenarios
are straight-line-feasible by design (corridor/doorway/blind-corner centerlines),
so the plan's A* option is deliberately deferred; path_curvature in the observation
is 0 until then. Walls are fed to the MPC as the per-solve nearest point on each
segment (radius-0 circles -- r_robot supplies the clearance), capacity-limited to
the 6 nearest together with the scenario's post obstacles.

Metrics (plan 4.3) are accumulated per episode in `info["episode_metrics"]` on the
final step (NOT `info["episode"]` -- SB3's VecMonitor owns that key): safety /
efficiency / comfort / filter-behavior / compute -- the baseline battery and the
evaluation ladder all read from this one place.

Domain randomization (plan D6, curriculum stage B+): pass `domain_rand` (the
rl.yaml `domain_randomization` block). Per episode: tau ~ nominal +- tau_jitter
(applied to BOTH the sim latency buffer and the filter, same physical quantity),
filter a_brake ~ nominal * (1 +- a_brake_frac); per step: N(0, sigma) noise on the
tracked human VELOCITIES as consumed by MPC/CBF/observation (metrics keep ground
truth). The MPC keeps nominal params (it uses neither tau nor a_brake).
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Optional, Sequence

import gymnasium as gym
import numpy as np

from core.cbf.cbf_filter import CbfFilter, d_stop
from core.common.observation import (build_observation, corner_sight_distance,
                                     obs_dim)
from core.common.params import (CbfParams, MpcParams, RlParams, RobotParams)
from core.mpc.mpc_controller import MpcController
from core.rl.reward import reward_terms, total_reward
from core.sim2d.kinematic_sim import KinematicSim
from core.sim2d.pedestrians import closest_point_on_segment
from core.sim2d.scenarios import SCENARIO_NAMES, ScenarioSpec, make_scenario

GOAL_TOL = 0.15        # m   success radius
FULL_STOP_V = 0.02     # m/s below this the robot counts as fully stopped (plan 4.3)
INTERVENTION_EPS = 1e-3  # ||u_safe - u_mpc|| above this counts as an intervention
V_MOVING = 1e-3        # accountability threshold (same split as the G2 battery)


class EpisodeMetrics:
    """Per-episode accumulator for the plan-4.3 metric families."""

    def __init__(self, straight_dist: float):
        self.straight_dist = max(straight_dist, 1e-9)
        self.t = 0.0
        self.path_len = 0.0
        self.energy = 0.0
        self.jerks: list[float] = []
        self.min_human_dist = np.inf
        self.min_wall_clear = np.inf
        self.min_h = np.inf
        self.violation_steps = 0        # h < 0 while MOVING (filter-accountable)
        self.collision = False
        self.wall_contact = False
        self.protective_stops = 0
        self.intrusion_time = 0.0
        self.full_stops = 0
        self.interventions: list[float] = []
        self.solve_ms: list[float] = []
        self.success = False

    def summary(self) -> dict:
        iv = np.asarray(self.interventions) if self.interventions else np.zeros(1)
        sm = np.asarray(self.solve_ms) if self.solve_ms else np.zeros(1)
        return {
            # safety
            "success": self.success, "collision": self.collision,
            "wall_contact": self.wall_contact,
            "min_human_dist": float(self.min_human_dist),
            "min_h": float(self.min_h), "violation_steps": self.violation_steps,
            "protective_stops": self.protective_stops,
            # efficiency
            "time_to_goal": self.t if self.success else np.nan,
            "episode_time": self.t, "energy": self.energy,
            "full_stops": self.full_stops,
            "path_length_ratio": self.path_len / self.straight_dist,
            # comfort / social
            "rms_jerk": float(np.sqrt(np.mean(np.square(self.jerks)))) if self.jerks else 0.0,
            "intrusion_time": self.intrusion_time,
            # filter behaviour
            "intervention_rate": float(np.mean(iv > INTERVENTION_EPS)),
            "mean_intervention": float(np.mean(iv)),
            # compute
            "mpc_solve_ms_median": float(np.median(sm)),
            "mpc_solve_ms_p99": float(np.percentile(sm, 99)),
        }


class NavEnv(gym.Env):
    """Supervisor-level Gymnasium env over the MPC+CBF+2D-sim stack (plan D6)."""

    metadata = {"render_modes": []}

    def __init__(self,
                 scenarios: Sequence[str] = SCENARIO_NAMES,
                 scenario_sampler: Optional[Callable[[np.random.Generator], ScenarioSpec]] = None,
                 use_cbf: bool = True,
                 fixed_params: Optional[tuple] = None,
                 robot: Optional[RobotParams] = None,
                 mpc: Optional[MpcParams] = None,
                 cbf: Optional[CbfParams] = None,
                 rl: Optional[RlParams] = None,
                 domain_rand: Optional[dict] = None,
                 record: bool = False,
                 obs_version: int = 1,
                 obs_scale: Optional[dict] = None,
                 scenario_platform: str = "tb3"):
        super().__init__()
        # obs v2 (industrial track): +3 map-derived wall/occlusion features and a
        # platform scale dict; v1 (default) stays byte-identical for TB3.
        self.obs_version = obs_version
        self.obs_scale = obs_scale
        # named-scenario sampling builds this platform's arena geometry (guards
        # against industrial params silently running in TB3-scale arenas)
        self.scenario_platform = scenario_platform
        # record=True: keep a per-inner-step trace in self.trajectory (list of
        # dicts) for the week-6 money plots -- velocity-vs-distance overlays
        # (plot 1) and barrier h(t) traces (plot 3) need per-step data that the
        # episode metrics deliberately aggregate away.
        self.record = record
        self.robot = robot or RobotParams.from_yaml()
        self.mpc_cfg = mpc or MpcParams.from_yaml()
        self.cbf_cfg = cbf or CbfParams.from_yaml()
        self.rl = rl or RlParams.from_yaml()
        self.scenario_names = list(scenarios)
        self.scenario_sampler = scenario_sampler
        self.use_cbf = use_cbf
        self.fixed_params = fixed_params
        self.domain_rand = domain_rand

        self.mpc = MpcController(self.robot, self.mpc_cfg)
        self.filt = CbfFilter(self.robot, self.cbf_cfg)   # acts AND/OR measures
        self.sim = KinematicSim(self.robot)

        self.action_space = gym.spaces.Box(
            low=np.array([self.rl.v_max_low, self.rl.d_margin_low], dtype=np.float32),
            high=np.array([self.rl.v_max_high, self.rl.d_margin_high], dtype=np.float32))
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim(self.rl.K_nearest, self.obs_version),),
            dtype=np.float32)

        self.spec_: Optional[ScenarioSpec] = None   # trailing _ : gym.Env owns .spec

    # ----------------------------------------------------------------- helpers
    def _carrot(self, pos_xy: np.ndarray) -> np.ndarray:
        """Moving carrot on the straight start->goal line (lookahead from config)."""
        a = self.spec_.robot_start[:2]
        b = self.spec_.goal
        ab = b - a
        L = float(np.hypot(*ab))
        e = ab / max(L, 1e-9)
        s = float(np.dot(pos_xy - a, e))
        s_carrot = np.clip(s + self.mpc_cfg.carrot_lookahead, 0.0, L)
        return a + s_carrot * e

    def _progress_coord(self, pos_xy: np.ndarray) -> float:
        """Arc-length coordinate along the reference line (for the w1 term)."""
        a, b = self.spec_.robot_start[:2], self.spec_.goal
        ab = b - a
        return float(np.dot(pos_xy - a, ab) / max(np.hypot(*ab), 1e-9))

    def _mpc_obstacles(self, pos_xy: np.ndarray) -> np.ndarray:
        """Scenario posts + nearest point on each wall (r=0), 6 nearest overall."""
        obs = [list(o) for o in self.spec_.static_obstacles]
        for w in self.spec_.walls:
            p = closest_point_on_segment(pos_xy, w[:2], w[2:])
            obs.append([p[0], p[1], 0.0])
        if not obs:
            return np.zeros((0, 3))
        obs = np.asarray(obs, dtype=float)
        d = np.hypot(obs[:, 0] - pos_xy[0], obs[:, 1] - pos_xy[1])
        return obs[np.argsort(d)[:self.mpc_cfg.max_static_obstacles]]

    def _wall_clearance(self, pos_xy: np.ndarray) -> float:
        if len(self.spec_.walls) == 0:
            return np.inf
        return min(float(np.hypot(*(pos_xy - closest_point_on_segment(pos_xy, w[:2], w[2:]))))
                   for w in self.spec_.walls)

    def _corner_speed_excess(self, v: float) -> float:
        """metres by which the robot's (sigma-inflated) stopping distance exceeds the
        sight distance to the nearest mapped blind constriction ahead -- the w9
        anticipatory-speed signal (0 unless obs v2 + a post is ahead). Uses the SAME
        longitudinal post_ahead the policy observes, so the incentive is actionable.
        The sight distance is max(post_ahead, sight_floor): you can see up to the
        corner, and sight_floor around it once there (ISO limited-visibility speed).
        """
        if self.obs_version < 2:
            return 0.0
        posts = self.spec_.static_obstacles
        if posts is None or len(posts) == 0:
            return 0.0
        post_ahead = corner_sight_distance(self.s, posts)
        sight = max(post_ahead, self.rl.blind_corner_sight_floor)
        d_stop_v = d_stop(self.cbf_cfg.sigma * v, self.cbf_cfg.tau, self.cbf_cfg.a_brake)
        return max(0.0, d_stop_v - sight)

    def _observe(self) -> np.ndarray:
        return build_observation(self.s, self.spec_.goal, self._tracked_humans(),
                                 self.v_max_cmd, self.d_margin_cmd,
                                 k_nearest=self.rl.K_nearest,
                                 version=self.obs_version,
                                 walls=self.spec_.walls,
                                 posts=self.spec_.static_obstacles,
                                 scale=self.obs_scale)

    def _apply_domain_rand(self) -> None:
        """Per-episode DR (see module doc). Rebuilds sim + filter (cheap); the MPC
        NLP is untouched (nominal robot bounds, no tau/a_brake dependence)."""
        dr = self.domain_rand
        tau = self.robot.tau_latency + self.np_random.uniform(
            -dr.get("tau_jitter", 0.0), dr.get("tau_jitter", 0.0))
        brake = self.cbf_cfg.a_brake * (1.0 + self.np_random.uniform(
            -dr.get("a_brake_frac", 0.0), dr.get("a_brake_frac", 0.0)))
        robot_dr = dataclasses.replace(self.robot, tau_latency=tau)
        cbf_dr = dataclasses.replace(self.cbf_cfg, tau=tau, a_brake=brake)
        self.sim = KinematicSim(robot_dr)
        self.filt = CbfFilter(robot_dr, cbf_dr)
        self._vel_noise = float(dr.get("human_vel_obs_noise_std", 0.0))

    def _tracked_humans(self) -> np.ndarray:
        """Tracker view fed to MPC/CBF/observation: occlusions applied by the
        scenario, velocity noise applied by DR. Metrics use ground truth instead."""
        humans = self.spec_.visible_humans(self.s[:2])
        if self._vel_noise > 0.0 and len(humans):
            humans = humans.copy()
            humans[:, 2:] += self.np_random.normal(0.0, self._vel_noise,
                                                   size=humans[:, 2:].shape)
        return humans

    # --------------------------------------------------------------- gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.scenario_sampler is not None:
            self.spec_ = self.scenario_sampler(self.np_random)
        else:
            name = self.scenario_names[self.np_random.integers(len(self.scenario_names))]
            self.spec_ = make_scenario(name, seed=int(self.np_random.integers(2 ** 31)),
                                       platform=self.scenario_platform)

        self._vel_noise = 0.0
        if self.domain_rand is not None:
            self._apply_domain_rand()
        self.s = self.sim.reset(self.spec_.robot_start)
        self.mpc.reset()
        self.filt.reset()
        self.t = 0.0
        self._u_prev = np.zeros(2)
        self._dv_prev = 0.0
        self._was_stopped = True        # episodes start at rest: no phantom stop
        self._s_along = self._progress_coord(self.s[:2])
        # params in effect before the first action: conservative floor
        self.v_max_cmd, self.d_margin_cmd = self.rl.v_max_low, self.rl.d_margin_high
        if self.fixed_params is not None:
            self.v_max_cmd, self.d_margin_cmd = self.fixed_params
        self.metrics = EpisodeMetrics(
            float(np.hypot(*(self.spec_.goal - self.spec_.robot_start[:2]))))
        self.trajectory: list[dict] = []
        return self._observe(), {"scenario": self.spec_.name, "seed": self.spec_.seed}

    def step(self, action):
        if self.fixed_params is None:
            a = np.clip(np.asarray(action, dtype=float),
                        self.action_space.low, self.action_space.high)
            self.v_max_cmd, self.d_margin_cmd = float(a[0]), float(a[1])

        dt, m = self.robot.dt, self.metrics
        window_terms: list[dict] = []
        terminated = truncated = False

        for _ in range(self.rl.decision_every):
            self.spec_.tick(self.t, robot_xy=self.s[:2])
            humans = self._tracked_humans()
            mpc_humans = humans[np.argsort(np.hypot(humans[:, 0] - self.s[0],
                                                    humans[:, 1] - self.s[1]))
                                [:self.mpc_cfg.max_humans]] if len(humans) else None

            u_mpc, info_mpc = self.mpc.solve(
                x0=self.s[:3], carrot=self._carrot(self.s[:2]),
                static_obs=self._mpc_obstacles(self.s[:2]), humans=mpc_humans,
                v_max_cmd=self.v_max_cmd, d_margin_cmd=self.d_margin_cmd,
                u_prev=self._u_prev)
            m.solve_ms.append(info_mpc["solve_ms"])

            if self.use_cbf:
                u_safe, info_f = self.filt.filter(self.s, u_mpc, humans)
            else:  # ablation/baseline: the filter never acts (it only measures h)
                u_safe = u_mpc
                info_f = {"intervention": 0.0, "protective_stop": False}

            v_before = float(self.s[3])
            pos_before = self.s[:2].copy()
            self.s = self.sim.step(u_safe)
            self.spec_.crowd.step(dt, robot_xy=self.s[:2])
            self.t += dt
            self._u_prev = np.asarray(u_safe, dtype=float)

            # ---- metrics + reward inputs (applied-state quantities) ----
            humans_true = self.spec_.crowd.state()      # metrics use ground truth
            d_human = (float(np.min(np.hypot(humans_true[:, 0] - self.s[0],
                                             humans_true[:, 1] - self.s[1])))
                       if len(humans_true) else np.inf)
            h = self.filt.min_barrier(self.s, humans_true)
            v = float(self.s[3])
            dv = v - v_before
            jerk = dv - self._dv_prev
            self._dv_prev = dv
            step_len = float(np.hypot(*(self.s[:2] - pos_before)))
            wall_clear = self._wall_clearance(self.s[:2])
            moving = v > V_MOVING
            corner_excess = self._corner_speed_excess(v)   # w9 anticipatory term (0 on v1)
            # w10 anticipatory human-approach margin: shortfall of the barrier h below
            # the buffer (0 when people far/beside since h large; industrial only)
            buf = self.rl.human_approach_buffer
            approach_buf = (float(np.clip(buf - h, 0.0, buf))
                            if (self.obs_version >= 2 and np.isfinite(h)) else 0.0)

            m.t = self.t
            m.path_len += step_len
            m.energy += abs(dv) * v            # |a|*v*dt with a*dt = dv
            m.jerks.append(jerk / (dt * dt))
            m.min_human_dist = min(m.min_human_dist, d_human)
            m.min_wall_clear = min(m.min_wall_clear, wall_clear)
            m.min_h = min(m.min_h, h)
            if h < -1e-6 and moving:
                m.violation_steps += 1
            if (d_human < self.cbf_cfg.d_hard and moving) or d_human < self.robot.robot_radius:
                m.collision = True
            if wall_clear < self.robot.robot_radius:
                m.wall_contact = True
            m.protective_stops += int(info_f["protective_stop"])
            m.intrusion_time += dt * float(d_human < self.rl.personal_space)
            m.interventions.append(float(info_f["intervention"]))
            stopped = v < FULL_STOP_V
            if stopped and not self._was_stopped:
                m.full_stops += 1
            self._was_stopped = stopped

            s_along = self._progress_coord(self.s[:2])
            progress = s_along - self._s_along
            self._s_along = s_along

            if self.record:
                # closing geometry to the NEAREST ground-truth human: v_los is the
                # robot's own closing speed along the line of sight (v*max(0,cos)),
                # human_closing is the pedestrian's. A breach with v_los ~ 0 but
                # human_closing > 0 is a pedestrian walking INTO the robot, not the
                # robot driving in -- the CBF caps closing motion, it cannot forbid
                # a human from approaching a robot that is not moving toward them.
                v_los = human_closing = 0.0
                if len(humans_true):
                    j = int(np.argmin(np.hypot(humans_true[:, 0] - self.s[0],
                                               humans_true[:, 1] - self.s[1])))
                    lx = humans_true[j, 0] - self.s[0]
                    ly = humans_true[j, 1] - self.s[1]
                    dd = max(float(np.hypot(lx, ly)), 1e-6)
                    nx, ny = lx / dd, ly / dd
                    cos = np.cos(self.s[2]) * nx + np.sin(self.s[2]) * ny
                    v_los = v * max(0.0, float(cos))
                    human_closing = -float(humans_true[j, 2] * nx + humans_true[j, 3] * ny)
                self.trajectory.append({
                    "t": self.t, "x": float(self.s[0]), "y": float(self.s[1]),
                    "theta": float(self.s[2]),
                    "v_mpc": float(u_mpc[0]), "v_safe": float(u_safe[0]),
                    "v_applied": v, "h": float(h), "d_human": float(d_human),
                    "v_los": v_los, "human_closing": human_closing,
                    "v_max_cmd": self.v_max_cmd, "d_margin_cmd": self.d_margin_cmd,
                    "intervention": float(info_f["intervention"]),
                    "protective_stop": bool(info_f["protective_stop"]),
                })

            success_now = float(np.hypot(*(self.spec_.goal - self.s[:2]))) < GOAL_TOL
            # w5 accountability (week-4 audit): charge a protective stop only if
            # the robot was MOVING when the field fired -- it drove in. A human
            # brushing past a stopped, correctly-yielding robot is unpreventable
            # (same split as the G2 battery); charging it per step made w5 pure
            # exposure noise that punished SLOW policies hardest (-15 vs -12 in
            # the corridor) and inverted the intended incentive. The headline
            # METRIC still counts every protective stop.
            pstop_charged = bool(info_f["protective_stop"]) and v > FULL_STOP_V
            window_terms.append(reward_terms(
                self.rl.weights, progress=progress, dv=dv, v=v, jerk=jerk,
                intervention=float(info_f["intervention"]),
                protective_stop=pstop_charged,
                min_human_dist=d_human, dt=dt, success=success_now,
                personal_space=self.rl.personal_space,
                barrier=h, moving=moving,      # h vs ground truth (see reward.py)
                corner_speed_excess=corner_excess,
                approach_buffer=approach_buf))

            if success_now:
                m.success = True
                terminated = True
            elif m.collision or m.wall_contact:
                terminated = True
            elif self.t >= self.rl.episode_timeout_s:
                truncated = True
            if terminated or truncated:
                break

        terms = {k: float(np.sum([w[k] for w in window_terms]))
                 for k in window_terms[0]}
        reward = total_reward(terms)
        info = {"reward_terms": terms, "params": (self.v_max_cmd, self.d_margin_cmd)}
        if terminated or truncated:
            info["episode_metrics"] = m.summary()   # "episode" belongs to VecMonitor
        return self._observe(), reward, terminated, truncated, info
