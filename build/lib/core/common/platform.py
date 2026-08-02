"""Platform stacks: one loader for every (robot, mpc, cbf, rl) parameter set.

Two platforms exist (plan D2 + the 2026-07 supervision-headroom replan):

    tb3         : TurtleBot3 Waffle, the deployed/Gazebo-validated platform.
                  v_max 0.26 -> d_stop ~0.25 m: NO supervision headroom (proven four
                  independent ways) -- the exact-braking CBF + MPC human term already
                  regulate speed near-optimally. Observation v1 (32-dim, human-only),
                  FROZEN together with all TB3 results/models on disk.
    industrial  : MiR-class AMR dynamics from robot.yaml `industrial_appendix`
                  (grounding table there and in the replan). v_max 1.5 ->
                  d_stop ~2.5 m: the regime where fixed tunings are Pareto-dominated
                  and supervision is NECESSARY (headroom probe: always-max breaches
                  28/30 crowd episodes despite the CBF). Observation v2 adds
                  wall/occlusion features so a policy CAN learn corner anticipation.

The CBF safety-argument invariant is untouched: within a platform, the filter
constants remain frozen and RL-independent; a platform swaps the WHOLE certified
parameter set, it never lets the policy touch one.

Usage:
    from core.common.platform import load_platform
    p = load_platform("industrial")
    env = NavEnv(robot=p.robot, mpc=p.mpc, cbf=p.cbf, rl=p.rl, ...)
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from core.common.params import (CbfParams, MpcParams, RlParams, RobotParams,
                                load_yaml)

PLATFORMS = ("tb3", "industrial")


@dataclass(frozen=True)
class Platform:
    """A complete, internally-consistent parameter stack for one robot class."""

    name: str
    robot: RobotParams
    mpc: MpcParams
    cbf: CbfParams
    rl: RlParams
    obs_version: int      # 1 = human-only 32-dim (TB3, frozen); 2 = +occlusion
    obs_scale: dict | None = None   # None = frozen v1 SCALE (observation.py)


def load_platform(name: str = "tb3") -> Platform:
    """Build the parameter stack for a platform. 'tb3' is byte-identical to the
    historical from_yaml() path so every existing result stays reproducible."""
    robot = RobotParams.from_yaml()
    mpc = MpcParams.from_yaml()
    cbf = CbfParams.from_yaml()
    rl = RlParams.from_yaml()
    if name == "tb3":
        return Platform("tb3", robot, mpc, cbf, rl, obs_version=1)
    if name == "industrial":
        ind = load_yaml("robot")["industrial_appendix"]
        act = ind["action"]
        robot = dataclasses.replace(
            robot, v_max=ind["v_max"], tau_latency=ind["tau_latency"],
            a_max_physical=ind["a_max_physical"], a_max_mpc=ind["a_max_mpc"])
        cbf = dataclasses.replace(
            cbf, a_brake=ind["a_brake"], tau=ind["tau_latency"])
        mpc = dataclasses.replace(mpc, carrot_lookahead=ind["carrot_lookahead"])
        rl = dataclasses.replace(
            rl, v_max_low=act["v_max_low"], v_max_high=act["v_max_high"],
            d_margin_high=act["d_margin_high"])
        rew = ind.get("reward", {})              # platform reward overrides
        if rew:
            rl = dataclasses.replace(rl, weights=dataclasses.replace(
                rl.weights, **rew))
        # v2 normalization from the platform itself: O(1) entries at 1.5 m/s in
        # 10+ m arenas (v1's constants are frozen with the TB3 models).
        from core.common.observation import SCALE, TTCA_MAX
        obs_scale = {**SCALE, "dist": 10.0, "v_robot": robot.v_max,
                     "v_human": 2.0 + robot.v_max, "margin": rl.d_margin_high,
                     "ttca": TTCA_MAX}
        return Platform("industrial", robot, mpc, cbf, rl,
                        obs_version=2, obs_scale=obs_scale)
    raise ValueError(f"unknown platform '{name}'; choose from {PLATFORMS}")
