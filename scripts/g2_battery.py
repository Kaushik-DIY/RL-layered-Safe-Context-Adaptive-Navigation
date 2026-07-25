"""Gate G2 (ABSOLUTE, plan sec. 5 & 4.1-S5): the CBF safety layer, verified before
any learning exists.

Feed the filter an ADVERSARIAL ROBOT POLICY (always full speed toward goal, or
uniformly random) while humans move on fair constant-velocity paths (offset head-on
passers + perpendicular crossers). If the robot yields, every path clears the robot
by >= d_hard, so any violation is the filter's fault, not a human walking into a
stopped robot (which no controller can prevent).

Pass criterion / metric classification (the honest split): the stopping-distance
constraint governs ROBOT MOTION, so accountability follows the robot's state:

  * barrier violation  h < 0 while the robot is MOVING           -> filter's fault
  * collision          d < d_hard while MOVING, or physical
                       contact d < robot_radius at any time      -> filter's fault
  * stopped intrusion  h < 0 (i.e. d < d_hard) while the robot is
                       FULLY STOPPED -> a human stepped within the hard floor of a
                       stationary robot. No control action can prevent this; the CV
                       battery excludes it by fair-path construction, SFM crowds
                       occasionally produce it (pedestrians press to the SFM
                       equilibrium ~0.30 m from a parked robot). Counted and
                       REPORTED separately -- never hidden -- but not a failure.

Run:

    python scripts/g2_battery.py            # full 1000-episode battery (CV humans)
    python scripts/g2_battery.py 200        # quick check
    python scripts/g2_battery.py 1000 --sfm # SFM-crowd variant (see below)

Two human models, deliberately:
  * CV (default, THE gate): humans on constant-velocity paths exactly matching the
    filter's one-step prediction model -- isolates the filter's own logic, and the
    fair-path construction makes any violation attributable to the filter.
  * SFM (--sfm, robustness extension): Helbing-Molnar pedestrians who accelerate,
    swerve, and react to the robot -- stress-tests the filter's constant-velocity
    assumption (claimed to be absorbed by the sigma inflation + conservative
    a_brake; this battery is what turns that claim into evidence).

The pytest in tests/test_cbf.py reuses run_adversary_episode for a smaller CI battery.
"""
from __future__ import annotations

import sys

import numpy as np

from core.cbf.cbf_filter import CbfFilter
from core.common.params import CbfParams, RobotParams
from core.sim2d.kinematic_sim import KinematicSim, wrap_angle
from core.sim2d.pedestrians import SfmParams, SocialForceCrowd

GOAL = np.array([6.0, 0.0])
V_STOPPED = 1e-3  # m/s: at/below this the robot counts as fully stopped


class SafetyLog:
    """Per-episode safety bookkeeping with the accountability split (module doc)."""

    def __init__(self, robot, cbf):
        self.robot, self.cbf = robot, cbf
        self.min_h, self.min_d = np.inf, np.inf
        self.violated = self.collided = False
        self.stopped_intrusions = self.pstops = 0

    def update(self, filt, s, humans, info) -> None:
        h = filt.min_barrier(s, humans)
        d = float(np.min(np.hypot(humans[:, 0] - s[0], humans[:, 1] - s[1])))
        moving = float(s[3]) > V_STOPPED
        self.min_h, self.min_d = min(self.min_h, h), min(self.min_d, d)
        if h < -1e-6:
            if moving:
                self.violated = True          # filter-accountable
            else:
                self.stopped_intrusions += 1  # human within d_hard of a parked robot
        if (d < self.cbf.d_hard and moving) or d < self.robot.robot_radius:
            self.collided = True
        self.pstops += int(info["protective_stop"])

    def summary(self, steps: int) -> dict:
        return {"min_h": float(self.min_h), "min_d": float(self.min_d),
                "violated": self.violated, "collided": self.collided,
                "stopped_intrusions": self.stopped_intrusions,
                "pstops": self.pstops, "steps": steps}


def _spawn_humans(rng, n, d_hard):
    """Fair CV humans: each clears the x-axis corridor by >= d_hard if untouched."""
    humans = []
    for _ in range(n):
        speed = rng.uniform(0.6, 1.5)
        if rng.random() < 0.5:
            # perpendicular crosser: crosses the robot's path at x_c
            x_c = rng.uniform(1.5, 4.5)
            side = rng.choice([-1.0, 1.0])
            y0 = side * rng.uniform(1.0, 1.8)
            humans.append([x_c, y0, 0.0, -side * speed])
        else:
            # offset head-on passer: walks toward the robot, off-centre by > d_hard
            x0 = rng.uniform(4.0, 6.0)
            offset = rng.choice([-1.0, 1.0]) * rng.uniform(d_hard + 0.15, 0.9)
            humans.append([x0, offset, -speed, 0.0])
    return np.array(humans, dtype=float)


def run_adversary_episode(seed, robot, cbf, n_humans=2, steps=220, adversary="max"):
    """One adversarial-robot episode. Returns per-episode safety summary."""
    rng = np.random.default_rng(seed)
    sim = KinematicSim(robot)
    filt = CbfFilter(robot, cbf)
    filt.reset()
    s = sim.reset([0.0, 0.0, 0.0])
    humans = _spawn_humans(rng, n_humans, cbf.d_hard)

    log = SafetyLog(robot, cbf)
    for t in range(steps):
        # The adversary is a policy over SPEED that always drives toward the goal
        # with sane pursuit steering and IGNORES humans entirely (maximally
        # aggressive: the CBF must do all the safety work). It never spins/loiters,
        # so a violation can only be the robot driving too fast into a human -- the
        # thing the filter must prevent -- not a human walking into a parked robot.
        heading_err = wrap_angle(np.arctan2(GOAL[1] - s[1], GOAL[0] - s[0]) - s[2])
        omega = float(np.clip(2.0 * heading_err, robot.omega_min, robot.omega_max))
        v_cmd = robot.v_max if adversary == "max" else rng.uniform(0.05, robot.v_max)
        u_mpc = np.array([v_cmd, omega])

        u_safe, info = filt.filter(s, u_mpc, humans)
        s = sim.step(u_safe)
        humans[:, 0] += humans[:, 2] * robot.dt
        humans[:, 1] += humans[:, 3] * robot.dt

        log.update(filt, s, humans, info)
        if np.hypot(*(GOAL - s[:2])) < 0.2:
            break

    return log.summary(t + 1)


def run_sfm_episode(seed, robot, cbf, n_humans=2, steps=260, adversary="max"):
    """Adversarial robot vs SFM pedestrians (CV-assumption stress, see module doc).

    Same fair spawn geometry as the CV battery (crossers + offset passers, expressed
    as SFM goals), but the pedestrians now accelerate from rest, curve, and repel off
    the robot -- everything the filter's one-step CV prediction does NOT model.
    """
    rng = np.random.default_rng(seed)
    sim = KinematicSim(robot)
    filt = CbfFilter(robot, cbf)
    filt.reset()
    s = sim.reset([0.0, 0.0, 0.0])
    sfm = SfmParams.from_yaml()

    positions, goals = [], []
    for h in _spawn_humans(rng, n_humans, cbf.d_hard):
        positions.append(h[:2])
        # same intent as the CV velocity, expressed as an SFM goal well past the path
        heading = np.array(h[2:]) / max(np.hypot(h[2], h[3]), 1e-9)
        goals.append(np.array(h[:2]) + heading * 6.0)
    speeds = rng.uniform(0.6, 1.5, size=n_humans)
    crowd = SocialForceCrowd(sfm, positions, goals, speeds, rng=rng)

    log = SafetyLog(robot, cbf)
    for t in range(steps):
        heading_err = wrap_angle(np.arctan2(GOAL[1] - s[1], GOAL[0] - s[0]) - s[2])
        omega = float(np.clip(2.0 * heading_err, robot.omega_min, robot.omega_max))
        v_cmd = robot.v_max if adversary == "max" else rng.uniform(0.05, robot.v_max)
        u_mpc = np.array([v_cmd, omega])

        humans = crowd.state()
        u_safe, info = filt.filter(s, u_mpc, humans)
        s = sim.step(u_safe)
        humans = crowd.step(robot.dt, robot_xy=s[:2])

        log.update(filt, s, humans, info)
        if np.hypot(*(GOAL - s[:2])) < 0.2:
            break

    return log.summary(t + 1)


def run_battery(n_episodes, robot, cbf, mode="cv"):
    run_episode = run_sfm_episode if mode == "sfm" else run_adversary_episode
    results = []
    for i in range(n_episodes):
        adv = "max" if i % 2 == 0 else "random"
        n_h = 1 + (i % 3)
        results.append(run_episode(i, robot, cbf, n_humans=n_h, adversary=adv))
    n_viol = sum(r["violated"] for r in results)
    n_coll = sum(r["collided"] for r in results)
    return {
        "n_episodes": n_episodes,
        "n_barrier_violations": n_viol,        # h < 0 while MOVING (filter's fault)
        "n_collisions": n_coll,                # moving breach or physical contact
        "n_stopped_intrusions": sum(r["stopped_intrusions"] > 0 for r in results),
        "global_min_h": min(r["min_h"] for r in results),
        "global_min_d": min(r["min_d"] for r in results),
        "total_protective_stops": sum(r["pstops"] for r in results),
        "passed": n_viol == 0 and n_coll == 0,
    }


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--sfm"]
    mode = "sfm" if "--sfm" in sys.argv[1:] else "cv"
    n = int(args[0]) if args else 1000
    robot = RobotParams.from_yaml()
    cbf = CbfParams.from_yaml()
    print(f"Running G2 battery: {n} adversarial episodes ({mode.upper()} humans) ...")
    r = run_battery(n, robot, cbf, mode=mode)
    for k, v in r.items():
        print(f"  {k:24s}: {v}")
    print(f"\nGate G2 (zero violations, zero collisions): "
          f"{'PASS' if r['passed'] else 'FAIL'}")
    raise SystemExit(0 if r["passed"] else 1)


if __name__ == "__main__":
    main()
