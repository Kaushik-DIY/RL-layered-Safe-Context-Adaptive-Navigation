"""Supervision-headroom probe (plan D2 appendix params): WHERE does adaptive
parameter modulation beat a fixed tuning, given the same MPC+CBF stack?

Finding so far: at TB3 scale (v_max 0.26, d_stop ~0.22 m) there is NO headroom --
the MPC human term + exact-braking CBF already regulate speed near-optimally, and
the trained policy converges to always-max (three independent confirmations).
Hypothesis: headroom appears when the stopping envelope is LARGE relative to
personal space -- the industrial_appendix platform (v_max 1.5, a_brake 0.8,
tau 0.5 -> d_stop ~2.2 m). There, riding the filter's cap through a crowd should
cost stop-and-go, jerk, protective stops, and deep incursions that preemptive
slowing avoids.

Probe: open_hall (6-8 free-roaming pedestrians), three HAND-BUILT supervisors on
the identical stack, paired seeds, both platform scales:

    always-max : [v_max, margin_floor]          (what the trained policy became)
    fixed-mid  : [0.55*v_max, 0.5]              (a sensible compromise tuning)
    heuristic  : density-aware speed + margin   (the ceiling a policy could learn)

If heuristic beats always-max at industrial scale on task metrics, THAT is the
regime where a learned supervisor is meaningful -- and the training target.

    python scripts/headroom_probe.py            # 30 paired eps x 3 sups x 2 scales
    python scripts/headroom_probe.py 5          # quick
    python scripts/headroom_probe.py 30 industrial   # one scale only
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.cbf.cbf_filter import d_stop  # noqa: E402
from core.common.params import CbfParams, RobotParams, load_yaml  # noqa: E402
from core.common.platform import load_platform  # noqa: E402
from core.rl.nav_env import NavEnv  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results"

METRICS = ["success", "time_to_goal", "violation_steps", "collision",
           "protective_stops", "full_stops", "rms_jerk", "energy",
           "intrusion_time", "min_human_dist", "intervention_rate"]


def build_stack(scale: str):
    """(robot, mpc, cbf, rl) for a platform scale -- shared loader (P1.1)."""
    p = load_platform(scale)
    return p.robot, p.mpc, p.cbf, p.rl


def make_supervisors(env: NavEnv, robot: RobotParams, cbf: CbfParams):
    """Three hand-built supervisors. The heuristic reads the same tracker view the
    RL observation is built from (visible humans + occlusion), no ground truth."""
    v_hi = robot.v_max
    D = d_stop(cbf.sigma * v_hi, cbf.tau, cbf.a_brake)   # full-speed stop envelope

    def always_max(_obs):
        return np.array([v_hi, 0.30])

    def fixed_mid(_obs):
        return np.array([0.55 * v_hi, 0.50])

    def heuristic(_obs):
        humans = env._tracked_humans()
        if len(humans) == 0:
            return np.array([v_hi, 0.30])
        d = np.hypot(humans[:, 0] - env.s[0], humans[:, 1] - env.s[1])
        d_near = float(d.min())
        n_close = int((d < 1.5 * D).sum())
        if d_near > 3.0 * D:
            v = v_hi
        elif d_near > 1.5 * D:
            v = 0.6 * v_hi
        else:
            v = 0.3 * v_hi
        margin = float(np.clip(0.30 + 0.10 * n_close, 0.30, 0.80))
        return np.array([v, margin])

    return {"always-max": always_max, "fixed-mid": fixed_mid,
            "heuristic": heuristic}


def run(scale: str, n: int, seed_base: int) -> pd.DataFrame:
    robot, mpc, cbf, rl = build_stack(scale)
    env = NavEnv(scenarios=["open_hall"], use_cbf=True,
                 robot=robot, mpc=mpc, cbf=cbf, rl=rl)
    sups = make_supervisors(env, robot, cbf)
    D = d_stop(cbf.sigma * robot.v_max, cbf.tau, cbf.a_brake)
    print(f"\n### scale={scale}  v_max={robot.v_max}  d_stop(full)={D:.2f} m ###")
    rows = []
    for name, pol in sups.items():
        for i in range(n):
            obs, _ = env.reset(seed=seed_base + i)
            done = False
            while not done:
                obs, _, term, trunc, info = env.step(pol(obs))
                done = term or trunc
            ep = info["episode_metrics"]
            ep.update(scale=scale, supervisor=name, episode=i)
            rows.append(ep)
        sub = [r for r in rows if r["supervisor"] == name and r["scale"] == scale]
        print(f"  [{name:10s}] succ {sum(r['success'] for r in sub)}/{n}"
              f"  coll {sum(r['collision'] for r in sub)}"
              f"  viol_eps {sum(r['violation_steps'] > 0 for r in sub)}"
              f"  t {np.nanmean([r['time_to_goal'] for r in sub]):.1f}s"
              f"  pstops {np.mean([r['protective_stops'] for r in sub]):.1f}"
              f"  jerk {np.mean([r['rms_jerk'] for r in sub]):.2f}")
    return pd.DataFrame(rows)


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    scales = [sys.argv[2]] if len(sys.argv) > 2 else ["industrial", "tb3"]
    seed_base = load_yaml("scenarios")["seed_base"] + 3000     # disjoint block

    dfs = [run(sc, n, seed_base) for sc in scales]
    df = pd.concat(dfs, ignore_index=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "headroom_probe.csv", index=False)

    print("\n=== summary (mean per scale x supervisor) ===")
    with pd.option_context("display.width", 140):
        print(df.groupby(["scale", "supervisor"])[METRICS].mean().round(3))
    print(f"\nrows -> {RESULTS / 'headroom_probe.csv'}")


if __name__ == "__main__":
    main()
