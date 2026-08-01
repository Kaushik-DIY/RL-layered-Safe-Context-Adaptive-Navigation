"""Known limitation, measured: the supervisor's speed cap is a HARD MPC bound it can slam.

Documented rather than fixed (the trained policy learned against this interface, so
changing it invalidates every downstream artifact until re-validated). Recorded here so the
number is reproducible if the question is ever asked.

The defect. `v_max_cmd` enters the MPC as a hard bound `v_k <= v_max_cmd`
(`core/mpc/mpc_controller.py:8`), while the same program also carries the hard rate limit
`|dv| <= a_max_mpc*dt`. The supervisor re-decides every `decision_every` steps, so within
one window the plant's authority to shed speed is only `a_max_mpc * dt * decision_every`
= 0.6 * 0.1 * 5 = 0.30 m/s. When the policy drops its cap by more than that, the two hard
constraints are jointly infeasible and IPOPT returns its best infeasible iterate -- a
command that violates the cap it was given.

Consequences, all measured below: an inflated hard-brake / jerk count for the RL arm, and
a hard bound that is not in fact hard in a few percent of steps. It is NOT a safety
problem: the CBF is downstream of the MPC and enforces the stopping-distance constraint on
whatever the MPC emits, which is why arm C's breaches do not trace back to these steps.

The fix, when it is taken: clamp the commanded cap to the reachable set before it reaches
the solver, `v_max_eff = max(v_max_cmd, v_prev - a_max_mpc*dt)`, so "slow down" becomes a
ramp at the physical limit instead of an infeasible step. Requires re-running the
evaluation, the offline showcase gate and the Gazebo demo.

    PYTHONPATH=$PWD .venv-navrl/bin/python scripts/audit_cap_interface.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core.common.platform import load_platform

RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results"


def main() -> None:
    plat = load_platform("industrial")
    dt, a_mpc = plat.robot.dt, plat.robot.a_max_mpc
    window = a_mpc * dt * plat.rl.decision_every
    tr = pd.read_csv(RESULTS / "three_arms_traces.csv.gz")

    print(f"plant authority to shed speed in one decision window: "
          f"a_max_mpc {a_mpc} * dt {dt} * decision_every {plat.rl.decision_every} "
          f"= {window:.2f} m/s\n")

    print("=== MPC commands that violate their own hard v <= v_max_cmd bound ===")
    for arm, s in tr.groupby("arm"):
        bad = s.v_mpc > s.v_max_cmd + 1e-3
        print(f"  {arm:16s} {int(bad.sum()):6d}/{len(s):6d} steps "
              f"({100 * bad.mean():5.2f} %)   worst excess "
              f"{float((s.v_mpc - s.v_max_cmd).max()):.3f} m/s")

    print("\n=== cap changes larger than the window authority ===")
    for arm, s in tr.groupby("arm"):
        d = np.concatenate([np.abs(np.diff(g.sort_values("step").v_max_cmd.to_numpy()))
                            for _, g in s.groupby(["scenario", "episode"])])
        d = d[d > 1e-6]
        if not len(d):
            print(f"  {arm:16s} cap never changes")
            continue
        print(f"  {arm:16s} {len(d):5d} changes   median {np.median(d):.3f}   "
              f"p90 {np.percentile(d, 90):.3f}   max {d.max():.3f}   "
              f"exceeding {window:.2f}: {100 * (d > window).mean():.0f} %")

    print("\n=== commanded speed drops faster than a_brake allows ===")
    for arm, s in tr.groupby("arm"):
        n = tot = 0
        for _, g in s.groupby(["scenario", "episode"]):
            dv = -np.diff(g.sort_values("step").v_safe.to_numpy())
            n += int((dv > plat.cbf.a_brake * dt + 1e-4).sum())
            tot += len(dv)
        print(f"  {arm:16s} {n:6d}/{tot:6d} steps ({100 * n / max(tot, 1):5.2f} %)")


if __name__ == "__main__":
    main()
