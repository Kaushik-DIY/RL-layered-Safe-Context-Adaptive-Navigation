"""Industrial fixed-tuning Pareto sweep (P1.5): the frontier for money-plot #2.

Grid of FIXED (v_max_cmd, d_margin_cmd) tunings driven through the identical
MPC+CBF industrial stack over the industrial scenario suite, paired seeds. The
resulting safety-vs-throughput scatter is the frontier the plan's money-plot #2
requires; the heuristic / trained-policy points are produced by their own scripts
(corner_breach_demo, headroom_probe, the P4 battery) on the SAME seeds.

Speeds cover the tunable range of the 1.5 m/s MiR-class platform (grounding table
in robot.yaml); margins from the CBF floor to a wide berth.

    python scripts/pareto_sweep.py             # 5 speeds x 3 margins x 6 scenarios x 20 seeds
    python scripts/pareto_sweep.py 3 2         # smoke: 3 seeds, first 2 scenarios
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.common.params import load_yaml  # noqa: E402
from core.common.platform import load_platform  # noqa: E402
from core.rl.nav_env import NavEnv  # noqa: E402
from core.sim2d.scenarios import INDUSTRIAL_SCENARIOS  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results"

SPEEDS = (0.5, 0.75, 1.0, 1.25, 1.5)     # m/s commanded cap (platform tops at 1.5)
MARGINS = (0.3, 0.6, 0.9)                # m   MPC human-margin command


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    n_scen = int(sys.argv[2]) if len(sys.argv) > 2 else len(INDUSTRIAL_SCENARIOS)
    scenarios = INDUSTRIAL_SCENARIOS[:n_scen]
    p = load_platform("industrial")
    seed_base = load_yaml("scenarios")["seed_base"] + 5000

    t0 = time.time()
    rows = []
    for v in SPEEDS:
        for m in MARGINS:
            for scenario in scenarios:
                env = NavEnv(scenarios=[scenario], scenario_platform="industrial",
                             use_cbf=True, fixed_params=(v, m),
                             robot=p.robot, mpc=p.mpc, cbf=p.cbf, rl=p.rl,
                             obs_version=p.obs_version, obs_scale=p.obs_scale)
                for i in range(n):
                    obs, _ = env.reset(seed=seed_base + i)
                    done = False
                    while not done:
                        obs, _, term, trunc, info = env.step(
                            np.array([v, m]))          # ignored (fixed_params)
                        done = term or trunc
                    ep = info["episode_metrics"]
                    ep.update(v_cmd=v, margin_cmd=m, scenario=scenario, episode=i)
                    rows.append(ep)
            sub = [r for r in rows if r["v_cmd"] == v and r["margin_cmd"] == m]
            print(f"[v={v:4.2f} m={m:3.1f}] "
                  f"succ {np.mean([r['success'] for r in sub]):.2f}  "
                  f"viol_eps {sum(r['violation_steps'] > 0 for r in sub)}"
                  f"/{len(sub)}  "
                  f"t {np.nanmean([r['time_to_goal'] for r in sub]):.1f}s  "
                  f"[{(time.time()-t0)/60:.0f} min]", flush=True)

    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "pareto_industrial.csv", index=False)
    print(f"\nrows -> {RESULTS / 'pareto_industrial.csv'}   "
          f"wall {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
