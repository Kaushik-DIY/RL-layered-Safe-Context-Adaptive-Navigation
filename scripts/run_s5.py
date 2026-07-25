"""S5 — the guarantee stress test (plan 4.1): MPC + ADVERSARIAL/RANDOM supervisor
+ CBF, through the identical seeded battery as S1/S2/S4.

This is the headline experiment: feed the full stack a supervisor that actively
tries to hurt it --

    adversarial : always [v_max, d_margin floor]  (max speed, minimum margin)
    random      : uniform draw over the action box every 2 Hz decision

-- and show ZERO stopping-distance violations and zero collisions anyway. One
table, whole thesis proven (the G2 batteries verified the bare filter; S5
verifies the DEPLOYED stack: RL slot -> MPC -> CBF -> sim, adversary included).

    python scripts/run_s5.py            # full battery, both adversaries
    python scripts/run_s5.py 5          # smoke

Outputs: experiments/results/s5_2d.csv + summary + PASS/FAIL headline.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core.common.params import RlParams, load_yaml
from core.rl.nav_env import NavEnv
from core.sim2d.scenarios import SCENARIO_NAMES

RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results"


def make_adversary(kind: str, rl: RlParams, rng: np.random.Generator):
    lo = np.array([rl.v_max_low, rl.d_margin_low])
    hi = np.array([rl.v_max_high, rl.d_margin_high])
    if kind == "adversarial":       # max speed, margin at the hard floor
        a = np.array([rl.v_max_high, rl.d_margin_low])
        return lambda obs: a
    if kind == "random":
        return lambda obs: rng.uniform(lo, hi)
    raise ValueError(kind)


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    cfg = load_yaml("scenarios")
    n_episodes = n if n is not None else cfg["episodes_2d"]
    seed_base = cfg["seed_base"]
    rl = RlParams.from_yaml()

    t0 = time.time()
    rows = []
    for kind in ("adversarial", "random"):
        print(f"[S5:{kind}]")
        for scenario in SCENARIO_NAMES:
            env = NavEnv(scenarios=[scenario], use_cbf=True)
            for i in range(n_episodes):
                rng = np.random.default_rng(10_000 + i)   # adversary's own seed
                policy = make_adversary(kind, rl, rng)
                obs, _ = env.reset(seed=seed_base + i)    # paired with S1/S2/S4
                ep = None
                while ep is None:
                    obs, _, term, trunc, info = env.step(policy(obs))
                    if term or trunc:
                        ep = info["episode_metrics"]
                ep.update(system="S5", adversary=kind, scenario=scenario, episode=i)
                rows.append(ep)
            sub = [r for r in rows if r["adversary"] == kind and r["scenario"] == scenario]
            print(f"  {scenario:24s}: viol_eps={sum(r['violation_steps'] > 0 for r in sub)}"
                  f" coll={sum(r['collision'] for r in sub)}"
                  f" succ={sum(r['success'] for r in sub)}/{n_episodes}"
                  f" min_h={min(r['min_h'] for r in sub):+.3f}")
    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "s5_2d.csv", index=False)

    n_viol = int((df["violation_steps"] > 0).sum())
    n_coll = int(df["collision"].sum())
    print(f"\nS5 STRESS TEST over {len(df)} adversarial episodes:")
    print(f"  stopping-distance violation episodes : {n_viol}")
    print(f"  collisions                           : {n_coll}")
    print(f"  global min barrier h                 : {df['min_h'].min():+.4f}")
    print(f"  protective stops (total)             : {int(df['protective_stops'].sum())}")
    verdict = ("PASS — the filter refused the adversary, every time"
               if n_viol == 0 and n_coll == 0
               else "FAIL — investigate before any claim")
    print(f"\n  {verdict}")
    print(f"rows -> {RESULTS / 's5_2d.csv'}   wall {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
