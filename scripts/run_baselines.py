"""Week-3 baseline battery (plan sec. 5, Gate G3): S1/S2 over the five scenarios.

Runs the FIXED-TUNING systems from scenarios.yaml through the identical NavEnv
code path the RL system will use (same MPC, same metrics, same seeds), so the
eventual S4 comparison is apples-to-apples by construction:

    S1  MPC conservative (v_max 0.13, d_margin 1.0),  no CBF   safe-but-slow
    S2  MPC aggressive   (v_max 0.26, d_margin 0.35), no CBF   fast-but-unsafe

Episodes are PAIRED across systems: episode i of a scenario uses the same reset
seed for every system, so differences are attributable to tuning, not draw luck.

    python scripts/run_baselines.py            # full battery (100 eps/scenario)
    python scripts/run_baselines.py 5          # smoke run

Outputs: experiments/results/baselines_2d.csv          (one row per episode)
         experiments/results/baselines_2d_summary.csv  (mean +- std per cell)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core.common.params import load_yaml
from core.rl.nav_env import NavEnv
from core.sim2d.scenarios import SCENARIO_NAMES

RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results"

# metrics shown in the printed summary (full set goes to the CSV)
HEADLINE = ["success", "collision", "time_to_goal", "min_human_dist",
            "violation_steps", "energy", "full_stops", "intrusion_time"]


def run_system(system_id: str, sys_cfg: dict, n_episodes: int, seed_base: int):
    env_cache: dict[str, NavEnv] = {}
    rows = []
    for scenario in SCENARIO_NAMES:
        if scenario not in env_cache:  # one env (one built NLP) per scenario
            env_cache[scenario] = NavEnv(scenarios=[scenario],
                                         use_cbf=bool(sys_cfg["cbf"]),
                                         fixed_params=(sys_cfg["v_max"],
                                                       sys_cfg["d_margin"]))
        env = env_cache[scenario]
        for i in range(n_episodes):
            env.reset(seed=seed_base + i)   # paired seeds across systems
            ep = None
            while ep is None:
                _, _, term, trunc, info = env.step(env.action_space.low)  # ignored
                if term or trunc:
                    ep = info["episode_metrics"]
            ep.update(system=system_id, scenario=scenario, episode=i,
                      v_max=sys_cfg["v_max"], d_margin=sys_cfg["d_margin"],
                      cbf=bool(sys_cfg["cbf"]))
            rows.append(ep)
        done = sum(r["success"] for r in rows if r["scenario"] == scenario
                   and r["system"] == system_id)
        print(f"  {system_id} {scenario:24s}: {done}/{n_episodes} success")
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby(["system", "scenario"]).agg(
        success_rate=("success", "mean"),
        collisions=("collision", "sum"),
        violations=("violation_steps", lambda s: (s > 0).sum()),
        t_goal_mean=("time_to_goal", "mean"),
        t_goal_std=("time_to_goal", "std"),
        min_dist_mean=("min_human_dist", "mean"),
        energy_mean=("energy", "mean"),
        full_stops_mean=("full_stops", "mean"),
        intrusion_mean=("intrusion_time", "mean"),
        solve_ms_median=("mpc_solve_ms_median", "median"),
    )
    return agg.round(3)


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    cfg = load_yaml("scenarios")
    n_episodes = n if n is not None else cfg["episodes_2d"]
    seed_base = cfg["seed_base"]
    systems = {k: v for k, v in cfg["systems"].items() if v.get("rl") is False}

    print(f"Baseline battery: {list(systems)} x {len(SCENARIO_NAMES)} scenarios "
          f"x {n_episodes} episodes (paired seeds from {seed_base})")
    t0 = time.time()
    rows = []
    for sid, scfg in systems.items():
        print(f"[{sid}] {scfg['desc']}")
        rows += run_system(sid, scfg, n_episodes, seed_base)
    df = pd.DataFrame(rows)

    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "baselines_2d.csv", index=False)
    summary = summarize(df)
    summary.to_csv(RESULTS / "baselines_2d_summary.csv")
    print(f"\n{summary.to_string()}")
    print(f"\nwall time: {(time.time() - t0) / 60:.1f} min")
    print(f"rows -> {RESULTS / 'baselines_2d.csv'}")
    print(f"summary -> {RESULTS / 'baselines_2d_summary.csv'}")


if __name__ == "__main__":
    main()
