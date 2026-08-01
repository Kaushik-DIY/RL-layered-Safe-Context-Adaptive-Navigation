"""Three-arm comparison: what a commissioned AMR pays for its safety, and what we pay.

The claim under test is NOT "the RL is safer". A certified AMR does not violate ISO --
that is the precondition for deploying it. It is compliant because a safety-rated scanner
trips a protective stop, and because an integrator hand-capped the speed at commissioning
so the stopping distance fits inside what the scanner can actually see. It buys compliance
by CRAWLING and STOPPING. The claim is that a learned supervisor buys the same certifiable
compliance far more cheaply.

    A  industrial reference   MPC + CBF, commissioned cap 0.50 m/s, speed-dependent
                              warning/protective fields with a latched protective stop
    B  rated speed            MPC + CBF at the platform's 1.5 m/s, no commissioning
                              -- shows WHY the commissioning conservatism exists
    C  ours                   MPC + CBF + the trained RL supervisor

Why 0.50 m/s is the right commissioned cap, and not an arbitrary handicap: at 0.50 m/s the
stopping distance is d_hard + d_stop(sigma*v) = 0.30 + 0.46 = 0.76 m, which fits inside the
1.2 m reveal distance at a blind corner. At the rated 1.5 m/s it is 2.53 m, which does not.
That is exactly the calculation an integrator does by hand, and the sweep confirms it:
`pareto_industrial.csv` has (v=0.5, m=0.9) at 100 % raw compliance across all five
scenarios. Arm A is that operating point plus the scanner behaviour.

Arm A's fields are sized at the COMMISSIONED speed and held fixed, which is how real
field switching works (a few discrete configured fields), not recomputed from the
instantaneous speed -- otherwise the field would shrink to nothing while stopped and the
robot would resume into the person it just stopped for.

Everything runs on the same paired seeds as `pareto_industrial.csv` and
`s4_industrial_full.csv` (seed_base + 5000), so the sanity gates in the plan hold:
arm B must reproduce the `always-max` row and arm C the `trained` row.

Per-step traces are recorded for every episode because the compliance statement needs the
filter-accountable / unavoidable split, which cannot be computed from episode summaries.

    python scripts/eval_three_arms.py            # 12 paired seeds, ~50 min
    python scripts/eval_three_arms.py 2 2        # smoke: 2 seeds, first 2 scenarios
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.cbf.cbf_filter import d_stop  # noqa: E402
from core.common.params import load_yaml  # noqa: E402
from core.demo import scanner_amr  # noqa: E402
from core.demo.scanner_amr import ScannerAMR  # noqa: E402
from core.common.platform import load_platform  # noqa: E402
from core.rl.nav_env import NavEnv  # noqa: E402
from core.sim2d.scenarios import (HOLDOUT_CLEAN, HOLDOUT_SCENARIO,  # noqa: E402
                                  INDUSTRIAL_SCENARIOS)

from eval_industrial import load_industrial  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results"
DEFAULT_MODEL = "experiments/models/ppo_ind_C_s0_full_final.zip"

# --- arm A: the commissioned machine (core/demo/scanner_amr.py) ---------------
A_SPEED, A_MARGIN, A_CREEP = scanner_amr.SPEED, scanner_amr.MARGIN, scanner_amr.CREEP

# --- arm B: the platform's rated operating point ------------------------------
B_SPEED, B_MARGIN = 1.50, 0.30

TRACE_COLS = ["t", "x", "y", "v_applied", "v_safe", "v_mpc", "h", "h_seen", "d_human",
              "v_los", "human_closing", "v_los_crit", "closing_crit",
              "v_max_cmd", "intervention", "protective_stop"]


def run_episode(env, arm, model, scanner, seed):
    """One mission. Returns (episode_metrics, trajectory)."""
    obs, _ = env.reset(seed=seed)
    if scanner is not None:
        scanner.reset()
    ep = None
    while ep is None:
        if arm == "A_commissioned":
            # fixed_params is set, so step() ignores the action -- drive the machine by
            # writing the commanded parameters directly (this is the only way to command
            # a true v = 0 stop; the RL action space floor is 0.1 m/s).
            env.v_max_cmd, env.d_margin_cmd = scanner(env.s[:2], env._tracked_humans())
            action = np.zeros(2, dtype=np.float32)
        elif arm == "B_rated":
            action = np.array([B_SPEED, B_MARGIN], dtype=np.float32)
        else:
            action = model.predict(obs, deterministic=True)[0]
        obs, _, term, trunc, info = env.step(action)
        if term or trunc:
            ep = info["episode_metrics"]
    return ep, env.trajectory


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    n_scen = int(sys.argv[2]) if len(sys.argv) > 2 else None
    plat = load_platform("industrial")
    seed_base = load_yaml("scenarios")["seed_base"] + 5000      # == Pareto sweep block

    # HOLDOUT_CLEAN carries the generalization claim (unseen combination, nothing
    # adversarial); HOLDOUT_SCENARIO is the same junction WITH a seeker and is reported
    # only as a probe, alongside `interferer`.
    scenarios = list(INDUSTRIAL_SCENARIOS) + [HOLDOUT_CLEAN, HOLDOUT_SCENARIO]
    if n_scen is not None:
        scenarios = scenarios[:n_scen]
    model = load_industrial(DEFAULT_MODEL, plat)
    scanner = ScannerAMR(plat)
    print(f"[three-arms] {n} paired seeds/scenario, {len(scenarios)} scenarios")
    print(f"  arm A commissioned {A_SPEED} m/s (field {scanner.r_prot:.2f} m)"
          f"  ->  creep {A_CREEP} m/s (field {scanner.r_creep:.2f} m)"
          f"  ->  stop;  warning field {scanner.r_warn:.2f} m")
    print(f"  arm B rated        {B_SPEED} m/s   (stopping distance "
          f"{plat.cbf.d_hard + d_stop(plat.cbf.sigma*B_SPEED, plat.cbf.tau, plat.cbf.a_brake):.2f} m)")
    print(f"  arm C {DEFAULT_MODEL}  ({model.num_timesteps} steps)\n")

    t0, rows, traces = time.time(), [], []
    for scenario in scenarios:
        for arm in ("A_commissioned", "B_rated", "C_rl"):
            # arm A needs fixed_params so step() ignores the action; B and C go through
            # the action path exactly as eval_industrial.py drives them, which is what
            # makes the "must reproduce always-max / trained" sanity gates meaningful.
            fixed = (A_SPEED, A_MARGIN) if arm == "A_commissioned" else None
            env = NavEnv(scenarios=[scenario], scenario_platform="industrial",
                         use_cbf=True, record=True, fixed_params=fixed,
                         robot=plat.robot, mpc=plat.mpc, cbf=plat.cbf, rl=plat.rl,
                         obs_version=plat.obs_version, obs_scale=plat.obs_scale)
            for i in range(n):
                ep, traj = run_episode(env, arm, model, scanner, seed_base + i)
                ep.update(arm=arm, scenario=scenario, episode=i)
                rows.append(ep)
                for k, r in enumerate(traj):
                    traces.append({"arm": arm, "scenario": scenario, "episode": i,
                                   "step": k, **{c: r.get(c) for c in TRACE_COLS}})
        sub = pd.DataFrame([r for r in rows if r["scenario"] == scenario])
        line = "   ".join(
            f"{a.split('_')[0]}: viol {int((g.violation_steps > 0).sum())}/{len(g)}"
            f" t {g.time_to_goal.mean():.1f}s"
            for a, g in sub.groupby("arm"))
        print(f"  {scenario:24s} {line}   [{(time.time()-t0)/60:.0f} min]")

    RESULTS.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "three_arms.csv", index=False)
    pd.DataFrame(traces).to_csv(RESULTS / "three_arms_traces.csv.gz",
                                index=False, compression="gzip")

    train = df[df.scenario != HOLDOUT_SCENARIO]
    print("\n=== per-arm totals, training scenarios (holdout excluded) ===")
    print(train.groupby("arm").agg(
        success=("success", "mean"),
        raw_viol_eps=("violation_steps", lambda s: (s > 0).mean()),
        collisions=("collision", "sum"),
        t_goal=("time_to_goal", "mean"),
        pstops=("protective_stops", "mean"),
        full_stops=("full_stops", "mean"),
        energy=("energy", "mean"),
    ).round(3).to_string())
    print("\nNOTE: raw_viol_eps counts every episode with h < 0, INCLUDING breaches where "
          "the robot was already braking maximally.\n      The compliance statement is the "
          "filter-accountable subset -- run scripts/analyse_three_arms.py.")
    print(f"\nrows -> {RESULTS / 'three_arms.csv'}   "
          f"traces -> {RESULTS / 'three_arms_traces.csv.gz'}   "
          f"wall {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
