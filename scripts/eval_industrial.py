"""P4 industrial evaluation: does the TRAINED supervisor beat every fixed tuning?

Runs the trained industrial policy + the three hand-built supervisors through the
6-scenario industrial suite on the SAME paired seeds as the Pareto sweep
(seed_base + 5000), so every point is directly comparable to
`pareto_industrial.csv`. One CSV out; the money plots (P4) read it + the frontier.

    python scripts/eval_industrial.py experiments/models/ppo_ind_C_s0_final.zip
    python scripts/eval_industrial.py <model> 5      # smoke
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stable_baselines3 import PPO  # noqa: E402

from core.cbf.cbf_filter import d_stop  # noqa: E402
from core.common.observation import obs_dim  # noqa: E402
from core.common.params import load_yaml  # noqa: E402
from core.common.platform import load_platform  # noqa: E402
from core.rl.nav_env import NavEnv  # noqa: E402
from core.sim2d.scenarios import INDUSTRIAL_SCENARIOS, HOLDOUT_SCENARIO  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results"


def load_industrial(path: str, p) -> PPO:
    """Cross-version tolerant load with the industrial (obs-v2, 1.5 m/s) spaces."""
    custom = {
        "observation_space": gym.spaces.Box(-np.inf, np.inf,
            (obs_dim(p.rl.K_nearest, p.obs_version),), np.float32),
        "action_space": gym.spaces.Box(
            np.array([p.rl.v_max_low, p.rl.d_margin_low], np.float32),
            np.array([p.rl.v_max_high, p.rl.d_margin_high], np.float32)),
        "lr_schedule": lambda _: 3e-4, "clip_range": lambda _: 0.2,
        "_last_obs": None, "_last_episode_starts": None,
        "ep_info_buffer": None, "ep_success_buffer": None,
    }
    return PPO.load(path, device="cpu", custom_objects=custom)


def hand_supervisors(env: NavEnv, p):
    """always-max / fixed-mid / density-heuristic / corner-aware -- the same rules
    the earlier probes used, so the comparison is apples-to-apples."""
    v_hi = p.robot.v_max
    D = d_stop(p.cbf.sigma * v_hi, p.cbf.tau, p.cbf.a_brake)
    V_CORNER, SLOW_RANGE = 0.6, 4.0

    def always_max(_o):
        return np.array([v_hi, 0.30])

    def fixed_mid(_o):
        return np.array([0.55 * v_hi, 0.50])

    def density(_o):
        humans = env._tracked_humans()
        if len(humans) == 0:
            return np.array([v_hi, 0.30])
        d = np.hypot(humans[:, 0] - env.s[0], humans[:, 1] - env.s[1])
        n_close = int((d < 1.5 * D).sum())
        dn = float(d.min())
        v = v_hi if dn > 3 * D else (0.6 * v_hi if dn > 1.5 * D else 0.3 * v_hi)
        return np.array([v, float(np.clip(0.30 + 0.10 * n_close, 0.30, 0.80))])

    def corner_aware(_o):
        x, y, th = env.s[0], env.s[1], env.s[2]
        v = v_hi
        for pt in env.spec_.static_obstacles:
            along = np.cos(th) * (pt[0] - x) + np.sin(th) * (pt[1] - y)
            if 0.0 < along and np.hypot(pt[0] - x, pt[1] - y) < SLOW_RANGE:
                v = V_CORNER
                break
        return np.array([v, 0.30])

    return {"always-max": always_max, "fixed-mid": fixed_mid,
            "density": density, "corner-aware": corner_aware}


def main() -> None:
    args = [a for a in sys.argv[1:]]
    model_path = args[0]
    n = int(args[1]) if len(args) > 1 else 30
    p = load_platform("industrial")
    seed_base = load_yaml("scenarios")["seed_base"] + 5000    # == Pareto sweep block
    model = load_industrial(model_path, p)
    print(f"[P4] {model_path}  ({model.num_timesteps} steps)  {n} eps/scenario\n")

    # include the held-out scenario to test generalization
    scenarios = INDUSTRIAL_SCENARIOS + (HOLDOUT_SCENARIO,)
    t0, rows = time.time(), []
    for scenario in scenarios:
        env = NavEnv(scenarios=[scenario], scenario_platform="industrial",
                     use_cbf=True, robot=p.robot, mpc=p.mpc, cbf=p.cbf, rl=p.rl,
                     obs_version=p.obs_version, obs_scale=p.obs_scale)
        sups = {"trained": lambda o: model.predict(o, deterministic=True)[0],
                **hand_supervisors(env, p)}
        for name, pol in sups.items():
            for i in range(n):
                obs, _ = env.reset(seed=seed_base + i)
                done = False
                while not done:
                    obs, _, term, trunc, info = env.step(pol(obs))
                    done = term or trunc
                ep = info["episode_metrics"]
                ep.update(supervisor=name, scenario=scenario, episode=i)
                rows.append(ep)
        sub = [r for r in rows if r["scenario"] == scenario]
        line = "  ".join(
            f"{nm[:4]}:v{sum(r['violation_steps']>0 for r in sub if r['supervisor']==nm)}"
            f"/t{np.nanmean([r['time_to_goal'] for r in sub if r['supervisor']==nm]):.0f}"
            for nm in sups)
        print(f"  {scenario:22s} (viol-eps/t_goal)  {line}")

    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "s4_industrial.csv", index=False)
    print(f"\n=== per-supervisor totals over the 6 training scenarios "
          f"(holdout excluded) ===")
    train_df = df[df.scenario != HOLDOUT_SCENARIO]
    summ = train_df.groupby("supervisor").agg(
        success=("success", "mean"),
        viol_eps=("violation_steps", lambda s: (s > 0).mean()),
        collisions=("collision", "sum"),
        t_goal=("time_to_goal", "mean"),
        pstops=("protective_stops", "mean"),
        jerk=("rms_jerk", "mean"),
        energy=("energy", "mean"),
    ).round(3)
    print(summ.to_string())
    print(f"\nrows -> {RESULTS / 's4_industrial.csv'}   wall {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
