"""Evaluate a trained supervisor policy as system S4 (plan 4.1) + Gate G4 check.

Runs the policy through the IDENTICAL seeded battery as the baselines
(scripts/run_baselines.py: same scenarios, same paired seeds `seed_base + i`,
same NavEnv code path, clean conditions -- no domain randomization), so every
S4-vs-S1/S2 difference is attributable to the policy.

    python scripts/eval_policy.py experiments/models/ppo_B_s0_final.zip        # 100 eps
    python scripts/eval_policy.py experiments/models/ppo_B_s0_final.zip 5      # smoke
    python scripts/eval_policy.py <model> --no-cbf                             # S3 ablation

Gate G4 (plan sec. 5, after week-4 training): on scenario 1 (corridor_passby),
beat S1 on time-to-goal AND S2 on min-human-distance. References come from
experiments/results/baselines_2d.csv (the definitive post-barrier-fix battery).

Outputs: experiments/results/{s4|s3}_2d.csv + comparison table + G4 verdict.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from core.common.observation import obs_dim
from core.common.params import RlParams, load_yaml
from core.rl.nav_env import NavEnv
from core.sim2d.scenarios import SCENARIO_NAMES

RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results"


def load_model(path: str) -> PPO:
    """PPO.load with custom_objects: Kaggle saves under NumPy 2, whose pickled
    class layout (numpy._core.*) cannot unpickle under this venv's NumPy 1. The
    policy weights are torch tensors (version-agnostic); only spaces/schedules
    are pickled, so we substitute them instead of deserializing (SB3's
    documented cross-version workaround). Spaces mirror NavEnv exactly."""
    rl = RlParams.from_yaml()
    custom = {
        "observation_space": gym.spaces.Box(
            -np.inf, np.inf, (obs_dim(rl.K_nearest),), np.float32),
        "action_space": gym.spaces.Box(
            np.array([rl.v_max_low, rl.d_margin_low], dtype=np.float32),
            np.array([rl.v_max_high, rl.d_margin_high], dtype=np.float32)),
        "lr_schedule": lambda _: 3e-4,
        "clip_range": lambda _: 0.2,
        # training-resume state pickled as raw NumPy-2 arrays -- irrelevant for
        # inference, so substitute rather than unpickle:
        "_last_obs": None,
        "_last_episode_starts": None,
        "ep_info_buffer": None,
        "ep_success_buffer": None,
    }
    return PPO.load(path, device="cpu", custom_objects=custom)


def run_policy_battery(model, n_episodes, seed_base, use_cbf=True):
    rows = []
    for scenario in SCENARIO_NAMES:
        env = NavEnv(scenarios=[scenario], use_cbf=use_cbf)   # clean: no DR
        for i in range(n_episodes):
            obs, _ = env.reset(seed=seed_base + i)   # paired with the baselines
            ep = None
            while ep is None:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, info = env.step(action)
                if term or trunc:
                    ep = info["episode_metrics"]
            ep.update(scenario=scenario, episode=i)
            rows.append(ep)
        done = sum(r["success"] for r in rows if r["scenario"] == scenario)
        print(f"  {scenario:24s}: {done}/{n_episodes} success")
    return rows


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    use_cbf = "--no-cbf" not in sys.argv
    system = "S4" if use_cbf else "S3"
    model_path = args[0]
    cfg = load_yaml("scenarios")
    n_episodes = int(args[1]) if len(args) > 1 else cfg["episodes_2d"]
    seed_base = cfg["seed_base"]

    model = load_model(model_path)
    print(f"[{system}] {model_path}  ({n_episodes} eps/scenario, cbf={use_cbf})")
    t0 = time.time()
    rows = run_policy_battery(model, n_episodes, seed_base, use_cbf=use_cbf)
    df = pd.DataFrame(rows)
    df.insert(0, "system", system)
    out = RESULTS / f"{system.lower()}_2d.csv"
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    # ---------------- comparison vs the baseline battery ----------------
    base = pd.read_csv(RESULTS / "baselines_2d.csv")
    both = pd.concat([base, df], ignore_index=True)
    summary = both.groupby(["system", "scenario"]).agg(
        success=("success", "mean"),
        collisions=("collision", "sum"),
        violations=("violation_steps", lambda s: (s > 0).sum()),
        t_goal=("time_to_goal", "mean"),
        min_dist=("min_human_dist", "mean"),
        interv_rate=("intervention_rate", "mean"),
        pstops=("protective_stops", "mean"),
        energy=("energy", "mean"),
        intrusion=("intrusion_time", "mean"),
    ).round(3)
    print(f"\n{summary.to_string()}")

    # ------------------------------- Gate G4 -----------------------------
    sc = "corridor_passby"
    s1_t = base[(base.system == "S1") & (base.scenario == sc)]["time_to_goal"].mean()
    s2_d = base[(base.system == "S2") & (base.scenario == sc)]["min_human_dist"].mean()
    s4 = df[df.scenario == sc]
    s4_t, s4_d = s4["time_to_goal"].mean(), s4["min_human_dist"].mean()
    beat_t, beat_d = s4_t < s1_t, s4_d > s2_d
    print(f"\nGate G4 (scenario 1 = {sc}):")
    print(f"  time_to_goal : {system} {s4_t:6.2f} s  vs S1 {s1_t:6.2f} s  "
          f"-> {'BEAT' if beat_t else 'MISS'}")
    print(f"  min_distance : {system} {s4_d:6.3f} m  vs S2 {s2_d:6.3f} m  "
          f"-> {'BEAT' if beat_d else 'MISS'}")
    print(f"  Gate G4: {'PASS' if beat_t and beat_d else 'FAIL'}")
    print(f"\nrows -> {out}   wall {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
