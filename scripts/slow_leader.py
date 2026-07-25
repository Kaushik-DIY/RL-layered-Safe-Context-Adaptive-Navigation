"""Slow-leader probe: does the trained supervisor ANTICIPATE, or does the filter
do all the work? (Proposed as the honest discriminator after the G4 analysis:
the corridor min-dist criterion cannot distinguish these, this scenario can.)

Setup: a narrow corridor (1.2 m -- too tight to overtake) with one pedestrian
2.5 m AHEAD walking the SAME direction at ~0.10 m/s, slower than the robot's
0.26 m/s. The robot must close in, slow down, and follow. Two supervisors drive
the IDENTICAL MPC+CBF stack on paired scenario seeds:

    trained    : the stage-C PPO policy (deterministic)
    always-max : constant [v_max_high, d_margin_low]  (the S5 adversary)

If the policy learned anticipation it starts shedding speed EARLIER (larger
deceleration-onset distance) and hands the filter less work (lower intervention
rate, fewer protective stops). If it saturates like in the eval corridors, the
two columns come out identical -- a clean negative result.

    python scripts/slow_leader.py            # 20 paired episodes + trace figure
    python scripts/slow_leader.py 5          # quicker
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_policy import load_model  # noqa: E402  (cross-numpy custom_objects loader)

from core.common.params import RobotParams, load_yaml  # noqa: E402
from core.rl.nav_env import NavEnv  # noqa: E402
from core.sim2d.pedestrians import SfmParams, SocialForceCrowd  # noqa: E402
from core.sim2d.scenarios import ScenarioSpec  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results"
MODEL = Path(__file__).resolve().parents[1] / "experiments" / "models" / "ppo_C_s0_final.zip"

V_CRUISE = 0.25     # m/s  robot counts as "at cruise" above this
V_ONSET = 0.22      # m/s  deceleration onset = first drop below this after cruising


def slow_leader(seed: int) -> ScenarioSpec:
    """Narrow corridor, one slow pedestrian ahead walking the same direction."""
    rng = np.random.default_rng(seed)
    half = 0.6                                     # 1.2 m corridor: no overtaking
    walls = np.array([[-0.5, -half, 8.0, -half], [-0.5, half, 8.0, half]])
    y0 = rng.uniform(-0.05, 0.05)
    speed = rng.uniform(0.08, 0.12)                # slower than the robot's 0.26
    crowd = SocialForceCrowd(SfmParams.from_yaml(), [[2.5, y0]], [[9.0, y0]],
                             [speed], walls=walls, rng=rng)
    return ScenarioSpec("slow_leader", seed, np.array([0.0, 0.0, 0.0]),
                        np.array([6.5, 0.0]), walls, np.zeros((0, 3)), crowd)


def run_episode(env: NavEnv, policy, seed: int):
    obs, _ = env.reset(seed=seed)
    done, ep = False, None
    while not done:
        obs, _, term, trunc, info = env.step(policy(obs))
        done = term or trunc
    return info["episode_metrics"], list(env.trajectory)


def onset_distance(traj) -> float:
    """d_human at the first sustained speed drop after reaching cruise (m)."""
    cruised = False
    for r in traj:
        if r["v_applied"] >= V_CRUISE:
            cruised = True
        elif cruised and r["v_applied"] < V_ONSET:
            return r["d_human"]
    return np.nan


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    seed_base = load_yaml("scenarios")["seed_base"] + 2000   # disjoint from batteries
    robot = RobotParams.from_yaml()

    model = load_model(str(MODEL))
    supervisors = {
        "trained": lambda obs: model.predict(obs, deterministic=True)[0],
        "always-max": lambda obs: np.array([0.26, 0.30]),
    }

    env = NavEnv(scenario_sampler=lambda rng: slow_leader(int(rng.integers(2 ** 31))),
                 use_cbf=True, record=True)

    rows, traces = [], {}
    for name, pol in supervisors.items():
        for i in range(n):
            ep, traj = run_episode(env, pol, seed_base + i)
            ep.update(supervisor=name, episode=i, onset_d=onset_distance(traj))
            rows.append(ep)
            if i == 0:
                traces[name] = traj
        sub = [r for r in rows if r["supervisor"] == name]
        print(f"[{name:10s}] {sum(r['success'] for r in sub)}/{n} success  "
              f"interv_rate {np.mean([r['intervention_rate'] for r in sub]):.3f}  "
              f"pstops/ep {np.mean([r['protective_stops'] for r in sub]):.2f}")

    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "slow_leader.csv", index=False)

    print("\n=== slow-leader probe (paired seeds, identical MPC+CBF stack) ===")
    print(f"{'metric':26s} {'trained':>10s} {'always-max':>11s}")
    for key, fmt in [("success", ".2f"), ("time_to_goal", ".1f"),
                     ("onset_d", ".3f"), ("intervention_rate", ".3f"),
                     ("mean_intervention", ".4f"), ("protective_stops", ".2f"),
                     ("full_stops", ".2f"), ("rms_jerk", ".2f"),
                     ("min_human_dist", ".3f"), ("min_h", ".3f"),
                     ("violation_steps", ".2f"), ("energy", ".3f")]:
        a = df[df.supervisor == "trained"][key].mean()
        b = df[df.supervisor == "always-max"][key].mean()
        print(f"{key:26s} {a:>10{fmt}} {b:>11{fmt}}")

    # per-decision supervisor commands of the trained policy (saturation check)
    obs, _ = env.reset(seed=seed_base)
    vs, ds, done = [], [], False
    while not done:
        a = model.predict(obs, deterministic=True)[0]
        vs.append(float(a[0])); ds.append(float(a[1]))
        obs, _, term, trunc, _ = env.step(a)
        done = term or trunc
    print(f"\ntrained policy commands (ep 0): v_max_cmd mean {np.mean(vs):.3f} "
          f"range [{min(vs):.3f}, {max(vs):.3f}]   "
          f"d_margin_cmd mean {np.mean(ds):.3f} range [{min(ds):.3f}, {max(ds):.3f}]")

    # ---- figure: speed + distance vs time, both supervisors, paired seed 0 ----
    colors = {"trained": "C0", "always-max": "C1"}
    fig, (ax_v, ax_d) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for name, traj in traces.items():
        t = [r["t"] for r in traj]
        c = colors[name]
        ax_v.plot(t, [r["v_safe"] for r in traj], color=c, lw=2, label=name)
        ax_v.plot(t, [r["v_mpc"] for r in traj], color=c, lw=1, ls=":", alpha=0.6)
        ax_d.plot(t, [r["d_human"] for r in traj], color=c, lw=2, label=name)
    ax_v.set_ylabel("commanded v [m/s]")
    ax_v.set_title("Slow leader: filtered speed (solid) vs MPC request (dotted) -- "
                   "who initiates the slowdown?")
    ax_v.legend(loc="upper right")
    ax_d.axhline(0.4, color="C3", ls=":", lw=1, label="protective field 0.4 m")
    ax_d.axhline(0.3, color="C3", ls="--", lw=1, label="d_hard 0.3 m")
    ax_d.set_ylabel("distance to pedestrian [m]")
    ax_d.set_xlabel("time [s]")
    ax_d.set_title("Distance to the slow leader")
    ax_d.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(RESULTS / "slow_leader.png", dpi=120)
    print(f"\nfigure -> {RESULTS / 'slow_leader.png'}")
    print(f"rows   -> {RESULTS / 'slow_leader.csv'}")


if __name__ == "__main__":
    main()
