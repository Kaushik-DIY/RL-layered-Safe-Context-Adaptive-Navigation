"""Guaranteed-breach demo (P1.4): blind corner at industrial scale.

The physics: the corner reveals the emerging pedestrian at reveal ~1.2 m, but the
platform's stopping envelope at cruise is d_stop(1.1*1.5) ~2.5 m. The filter
CANNOT save what it cannot see with room to brake -- an always-max supervisor is
structurally guaranteed to breach. Only slowing BEFORE the corner is clean:
d_stop(1.1*v) <= 1.2 m  =>  v ~<= 0.7 m/s at the opening.

Supervisors (identical MPC+CBF stack, paired seeds):
    always-max   : [v_max, 0.30]      -- arrives at the corner at 1.5 m/s
    corner-aware : slows to v_corner when a mapped constriction post is ahead
                   within slow_range (the v2 `post_ahead` feature, hand-rule form)

This is run BEFORE any training spend: it validates the killer figure and defines
the industrial policy's training target (learn the corner-aware behavior + beat
its time cost).

    python scripts/corner_breach_demo.py          # 30 paired seeds + trace figure
    python scripts/corner_breach_demo.py 5
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

from core.cbf.cbf_filter import d_stop  # noqa: E402
from core.common.platform import load_platform  # noqa: E402
from core.common.params import load_yaml  # noqa: E402
from core.rl.nav_env import NavEnv  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results"

V_CORNER = 0.6      # m/s  d_stop(1.1*0.6) ~ 0.68 m < 1.2 m reveal -> clean
SLOW_RANGE = 4.0    # m    begin slowing this far from the constriction post


def make_supervisors(env: NavEnv, v_max: float):
    def always_max(_obs):
        return np.array([v_max, 0.30])

    def corner_aware(_obs):
        # hand-rule form of the v2 post_ahead feature: mapped constriction ahead
        x, y, th = env.s[0], env.s[1], env.s[2]
        posts = env.spec_.static_obstacles
        v = v_max
        if len(posts):
            cos_t, sin_t = np.cos(th), np.sin(th)
            for p in posts:
                along = cos_t * (p[0] - x) + sin_t * (p[1] - y)
                if 0.0 < along and np.hypot(p[0] - x, p[1] - y) < SLOW_RANGE:
                    v = V_CORNER
                    break
        return np.array([v, 0.30])

    return {"always-max": always_max, "corner-aware": corner_aware}


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    p = load_platform("industrial")
    seed_base = load_yaml("scenarios")["seed_base"] + 4000
    D = d_stop(p.cbf.sigma * p.robot.v_max, p.cbf.tau, p.cbf.a_brake)
    Dc = d_stop(p.cbf.sigma * V_CORNER, p.cbf.tau, p.cbf.a_brake)
    print(f"industrial blind corner: reveal 1.2 m | d_stop(cruise {p.robot.v_max}) "
          f"= {D:.2f} m (>{1.2}: CANNOT stop in time) | d_stop({V_CORNER}) "
          f"= {Dc:.2f} m (<1.2: clean)")

    env = NavEnv(scenarios=["blind_corner"], scenario_platform="industrial",
                 use_cbf=True,
                 robot=p.robot, mpc=p.mpc, cbf=p.cbf, rl=p.rl,
                 obs_version=p.obs_version, obs_scale=p.obs_scale, record=True)

    sups = make_supervisors(env, p.robot.v_max)
    rows, traces = [], {}
    for name, pol in sups.items():
        for i in range(n):
            obs, _ = env.reset(seed=seed_base + i)
            done = False
            while not done:
                obs, _, term, trunc, info = env.step(pol(obs))
                done = term or trunc
            ep = info["episode_metrics"]
            ep.update(supervisor=name, episode=i)
            rows.append(ep)
            if i == 0:
                traces[name] = list(env.trajectory)
        sub = [r for r in rows if r["supervisor"] == name]
        print(f"  [{name:12s}] succ {sum(r['success'] for r in sub)}/{n}"
              f"  coll {sum(r['collision'] for r in sub)}"
              f"  viol_eps {sum(r['violation_steps'] > 0 for r in sub)}"
              f"  min_h {min(r['min_h'] for r in sub):+.3f}"
              f"  min_d {min(r['min_human_dist'] for r in sub):.3f}"
              f"  t {np.nanmean([r['time_to_goal'] for r in sub]):.1f}s")

    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "corner_breach.csv", index=False)

    am = df[df.supervisor == "always-max"]
    ca = df[df.supervisor == "corner-aware"]
    print(f"\nviolation episodes: always-max {int((am.violation_steps > 0).sum())}/{n}"
          f"  corner-aware {int((ca.violation_steps > 0).sum())}/{n}")
    print(f"time cost of clean corners: {ca.time_to_goal.mean() - am.time_to_goal.mean():+.1f} s"
          f" ({ca.time_to_goal.mean():.1f} vs {am.time_to_goal.mean():.1f})")

    # ---- trace figure: speed vs x-position through the corner, paired seed 0 ----
    colors = {"always-max": "C1", "corner-aware": "C0"}
    spec = env.spec_
    open_x = 6.0                                  # industrial_geometry open_x
    fig, (ax_v, ax_d) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for name, traj in traces.items():
        xs = [r["x"] for r in traj]
        ax_v.plot(xs, [r["v_applied"] for r in traj], color=colors[name],
                  lw=2, label=name)
        ax_d.plot(xs, [r["h"] for r in traj], color=colors[name], lw=2, label=name)
    ax_v.axvline(open_x, color="gray", ls=":", lw=1)
    ax_v.text(open_x, ax_v.get_ylim()[1] * 0.95, " blind corner", va="top",
              color="gray", fontsize=9)
    ax_v.set_ylabel("speed [m/s]")
    ax_v.set_title("Blind corner at AMR speed: arrive fast and the filter cannot "
                   "save you -- reveal (1.2 m) < stopping envelope (2.5 m)")
    ax_v.legend(loc="lower left")
    ax_d.axhline(0.0, color="C3", ls="--", lw=1, label="violation boundary h=0")
    ax_d.axvline(open_x, color="gray", ls=":", lw=1)
    ax_d.set_xlabel("x position [m]")
    ax_d.set_ylabel("stopping-distance barrier h [m]")
    ax_d.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(RESULTS / "corner_breach.png", dpi=120)
    print(f"figure -> {RESULTS / 'corner_breach.png'}")
    print(f"rows   -> {RESULTS / 'corner_breach.csv'}")


if __name__ == "__main__":
    main()
