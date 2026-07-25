"""Render a side-by-side demo GIF/MP4 of the industrial blind-corner scenario:
LEFT an aggressive fixed supervisor (drives fast, breaches the stopping-distance
limit at the occluded corner), RIGHT an adaptive supervisor (slows before the
blind corner, stays safe) -- on the SAME seed, so the only difference is behavior.

This is the headline visual: at AMR speed the corner reveal (1.2 m) is inside the
2.5 m stopping envelope, so arriving fast is uncatchable -- only anticipatory
slowing is safe. Robots are drawn with their footprint + protective field; the
occluded pedestrian is greyed until the tracker reveals it; a live speed +
barrier-h readout shows the violation (h<0, red) vs safe (h>=0, green).

    python scripts/render_demo.py            # writes experiments/results/industrial_demo.gif
    python scripts/render_demo.py --seed 5   # pick a clean illustrative seed
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Circle, Rectangle

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.common.platform import load_platform  # noqa: E402
from core.rl.nav_env import NavEnv  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results"
V_CORNER, SLOW = 0.6, 4.0


def roll(seed: int, kind: str):
    """Run one industrial blind-corner episode; return the full per-step record
    plus the ground-truth pedestrian track (record only stores nearest human)."""
    p = load_platform("industrial")
    env = NavEnv(scenarios=["blind_corner"], scenario_platform="industrial",
                 use_cbf=True, robot=p.robot, mpc=p.mpc, cbf=p.cbf, rl=p.rl,
                 obs_version=p.obs_version, obs_scale=p.obs_scale, record=True)

    def aggressive(_o):
        return np.array([p.robot.v_max, 0.30])

    def adaptive(_o):
        x, y, th = env.s[0], env.s[1], env.s[2]
        v = p.robot.v_max
        for pt in env.spec_.static_obstacles:
            if 0 < np.cos(th) * (pt[0]-x) + np.sin(th) * (pt[1]-y) \
               and np.hypot(pt[0]-x, pt[1]-y) < SLOW:
                v = V_CORNER
                break
        return np.array([v, 0.30])

    pol = aggressive if kind == "aggressive" else adaptive
    obs, _ = env.reset(seed=seed)
    ped = []
    done = False
    while not done:
        obs, _, term, trunc, info = env.step(pol(obs))
        ped.append(env.spec_.crowd.state()[0, :2].copy())   # the emerging worker
        done = term or trunc
    return env.spec_, env.trajectory, np.array(ped), p, info["episode_metrics"]


def draw_static(ax, spec, p, title):
    ax.set_aspect("equal")
    for w in spec.walls:
        ax.plot([w[0], w[2]], [w[1], w[3]], color="#444", lw=3)
    ax.plot(*spec.goal, "*", color="#2FA84F", ms=20, zorder=3)
    ax.set_xlim(-0.5, spec.goal[0] + 1.0)
    ax.set_ylim(-2.2, 3.2)
    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_xticks([]); ax.set_yticks([])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1005000)
    ap.add_argument("--mp4", action="store_true", help="also try an .mp4 (needs ffmpeg)")
    args = ap.parse_args()

    runs = {k: roll(args.seed, k) for k in ("aggressive", "adaptive")}
    T = min(len(runs["aggressive"][1]), len(runs["adaptive"][1]))
    rr = float(runs["aggressive"][3].robot.robot_radius)
    pr = float(runs["aggressive"][3].cbf.protective_radius)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    arts = {}
    for ax, kind in zip(axes, ("aggressive", "adaptive")):
        spec, traj, ped, p, ep = runs[kind]
        label = ("AGGRESSIVE fixed supervisor  (full speed into the blind corner)"
                 if kind == "aggressive" else
                 "ADAPTIVE supervisor  (slows before the blind corner)")
        draw_static(ax, spec, p, label)
        body = Circle((0, 0), rr, color="#3B6FE0", zorder=5)
        field = Circle((0, 0), pr, color="#E5A02E", fill=False, ls=":", lw=1.5)
        pedd = Circle((0, 0), 0.25, color="#E1575A", zorder=5)
        ax.add_patch(body); ax.add_patch(field); ax.add_patch(pedd)
        txt = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top",
                      fontsize=11, family="monospace",
                      bbox=dict(boxstyle="round", fc="white", ec="#ccc"))
        arts[kind] = (traj, ped, body, field, pedd, txt)

    def update(k):
        out = []
        for kind, (traj, ped, body, field, pedd, txt) in arts.items():
            i = min(k, len(traj) - 1)
            r = traj[i]
            body.center = (r["x"], r["y"]); field.center = (r["x"], r["y"])
            pedd.center = tuple(ped[min(i, len(ped) - 1)])
            h = r["h"]
            state = "VIOLATION h<0" if h < 0 else "safe"
            body.set_color("#E1575A" if h < 0 else "#3B6FE0")
            txt.set_text(f"t={r['t']:4.1f}s\nv={r['v_applied']:.2f} m/s\n"
                         f"h={h:+.2f} m  {state}")
            out += [body, field, pedd, txt]
        return out

    anim = animation.FuncAnimation(fig, update, frames=T + 8, interval=80, blit=True)
    fig.suptitle("Industrial AMR at a blind corner (1.5 m/s, 2.5 m stopping distance): "
                 "reveal < stopping distance, so only anticipation is safe",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    gif = RESULTS / "industrial_demo.gif"
    anim.save(gif, writer=animation.PillowWriter(fps=12))
    print(f"GIF -> {gif}")
    for kind in ("aggressive", "adaptive"):
        ep = runs[kind][4]
        print(f"  {kind:11s}: viol_steps={ep['violation_steps']}  min_h={ep['min_h']:+.3f}"
              f"  min_dist={ep['min_human_dist']:.2f}  t={ep['time_to_goal']:.1f}s")
    if args.mp4:
        try:
            anim.save(str(RESULTS / "industrial_demo.mp4"),
                      writer=animation.FFMpegWriter(fps=12))
            print(f"MP4 -> {RESULTS / 'industrial_demo.mp4'}")
        except Exception as e:
            print("mp4 skipped:", type(e).__name__)


if __name__ == "__main__":
    main()
