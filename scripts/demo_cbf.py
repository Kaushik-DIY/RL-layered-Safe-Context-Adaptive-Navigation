"""Week-2 CBF safety-filter demo / visualization (Gate G2 made visible).

Runs ONE illustrative episode: an adversarial robot policy commanding FULL SPEED
toward the goal (it ignores humans entirely), with pedestrians on scripted paths.
The CBF filter must do all the safety work -- you watch it clamp the commanded
velocity so the stopping-distance barrier h(t) never crosses zero and the robot
never breaches d_hard.

    python scripts/demo_cbf.py                 # save a 4-panel figure (PNG)
    python scripts/demo_cbf.py --show          # also pop up an interactive window
    python scripts/demo_cbf.py --animate       # also save an animated GIF

No RL, no ROS -- just Layer 1 (CBF) on top of the 2D sim.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

import numpy as np

from core.cbf.cbf_filter import CbfFilter
from core.common.params import CbfParams, RobotParams
from core.sim2d.kinematic_sim import KinematicSim, wrap_angle

RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results"
GOAL = np.array([5.0, 0.0])

# Scripted pedestrians (the TB3 at 0.26 m/s is too slow for a moving pedestrian to
# ever be in its path at the right moment, so we script the interaction):
#   A: stands in the robot's path, then steps aside once the robot has yielded
#      -> a clean, repeatable "slow down, wait, proceed" demonstration.
#   B: an offset head-on passer for context (never forces a slowdown).
#   [x, y, vx, vy]
HUMANS0 = np.array([
    [2.6, 0.05, 0.0, 0.0],     # A: standing in the path
    [4.6, 0.55, -0.40, 0.0],   # B: head-on passer
])
A_STEPS_AWAY = 95              # step at which pedestrian A starts stepping aside
A_AWAY_VEL = np.array([0.20, 0.55])  # A walks forward-and-up, clearing the lane


def _script_humans(humans, t):
    """Apply the scripted pedestrian motion for step t (A waits, then steps aside)."""
    if t == A_STEPS_AWAY:
        humans[0, 2:] = A_AWAY_VEL


def run_episode(steps=260):
    robot = RobotParams.from_yaml()
    cbf = CbfParams.from_yaml()
    sim = KinematicSim(robot)
    filt = CbfFilter(robot, cbf)
    filt.reset()
    s = sim.reset([0.0, 0.0, 0.0])
    humans = HUMANS0.copy()

    log = {k: [] for k in ("t", "rx", "ry", "v_cmd", "v_safe", "h", "d", "pstop")}
    human_traj = []
    for t in range(steps):
        heading_err = wrap_angle(np.arctan2(GOAL[1] - s[1], GOAL[0] - s[0]) - s[2])
        omega = float(np.clip(2.0 * heading_err, robot.omega_min, robot.omega_max))
        u_mpc = np.array([robot.v_max, omega])          # adversary: full speed to goal
        u_safe, info = filt.filter(s, u_mpc, humans)

        log["t"].append(t * robot.dt)
        log["rx"].append(s[0]); log["ry"].append(s[1])
        log["v_cmd"].append(u_mpc[0]); log["v_safe"].append(u_safe[0])
        log["h"].append(info["h_min"]); log["d"].append(info["n_active"])
        log["pstop"].append(info["protective_stop"])
        human_traj.append(humans.copy())

        s = sim.step(u_safe)
        _script_humans(humans, t)
        humans[:, 0] += humans[:, 2] * robot.dt
        humans[:, 1] += humans[:, 3] * robot.dt
        # true min distance to a human (for the distance panel)
        log["d"][-1] = float(np.min(np.hypot(humans[:, 0] - s[0], humans[:, 1] - s[1])))
        if np.hypot(*(GOAL - s[:2])) < 0.12:
            break

    out = {k: np.array(v) for k, v in log.items()}
    out["humans"] = np.array(human_traj)          # (T, n_humans, 4)
    out["robot"] = robot
    out["cbf"] = cbf
    return out


def make_figure(ep, path):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    robot, cbf = ep["robot"], ep["cbf"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    (ax_xy, ax_v), (ax_h, ax_d) = axes

    # --- trajectory, robot path coloured by speed ---
    pts = np.column_stack([ep["rx"], ep["ry"]]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap="viridis", array=ep["v_safe"][:-1], lw=3)
    ax_xy.add_collection(lc)
    cbar = fig.colorbar(lc, ax=ax_xy, fraction=0.046, pad=0.02)
    cbar.set_label("robot speed [m/s]")
    n_h = ep["humans"].shape[1]
    for j in range(n_h):
        hx, hy = ep["humans"][:, j, 0], ep["humans"][:, j, 1]
        ax_xy.plot(hx, hy, "--", color="C3", alpha=0.6)
        ax_xy.plot(hx[0], hy[0], "o", color="C3")                 # start
        ax_xy.plot(hx[-1], hy[-1], "X", color="C3", ms=9)         # end
    ax_xy.plot(*GOAL, "g*", ms=18, label="goal")
    ax_xy.plot(ep["rx"][0], ep["ry"][0], "ks", label="robot start")
    ax_xy.set_xlim(-0.5, 5.5); ax_xy.set_ylim(-1.3, 1.3)
    ax_xy.set_aspect("equal")
    ax_xy.set_title("Trajectory (robot coloured by speed; red = pedestrians)")
    ax_xy.set_xlabel("x [m]"); ax_xy.set_ylabel("y [m]"); ax_xy.legend(loc="upper left")

    # --- commanded vs filtered velocity ---
    ax_v.plot(ep["t"], ep["v_cmd"], color="C3", label="commanded (adversary, full speed)")
    ax_v.plot(ep["t"], ep["v_safe"], color="C0", lw=2, label="filtered (CBF output)")
    ax_v.fill_between(ep["t"], ep["v_safe"], ep["v_cmd"], color="C3", alpha=0.15,
                      label="filter intervention")
    ax_v.set_title("The filter clamps commanded velocity")
    ax_v.set_xlabel("time [s]"); ax_v.set_ylabel("v [m/s]"); ax_v.legend(loc="lower right")

    # --- barrier h(t) ---
    ax_h.axhline(0.0, color="C3", ls="--", label="violation boundary (h = 0)")
    ax_h.plot(ep["t"], ep["h"], color="C0", lw=2, label="min stopping-distance barrier h(t)")
    ax_h.fill_between(ep["t"], 0, ep["h"], where=ep["h"] >= 0, color="C2", alpha=0.15)
    # h -> 0 while yielding is BY DESIGN: the robot rides the ISO stopping-distance
    # limit (go as fast as you can still stop), maximally efficient while safe.
    ax_h.set_title(f"Barrier rides the stopping limit but never crosses it "
                   f"(min h = {ep['h'].min():.3f} m)")
    ax_h.set_xlabel("time [s]"); ax_h.set_ylabel("h [m]"); ax_h.legend(loc="upper right")

    # --- distance to nearest human ---
    ax_d.axhline(cbf.d_hard, color="C3", ls="--", label=f"d_hard = {cbf.d_hard} m")
    ax_d.axhline(cbf.protective_radius, color="C1", ls=":",
                 label=f"protective field = {cbf.protective_radius} m")
    ax_d.plot(ep["t"], ep["d"], color="C0", lw=2, label="distance to nearest human")
    ax_d.set_title(f"Nearest-human distance  (min = {ep['d'].min():.3f} m)")
    ax_d.set_xlabel("time [s]"); ax_d.set_ylabel("distance [m]"); ax_d.legend(loc="upper right")

    RESULTS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return fig


def make_animation(ep, path):
    import matplotlib.pyplot as plt
    from matplotlib import animation

    robot, cbf = ep["robot"], ep["cbf"]
    T = len(ep["t"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(-0.5, 5.5); ax.set_ylim(-1.6, 1.6); ax.set_aspect("equal")
    ax.set_title("CBF safety filter — robot yields to pedestrians")
    ax.plot(*GOAL, "g*", ms=18)
    robot_dot, = ax.plot([], [], "o", color="C0", ms=12)
    robot_ring = plt.Circle((0, 0), cbf.protective_radius, color="C1", fill=False, ls=":")
    ax.add_patch(robot_ring)
    human_dots = [ax.plot([], [], "o", color="C3", ms=11)[0]
                  for _ in range(ep["humans"].shape[1])]
    txt = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top", fontsize=9)

    def update(k):
        ax.collections and None
        robot_dot.set_data([ep["rx"][k]], [ep["ry"][k]])
        robot_ring.center = (ep["rx"][k], ep["ry"][k])
        for j, hd in enumerate(human_dots):
            hd.set_data([ep["humans"][k, j, 0]], [ep["humans"][k, j, 1]])
        txt.set_text(f"t={ep['t'][k]:4.1f}s  v={ep['v_safe'][k]:.2f} "
                     f"(cmd {ep['v_cmd'][k]:.2f})  h={ep['h'][k]:.2f}")
        return [robot_dot, robot_ring, txt, *human_dots]

    anim = animation.FuncAnimation(fig, update, frames=T, interval=60, blit=True)
    anim.save(path, writer=animation.PillowWriter(fps=15))
    return anim


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--show", action="store_true", help="pop up an interactive window")
    p.add_argument("--animate", action="store_true", help="also save an animated GIF")
    args = p.parse_args()
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = run_episode()
    reached = np.hypot(*(GOAL - np.array([ep["rx"][-1], ep["ry"][-1]]))) < 0.15
    print(f"steps           : {len(ep['t'])}  ({ep['t'][-1]:.1f} s)")
    print(f"reached goal    : {reached}")
    print(f"min barrier h   : {ep['h'].min():.3f} m   (>= 0 => no stopping-dist violation)")
    print(f"min human dist  : {ep['d'].min():.3f} m   (d_hard = {ep['cbf'].d_hard} m)")
    print(f"protective stops: {int(np.sum(ep['pstop']))}")
    yield_cut = np.max((ep["v_cmd"] - ep["v_safe"])[ep["t"] > 1.5])  # exclude startup ramp
    print(f"yield slowdown  : {yield_cut:.3f} m/s  (how much the filter cut for the person)")

    fig_path = RESULTS / "cbf_demo.png"
    make_figure(ep, fig_path)
    print(f"figure saved    : {fig_path}")
    if args.animate:
        gif_path = RESULTS / "cbf_demo.gif"
        make_animation(ep, gif_path)
        print(f"animation saved : {gif_path}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
