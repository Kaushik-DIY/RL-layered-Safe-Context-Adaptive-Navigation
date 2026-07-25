"""Week-1 MPC standalone demo (Gate G1 evidence, plan sec. 5).

Drives the hand-built NMPC + kinematic sim along a polyline path through static
clutter, following a moving carrot (plan D4). Prints solve-time stats (the
industry-legible result) and saves a trajectory + timing figure.

    python scripts/demo_mpc.py

No RL, no CBF, no ROS -- just the Layer-2 controller in the 2D sim.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.common.params import MpcParams, RobotParams
from core.mpc.mpc_controller import MpcController
from core.sim2d.kinematic_sim import KinematicSim

RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results"


def carrot_on_path(pos: np.ndarray, path: np.ndarray, lookahead: float) -> np.ndarray:
    """Nearest point on the polyline, advanced `lookahead` metres along it."""
    seg = path[1:] - path[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    # nearest point across all segments
    best_d, best_i, best_t = np.inf, 0, 0.0
    for i in range(len(seg)):
        if seg_len[i] < 1e-9:
            continue
        t = np.clip(np.dot(pos - path[i], seg[i]) / seg_len[i] ** 2, 0.0, 1.0)
        proj = path[i] + t * seg[i]
        d = np.linalg.norm(pos - proj)
        if d < best_d:
            best_d, best_i, best_t = d, i, t
    # advance `lookahead` metres forward along the path from that projection
    remaining, i, t = lookahead, best_i, best_t
    while i < len(seg):
        seg_remaining = (1.0 - t) * seg_len[i]
        if remaining <= seg_remaining or i == len(seg) - 1:
            frac = np.clip(t + remaining / max(seg_len[i], 1e-9), 0.0, 1.0)
            return path[i] + frac * seg[i]
        remaining -= seg_remaining
        i, t = i + 1, 0.0
    return path[-1]


def main() -> None:
    robot = RobotParams.from_yaml()
    mpc = MpcParams.from_yaml()
    ctrl = MpcController(robot, mpc)
    sim = KinematicSim(robot)

    # S-curve path through an obstacle field
    path = np.array([[0.0, 0.0], [2.0, 0.0], [3.5, 1.5], [5.0, 1.5], [6.5, 0.0]])
    obstacles = [[2.0, 0.6, 0.3], [3.5, 0.7, 0.25], [5.2, 0.9, 0.3]]  # [x, y, r]
    goal = path[-1]

    s = sim.reset([0.0, 0.0, 0.0])
    uprev = np.zeros(2)
    traj, carrots, vels, solve_ms = [s.copy()], [], [], []

    for _ in range(400):
        carrot = carrot_on_path(s[:2], path, mpc.carrot_lookahead)
        u, info = ctrl.solve(x0=s[:3], carrot=carrot, static_obs=obstacles, u_prev=uprev)
        carrots.append(carrot)
        vels.append(u[0])
        solve_ms.append(info["solve_ms"])
        uprev = u
        s = sim.step(u)
        traj.append(s.copy())
        if np.linalg.norm(s[:2] - goal) < 0.12:
            break

    traj = np.array(traj)
    solve_ms = np.array(solve_ms)
    warm = solve_ms[5:]
    reached = np.linalg.norm(traj[-1, :2] - goal) < 0.12

    print(f"steps           : {len(traj) - 1}  ({(len(traj) - 1) * robot.dt:.1f} s)")
    print(f"reached goal    : {reached}  (final err "
          f"{np.linalg.norm(traj[-1, :2] - goal):.3f} m)")
    print(f"solve time (ms) : median {np.median(warm):.2f}  "
          f"mean {warm.mean():.2f}  p99 {np.percentile(warm, 99):.2f}  "
          f"max {warm.max():.2f}")
    print(f"Gate G1 (<30 ms median): {'PASS' if np.median(warm) < 30 else 'FAIL'}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(path[:, 0], path[:, 1], "k--", lw=1, alpha=0.5, label="reference path")
    ax1.plot(traj[:, 0], traj[:, 1], "C0", lw=2, label="robot")
    for ox, oy, r in obstacles:
        ax1.add_patch(plt.Circle((ox, oy), r, color="C3", alpha=0.35))
        ax1.add_patch(plt.Circle((ox, oy), r + robot.robot_radius, color="C3",
                                 fill=False, ls=":", alpha=0.5))
    ax1.plot(*goal, "g*", ms=16, label="goal")
    ax1.set_aspect("equal")
    ax1.set_title("MPC path tracking through static clutter (G1)")
    ax1.set_xlabel("x [m]"); ax1.set_ylabel("y [m]"); ax1.legend(loc="best")

    ax2.hist(warm, bins=25, color="C0", alpha=0.8)
    ax2.axvline(30, color="C3", ls="--", label="G1 budget (30 ms)")
    ax2.axvline(np.median(warm), color="k", ls="-",
                label=f"median {np.median(warm):.1f} ms")
    ax2.set_title("Solve time distribution")
    ax2.set_xlabel("solve time [ms]"); ax2.set_ylabel("count"); ax2.legend()

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "mpc_demo.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"figure saved    : {out}")


if __name__ == "__main__":
    main()
