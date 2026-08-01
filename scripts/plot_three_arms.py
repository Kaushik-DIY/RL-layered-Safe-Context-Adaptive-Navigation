"""The two figures: how each arm pays for its safety, and the scorecard.

Figure 1 is the argument. Speed and the ISO barrier against position through the blind
corner, all 12 seeds per arm: median line plus interquartile band, so nothing is
cherry-picked. The commissioned machine crawls the whole aisle; the rated-speed machine
holds 1.5 m/s and drives its barrier through zero; ours eases down before the corner and
recovers afterwards.

Position is the x-axis rather than time because the hazard is at a FIXED place (the side
passage at x = open_x) while the three arms reach it at completely different times -- a
time axis would put the three encounters in three different places and destroy the
comparison.

    PYTHONPATH=$PWD .venv-navrl/bin/python scripts/plot_three_arms.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.cbf.cbf_filter import d_stop
from core.common.params import load_yaml
from core.common.platform import load_platform

RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results"
REAL = ["blind_corner", "corridor_passby", "doorway_negotiation", "open_hall",
        "perpendicular_crossing"]
ARMS = ["A_commissioned", "B_rated", "C_rl"]
COLOR = {"A_commissioned": "#2e8b57", "B_rated": "#c1443c", "C_rl": "#1b6ca8"}
SHORT = {"A_commissioned": "A  commissioned AMR\n0.5 m/s + field switching",
         "B_rated": "B  rated speed\n1.5 m/s, no commissioning",
         "C_rl": "C  MPC + CBF + RL\n(ours)"}
TAG = {"A_commissioned": "A commissioned", "B_rated": "B rated speed", "C_rl": "C ours"}


def profile(g: pd.DataFrame, col: str, grid: np.ndarray) -> np.ndarray:
    """One episode's `col` resampled onto the x grid.

    The robot can stop dead (x stops advancing) and can back off slightly, so x is not a
    valid interpolation axis as recorded: force it monotone with a running max and keep the
    LAST sample at each x, which is the value it held while sitting there.
    """
    x = np.maximum.accumulate(g["x"].to_numpy())
    v = g[col].to_numpy()
    keep = np.r_[np.diff(x) > 1e-9, True]
    return np.interp(grid, x[keep], v[keep], left=np.nan, right=np.nan)


def band(ax, tr, arm, scen, col, grid, lw=2.0):
    eps = [profile(g.sort_values("step"), col, grid)
           for _, g in tr[(tr.arm == arm) & (tr.scenario == scen)].groupby("episode")]
    if not eps:
        return
    M = np.vstack(eps)
    lo, mid, hi = (np.nanpercentile(M, q, axis=0) for q in (25, 50, 75))
    ax.fill_between(grid, lo, hi, color=COLOR[arm], alpha=0.15, lw=0)
    ax.plot(grid, mid, color=COLOR[arm], lw=lw, label=TAG[arm], zorder=3)


def fig_traces(tr, plat, cfg) -> None:
    geom = cfg["industrial_geometry"]["blind_corner"]
    corner_x, goal_x = geom["open_x"], geom["goal_x"]
    reveal = cfg["scenarios"]["blind_corner"]["reveal_distance"]
    d_cruise = plat.cbf.d_hard + d_stop(plat.cbf.sigma * plat.robot.v_max,
                                        plat.cbf.tau, plat.cbf.a_brake)
    grid = np.linspace(0.0, goal_x - 0.2, 400)

    fig, axes = plt.subplots(2, 1, figsize=(11.0, 7.2), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.0, 0.9], hspace=0.10))
    for arm in ARMS:
        band(axes[0], tr, arm, "blind_corner", "v_safe", grid)
        band(axes[1], tr, arm, "blind_corner", "h", grid)

    for ax in axes:
        ax.axvline(corner_x, color="#7a5c00", ls="--", lw=1.2, zorder=1)
        ax.grid(alpha=0.25, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].axhline(plat.robot.v_max, color="#999", ls=":", lw=1.0)
    axes[0].text(0.15, plat.robot.v_max - 0.07, "platform max 1.5 m/s", fontsize=8,
                 color="#777", va="top")
    axes[0].set_ylabel("commanded speed  [m/s]", fontsize=10)
    axes[0].set_ylim(-0.05, plat.robot.v_max + 0.12)
    axes[0].text(corner_x - 0.12, plat.robot.v_max + 0.05, "blind corner ", ha="right",
                 fontsize=9, color="#7a5c00", weight="bold")
    axes[0].legend(loc="lower right", fontsize=9, frameon=False)

    axes[1].axhline(0.0, color="#cc2b1d", ls="--", lw=1.3, zorder=2)
    axes[1].text(0.15, -0.06, "h = 0   ISO stopping-distance violation", fontsize=8.5,
                 color="#cc2b1d", va="top")
    axes[1].set_ylabel("ISO barrier  h  [m]", fontsize=10)
    axes[1].set_ylim(-1.1, 4.2)
    axes[1].set_xlabel("position along the aisle  x  [m]", fontsize=10)

    fig.suptitle("Blind corner: the worker is revealed at "
                 f"{reveal:.1f} m, but stopping from 1.5 m/s needs {d_cruise:.2f} m\n"
                 "median of 12 paired seeds, shaded = interquartile range",
                 fontsize=12, weight="bold", x=0.012, ha="left", y=0.985)
    fig.text(0.075, 0.020,
             "Only an arm that is ALREADY slow when the worker appears can keep h >= 0. The "
             "commissioned machine does it by crawling the whole aisle at 0.5 m/s;\nthe RL "
             "supervisor starts easing down ~4 m out, is at corner speed before anything is "
             "visible, and recovers to 1.5 m/s afterwards -- 31 % faster to the goal.",
             fontsize=8.5, color="#333", linespacing=1.5)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.135)
    fig.savefig(RESULTS / "three_arms_corner.png", dpi=150)
    print(f"wrote {RESULTS / 'three_arms_corner.png'}")


HAZARDS = ["blind_corner", "perpendicular_crossing"]


def fig_scorecard(cls) -> None:
    """Scored on the two hazards the work is about, plus availability over everything.

    The headline is deliberately scoped: on occluded corners and blind intersections the
    learned supervisor matches the commissioned machine's ZERO breaches while running much
    faster. Crowds are not there yet and are reported as a limitation, not buried.
    """
    haz = cls[cls.scenario.isin(HAZARDS)]
    agg = haz.groupby("arm").agg(fail=("breached", "sum"), t=("time_to_goal", "mean"))
    agg["all5"] = cls[cls.scenario.isin(REAL)].groupby("arm").breached.sum()
    agg["succ"] = cls.groupby("arm").success.mean()      # availability over ALL 96
    # Ride-quality metrics (hard brakes, peak decel) are deliberately NOT here: they are
    # inflated for arm C by the documented cap-slew interface defect
    # (scripts/audit_cap_interface.py), so they belong in a table with that caveat
    # attached, not in a headline bar chart that invites a like-for-like reading.
    panels = [("fail", "ISO breaches at corner + crossing\nof 24 missions  (lower better)",
               "{:.0f}"),
              ("t", "time to goal, corner + crossing  [s]\n(lower better)", "{:.1f}"),
              ("all5", "ISO breaches, all 5 scenarios\nof 60 missions  (lower better)",
               "{:.0f}"),
              ("succ", "missions completed, all 96\n(higher better)", "{:.2f}")]

    fig, axes = plt.subplots(1, len(panels), figsize=(13.0, 4.0))
    for ax, (key, title, fmt) in zip(axes, panels):
        vals = [agg.loc[a, key] for a in ARMS]
        ax.bar(range(3), vals, color=[COLOR[a] for a in ARMS], width=0.62)
        for i, v in enumerate(vals):
            ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=10,
                    weight="bold", color=COLOR[ARMS[i]])
        ax.set_title(title, fontsize=9.5)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["A", "B", "C"], fontsize=11, weight="bold")
        ax.set_ylim(0, max(vals) * 1.28 or 1)
        ax.grid(axis="y", alpha=0.25, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(left=False, labelleft=False)

    handles = [plt.Line2D([], [], color=COLOR[a], lw=8, label=SHORT[a].replace("\n", "  "))
               for a in ARMS]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Occluded corner + blind intersection, 12 paired seeds each: identical "
                 "MPC + CBF stack, three ways of choosing the speed",
                 fontsize=12, weight="bold", x=0.008, ha="left")
    fig.subplots_adjust(left=0.015, right=0.99, top=0.78, bottom=0.20, wspace=0.12)
    fig.savefig(RESULTS / "three_arms_scorecard.png", dpi=150)
    print(f"wrote {RESULTS / 'three_arms_scorecard.png'}")


def main() -> None:
    plat, cfg = load_platform("industrial"), load_yaml("scenarios")
    tr = pd.read_csv(RESULTS / "three_arms_traces.csv.gz")
    cls = pd.read_csv(RESULTS / "three_arms_classified.csv")
    fig_traces(tr, plat, cfg)
    fig_scorecard(cls)


if __name__ == "__main__":
    main()
