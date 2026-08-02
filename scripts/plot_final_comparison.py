"""The portfolio comparison figure: three machines, four metrics that decide the claim.

    PYTHONPATH=$PWD .venv-navrl/bin/python scripts/plot_final_comparison.py

WHY THESE FOUR METRICS AND NOT OTHERS. The figure has to carry the video's claim without
overstating it, so each panel answers one question a sceptical reader would actually ask:

  1  "What does it save?"      site parameters configured by hand. The whole argument.
  2  "Is it as safe?"          worst ISO stopping-distance margin against the limit,
                               with contacts and violations stated on the panel.
  3  "What does it cost?"      mission time over 6 randomised presentations, with the
                               spread, plus the Gazebo run as a sim-to-sim check.
  4  "Does it actually adapt?" the SAME pedestrian pass twice, where only the geometry
                               differs. This is the one the other two machines cannot do
                               and the one the video is built around.

DELIBERATELY NOT PLOTTED. Speed-at-pass alone (it flatters us at station C and says
nothing at A); `min_h` as a "safety score" (the three are equal to within 0.01 m, so a
bar chart of it invites a difference that is not there -- it is drawn against the LIMIT
instead, which is the honest reading); and anything from the `crowd` station, which is
not on this route and where ours is beaten outright.

COLOUR. Slots 1-3 of the data-viz reference palette, unchanged. They are documented as
the three that clear the all-pairs colour-blindness floors in both light and dark, which
is the case that applies here (small multiples). They are NOT re-picked to match the
video's greens and blues: the palette is validated as a set, and re-stepping it by eye is
exactly the mistake the method exists to prevent. Every bar is direct-labelled, so
identity never rests on colour alone.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

OUT = Path("experiments/results/final_comparison.png")

# --- reference palette, slots 1-3, light mode. Colour follows the ENTITY everywhere. ---
SERIES = {"ours": "#2a78d6", "commissioned": "#eb6834", "scanner": "#1baf7a"}
SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
INK_3, GRID = "#8a8a86", "#e6e6e2"

ORDER = ["scanner", "commissioned", "ours"]          # story order, top to bottom
NAME = {"scanner": "Scanner only\n(no marked zones)",
        "commissioned": "Commissioned AMR\n(scanner + marked zones)",
        "ours": "MPC + CBF + RL supervised\n(nothing configured)"}
SHORT = {"scanner": "Scanner only", "commissioned": "Commissioned AMR",
         "ours": "MPC + CBF + RL supervised"}

# --- measured: 6 randomised presentations, scripts/verify_final.py 6 -----------------
PARAMS = {"scanner": 11, "commissioned": 13, "ours": 0}
TIME = {"scanner": (31.4, 1.3), "commissioned": (32.9, 0.1), "ours": (32.6, 0.1)}
MIN_H = {"scanner": 0.36, "commissioned": 0.36, "ours": 0.37}
PSTOPS = {"scanner": 0.17, "commissioned": 0.00, "ours": 0.00}
GAZEBO_OURS = 31.8                                    # scripts/check_final_gazebo.py


def _bare(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=0)
    ax.set_axisbelow(True)


def _title(ax, n, q, a):
    """Question then answer, both in axes coords so they stack predictably. set_title's
    pad is measured from the axes and does not know about a second line, which put the
    two on top of each other."""
    ax.text(0, 1.235, f"{n}   {q}", transform=ax.transAxes, fontsize=11,
            weight="bold", color=INK, va="top")
    ax.text(0, 1.115, a, transform=ax.transAxes, fontsize=8.6, color=INK_2, va="top")


def panel_params(ax):
    _title(ax, "1", "What does it save?",
           "Site parameters an integrator must measure, enter and re-validate")
    y = np.arange(len(ORDER))[::-1]
    for yi, k in zip(y, ORDER):
        v = PARAMS[k]
        ax.barh(yi, v, height=0.52, color=SERIES[k], edgecolor=SURFACE, linewidth=2)
        # a zero bar is invisible, so the value carries it and a tick marks the origin
        ax.text(v + 0.35 if v else 0.35, yi, f"{v}", va="center", fontsize=11,
                weight="bold", color=INK)
        if not v:
            ax.plot([0, 0], [yi - 0.26, yi + 0.26], color=SERIES[k], lw=3,
                    solid_capstyle="butt")
    ax.set_yticks(y, [SHORT[k] for k in ORDER], fontsize=8.8, color=INK)
    ax.set_xlim(0, 16)
    ax.set_xlabel("parameters configured by hand", fontsize=8.5, color=INK_2)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    _bare(ax)


def panel_safety(ax):
    _title(ax, "2", "Is it as safe?",
           "Worst stopping-distance margin; below zero = too close to stop in time")
    y = np.arange(len(ORDER))[::-1]
    for yi, k in zip(y, ORDER):
        ax.barh(yi, MIN_H[k], height=0.52, color=SERIES[k], edgecolor=SURFACE,
                linewidth=2)
        ax.text(MIN_H[k] + 0.012, yi, f"+{MIN_H[k]:.2f} m", va="center", fontsize=10,
                weight="bold", color=INK)
    ax.axvline(0, color="#c0392b", lw=1.6)
    ax.text(0.006, 2.52, "the limit", fontsize=8.2, color="#c0392b", va="center")
    ax.set_yticks(y, [SHORT[k] for k in ORDER], fontsize=8.8, color=INK)
    ax.set_xlim(-0.02, 0.52)
    ax.set_xlabel("barrier margin  h  [m]   (higher = more room to stop)",
                  fontsize=8.5, color=INK_2)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    _bare(ax)
    ax.text(0, -0.34, "Zero contacts and zero stopping-distance\nviolations on all "
                      "three, over every run.\nProtective stops per mission:   "
            f"{PSTOPS['scanner']:.2f}  /  {PSTOPS['commissioned']:.2f}  /  "
            f"{PSTOPS['ours']:.2f}",
            transform=ax.transAxes, fontsize=8.4, color=INK_2, va="top", linespacing=1.6)


def panel_time(ax):
    _title(ax, "3", "What does it cost?",
           "Mission time over 6 randomised presentations, mean and spread")
    y = np.arange(len(ORDER))[::-1]
    for yi, k in zip(y, ORDER):
        m, sd = TIME[k]
        ax.barh(yi, m, height=0.52, color=SERIES[k], edgecolor=SURFACE, linewidth=2)
        ax.errorbar(m, yi, xerr=sd, fmt="none", ecolor=INK_2, elinewidth=1.4, capsize=4)
        ax.text(m + sd + 1.1, yi, f"{m:.1f} s", va="center", fontsize=10,
                weight="bold", color=INK)
    # the Gazebo run: same stack, physically simulated. Marker, not a bar -- it is one
    # run against a 6-run distribution and must not read as a fourth machine.
    ax.plot([GAZEBO_OURS], [-0.30], marker="D", ms=7, color=SURFACE,
            markeredgecolor=INK, markeredgewidth=1.6, zorder=5)
    ax.annotate(f"same stack in Gazebo 3D: {GAZEBO_OURS:.1f} s", xy=(GAZEBO_OURS, -0.28),
                xytext=(11.0, -0.80), fontsize=8.3, color=INK_2,
                arrowprops=dict(arrowstyle="-", color=INK_3, lw=0.9))
    ax.set_yticks(y, [SHORT[k] for k in ORDER], fontsize=8.8, color=INK)
    ax.set_xlim(0, 40)
    ax.set_ylim(-1.05, 2.6)
    ax.set_xlabel("time to complete the 31 m route  [s]   (lower is quicker)",
                  fontsize=8.5, color=INK_2)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    _bare(ax)


def panel_adapt(ax, beh):
    _title(ax, "4", "Does it actually adapt?",
           "The same pedestrian pass twice. Only the geometry differs.")
    groups = ["Escape side is an\nOPEN cross-aisle", "Escape side is\nSOLID racking"]
    idx = (0, 2)                       # stations A and C
    xw, x = 0.24, np.arange(2)
    for j, k in enumerate(ORDER):
        vals = [beh[k][i]["v_pass"] for i in idx]
        off = (j - 1) * xw
        ax.bar(x + off, vals, width=xw - 0.03, color=SERIES[k], edgecolor=SURFACE,
               linewidth=2, label=SHORT[k])
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", fontsize=8.6,
                    weight="bold", color=INK)
    # what the speed alone does not say: only ours uses the width, and only where it may
    lat = beh["ours"][2]["lat"]
    ax.annotate(f"ours steps aside {lat:.2f} m\ninstead of slowing",
                xy=(1 + xw, beh["ours"][2]["v_pass"] - 0.06), xytext=(0.86, 1.30),
                fontsize=8.4, color=INK, ha="right", linespacing=1.4,
                arrowprops=dict(arrowstyle="->", color=INK_3, lw=1.0))
    # clear of the 0.60 bars it describes -- at 0.24 it sat on top of them
    ax.text(0, 0.86, "nowhere safe to go:\nall three slow", fontsize=8.4,
            color=INK_2, ha="center", linespacing=1.4)
    ax.set_xticks(x, groups, fontsize=8.8, color=INK)
    ax.set_ylim(0, 1.5)
    ax.set_ylabel("speed passing the picker  [m/s]", fontsize=8.5, color=INK_2)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    _bare(ax)


BEH_CACHE = Path("experiments/results/final_behaviour.json")


def behaviour():
    """Per-station response of all three arms, measured through the same gate.

    Computed here rather than tracked as a data file: the figure is then reproducible
    from the repo alone, and the only output this project version-controls stays the
    finished video. Cached because it costs about two minutes of controller time.
    """
    if BEH_CACHE.exists():
        return json.loads(BEH_CACHE.read_text())
    import verify_final as F
    from core.common.platform import load_platform
    from core.rl.supervisor import SupervisorPolicy
    print("measuring per-station behaviour (once; cached afterwards) ...")
    plat, scene = load_platform("industrial"), F.build_scene()
    sup = SupervisorPolicy(F.MODEL, platform="industrial",
                           walls=scene["walls"], posts=scene["posts"])
    out = {}
    for arm in F.ARMS:
        rec = []
        F.run(arm, plat, scene, sup=sup if arm == "ours" else None, record=rec)
        out[arm] = F.encounters(rec)
        print(f"  {arm:<13} "
              + "  ".join(f"{e['v_pass']:.2f}/{e['lat']:.2f}" for e in out[arm]))
    BEH_CACHE.write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    beh = behaviour()

    fig = plt.figure(figsize=(13.6, 9.2), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, left=0.185, right=0.965, top=0.775, bottom=0.125,
                          hspace=0.95, wspace=0.60)
    fig.text(0.012, 0.968, "Safe Context-Adaptive Navigation for Industrial AMRs",
             fontsize=17, weight="bold", color=INK, va="top")
    fig.text(0.012, 0.925,
             "One 31 m shared-aisle route, three machines, six randomised presentations "
             "each. Identical protective field and 1.20 m/s site limit on all three.",
             fontsize=9.6, color=INK_2, va="top")

    panel_params(fig.add_subplot(gs[0, 0]))
    panel_safety(fig.add_subplot(gs[0, 1]))
    panel_time(fig.add_subplot(gs[1, 0]))
    ax4 = fig.add_subplot(gs[1, 1])
    panel_adapt(ax4, beh)

    handles = [Patch(facecolor=SERIES[k], label=SHORT[k]) for k in ORDER]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.885),
               ncol=3, frameon=False, fontsize=9.2, handlelength=1.1,
               handleheight=0.9, columnspacing=1.6,
               labelcolor=INK)

    fig.text(0.012, 0.030,
             "Measured with scripts/verify_final.py (2D, 6 seeds) and "
             "scripts/check_final_gazebo.py (Gazebo). The commissioned machine is the "
             "same scanner AMR with a reduced-speed zone marked at each cross-aisle; the "
             "zone speed is derived from\nthe surveyed sight line, not chosen. Ours "
             "carries the identical protective field and no warning tier. Not shown: the "
             "`crowd` encounter, which is off this route and where ours is beaten "
             "outright.",
             fontsize=8.0, color=INK_3, va="bottom", linespacing=1.6)

    fig.savefig(OUT, dpi=170, facecolor=SURFACE)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
