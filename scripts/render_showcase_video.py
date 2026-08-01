"""Turn two showcase telemetry CSVs into the panel that carries the argument.

The Gazebo capture shows motion; this shows WHY. Renders, synchronised on mission time:

  * commanded speed cap (the RL supervisor's action) vs the speed actually achieved
  * the ISO stopping-distance barrier h, with h < 0 shaded red -- a violation is the
    robot being closer to a person than it can stop in, which is a certification failure
  * clearance to the nearest tracked worker
  * the three hazard stations marked, so each dip lines up with a cause

Usage:
    PYTHONPATH=$PWD .venv-navrl/bin/python scripts/render_showcase_video.py \
        --rl experiments/results/showcase_rl.csv \
        --baseline experiments/results/showcase_baseline.csv \
        --out experiments/results/showcase_panel.mp4

`--static` writes a PNG summary instead of a video (useful for the README / slides).
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from core.demo.showcase_scene import EVENT_X

STATIONS = [("A  blind corner\n(nobody there)", EVENT_X[0]),
            ("B  worker crosses\nthe intersection", EVENT_X[1]),
            ("C  occluded worker\nsteps out", EVENT_X[2])]

# measured over 12 seeds, experiments/results/s4_industrial_full.csv
STATS = ("Over 12 randomised runs:  ISO stopping-distance violations\n"
         "blind corner   fixed params 8/12  ->  RL supervisor 0/12\n"
         "crossing       fixed params 6/12  ->  RL supervisor 0/12")


def load(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for k in rows[0]:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[k]))
            except (ValueError, TypeError):
                vals.append(np.nan)
        out[k] = np.asarray(vals, dtype=float)
    return out


def _style(ax):
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def summarise(d, label):
    h = d["h_min"][np.isfinite(d["h_min"])]
    viol = int(np.sum(h < 0.0)) if h.size else 0
    moving = d["v"] > 0.05
    stops = int(np.sum(moving[:-1] & ~moving[1:])) if moving.size > 1 else 0
    nd = d["nearest_human_d"][np.isfinite(d["nearest_human_d"])]
    return dict(label=label, t=float(d["t"][-1]), viol=viol, stops=stops,
                min_h=float(np.min(h)) if h.size else float("nan"),
                closest=float(np.min(nd)) if nd.size else float("nan"))


def render(rl, bl, out_path, static=False, fps=25):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    plt.rcParams.update({"font.size": 9, "figure.facecolor": "white"})
    fig, axes = plt.subplots(3, 1, figsize=(11.0, 7.4), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.05, 1.0, 0.85],
                                              hspace=0.16))
    ax_v, ax_h, ax_d = axes
    C_RL, C_BL = "#1b6ca8", "#c1443c"

    for d, c, lab, ls in ((bl, C_BL, "MPC + CBF, fixed parameters (no RL)", "--"),
                          (rl, C_RL, "MPC + CBF + RL supervisor", "-")):
        ax_v.plot(d["x"], d["v_max_cmd"], color=c, ls=ls, lw=1.9, alpha=0.95,
                  label=f"{lab}: commanded cap")
        ax_v.plot(d["x"], d["v"], color=c, ls=":", lw=1.2, alpha=0.7)
        ax_h.plot(d["x"], d["h_min"], color=c, ls=ls, lw=1.9, label=lab)
        ax_d.plot(d["x"], d["nearest_human_d"], color=c, ls=ls, lw=1.9, label=lab)

    ax_h.axhline(0.0, color="black", lw=1.0)
    for d, c in ((bl, C_BL), (rl, C_RL)):
        bad = np.isfinite(d["h_min"]) & (d["h_min"] < 0)
        if bad.any():
            ax_h.fill_between(d["x"], d["h_min"], 0.0, where=bad, color=c, alpha=0.35,
                              interpolate=True)

    for ax in axes:
        for name, xs in STATIONS:
            ax.axvspan(xs - 0.85, xs + 0.85, color="#f0c419", alpha=0.16, lw=0)
        _style(ax)
    ax_v.set_ylabel("speed  (m/s)")
    ax_v.set_ylim(0, 2.35)
    for name, xs in STATIONS:                # sits above the traces, not across them
        ax_v.annotate(name, xy=(xs, 2.28), ha="center", va="top", fontsize=8,
                      color="#7a5c00", linespacing=1.15)
    ax_v.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax_v.set_title("Identical MPC + CBF stack, with and without the RL supervisor "
                   "- same world, same workers, same goal",
                   fontsize=11, loc="left", weight="bold")
    ax_h.set_ylabel("ISO barrier  h  (m)")
    # Clip both lower panels. Un-clipped they are dominated by the robot simply
    # approaching a worker who is still 16 m away down a side aisle, which dwarfs the
    # encounters that actually matter and makes the violations invisible.
    ax_h.set_ylim(-1.2, 3.0)
    ax_h.annotate("h < 0 = closer than it can stop in  (stopping-distance violation)",
                  xy=(0.02, 0.05), xycoords="axes fraction", fontsize=8, color="#7a1d17")
    ax_d.set_ylabel("clearance to\nworker (m)")
    ax_d.set_ylim(0, 5.0)
    ax_d.set_xlabel("position along the aisle  x  (m)")
    for d, c in ((bl, C_BL), (rl, C_RL)):           # mark each run's worst approach
        nd = d["nearest_human_d"]
        if np.isfinite(nd).any():
            i = int(np.nanargmin(nd))
            ax_d.plot(d["x"][i], nd[i], "o", color=c, ms=6, zorder=5)
            ax_d.annotate(f"{nd[i]:.2f} m", xy=(d["x"][i], nd[i]), xytext=(6, 6),
                          textcoords="offset points", fontsize=8, color=c, weight="bold")

    srl, sbl = (summarise(rl, "MPC+CBF+RL "), summarise(bl, "MPC+CBF fixed"))
    fig.text(0.012, 0.017,
             f"{sbl['label']}: {sbl['t']:.0f} s | violations {sbl['viol']} | "
             f"stops {sbl['stops']} | min h {sbl['min_h']:+.2f} m | closest {sbl['closest']:.2f} m\n"
             f"{srl['label']}: {srl['t']:.0f} s | violations {srl['viol']} | "
             f"stops {srl['stops']} | min h {srl['min_h']:+.2f} m | closest {srl['closest']:.2f} m",
             fontsize=8, family="monospace", va="bottom")
    fig.text(0.985, 0.017, STATS, fontsize=8, family="monospace", va="bottom", ha="right",
             color="#333333")
    fig.subplots_adjust(left=0.085, right=0.985, top=0.94, bottom=0.14)

    if static:
        fig.savefig(out_path, dpi=150)
        print(f"wrote {out_path}")
        return

    cursors = [ax.axvline(0.0, color="#222222", lw=1.2, alpha=0.85) for ax in axes]
    t_end = float(max(rl["t"][-1], bl["t"][-1]))
    n = int(t_end * fps) + 1
    writer = FFMpegWriter(fps=fps, bitrate=2600)
    with writer.saving(fig, out_path, dpi=130):
        for i in range(n):
            t = i / fps
            x = float(np.interp(t, rl["t"], rl["x"]))
            for c in cursors:
                c.set_xdata([x, x])
            writer.grab_frame()
    print(f"wrote {out_path}  ({n} frames, {t_end:.1f} s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl", default="experiments/results/showcase_rl.csv")
    ap.add_argument("--baseline", default="experiments/results/showcase_baseline.csv")
    ap.add_argument("--out", default="experiments/results/showcase_panel.mp4")
    ap.add_argument("--static", action="store_true", help="write a PNG instead")
    args = ap.parse_args()

    for p in (args.rl, args.baseline):
        if not os.path.isfile(p):
            raise SystemExit(f"missing telemetry: {p}\n"
                             "run both launches first (run:=rl and run:=baseline)")
    out = args.out
    if args.static and out.endswith(".mp4"):
        out = out[:-4] + ".png"
    render(load(args.rl), load(args.baseline), out, static=args.static)


if __name__ == "__main__":
    main()
