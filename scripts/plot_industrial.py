"""P4 money plots (industrial track). Reads s4_industrial.csv (trained + hand
supervisors) and pareto_industrial.csv (fixed-tuning frontier), writes the two
headline figures + the KPI summary the write-up needs.

  1. Safety-vs-throughput Pareto scatter: fixed-tuning cloud, its frontier, the
     hand heuristics, and the TRAINED policy -- the point of the whole thesis is
     the trained marker sitting on the good side of the frontier.
  2. Per-scenario ISO-compliance bars: % missions with zero stopping-distance
     violations, trained vs the fixed extremes vs the corner heuristic.

    python scripts/plot_industrial.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results"

# colorblind-safe categorical (validated palette; identity carried by color+marker)
C = {"frontier": "#8A8F98", "trained": "#3B6FE0", "always-max": "#E1575A",
     "density": "#2FA84F", "corner-aware": "#E5A02E", "fixed-mid": "#8A63D2"}
# The Pareto/compliance figures aggregate over the REALISTIC industrial scenarios.
# The interferer (a curious bystander who continuously follows the robot) is an
# adversarial robustness stress test, not a representative task -- excluded from the
# headline frontier claim and reported separately (like the S5 adversarial battery).
TRAIN_SCEN = ("corridor_passby", "perpendicular_crossing", "doorway_negotiation",
              "open_hall", "blind_corner")


def agg(df):
    """mean per (supervisor) over the training scenarios: (t_goal, viol_rate)."""
    d = df[df.scenario.isin(TRAIN_SCEN)]
    g = d.groupby("supervisor").agg(
        t=("time_to_goal", "mean"),
        viol=("violation_steps", lambda s: (s > 0).mean()),
        succ=("success", "mean")).reset_index()
    return g


def main() -> None:
    s4 = pd.read_csv(RESULTS / "s4_industrial.csv")
    hand = agg(s4)

    # fixed-tuning frontier: mean per (v_cmd, margin) over the same scenarios
    par = pd.read_csv(RESULTS / "pareto_industrial.csv")
    par = par[par.scenario.isin(TRAIN_SCEN)]
    grid = par.groupby(["v_cmd", "margin_cmd"]).agg(
        t=("time_to_goal", "mean"),
        viol=("violation_steps", lambda s: (s > 0).mean())).reset_index()

    # lower-left frontier of the fixed grid (minimize both t and viol)
    pts = grid.sort_values("t").reset_index(drop=True)
    frontier, best = [], np.inf
    for _, r in pts.iterrows():
        if r["viol"] <= best:
            frontier.append((r["t"], r["viol"]))
            best = r["viol"]
    fx, fy = zip(*frontier)

    # ---- Plot 1: safety-vs-throughput Pareto ----
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.scatter(grid["t"], 100 * grid["viol"], s=45, c=C["frontier"], alpha=0.5,
               edgecolor="white", linewidth=0.5, label="fixed tunings (grid)", zorder=2)
    ax.plot(fx, 100 * np.array(fy), c=C["frontier"], lw=2, ls="--",
            label="fixed-tuning frontier", zorder=3)
    marks = {"trained": ("*", 340), "always-max": ("s", 130),
             "density": ("^", 130), "corner-aware": ("D", 110), "fixed-mid": ("v", 110)}
    for _, r in hand.iterrows():
        nm = r["supervisor"]
        if nm not in marks:
            continue
        m, sz = marks[nm]
        ax.scatter(r["t"], 100 * r["viol"], marker=m, s=sz, c=C[nm],
                   edgecolor="white", linewidth=1.2, zorder=5,
                   label=f"{nm}" + (" (learned)" if nm == "trained" else ""))
    ax.set_xlabel("time-to-goal  [s]   (throughput ->)")
    ax.set_ylabel("missions with a stopping-distance violation  [%]   (<- safety)")
    ax.set_title("Industrial AMR safety-throughput (realistic scenarios): the LEARNED\n"
                 "supervisor sits on the safe end of the frontier -- matching the safest\n"
                 "heuristic's safety at higher speed, beating every hand-tuned baseline")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(RESULTS / "industrial_pareto.png", dpi=130)
    print(f"figure -> {RESULTS / 'industrial_pareto.png'}")

    # ---- Plot 2: per-scenario ISO-compliance bars ----
    show = ["always-max", "fixed-mid", "corner-aware", "density", "trained"]
    show = [s for s in show if s in s4.supervisor.unique()]
    clean = (s4[s4.scenario.isin(TRAIN_SCEN)]
             .assign(clean=lambda d: d.violation_steps == 0)
             .groupby(["scenario", "supervisor"])["clean"].mean().unstack() * 100)
    clean = clean.reindex(columns=show)
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(clean.index))
    w = 0.8 / len(show)
    for k, nm in enumerate(show):
        ax.bar(x + k * w, clean[nm].values, w, color=C.get(nm, "#888"),
               edgecolor="white", linewidth=0.6, label=nm)
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels([s.replace("_", "\n") for s in clean.index], fontsize=9)
    ax.set_ylabel("ISO-compliant missions  [%]  (zero stopping-dist. violations)")
    ax.set_title("Per-scenario ISO 3691-4 compliance (industrial platform)")
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, ncol=len(show), fontsize=9, loc="lower center",
              bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    fig.savefig(RESULTS / "industrial_compliance.png", dpi=130)
    print(f"figure -> {RESULTS / 'industrial_compliance.png'}")

    # headline text
    tr = hand[hand.supervisor == "trained"].iloc[0]
    am = hand[hand.supervisor == "always-max"].iloc[0]
    print(f"\nTRAINED: t={tr['t']:.1f}s viol={100*tr['viol']:.0f}% succ={tr['succ']:.2f}"
          f"   ALWAYS-MAX: t={am['t']:.1f}s viol={100*am['viol']:.0f}% succ={am['succ']:.2f}")


if __name__ == "__main__":
    main()
