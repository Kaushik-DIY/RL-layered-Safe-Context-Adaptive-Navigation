"""Does the Gazebo run reproduce the 2D result? Run this after every recording.

The 2D gate is the measurement; the Gazebo run is the proof that the same stack drives a
physically simulated robot the same way. This script is what turns "it looked right" into
a number, by checking the three things that would each invalidate the demo:

  plant health    the machine must be able to brake at least as hard as the CBF plans
                  for. A Gazebo run where `max_wheel_acceleration` is too low is
                  plant-limited rather than policy-limited, which silently flattens every
                  comparison -- it cost a whole recording once.
  behaviour       station by station: does it slow and refuse to move at the blind
                  cross-aisle, and does it step aside in the plain aisle? That is the
                  argument; if 3D does not show it, the video is not proof of anything.
  agreement       mission time, barrier margin, protective stops against the 2D numbers.

MEASUREMENT TRAPS, both paid for already:
  * The recorder keeps writing after arrival, so the RECORDING LENGTH IS NOT THE MISSION
    TIME. A healthy 31 s run once read as 64 s of idle tail. Time is always measured to
    the first sample within 0.15 m of the goal.
  * The CSV contains repeated timestamps (the recorder samples faster than the clock
    resolution). Differencing without collapsing them gives impossible accelerations --
    -640 m/s^2 on a 1.2 m/s^2 machine. Always dedupe on `t` first.

    PYTHONPATH=$PWD .venv-navrl/bin/python scripts/check_final_gazebo.py
    PYTHONPATH=$PWD .venv-navrl/bin/python scripts/check_final_gazebo.py path/to.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.common.platform import load_platform
from core.demo.final_route import GOAL, STATION_X

DEFAULT = "experiments/results/final_gz_rl.csv"

# what the 2D gate measured on this route, for the agreement check
REF = dict(t=32.5, min_h=0.41, pstops=0,
           v_pass=(0.58, 0.80, 1.20), lat=(0.02, 0.01, 1.12))
LABEL = ("A  blind cross-aisle on the escape side",
         "B  4-way junction, occluded worker",
         "C  plain aisle, solid racking")


def load(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise SystemExit(f"{path} is empty")
    col = {k: np.array([float(r[k]) for r in rows]) for k in
           ("t", "x", "y", "v", "v_max_cmd", "h_min", "protective_stop",
            "nearest_human_d")}
    # collapse repeated timestamps: the recorder samples faster than the clock ticks,
    # and differencing duplicates manufactures impossible accelerations
    _, keep = np.unique(col["t"], return_index=True)
    return {k: v[np.sort(keep)] for k, v in col.items()}


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    plat = load_platform("industrial")
    c = load(path)
    t, x, y, v, h, d = (c["t"], c["x"], c["y"], c["v"], c["h_min"],
                        c["nearest_human_d"])

    reach = np.where(np.hypot(x - GOAL[0], y - GOAL[1]) < 0.15)[0]
    t_goal = float(t[reach[0]]) if len(reach) else float("nan")

    # Accelerations are a MISSION statistic, measured to arrival for the same reason the
    # clock is: the recorder keeps running afterwards, and anything the machine does once
    # it is parked says nothing about how it drove the route.
    mis = t <= (t_goal if np.isfinite(t_goal) else t.max())
    dt = np.diff(t[mis])
    acc = np.diff(v[mis])[dt > 1e-6] / dt[dt > 1e-6]
    peak_dec, peak_acc = float(acc.min()), float(acc.max())
    finite_h = h[np.isfinite(h)]
    min_h = float(finite_h.min()) if len(finite_h) else float("nan")
    pstops = int(c["protective_stop"].sum() > 0)

    print(f"{path}\n{'':-<66}")
    print(f"mission time      {t_goal:6.1f} s        (2D: {REF['t']:.1f} s, "
          f"{100 * (t_goal - REF['t']) / REF['t']:+.0f} %)")
    print(f"recording length  {t.max():6.1f} s        <- NOT the mission time")
    print(f"worst barrier h   {min_h:+6.2f} m        (2D: {REF['min_h']:+.2f} m)")
    print(f"protective stops  {pstops:6d}          (2D: {REF['pstops']})")
    print(f"closest approach  {d[d > 0].min():6.2f} m")
    print(f"peak decel/accel  {peak_dec:+6.2f} / {peak_acc:+.2f} m/s^2   "
          f"(limit {plat.robot.a_max_physical:.2f}, measured to arrival)")

    print(f"\n{'station':<40}{'v@pass':>9}{'|y|max':>9}{'2D v/y':>14}")
    beh = []
    for i, sx in enumerate(STATION_X):
        m = (x > sx - 5.0) & (x < sx + 2.5)
        if not m.any():
            continue
        kc = int(np.arange(len(x))[m][np.argmin(d[m])])
        vp, lat = float(v[kc]), float(np.abs(y[m]).max())
        beh.append((vp, lat))
        print(f"{LABEL[i]:<40}{vp:>9.2f}{lat:>9.2f}"
              f"{REF['v_pass'][i]:>8.2f} /{REF['lat'][i]:>5.2f}")

    checks = [
        ("mission completed", len(reach) > 0),
        ("plant can brake at least as hard as the CBF plans",
         abs(peak_dec) >= plat.cbf.a_brake - 0.05),
        ("no arm exceeds the platform acceleration limit",
         max(abs(peak_dec), peak_acc) <= plat.robot.a_max_physical + 0.35),
        ("barrier never crossed", min_h >= 0.0),
        ("no protective stop", pstops == 0),
        ("mission time within 15 % of the 2D result",
         abs(t_goal - REF["t"]) / REF["t"] <= 0.15),
        ("A: refuses to step aside beside the blind opening", beh[0][1] < 0.35),
        ("A: slows instead", beh[0][0] < 0.85),
        ("C: does step aside in the plain aisle", beh[2][1] > 0.60),
        ("C: keeps its speed up", beh[2][0] > beh[0][0] + 0.30),
    ]
    print()
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("\nGAZEBO CHECK:", "PASS" if all(ok for _, ok in checks) else "FAIL")
    sys.exit(0 if all(ok for _, ok in checks) else 1)


if __name__ == "__main__":
    main()
