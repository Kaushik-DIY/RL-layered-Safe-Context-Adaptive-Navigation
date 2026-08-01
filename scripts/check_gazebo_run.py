"""Reality check: does a recorded Gazebo run behave the way the offline gate predicts?

This is the step every earlier demo attempt skipped. It reads the telemetry the ROS
recorder wrote and answers three questions, in order of what usually goes wrong:

  1. IS THE PLANT HEALTHY?  If Gazebo cannot accelerate or brake as hard as the
     controller assumes, BOTH runs become plant-limited and the comparison collapses --
     the baseline never reaches 1.5 m/s, so it never looks reckless, and the supervisor's
     slow-downs stop standing out. This was the actual cause the first time:
     max_wheel_acceleration was 3.0 rad/s^2 = 0.30 m/s^2 against a CBF planning stops at
     0.8. The plant limit must be >= a_brake / wheel_radius.

  2. IS THE SPEED BEING MANAGED?  The supervised cap should step DOWN at each station and
     recover between them; the baseline's should be a flat line at v_max. And the robot
     should actually achieve what it is commanded.

  3. IS THE SAFETY RESULT THERE?  Baseline breaches, supervised does not.

    PYTHONPATH=$PWD .venv-navrl/bin/python scripts/check_gazebo_run.py
"""
from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from core.common.platform import load_platform
from core.demo.showcase_scene import EVENT_X, GOAL

NAMES = ["A blind corner (nobody there)", "B worker crosses intersection",
         "C occluded worker steps out"]


def load(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{path} is empty -- did the run reach the goal?")
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


def peak_accel(d):
    """Largest sustained accel / decel the plant actually delivered (m/s^2)."""
    t, v = d["t"], d["v"]
    ok = np.diff(t) > 1e-3
    a = np.diff(v)[ok] / np.diff(t)[ok]
    if a.size == 0:
        return 0.0, 0.0
    # 95th percentile, not the max: a single sample can spike on odometry noise
    return float(np.percentile(a[a > 0], 95)) if (a > 0).any() else 0.0, \
        float(-np.percentile(a[a < 0], 5)) if (a < 0).any() else 0.0


def mission_time(d):
    """Time to the goal, NOT the length of the recording.

    The launch keeps logging after the robot arrives, so the raw recording can be twice
    the mission. Reading the last timestamp made a healthy 30.9 s run look like 64.6 s.
    """
    dist = np.hypot(d["x"] - GOAL[0], d["y"] - GOAL[1])
    hit = np.where(dist < 0.15)[0]
    return float(d["t"][hit[0]]) if hit.size else None


def report(name, d, plat, is_rl):
    h = d["h_min"][np.isfinite(d["h_min"])]
    viol = int(np.sum(h < 0.0)) if h.size else 0
    acc, dec = peak_accel(d)
    moving = d["v"] > 0.05
    tg = mission_time(d)
    print(f"\n=== {name} ===")
    print(f"  mission {('%.1f s' % tg) if tg else 'DID NOT REACH GOAL'}"
          f"   (recording {d['t'][-1]:.1f} s, idle tail ignored)"
          f"   reached x = {np.nanmax(d['x']):.2f} m")
    print(f"  plant: peak accel {acc:.2f} m/s^2   peak decel {dec:.2f} m/s^2"
          f"   (controller assumes {plat.robot.a_max_mpc} / {plat.cbf.a_brake})")
    print(f"  speed: mean {np.nanmean(d['v'][moving]):.2f}   max {np.nanmax(d['v']):.2f}"
          f"   commanded cap mean {np.nanmean(d['v_max_cmd']):.2f}"
          f"   shortfall {np.nanmean((d['v_max_cmd'] - d['v'])[moving]):.2f} m/s")
    print(f"  safety: min h {np.min(h) if h.size else float('nan'):+.2f}   "
          f"violation samples {viol}   closest worker "
          f"{np.nanmin(d['nearest_human_d']):.2f} m")
    caps = []
    for i, ex in enumerate(EVENT_X):
        m = np.abs(d["x"] - ex) <= 2.5
        c = float(np.nanmin(d["v_max_cmd"][m])) if m.any() else float("nan")
        caps.append(c)
        print(f"    {NAMES[i]:<32} cap_min {c:.2f}   v_min "
              f"{float(np.nanmin(d['v'][m])) if m.any() else float('nan'):.2f}")
    return dict(viol=viol, caps=caps, dec=dec, acc=acc,
                vmax=float(np.nanmax(d["v"])), t=tg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl", default="experiments/results/showcase_rl.csv")
    ap.add_argument("--baseline", default="experiments/results/showcase_baseline.csv")
    args = ap.parse_args()

    plat = load_platform("industrial")
    rl = report("MPC + CBF + RL supervisor", load(args.rl), plat, True)
    bl = report("MPC + CBF, fixed parameters", load(args.baseline), plat, False)

    need = plat.cbf.a_brake                      # the plant must at least match this
    if rl["t"] and bl["t"]:
        print(f"\nvs the offline gate (27.4 s / 20.6 s): supervised {rl['t']:.1f} s "
              f"({100*(rl['t']/27.4-1):+.0f}%),  baseline {bl['t']:.1f} s "
              f"({100*(bl['t']/20.6-1):+.0f}%)")
    print("\n=== CHECKS ===")
    checks = [
        (f"plant can brake as hard as the CBF assumes (>= {need} m/s^2)",
         min(rl["dec"], bl["dec"]) >= 0.75 * need),
        ("baseline actually reaches full speed (>= 1.40 m/s)", bl["vmax"] >= 1.40),
        ("baseline cap is flat at max (no supervisor)", min(bl["caps"]) >= 1.40),
        ("supervised cap steps down at every station (<= 1.0)", max(rl["caps"]) <= 1.00),
        ("supervised recovers between stations (max speed >= 1.2)", rl["vmax"] >= 1.20),
        ("the dip is big enough to SEE (cap at least 1.6x lower at a station)",
         rl["vmax"] / max(1e-6, max(rl["caps"])) >= 1.6),
        ("supervised finishes (mission time recorded)", rl["t"] is not None),
        ("baseline breaches the stopping distance", bl["viol"] > 0),
        ("supervised does not breach", rl["viol"] == 0),
    ]
    ok = True
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        ok &= bool(passed)

    if not ok:
        print("\nFirst thing to check: the plant row above. If peak decel is well under "
              f"{need} m/s^2,\nraise <max_wheel_acceleration> in the world "
              f"(needs >= a_brake / wheel_radius = {need / 0.1:.0f} rad/s^2) and re-run --\n"
              "everything downstream is distorted by it.")
    else:
        print("\nGazebo run matches the offline gate -- safe to record.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
