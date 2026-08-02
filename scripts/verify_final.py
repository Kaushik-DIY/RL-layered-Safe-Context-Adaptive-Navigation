"""THE GATE for the final video: the commissioning route, with two pedestrian passes.

The same argument as `verify_commissioning.py` -- one machine hand-commissioned, one
reading the map it already has -- on a route that now also asks the harder question:
when somebody walks at you, do you slow down or do you go round?

The answer has to depend on the geometry, and that is the point of the route:

  A  x = 7.5   BLIND CROSS-AISLE ON THE SOUTH SIDE, and a picker walking at the robot up
               the NORTH side. Going round him means swinging SOUTH -- straight across
               the mouth of an opening nobody can see into. The machine must refuse and
               slow down instead. Trading a person it can see for a person it cannot is
               exactly the trade this project exists to avoid.
  B  x = 16.0  A TRUE 4-WAY JUNCTION -- openings on both sides -- with an occluded worker
               descending the north arm and crossing to the south one. He now walks out
               through a real gap rather than appearing to step through the racking.
  C  x = 24.5  THE SAME PEDESTRIAN PASS, in plain aisle with solid racking both sides.
               Nothing can emerge, so the room is real: the machine offsets and carries
               its speed through.

Same picker, same side, same closing speed at A and C. The only thing that differs is
what the map says about the space the machine would move into -- which is the whole claim,
run twice with the answer flipped.

The industrial machine slows at both, because slowing is all its warning tier can do.

    PYTHONPATH=$PWD .venv-navrl/bin/python scripts/verify_final.py
    PYTHONPATH=$PWD .venv-navrl/bin/python scripts/verify_final.py 6
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.common.platform import load_platform
from core.demo.final_route import (GOAL_X, HALF_W, STATION_LABEL,  # noqa: F401
                                   STATION_X, ZONE_X, build, site_zones)
from core.rl.supervisor import SupervisorPolicy

from verify_commissioning import (ARMS, MODEL,  # noqa: E402
                                  commissioning_ledger as _ledger,
                                  run as _run)

# The route itself lives in `core/demo/final_route.py`, because the Gazebo world, the
# scene director and this gate all have to agree about it. Anything that draws the route
# imports it from there; nothing redefines it.
build_scene = build


def commissioning_ledger(plat):
    """This route marks only TWO zones -- station C has no cross-aisle -- so the count
    has to come from ZONE_X, not from the commissioning route's three."""
    return _ledger(plat, zones=site_zones(plat))


def run(arm, plat, scene, sup=None, jitter=0.0, record=None):
    """Delegates to the commissioning loop -- same controllers, same scoring, only the
    scene, the zones and the passing rule differ."""
    return _run(arm, plat, scene, sup=sup, jitter=jitter, record=record,
                zones=site_zones(plat) if arm == "commissioned" else [],
                lateral=True)


def battery(n, plat, scene, sup):
    res = {}
    for arm in ARMS:
        acc = [run(arm, plat, scene, sup=sup if arm == "ours" else None,
                   jitter=(i - (n - 1) / 2) * 0.28) for i in range(n)]
        ok = [a for a in acc if a["t"] is not None]
        res[arm] = dict(
            arrived=len(ok), n=n,
            t=float(np.mean([a["t"] for a in ok])) if ok else float("nan"),
            t_sd=float(np.std([a["t"] for a in ok])) if ok else float("nan"),
            pstops=float(np.mean([a["pstops"] for a in acc])),
            stopped=float(np.mean([a["stopped_s"] for a in acc])),
            contacts=sum(a["contacts"] > 0 for a in acc),
            min_h=float(np.min([a["min_h"] for a in acc])),
            violeps=sum(a["viol"] > 0 for a in acc))
    return res


def encounters(rec):
    """Per-station behaviour.

    Reporting the MINIMUM speed alone is misleading and was corrected after it misread
    once: ours dips to 0.36 m/s at station A while the picker is still 1.67 m away, then
    recovers to 0.58 by the time they are actually alongside -- so the minimum describes
    a 1.4 s brake pulse, not the speed it passes at. `v_pass` (speed at closest approach)
    and `v_mean` are what the behaviour claim should rest on; `v_min` is kept but read
    alongside them.
    """
    x = np.array([e["x"] for e in rec])
    y = np.array([e["y"] for e in rec])
    v = np.array([e["v"] for e in rec])
    gap = np.array([min((np.hypot(w[0] - e["x"], w[1] - e["y"])
                         for w in e["workers"]), default=np.inf) for e in rec])
    out = []
    for sx in STATION_X:
        m = (x > sx - 5.0) & (x < sx + 2.5)
        if not m.any():
            out.append(dict(v_min=np.nan, v_pass=np.nan, v_mean=np.nan, lat=np.nan))
            continue
        idx = np.arange(len(x))[m]
        kc = int(idx[np.argmin(gap[m])])
        out.append(dict(v_min=float(v[m].min()), v_pass=float(v[kc]),
                        v_mean=float(v[m].mean()), gap=float(gap[kc]),
                        lat=float(np.abs(y[m]).max()),
                        lat_signed=float(y[m][np.argmax(np.abs(y[m]))])))
    return out


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    plat = load_platform("industrial")
    scene = build_scene()
    sup = SupervisorPolicy(MODEL, platform="industrial",
                           walls=scene["walls"], posts=scene["posts"])
    print(f"route: {2 * HALF_W:.1f} m two-way aisle, {GOAL_X:.0f} m run")
    for i, lab in enumerate(STATION_LABEL):
        print(f"  x={STATION_X[i]:>5.1f}  {lab.replace(chr(10), ' ')}")
    for z in site_zones(plat):
        print(f"  marked {z}")
    print()

    res = battery(n, plat, scene, sup)
    print(f"{'arm':<14}{'arrived':>8}{'time':>9}{'sd':>6}{'prot.stops':>11}"
          f"{'contacts':>9}{'min_h':>8}{'viol_eps':>9}")
    for arm in ARMS:
        a = res[arm]
        print(f"{arm:<14}{a['arrived']:>4}/{a['n']:<3}{a['t']:>9.1f}{a['t_sd']:>6.1f}"
              f"{a['pstops']:>11.2f}{a['contacts']:>9}{a['min_h']:>8.2f}"
              f"{a['violeps']:>9}")

    # the behavioural claim, measured station by station on the nominal run
    beh = {}
    for arm in ("commissioned", "ours"):
        rec = []
        run(arm, plat, scene, sup=sup if arm == "ours" else None, record=rec)
        beh[arm] = encounters(rec)
    print(f"\n{'station':<30}{'commissioned':>30}{'ours':>30}")
    print(f"{'':<30}{'v@pass  v_mean  v_min  offset':>30}"
          f"{'v@pass  v_mean  v_min  offset':>30}")
    for i, lab in enumerate(STATION_LABEL):
        c, o = beh["commissioned"][i], beh["ours"][i]
        fmt = lambda d: (f"{d['v_pass']:>6.2f}{d['v_mean']:>8.2f}"
                         f"{d['v_min']:>7.2f}{d['lat']:>8.2f}")
        print(f"{lab.replace(chr(10), ' ')[:29]:<30}{fmt(c):>30}{fmt(o):>30}")

    i, o = res["commissioned"], res["ours"]
    cost = 100.0 * (o["t"] - i["t"]) / i["t"]
    a_ours, c_ours = beh["ours"][0], beh["ours"][2]
    print(f"\nthroughput vs commissioned: {cost:+.1f} % "
          f"({i['t']:.1f} s -> {o['t']:.1f} s)")
    print(f"site parameters configured: {len(commissioning_ledger(plat))} vs 0")

    checks = [
        ("every arm completes every mission",
         all(res[a]["arrived"] == n for a in ARMS)),
        ("zero contacts, every arm", all(res[a]["contacts"] == 0 for a in ARMS)),
        ("ours holds the strict barrier (min_h >= 0)", o["min_h"] >= 0.0),
        ("ours logs no stopping-distance violation", o["violeps"] == 0),
        ("throughput cost stays inside 20 %", -20.0 <= cost <= 20.0),
        ("at A, ours REFUSES to swerve into the blind opening", a_ours["lat"] < 0.35),
        ("at A, ours slows instead", a_ours["v_mean"] < 0.88),
        ("at C, ours DOES use the aisle", c_ours["lat"] > 0.60),
        ("at C, ours keeps its speed up",
         c_ours["v_pass"] > a_ours["v_pass"] + 0.30),
        ("the commissioned machine only ever slows, at both",
         beh["commissioned"][0]["lat"] < 0.35 and beh["commissioned"][2]["lat"] < 0.35),
    ]
    print()
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("\nGATE:", "PASS" if all(ok for _, ok in checks) else "FAIL")
    sys.exit(0 if all(ok for _, ok in checks) else 1)


if __name__ == "__main__":
    main()
