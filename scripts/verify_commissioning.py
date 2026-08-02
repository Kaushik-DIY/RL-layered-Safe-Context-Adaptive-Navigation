"""THE GATE for the commissioning video. Run this before rendering anything.

The claim the video makes, and therefore what this script has to establish:

    equivalent ISO-compliant safety with ZERO per-site commissioning
    -- no zone marking, no field-set sizing, no re-validation on layout change --
    at a throughput cost of roughly ten percent.

Three machines, one route, identical workers and identical presentation timing:

    scanner      : MPC + safety-rated scanner only (warning tier -> 0.60 m/s,
                   protective tier -> stop), commissioned speed 1.20 m/s, NO marked
                   zones. The fastest defensible configuration, and the baseline the
                   earlier negative result was measured against -- kept here so that
                   result stays visible rather than being quietly replaced.
    commissioned : the same machine as actually deployed -- plus hand-marked reduced
                   speed zones at every mapped cross-aisle, because a scanner cannot
                   see round a corner. Zone speed is DERIVED (stopping distance must
                   fit the sight line), not chosen.
    ours         : MPC + relaxed CBF governor + learned supervisor + the SAME
                   protective tier. No warning tier, no zones: the supervisor supplies
                   the anticipation both of those exist to provide.

Three things keep this honest and none of them may be quietly changed:

* **Both machines carry the same protective field.** It is mandatory equipment and
  it is sized from the service brake (the strict `plat.cbf`), not from the relaxed
  governor. Ours never gets a smaller mandatory field than the machine it is
  compared against.
* **Safety is scored on the STRICT barrier for both arms.** The relaxed governor
  (sigma 1.0, physical a_brake 1.2, gamma 0.8) is what ours *drives* by; it is not
  what it is *marked* by. Scoring ours on its own relaxed barrier would be marking
  its own homework.
* **The route was not selected to flatter us.** It is built from station types that
  were each validated one at a time in `scripts/probe_station.py` /
  `scripts/relax_sweep.py`. The one station type that ours demonstrably fails --
  `crowd`, 1.50 protective stops against 0.25 and min_h -0.01 in every governor
  setting -- is NOT on this route, and the script says so out loud rather than
  letting its absence pass unnoticed.

    PYTHONPATH=$PWD .venv-navrl/bin/python scripts/verify_commissioning.py
    PYTHONPATH=$PWD .venv-navrl/bin/python scripts/verify_commissioning.py 8
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.cbf.cbf_filter import CbfFilter, d_stop
from core.common.observation import geometry_features
from core.common.platform import load_platform
from core.demo import aisle_scene as sc
from core.demo.industrial_amr import (COMMISSIONED, DWELL_S, PROT_PAD, STOPPED,
                                      WARN_FACTOR, WARN_HALF_W, IndustrialAMR)
from core.demo.plant import apply_plant, reachable_cap
from core.demo.sight_limit import floor_speed, lateral_room
from core.demo.site_zones import mark_zones, zone_cap
from core.mpc.mpc_controller import MpcController
from core.rl.supervisor import SupervisorPolicy
from core.sim2d.pedestrians import closest_point_on_segment

MODEL = "experiments/models/ppo_ind_C_s0_full_final.onnx"

# ---------------------------------------------------------------------- the route
# Three cross-aisles at 8.5 m centres, then a clear run-out to the pick station. Every
# one of them is an identical SITE FEATURE -- same mouth, same jambs, same lost sight
# line -- and that is the point: an integrator has to survey and configure for the
# feature, so all three cost commissioning effort whether or not anyone is in them.
STATION_X = (7.5, 16.0, 24.5)
GOAL_X = 31.0
STATIONS = [sc.Station("blind_clear", STATION_X[0]),
            sc.Station("blind_cross", STATION_X[1]),
            sc.Station("crossing", STATION_X[2])]
STATION_LABEL = ["blind cross-aisle\nnobody there",
                 "occluded worker\ncrosses the aisle",
                 "visible worker\ncrosses the aisle"]

# The relaxed governor, adopted 2026-08-01: never worse than strict at any station.
# The protective field is the real guarantee, so the CBF may plan on the PHYSICAL
# brake rather than the service brake it reserves for the certified stop.
RELAXED = dict(sigma=1.0, a_brake=1.2, gamma=0.8)


ARMS = ("scanner", "commissioned", "ours")


def build_scene():
    return sc.build(STATIONS, goal_x=GOAL_X)


def site_zones(plat):
    """What the integrator marks on this site's map -- one zone per cross-aisle."""
    return mark_zones(STATION_X, plat, sc.REVEAL_DISTANCE, COMMISSIONED, sc.MOUTH)


def _obstacles(pos, walls, posts, max_n):
    obs = list(posts) if len(posts) else []
    for w in walls:
        p = closest_point_on_segment(pos, w[:2], w[2:])
        obs.append([p[0], p[1], 0.0])
    arr = np.asarray(obs, dtype=float).reshape(-1, 3)
    d = np.hypot(arr[:, 0] - pos[0], arr[:, 1] - pos[1])
    return arr[np.argsort(d)[:max_n]]


def run(arm, plat, scene, sup=None, jitter=0.0, horizon_s=90.0, record=None,
        zones=None, lateral=False):
    """One mission. `arm` is one of ARMS; `sup` is required for 'ours'.

    `zones` lets a different route supply its own marked zones; `lateral` turns on
    the map-derived passing rule. Both default to the commissioning route's
    behaviour so that gate is untouched.
    """
    assert arm in ARMS, arm
    ours = arm == "ours"
    if zones is None:
        zones = site_zones(plat) if arm == "commissioned" else []
    walls, posts, goal = scene["walls"], scene["posts"], scene["goal"]
    cues = [dict(c, present_time=max(0.4, c["present_time"] + jitter))
            for c in scene["cues"]]

    mpc = MpcController(plat.robot, plat.mpc)
    # governor = what ours drives by; scorer = the strict barrier BOTH arms are marked on
    governor = CbfFilter(plat.robot, dataclasses.replace(plat.cbf, **RELAXED))
    scorer = CbfFilter(plat.robot, plat.cbf)
    scanner = IndustrialAMR(plat, use_warning=not ours)     # same protective tier both
    scanner.reset()
    if ours:
        sup.walls = np.asarray(walls, float).reshape(-1, 4)
        sup.posts = np.asarray(posts, float).reshape(-1, 3)
        sup.reset()

    dt = plat.robot.dt
    s = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    u_prev = np.zeros(2)
    rl_cap = rl_margin = None
    fired = [None] * len(cues)
    seen = [False] * len(cues)
    contacts = viol = 0
    min_h = np.inf
    stopped_s = 0.0
    zone_excess = 0.0            # worst speed over the marked limit, inside a zone
    floor_steps = n_steps = 0    # how often the geometry floor, not the policy, set the cap
    peak_decel = 0.0
    reached = None

    for k in range(int(horizon_s / dt)):
        t = k * dt
        n_steps += 1
        rows, truth = [], []
        for i, cue in enumerate(cues):
            if fired[i] is None:
                if sc.should_fire(cue, s[0], s[3]):
                    fired[i] = t
                wx, wy, wyaw = sc.staged_pose(cue)
                vx = vy = 0.0
            else:
                el = t - fired[i]
                wx, wy, wyaw, _ = sc.walk_path(cue, el)
                nx, ny, _, _ = sc.walk_path(cue, el + dt)
                vx, vy = (nx - wx) / dt, (ny - wy) / dt
            truth.append((wx, wy, wyaw))
            if sc.visible(cue, wx, wy, s[:2]):
                seen[i] = True
            if seen[i]:
                rows.append([wx, wy, vx, vy])
        humans = np.asarray(rows, dtype=float).reshape(-1, 4)

        if ours and k % plat.rl.decision_every == 0:
            rl_cap, rl_margin = sup.compute(s, goal, humans)
        # The policy was trained against the strict governor and is systematically slow
        # under the relaxed one, so it is floored at the speed its own map already
        # justifies: fast enough to stop inside what it can see, and no faster than the
        # strict barrier allows against anyone tracked. A floor, never a ceiling.
        v_floor = None
        lat = None
        if ours:
            v_floor = floor_speed(s, walls, posts, plat, COMMISSIONED,
                                  humans=humans, scorer=scorer)
            if lateral:
                # step aside only into space the map says is clear; otherwise the
                # policy's own request stands and the machine slows instead
                lat = lateral_room(s, walls, posts, plat, humans,
                                   scene.get("half_w", sc.HALF_W))
            if v_floor > (rl_cap if rl_cap is not None else 0.0) + 1e-9:
                floor_steps += 1
            rl_cap = max(rl_cap, v_floor) if rl_cap is not None else v_floor
        v_scan = scanner(s, humans, t)[0]
        # The marked limit is APPROACHED, not stepped into: a real zoned AMR has the
        # zone on its map and must already be at the limit when it crosses the line.
        v_zone = zone_cap(zones, float(s[0]), COMMISSIONED, a_dec=plat.robot.a_max_mpc)
        v_want = min([v_scan] + ([rl_cap] if ours and rl_cap is not None
                                 else [v_zone]))
        # ...and no layer may ask for a speed change the machine cannot make. A
        # protective stop is exempt: the safety controller cuts the drives directly and
        # the field is sized on the service brake, not on the planner's comfort limit.
        v_max_cmd = reachable_cap(v_want, float(s[3]), plat,
                                  emergency=scanner.state == STOPPED)
        d_margin_cmd = rl_margin if ours else 0.30
        if lat is not None and lat["margin"] is not None:
            d_margin_cmd = max(d_margin_cmd, lat["margin"])

        to_goal = goal - s[:2]
        dist_goal = float(np.hypot(*to_goal))
        if dist_goal < 0.15:
            reached = t
            break
        carrot = s[:2] + to_goal / dist_goal * min(plat.mpc.carrot_lookahead, dist_goal)
        u, _ = mpc.solve(x0=s[:3], carrot=carrot,
                         static_obs=_obstacles(s[:2], walls, posts,
                                               plat.mpc.max_static_obstacles),
                         humans=humans if len(humans) else None,
                         v_max_cmd=v_max_cmd, d_margin_cmd=d_margin_cmd, u_prev=u_prev)
        if ours:
            u, _ = governor.filter(s, u, humans if len(humans) else None)
        u_prev = u
        # the plant, finally: whatever the stack asked for, the wheels are bounded by
        # the platform's acceleration. Without this the harness integrated the command
        # directly and every arm braked at 4-6 m/s^2 on a 1.2 m/s^2 machine.
        v_app = apply_plant(float(u[0]), float(s[3]), plat)
        peak_decel = min(peak_decel, (v_app - float(s[3])) / dt)
        s = np.array([s[0] + dt * v_app * np.cos(s[2]), s[1] + dt * v_app * np.sin(s[2]),
                      s[2] + dt * u[1], v_app, u[1]])
        if s[3] <= 1e-3:
            stopped_s += dt
        # did the machine actually honour the limit its own map imposed? A zoned AMR
        # that only gets down to the limit somewhere past the line is not compliant,
        # so this is measured rather than assumed.
        for z in zones:
            if z.contains(float(s[0])):
                zone_excess = max(zone_excess, float(s[3]) - z.v)

        h = None
        if truth:
            # ISO 3691-4 stopping-distance condition on GROUND-TRUTH humans, strict
            # barrier, both arms. h < 0 while moving = the person is inside the room
            # the machine needs to stop, whether or not anything was ever touched.
            arr = np.asarray([[wx, wy, 0.0, 0.0] for wx, wy, _ in truth], float)
            hb = scorer.min_barrier(s, arr)
            if np.isfinite(hb):
                h = float(hb)
                min_h = min(min_h, h)
                viol += int(h < 0.0 and s[3] > 1e-3)
            d_now = min(float(np.hypot(wx - s[0], wy - s[1])) for wx, wy, _ in truth)
            contacts += int(d_now < plat.robot.robot_radius + plat.cbf.d_hard)

        if record is not None:
            v_ref = max(scanner._vhist)
            wc, ff, pa = geometry_features(s, walls, posts)
            record.append(dict(
                t=t, x=float(s[0]), y=float(s[1]), yaw=float(s[2]), v=float(s[3]),
                cap=float(v_max_cmd), rl_cap=None if rl_cap is None else float(rl_cap),
                v_floor=None if v_floor is None else float(v_floor),
                scan_cap=float(v_scan), state=scanner.state,
                zone_cap=None if not zones else float(v_zone),
                d_margin=float(d_margin_cmd),
                lat_blind=None if lat is None else bool(lat["blind"]),
                lat_escape=None if lat is None else float(lat["escape"]),
                prot=float(scanner.protective_len(v_ref)),
                warn=float(WARN_FACTOR * scanner.protective_len(v_ref)),
                h=h, viol_s=viol * dt, stopped_s=stopped_s, n_stops=scanner.n_stops,
                dist_goal=dist_goal, wall_clear=wc, forward_free=ff, post_ahead=pa,
                workers=[(wx, wy, wyaw, bool(seen[i]))
                         for i, (wx, wy, wyaw) in enumerate(truth)]))

    return dict(t=reached, pstops=scanner.n_stops, stopped_s=stopped_s,
                contacts=contacts, min_h=float(min_h), viol=viol, viol_s=viol * dt,
                zone_excess=zone_excess, peak_decel=peak_decel,
                floor_frac=floor_steps / max(1, n_steps))


# ------------------------------------------------------------------ what was configured
def commissioning_ledger(plat, zones=None):
    """Every hand-set number the industrial arm actually uses, and what an integrator
    has to know about THIS site to choose it. These are not illustrative -- each one
    is read straight out of the configuration the simulated machine runs on.

    `zones` MUST be the zones of the route being shown. It defaulted to this module's
    own three, which silently over-counted by one on any route with fewer marked
    openings -- the final video claimed 14 parameters where it configures 13.
    """
    c, r = plat.cbf, plat.robot
    prot = c.d_hard + d_stop(c.sigma * COMMISSIONED, c.tau, c.a_brake)
    return [
        ("site speed limit", f"{COMMISSIONED:.2f} m/s",
         "aisle width, traffic mix, B56.5 hazard-zone rule"),
        ("service braking rate", f"{c.a_brake:.2f} m/s²",
         "measured loaded, on this floor surface"),
        ("system response time", f"{c.tau:.2f} s",
         "scanner + controller + brake engagement"),
        ("speed-measurement factor", f"{c.sigma:.2f}",
         "odometry tolerance, ISO 13855 chain"),
        ("hard keep-out", f"{c.d_hard:.2f} m",
         "footprint tolerance + localisation error"),
        ("protective field length", f"{prot:.2f} m",
         f"= stopping distance at {COMMISSIONED:.2f} m/s, re-sized per tier"),
        ("protective field width", f"{2 * (r.robot_radius + PROT_PAD):.2f} m",
         "footprint + load overhang + tracking tolerance"),
        ("warning field length", f"{WARN_FACTOR * prot:.2f} m",
         "chosen to pre-slow before the protective tier trips"),
        ("warning field width", f"{2 * WARN_HALF_W:.2f} m",
         "wide enough to see the aisle, narrow enough to ignore side racking"),
        ("warning-tier speed", "0.60 m/s",
         "must fit the reduced field set at the reduced speed"),
        ("resume dwell", f"{DWELL_S:.1f} s",
         "site rule after a protective stop clears"),
    ] + [(f"zone {z.label} speed / extent",
          f"{z.v:.2f} m/s over {z.x1 - z.x0:.1f} m",
          "corner sight line surveyed, entry set by braking distance")
         for z in (site_zones(plat) if zones is None else zones)]


# ------------------------------------------------------------------------ battery
def battery(n, plat, scene, sup):
    """Every arm over `n` presentations, the timing jittered either side of nominal.

    The single run the video shows is the nominal one. It is not allowed to speak for
    the distribution -- this is what the closing card's "over N presentations" line is
    computed from, so a good-looking nominal run cannot hide a worse spread.
    """
    res = {}
    for arm in ARMS:
        acc = [run(arm, plat, scene, sup=sup if arm == "ours" else None,
                   jitter=(i - (n - 1) / 2) * 0.28) for i in range(n)]
        arrived = [a for a in acc if a["t"] is not None]
        res[arm] = dict(
            arrived=len(arrived), n=n,
            t=float(np.mean([a["t"] for a in arrived])) if arrived else float("nan"),
            t_sd=float(np.std([a["t"] for a in arrived])) if arrived else float("nan"),
            pstops=float(np.mean([a["pstops"] for a in acc])),
            stopped=float(np.mean([a["stopped_s"] for a in acc])),
            contacts=sum(a["contacts"] > 0 for a in acc),
            min_h=float(np.min([a["min_h"] for a in acc])),
            violeps=sum(a["viol"] > 0 for a in acc),
            zone_excess=float(np.max([a["zone_excess"] for a in acc])),
            peak_decel=float(np.min([a["peak_decel"] for a in acc])),
            floor_frac=float(np.mean([a["floor_frac"] for a in acc])))
    return res


# ------------------------------------------------------------------------- gate
def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    plat = load_platform("industrial")
    scene = build_scene()
    sup = SupervisorPolicy(MODEL, platform="industrial",
                           walls=scene["walls"], posts=scene["posts"])

    print(f"route: {len(STATIONS)} cross-aisles at x = "
          f"{', '.join(f'{x:.1f}' for x in STATION_X)} m, goal {GOAL_X:.1f} m")
    print(f"all arms: same protective field, same {COMMISSIONED:.2f} m/s "
          f"commissioned cap, scored on the strict barrier")
    print(f"ours: relaxed governor {RELAXED}, no warning tier, no zones")
    for z in site_zones(plat):
        print(f"  marked {z}")
    print()

    res = battery(n, plat, scene, sup)

    hdr = f"{'arm':<14}{'arrived':>8}{'time':>9}{'sd':>6}{'prot.stops':>11}" \
          f"{'standstill':>11}{'contacts':>9}{'min_h':>8}{'viol_eps':>9}{'peak_dec':>10}"
    print(hdr)
    for arm in ARMS:
        a = res[arm]
        print(f"{arm:<14}{a['arrived']:>4}/{a['n']:<3}{a['t']:>9.1f}{a['t_sd']:>6.1f}"
              f"{a['pstops']:>11.2f}{a['stopped']:>11.1f}{a['contacts']:>9}"
              f"{a['min_h']:>8.2f}{a['violeps']:>9}{a['peak_decel']:>10.2f}")
    print(f"\nthe geometry floor set ours' cap on "
          f"{100 * res['ours']['floor_frac']:.0f} % of steps; the learned policy set it "
          f"on the other {100 * (1 - res['ours']['floor_frac']):.0f} %.")
    print(f"plant limit {plat.robot.a_max_physical:.2f} m/s^2; no arm may exceed it. "
          f"Zone approach: the commissioned machine's worst\nspeed excess anywhere "
          f"inside a marked zone is {res['commissioned']['zone_excess']:+.3f} m/s "
          f"against the {site_zones(plat)[0].v:.2f} m/s limit.")

    s_, i, o = res["scanner"], res["commissioned"], res["ours"]
    cost = 100.0 * (o["t"] - i["t"]) / i["t"]
    cost_scan = 100.0 * (o["t"] - s_["t"]) / s_["t"]
    n_par = len(commissioning_ledger(plat))
    # The video claims equivalent COMPLIANCE, not equivalent behaviour. These are the
    # conditions that claim rests on, and every one of them is checked over the whole
    # battery rather than over the nominal run the video happens to show.
    checks = [
        ("every arm completes every mission",
         all(res[a]["arrived"] == n for a in ARMS)),
        ("zero contacts, every arm", all(res[a]["contacts"] == 0 for a in ARMS)),
        ("ours holds the strict barrier (min_h >= 0)", o["min_h"] >= 0.0),
        ("ours logs no stopping-distance violation", o["violeps"] == 0),
        ("ours is not claimed faster than the commissioned machine", cost >= 0.0),
        ("throughput cost against the commissioned machine <= 20 %", cost <= 20.0),
        ("no arm brakes harder than the platform can",
         all(res[a]["peak_decel"] >= -plat.robot.a_max_physical - 1e-6 for a in ARMS)),
        ("the commissioned machine honours its own marked zones",
         i["zone_excess"] <= 0.05),
    ]
    print(f"\nthroughput cost of ours vs commissioned : {cost:+.1f} %  "
          f"({i['t']:.1f} s -> {o['t']:.1f} s)")
    print(f"throughput cost of ours vs scanner-only : {cost_scan:+.1f} %  "
          f"({s_['t']:.1f} s -> {o['t']:.1f} s)")
    print(f"site parameters configured: {n_par} (commissioned) vs 0 (ours)\n")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    # NOT a gate, and deliberately so: a protective stop is the safety device doing its
    # job, not a non-conformity. It costs AVAILABILITY, which is a claim the video does
    # not make -- so it is reported here and the closing card has to carry it whenever
    # ours is worse, instead of the nominal run being allowed to imply parity.
    if o["pstops"] > i["pstops"] + 1e-9:
        print(f"\n  [REPORT ON THE CARD] over {n} presentations ours averages "
              f"{o['pstops']:.2f} protective stops against {i['pstops']:.2f}, "
              f"and its worst barrier margin is {o['min_h']:+.2f} m.")
    print("\nNOT ON THIS ROUTE, and the reason: the `crowd` station -- ours takes "
          "1.50\nprotective stops against 0.25 and min_h -0.01 there in every governor "
          "setting.\nIt is a known open failure, reported in the write-up, not hidden "
          "by scene choice.")
    print("\nGATE:", "PASS" if all(ok for _, ok in checks) else "FAIL")
    sys.exit(0 if all(ok for _, ok in checks) else 1)


if __name__ == "__main__":
    main()
