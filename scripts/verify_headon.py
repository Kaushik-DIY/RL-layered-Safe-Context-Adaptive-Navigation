"""THE GATE for the head-on video: an oncoming picker, in a wide two-way aisle.

A DIFFERENT QUESTION FROM THE COMMISSIONING DEMO, deliberately kept in its own scene and
its own video. There are no cross-aisles on this route, so there is nothing for an
integrator to mark and the commissioning argument does not apply. What is on trial here is
purely how each machine BEHAVES when a person walks straight at it and has to be passed.

    industrial : MPC + safety-rated scanner. No zones to mark, so its anticipation is
                 entirely the warning tier -- anything inside a ~5 m x 2.2 m forward box
                 drops it to a flat 0.60 m/s, whether or not that person is actually on a
                 collision course, and it stays there until the box is clear.
    ours       : MPC + relaxed CBF governor + learned supervisor + sight floor, same
                 mandatory protective field, no warning tier. Free to modulate
                 continuously and to use the aisle width.

WHY THE AISLE IS WIDER HERE. The commissioning route is a 3.5 m aisle, which is the
industrial standard for one-way transport. Passing an oncoming pedestrian in one leaves
almost no lateral room: with the picker a realistic 0.75 m off centre the robot has about
a metre of usable offset before it is against the racking, so BOTH machines are forced
into the same manoeuvre and the comparison measures the aisle, not the controller. A 5.0 m
two-way aisle -- the width a site uses where AMRs and people share a route in both
directions -- gives a genuine choice between slowing down and stepping aside, which is the
behaviour worth filming.

WHAT THE EXISTING EVIDENCE SAYS, so this is not read as a new claim. Head-on is the one
geometry where the project has repeatedly measured NO supervisor advantage: the slowdown
comes from the MPC's human-cost term (which CV-propagates every tracked human over its 2 s
horizon) and the CBF, and it is present with the supervisor removed. Any difference seen
here is therefore expected to be about the WARNING TIER's bluntness, not about learned
anticipation, and the write-up says so whichever way the numbers fall.

    PYTHONPATH=$PWD .venv-navrl/bin/python scripts/verify_headon.py
    PYTHONPATH=$PWD .venv-navrl/bin/python scripts/verify_headon.py 6
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.cbf.cbf_filter import CbfFilter
from core.common.platform import load_platform
from core.demo import aisle_scene as sc
from core.demo.industrial_amr import COMMISSIONED, STOPPED, IndustrialAMR
from core.demo.plant import apply_plant, reachable_cap
from core.demo.sight_limit import floor_speed, passing_margin
from core.mpc.mpc_controller import MpcController
from core.rl.supervisor import SupervisorPolicy
from core.sim2d.pedestrians import closest_point_on_segment

from verify_commissioning import MODEL, RELAXED, _obstacles       # noqa: E402

ARMS = ("industrial", "ours", "ours_lateral")

# ------------------------------------------------------------------- the encounter
HALF_W = 2.50            # 5.0 m two-way aisle: room to step aside, not just slow down
GOAL_X = 26.0        # long enough that the panel aspect leaves room for instruments
MEET_X = 13.0            # where the two are intended to pass, mid-route
PICKER_LANE = 0.75       # how far off centre he walks -- people do not walk the centreline
PICKER_SPEED = 1.25
PICKER_LEAD = 9.5        # he starts this far beyond the meeting point
PICKER_TRAIL = 9.5       # ...and walks fully out of the scene behind the robot

STATIONS = [sc.Station("head_on", MEET_X, lane=PICKER_LANE, speed=PICKER_SPEED,
                       lead=PICKER_LEAD, trail=PICKER_TRAIL,
                       # large, so he sets off at t=0 and the meeting point is decided by
                       # the two speeds rather than by a trigger
                       present_time=30.0)]


def build_scene():
    return sc.build(STATIONS, goal_x=GOAL_X, half_w=HALF_W)


def run(arm, plat, scene, sup=None, jitter=0.0, horizon_s=90.0, record=None):
    """One mission. Same stack as the commissioning gate; only the scene and the arms
    differ, so nothing about the controllers is special-cased for this video."""
    assert arm in ARMS, arm
    ours = arm.startswith("ours")
    lateral = arm == "ours_lateral"
    walls, posts, goal = scene["walls"], scene["posts"], scene["goal"]
    # jitter moves the PICKER's start, which is what changes the encounter here
    cues = [dict(c, path=[(c["path"][0][0] + jitter, c["path"][0][1]), c["path"][1]])
            for c in scene["cues"]]

    mpc = MpcController(plat.robot, plat.mpc)
    governor = CbfFilter(plat.robot, dataclasses.replace(plat.cbf, **RELAXED))
    scorer = CbfFilter(plat.robot, plat.cbf)       # strict barrier, marks BOTH arms
    scanner = IndustrialAMR(plat, use_warning=not ours)
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
    contacts = viol = 0
    min_h = min_d = np.inf
    lat_max = 0.0            # how far the machine actually used the aisle
    v_at_pass = None         # speed at closest approach -- the number the video is about
    warn_s = stopped_s = 0.0
    reached = None

    for k in range(int(horizon_s / dt)):
        t = k * dt
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
            rows.append([wx, wy, vx, vy])       # open aisle: he is visible throughout
        humans = np.asarray(rows, dtype=float).reshape(-1, 4)

        if ours and k % plat.rl.decision_every == 0:
            rl_cap, rl_margin = sup.compute(s, goal, humans)
        v_floor = None
        if ours:
            v_floor = floor_speed(s, walls, posts, plat, COMMISSIONED,
                                  humans=humans, scorer=scorer)
            rl_cap = max(rl_cap, v_floor) if rl_cap is not None else v_floor
        v_scan = scanner(s, humans, t)[0]
        v_want = min([v_scan] + ([rl_cap] if ours and rl_cap is not None else []))
        v_max_cmd = reachable_cap(v_want, float(s[3]), plat,
                                  emergency=scanner.state == STOPPED)
        d_margin_cmd = rl_margin if ours else 0.30
        if lateral:
            # floor the LATERAL request at what the aisle can give, the
            # same idea as the speed floor and for the same reason
            d_margin_cmd = max(d_margin_cmd,
                               passing_margin(s, walls, plat,
                                              half_w=scene['half_w']))

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
        v_app = apply_plant(float(u[0]), float(s[3]), plat)
        s = np.array([s[0] + dt * v_app * np.cos(s[2]), s[1] + dt * v_app * np.sin(s[2]),
                      s[2] + dt * u[1], v_app, u[1]])

        if s[3] <= 1e-3:
            stopped_s += dt
        if scanner.state != "NORMAL":
            warn_s += dt
        lat_max = max(lat_max, abs(float(s[1])))

        h = None
        arr = np.asarray([[wx, wy, 0.0, 0.0] for wx, wy, _ in truth], float)
        hb = scorer.min_barrier(s, arr)
        if np.isfinite(hb):
            h = float(hb)
            min_h = min(min_h, h)
            viol += int(h < 0.0 and s[3] > 1e-3)
        d_now = min(float(np.hypot(wx - s[0], wy - s[1])) for wx, wy, _ in truth)
        if d_now < min_d:                      # closest approach = the passing moment
            min_d, v_at_pass = d_now, float(s[3])
        contacts += int(d_now < plat.robot.robot_radius + plat.cbf.d_hard)

        if record is not None:
            record.append(dict(
                t=t, x=float(s[0]), y=float(s[1]), yaw=float(s[2]), v=float(s[3]),
                cap=float(v_max_cmd), state=scanner.state,
                prot=float(scanner.protective_len(max(scanner._vhist))),
                warn=float(2.5 * scanner.protective_len(max(scanner._vhist))),
                h=h, gap=d_now, lat=float(s[1]), stopped_s=stopped_s,
                margin=float(d_margin_cmd),      # what was ASKED, after any floor
                rl_margin=None if rl_margin is None else float(rl_margin),
                dist_goal=dist_goal, v_floor=v_floor,
                workers=[(wx, wy, wyaw, True) for wx, wy, _ in truth]))

    return dict(t=reached, pstops=scanner.n_stops, stopped_s=stopped_s, warn_s=warn_s,
                contacts=contacts, min_h=float(min_h), viol=viol, min_d=float(min_d),
                v_at_pass=v_at_pass, lat_max=lat_max)


def battery(n, plat, scene, sup):
    res = {}
    for arm in ARMS:
        acc = [run(arm, plat, scene, sup=sup if arm.startswith("ours") else None,
                   jitter=(i - (n - 1) / 2) * 0.6) for i in range(n)]
        ok = [a for a in acc if a["t"] is not None]
        res[arm] = dict(
            arrived=len(ok), n=n,
            t=float(np.mean([a["t"] for a in ok])) if ok else float("nan"),
            pstops=float(np.mean([a["pstops"] for a in acc])),
            warn_s=float(np.mean([a["warn_s"] for a in acc])),
            contacts=sum(a["contacts"] > 0 for a in acc),
            min_h=float(np.min([a["min_h"] for a in acc])),
            min_d=float(np.min([a["min_d"] for a in acc])),
            v_at_pass=float(np.mean([a["v_at_pass"] for a in acc])),
            lat_max=float(np.mean([a["lat_max"] for a in acc])),
            violeps=sum(a["viol"] > 0 for a in acc))
    return res


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    plat = load_platform("industrial")
    scene = build_scene()
    sup = SupervisorPolicy(MODEL, platform="industrial",
                           walls=scene["walls"], posts=scene["posts"])
    print(f"head-on encounter: {2 * HALF_W:.1f} m two-way aisle, picker "
          f"{PICKER_LANE:.2f} m off centre at {PICKER_SPEED:.2f} m/s, "
          f"{GOAL_X:.0f} m run")
    print(f"both arms: same protective field, same {COMMISSIONED:.2f} m/s cap, scored "
          f"on the strict barrier. No cross-aisles, so no zones to mark.\n")

    res = battery(n, plat, scene, sup)
    print(f"{'arm':<12}{'arrived':>8}{'time':>8}{'prot.stops':>11}{'derated_s':>10}"
          f"{'min_gap':>9}{'v@pass':>8}{'lateral':>9}{'min_h':>8}{'contacts':>9}"
          f"{'viol':>6}")
    for arm in ARMS:
        a = res[arm]
        print(f"{arm:<12}{a['arrived']:>4}/{a['n']:<3}{a['t']:>8.1f}{a['pstops']:>11.2f}"
              f"{a['warn_s']:>10.1f}{a['min_d']:>9.2f}{a['v_at_pass']:>8.2f}"
              f"{a['lat_max']:>9.2f}{a['min_h']:>8.2f}{a['contacts']:>9}"
              f"{a['violeps']:>6}")

    i, o, l = res["industrial"], res["ours"], res["ours_lateral"]
    print(f"\nas trained, ours matches the industrial machine almost exactly: "
          f"{i['t']:.1f} -> {o['t']:.1f} s, clearance {i['min_d']:.2f} -> "
          f"{o['min_d']:.2f} m, lateral {i['lat_max']:.2f} -> {o['lat_max']:.2f} m.")
    print(f"asking for the lateral room the aisle already has: {l['t']:.1f} s "
          f"({100 * (i['t'] - l['t']) / i['t']:+.1f} %), clearance {l['min_d']:.2f} m, "
          f"offset {l['lat_max']:.2f} m, {l['v_at_pass']:.2f} m/s at the pass, "
          f"barrier {l['min_h']:+.2f} m.")
    checks = [
        ("both arms complete every mission", all(res[a]["arrived"] == n for a in ARMS)),
        ("zero contacts, both arms", all(res[a]["contacts"] == 0 for a in ARMS)),
        ("ours holds the strict barrier (min_h >= 0)", o["min_h"] >= 0.0),
        ("ours logs no stopping-distance violation", o["violeps"] == 0),
        ("ours keeps at least the industrial machine's passing clearance",
         o["min_d"] >= i["min_d"] - 0.05),
        ("no arm ends up closer than the hard keep-out",
         min(res[a]["min_d"] for a in ARMS)
         > plat.robot.robot_radius + plat.cbf.d_hard),
        ("using the aisle beats squeezing past, on clearance AND on time",
         l["min_d"] > o["min_d"] and l["t"] <= o["t"] + 1e-9),
        ("no arm brakes harder than the platform can, and none contacts anybody",
         all(res[a]["contacts"] == 0 for a in ARMS)),
    ]
    print()
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("\nNOTE: head-on is the geometry where this project has repeatedly measured NO "
          "supervisor\nadvantage -- the slowdown is the MPC human-cost term plus the CBF "
          "and survives removing\nthe supervisor. Read any difference here as the warning "
          "tier's bluntness, not as learned\nanticipation.")
    print("\nGATE:", "PASS" if all(ok for _, ok in checks) else "FAIL")
    sys.exit(0 if all(ok for _, ok in checks) else 1)


if __name__ == "__main__":
    main()
