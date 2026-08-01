"""Offline gate for the industrial showcase demo.

Replays the EXACT Gazebo scene -- same geometry, same position-triggered worker cues,
same tracker occlusion rule -- through the real MpcController + CbfFilter, for each arm of
the comparison.

Every previous demo attempt was built first and checked afterwards. This runs first:
if the contrast is not here, it will not be in Gazebo either.

    A   commissioned AMR   core/demo/scanner_amr.py: 0.50 m/s + field switching
    C   ours               the exported ONNX SupervisorPolicy
    B   rated speed        no supervisor, platform max -- kept for the old B-vs-C video

WHAT THE GATE NOW ASSERTS, AND WHY IT CHANGED
---------------------------------------------
It used to assert that the baseline BREACHES. That was arm B, rated speed with no
commissioning -- a machine nobody would deploy, so "we are safer than it" was never the
claim worth making. Against a properly commissioned AMR the result is different and
stronger: BOTH arms keep h >= 0 everywhere, and the difference is throughput. So the gate
now requires zero violations from both, and a decisive mission-time gap.

    PYTHONPATH=$PWD .venv-navrl/bin/python scripts/verify_showcase.py
    PYTHONPATH=$PWD .venv-navrl/bin/python scripts/verify_showcase.py --trace
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from core.cbf.cbf_filter import CbfFilter
from core.common.platform import load_platform
from core.demo import industrial_amr
from core.demo.industrial_amr import IndustrialAMR
from core.demo.showcase_scene import (CUES, EVENT_X, GOAL, OCCLUSION_Y, POSTS,
                                      REVEAL_DISTANCE, WALLS, should_fire,
                                      staged_pose, walk_path)
from core.mpc.mpc_controller import MpcController
from core.rl.supervisor import SupervisorPolicy
from core.sim2d.pedestrians import closest_point_on_segment

MODEL = "experiments/models/ppo_ind_C_s0_full_final.onnx"


def mpc_obstacles(pos, max_n):
    """Posts + nearest point on each wall -- identical to NavEnv._mpc_obstacles."""
    obs = POSTS.tolist()
    for w in WALLS:
        p = closest_point_on_segment(pos, w[:2], w[2:])
        obs.append([p[0], p[1], 0.0])
    arr = np.asarray(obs, dtype=float)
    d = np.hypot(arr[:, 0] - pos[0], arr[:, 1] - pos[1])
    return arr[np.argsort(d)[:max_n]]


def run(arm: str = "ours", horizon_s: float = 90.0, trace: bool = False,
        presents=None, record=None) -> dict:
    """Replay the scene for one arm.

        "industrial"  MPC + safety-rated scanner, commissioned 1.2 m/s. NO CBF -- a
                      fielded AMR does not have one; the scanner is what keeps it safe.
        "ours"        MPC + CBF + RL supervisor + THE SAME scanner, same 1.2 m/s cap.
        "B" / "C"     the older rated-speed / RL pair, kept so the previous video stays
                      reproducible.

    Both real arms carry the scanner because it is mandatory equipment: withholding it from
    either would be a strawman. Neither can therefore contact a person, so the comparison
    is about how often the scanner has to FIRE, not about who is safe.

    Pass `record=[]` to collect a full per-step trajectory for the renderers.
    """
    if arm not in ("industrial", "ours", "B", "C"):
        raise ValueError(f"unknown arm {arm!r}")
    cues = CUES
    if presents is not None:                      # tuning sweep override
        cues = [dict(c, present_distance=p) for c, p in zip(CUES, presents)]
    plat = load_platform("industrial")
    mpc = MpcController(plat.robot, plat.mpc)
    cbf = CbfFilter(plat.robot, plat.cbf)
    use_cbf = arm in ("ours", "B", "C")
    sup = scanner = None
    if arm in ("ours", "C"):
        sup = SupervisorPolicy(MODEL, platform="industrial", walls=WALLS, posts=POSTS)
        sup.reset()
    if arm in ("industrial", "ours"):
        scanner = IndustrialAMR(plat)
        scanner.reset()

    dt = plat.robot.dt
    s = np.array([0.0, 0.0, 0.0, 0.0, 0.0])          # x, y, yaw, v, omega
    u_prev = np.zeros(2)
    v_max_cmd = d_margin_cmd = None
    rl_cap = rl_margin = None

    fired = [None] * len(cues)                        # cue -> fire time
    seen = [False] * len(cues)                        # tracker occlusion latch
    per_event = [dict(v_cmd=np.inf, v=np.inf, min_h=np.inf, min_d=np.inf, viol=0)
                 for _ in EVENT_X]
    stops = 0
    was_moving = False
    reached = None
    stopped_s = 0.0                                   # cumulative time at a standstill
    pstops = 0
    was_pstop = False
    contacts = 0                                      # the safety floor: must stay 0
    min_d = np.inf

    for k in range(int(horizon_s / dt)):
        t = k * dt

        # ---- position-triggered worker cues (the ISO presentation test) -----
        rows = []
        for i, cue in enumerate(cues):
            if fired[i] is None:
                if should_fire(cue, s[0], s[3]):
                    fired[i] = t
                hx, hy, _ = staged_pose(cue)
                vx = vy = 0.0
            else:
                el = t - fired[i]
                hx, hy, yaw, _ = walk_path(cue, el)
                nx, ny, _, _ = walk_path(cue, el + dt)
                vx, vy = (nx - hx) / dt, (ny - hy) / dt
            d = float(np.hypot(hx - s[0], hy - s[1]))
            if hy <= OCCLUSION_Y or d <= REVEAL_DISTANCE:
                seen[i] = True
            if seen[i]:
                rows.append([hx, hy, vx, vy])
        humans = np.asarray(rows, dtype=float).reshape(-1, 4)

        if k % plat.rl.decision_every == 0 and sup is not None:
            rl_cap, rl_margin = sup.compute(s, GOAL, humans)
        # The scanner is safety-rated hardware and overrides the planner every cycle, not
        # just on the supervisor's 2 Hz tick. The two caps are recombined FRESH each step:
        # carrying `v_max_cmd` into a min() ratchets it down permanently, so the machine
        # never recovers speed after its first warning-field trip.
        v_scan = scanner(s, humans, t)[0] if scanner is not None else None
        caps = [c for c in (rl_cap, v_scan) if c is not None]
        v_max_cmd = min(caps) if caps else None
        d_margin_cmd = rl_margin if sup is not None else industrial_amr.MARGIN

        to_goal = GOAL - s[:2]
        dist_goal = float(np.hypot(*to_goal))
        if dist_goal < 0.15:
            reached = t
            break
        carrot = s[:2] + to_goal / dist_goal * min(plat.mpc.carrot_lookahead, dist_goal)
        u, _ = mpc.solve(x0=s[:3], carrot=carrot,
                         static_obs=mpc_obstacles(s[:2], plat.mpc.max_static_obstacles),
                         humans=humans if len(humans) else None,
                         v_max_cmd=v_max_cmd, d_margin_cmd=d_margin_cmd, u_prev=u_prev)
        if use_cbf:
            u_safe, info = cbf.filter(s, u, humans if len(humans) else None)
        else:                       # industrial arm: the scanner is the safety layer
            u_safe = u
            info = {"intervention": 0.0, "protective_stop": False,
                    "h_min": cbf.min_barrier(s, humans) if len(humans) else np.inf}
        u_prev = u_safe
        s = np.array([s[0] + dt * u_safe[0] * np.cos(s[2]),
                      s[1] + dt * u_safe[0] * np.sin(s[2]),
                      s[2] + dt * u_safe[1], u_safe[0], u_safe[1]])

        if was_moving and s[3] <= 1e-3:
            stops += 1
        was_moving = s[3] > 0.05
        if s[3] <= 1e-3:
            stopped_s += dt
        pstop_now = bool(info["protective_stop"]) or (
            scanner is not None and scanner.state == industrial_amr.STOPPED)
        if pstop_now and not was_pstop:
            pstops += 1
        was_pstop = pstop_now

        if len(humans):
            d_now = float(np.min(np.hypot(humans[:, 0] - s[0], humans[:, 1] - s[1])))
            min_d = min(min_d, d_now)
            # contact = the person is inside the robot's physical footprint plus the hard
            # keep-out. This is the floor neither machine may ever breach.
            contacts += int(d_now < plat.robot.robot_radius + plat.cbf.d_hard)
        h = float(info["h_min"])
        cmd = plat.robot.v_max if v_max_cmd is None else v_max_cmd
        for e, ex in enumerate(EVENT_X):                # attribute to nearest station
            if abs(s[0] - ex) <= 3.0:
                pe = per_event[e]
                pe["v_cmd"] = min(pe["v_cmd"], cmd)
                pe["v"] = min(pe["v"], float(s[3]))
                if np.isfinite(h):
                    pe["min_h"] = min(pe["min_h"], h)
                    pe["viol"] += int(h < 0.0)
                if len(humans):
                    pe["min_d"] = min(pe["min_d"], float(np.min(
                        np.hypot(humans[:, 0] - s[0], humans[:, 1] - s[1]))))
        if record is not None:
            # everything the offline renderer needs to draw the frame
            all_pos = []
            for i, cue in enumerate(cues):
                if fired[i] is None:
                    px, py, pyaw = staged_pose(cue)
                else:
                    px, py, pyaw, _ = walk_path(cue, t - fired[i])
                all_pos.append((px, py, pyaw, bool(seen[i])))
            record.append(dict(t=t, x=float(s[0]), y=float(s[1]), yaw=float(s[2]),
                               v=float(s[3]), cap=float(cmd),
                               h=h if np.isfinite(h) else None,
                               dist_goal=dist_goal, stopped_s=stopped_s,
                               state=(scanner.state if scanner is not None else None),
                               prot_len=(scanner.protective_len(max(s[3], 0.42))
                                         if scanner is not None else None),
                               pstop=pstop_now, n_stops=(scanner.n_stops if scanner
                                                         else pstops),
                               workers=all_pos))

        if trace and k % 10 == 0:
            print(f"   t={t:5.1f} x={s[0]:6.2f} v={s[3]:.2f} "
                  f"cmd={cmd:.2f} h={h if np.isfinite(h) else float('nan'):6.2f}")

    return dict(t=reached, x=float(s[0]), events=per_event, stops=stops, fired=fired,
                pstops=(scanner.n_stops if scanner is not None else pstops),
                stopped_s=stopped_s, arm=arm, contacts=contacts, min_d=min_d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--present", nargs=2, type=float, default=None,
                    metavar=("B", "C"), help="override presentation distances")
    args = ap.parse_args()

    labels = {"industrial": "INDUSTRIAL AMR  (MPC + safety scanner, 1.2 m/s commissioned)",
              "ours": "OURS  (MPC + CBF + RL supervisor + the same scanner)"}
    res = {}
    for arm in ("ours", "industrial"):
        print(f"\n=== {labels[arm]} ===")
        res[arm] = run(arm, trace=args.trace, presents=args.present)
        r = res[arm]
        arrive = f"{r['t']:.1f} s" if r["t"] else f"DID NOT ARRIVE (x={r['x']:.1f})"
        print(f"  mission: {arrive}   PROTECTIVE STOPS: {r['pstops']}   "
              f"standstill: {r['stopped_s']:.1f} s   contacts: {r['contacts']}   "
              f"closest approach: {r['min_d']:.2f} m")
        print(f"  cue fire times: {[None if f is None else round(f,1) for f in r['fired']]}")
        names = ["A blind corner (nobody there)", "B worker crosses intersection",
                 "C occluded worker steps out"]
        for nm, e in zip(names, r["events"]):
            md = "n/a" if not np.isfinite(e["min_d"]) else f"{e['min_d']:.2f} m"
            mh = "n/a" if not np.isfinite(e["min_h"]) else f"{e['min_h']:+.2f}"
            print(f"    {nm:<28} v_cmd_min {e['v_cmd']:.2f}  v_min {e['v']:.2f}  "
                  f"min_h {mh}  closest {md}  viol_steps {e['viol']}")

    o, i = res["ours"], res["industrial"]
    ratio = (i["t"] / o["t"]) if (i["t"] and o["t"]) else 0.0
    print(f"\n=== GATE ===   mission-time ratio industrial/ours = {ratio:.2f}")
    checks = [
        # Both machines carry the scanner, so NEITHER may ever touch a person. That is the
        # floor; if it fails, the field model is wrong and nothing else is trustworthy.
        ("no contact, industrial", i["contacts"] == 0),
        ("no contact, ours", o["contacts"] == 0),
        ("both complete the mission", o["t"] is not None and i["t"] is not None),
        # The claim: ours keeps the robot out of states where the scanner must fire.
        ("industrial trips the protective field at least twice", i["pstops"] >= 2),
        ("ours trips it strictly less often", o["pstops"] < i["pstops"]),
        ("ours loses less time at a standstill", o["stopped_s"] < i["stopped_s"]),
        ("ours is at least 25 % faster", ratio >= 1.25),
        ("ours slows on geometry alone at the blind corner (v_cmd <= 0.95)",
         o["events"][0]["v_cmd"] <= 0.95),
    ]
    ok = True
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok &= bool(passed)
    print(f"\n  protective stops : ours {o['pstops']}   vs   industrial {i['pstops']}")
    print(f"  standstill time  : ours {o['stopped_s']:.1f} s   vs   "
          f"industrial {i['stopped_s']:.1f} s")
    print(f"  mission time     : ours {o['t']}   vs   industrial {i['t']}")
    print(f"\n{'GATE PASSED' if ok else 'GATE FAILED - diagnose before rendering'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
