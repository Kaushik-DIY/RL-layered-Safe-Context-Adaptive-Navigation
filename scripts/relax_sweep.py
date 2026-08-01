"""How much can the supervisor be relaxed before it starts costing safety?

The measured problem: our policy is SAFER THAN NECESSARY and pays throughput for it. It
holds the CBF barrier well clear of zero while a conventional AMR, protected only by its
scanner, is already compliant. So the question is whether a relaxed cap recovers the
throughput without giving up the safety outcome.

The relaxation is a post-hoc scaling of the policy's commanded cap,
`v = min(commissioned, alpha * rl_cap)`, swept from alpha = 1 (the policy as trained) up to
alpha = inf (no supervision at all). No retraining.

CONFIGURATION UNDER TEST -- this is the point of the sweep. The warning field is NOT a
certified safety function; vendor and consultant sources are explicit that "the protective
field is the only field that contributes to AGV safety certification", and speed-dependent
zone selection is permissive (ISO 3691-4 cl. 4.8.2.6 "can be adaptive"). So:

    industrial : MPC + warning field (-> 0.60 m/s) + protective field (-> stop)
    ours       : MPC + CBF + relaxed supervisor + protective field ONLY

Ours therefore is not clamped to 0.60 m/s every time somebody is inside a 5 m box, and its
anticipation has to earn that freedom. It is kept only if it holds ZERO contacts and no
more protective stops than the industrial machine.

    PYTHONPATH=$PWD .venv-navrl/bin/python scripts/relax_sweep.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "probe_station", Path(__file__).resolve().parent / "probe_station.py")
pb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pb)

from core.cbf.cbf_filter import CbfFilter          # noqa: E402
from core.common.platform import load_platform     # noqa: E402
from core.demo import aisle_scene as sc            # noqa: E402
from core.demo.industrial_amr import IndustrialAMR  # noqa: E402
from core.mpc.mpc_controller import MpcController  # noqa: E402
from core.rl.supervisor import SupervisorPolicy    # noqa: E402

STATIONS = ["blind_cross", "crossing", "head_on", "crowd"]
ALPHAS = [1.0, 1.3, 1.7, 2.5, float("inf")]


def run(scene, plat, sup, alpha, use_warning, use_cbf=True, jitter=0.0,
        horizon_s=80.0):
    walls, posts, goal = scene["walls"], scene["posts"], scene["goal"]
    cues = [dict(c, present_time=max(0.4, c["present_time"] + jitter))
            for c in scene["cues"]]
    mpc = MpcController(plat.robot, plat.mpc)
    cbf = CbfFilter(plat.robot, plat.cbf)
    scanner = IndustrialAMR(plat, use_warning=use_warning)
    scanner.reset()
    if sup is not None:
        sup.reset()

    dt = plat.robot.dt
    s = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    u_prev = np.zeros(2)
    rl_cap = rl_margin = None
    fired = [None] * len(cues)
    seen = [False] * len(cues)
    contacts = viol = 0
    min_h = np.inf
    reached = None

    for k in range(int(horizon_s / dt)):
        t = k * dt
        rows, truth = [], []
        for i, cue in enumerate(cues):
            if fired[i] is None:
                if sc.should_fire(cue, s[0], s[3]):
                    fired[i] = t
                wx, wy, _ = sc.staged_pose(cue)
                vx = vy = 0.0
            else:
                el = t - fired[i]
                wx, wy, _, _ = sc.walk_path(cue, el)
                nx, ny, _, _ = sc.walk_path(cue, el + dt)
                vx, vy = (nx - wx) / dt, (ny - wy) / dt
            truth.append((wx, wy))
            if sc.visible(cue, wx, wy, s[:2]):
                seen[i] = True
            if seen[i]:
                rows.append([wx, wy, vx, vy])
        humans = np.asarray(rows, dtype=float).reshape(-1, 4)

        if sup is not None and k % plat.rl.decision_every == 0:
            rl_cap, rl_margin = sup.compute(s, goal, humans)
        v_scan = scanner(s, humans, t)[0]
        cap = v_scan if rl_cap is None else min(v_scan, alpha * rl_cap)
        margin = 0.30 if rl_margin is None else rl_margin

        to_goal = goal - s[:2]
        dist_goal = float(np.hypot(*to_goal))
        if dist_goal < 0.15:
            reached = t
            break
        carrot = s[:2] + to_goal / dist_goal * min(plat.mpc.carrot_lookahead, dist_goal)
        u, _ = mpc.solve(x0=s[:3], carrot=carrot,
                         static_obs=pb._obstacles(s[:2], walls, posts,
                                                  plat.mpc.max_static_obstacles),
                         humans=humans if len(humans) else None,
                         v_max_cmd=cap, d_margin_cmd=margin, u_prev=u_prev)
        if use_cbf and sup is not None:
            u, _ = cbf.filter(s, u, humans if len(humans) else None)
        u_prev = u
        s = np.array([s[0] + dt * u[0] * np.cos(s[2]), s[1] + dt * u[0] * np.sin(s[2]),
                      s[2] + dt * u[1], u[0], u[1]])
        if truth:
            arr = np.asarray([[wx, wy, 0.0, 0.0] for wx, wy in truth], float)
            hb = cbf.min_barrier(s, arr)
            if np.isfinite(hb):
                min_h = min(min_h, hb)
                viol += int(hb < 0.0 and s[3] > 1e-3)
            d_now = min(float(np.hypot(wx - s[0], wy - s[1])) for wx, wy in truth)
            contacts += int(d_now < plat.robot.robot_radius + plat.cbf.d_hard)

    return dict(t=reached, pstops=scanner.n_stops, contacts=contacts,
                min_h=float(min_h), viol=viol)


def main() -> None:
    plat = load_platform("industrial")
    sup = SupervisorPolicy(pb.MODEL, platform="industrial",
                           walls=np.zeros((0, 4)), posts=np.zeros((0, 3)))
    n = 4
    print("industrial = MPC + warning + protective.  "
          "ours = MPC + CBF + relaxed supervisor + PROTECTIVE ONLY.\n")
    totals = {}
    for kind in STATIONS:
        scene = sc.build(pb.CATALOG[kind](9.0), goal_x=17.0)
        sup.walls = np.asarray(scene["walls"], float).reshape(-1, 4)
        sup.posts = np.asarray(scene["posts"], float).reshape(-1, 3)
        base = [run(scene, plat, None, 1.0, True, jitter=(i - 1.5) * 0.28)
                for i in range(n)]
        bt = float(np.mean([b["t"] for b in base if b["t"]]))
        bs = float(np.mean([b["pstops"] for b in base]))
        print(f"{kind}:  industrial {bt:.1f} s, {bs:.2f} prot.stops, "
              f"{sum(b['contacts'] > 0 for b in base)} contact eps")
        for a in ALPHAS:
            acc = [run(scene, plat, sup, a, False, jitter=(i - 1.5) * 0.28)
                   for i in range(n)]
            t = float(np.mean([x["t"] for x in acc if x["t"]])) if any(
                x["t"] for x in acc) else float("nan")
            ps = float(np.mean([x["pstops"] for x in acc]))
            ct = sum(x["contacts"] > 0 for x in acc)
            vi = sum(x["viol"] > 0 for x in acc)
            mh = float(np.min([x["min_h"] for x in acc]))
            tag = "no supervision" if np.isinf(a) else f"alpha {a:.1f}"
            ok = "OK " if (ct == 0 and ps <= bs + 1e-9) else "!! "
            gain = 100.0 * (bt - t) / bt if t == t else float("nan")
            print(f"    {ok}{tag:<16} {t:5.1f} s ({gain:+5.1f} %)   "
                  f"prot.stops {ps:.2f}   contacts {ct}   min_h {mh:+.2f}   viol_eps {vi}")
            totals.setdefault(a, []).append((bt, t, ps <= bs + 1e-9 and ct == 0))
        print()
    print("=== across all stations ===")
    for a in ALPHAS:
        rows = totals[a]
        gain = 100.0 * (sum(b for b, _, _ in rows) - sum(t for _, t, _ in rows)) / \
            sum(b for b, _, _ in rows)
        safe = all(ok for _, _, ok in rows)
        tag = "no supervision" if np.isinf(a) else f"alpha {a:.1f}"
        print(f"  {tag:<16} throughput {gain:+5.1f} %   safety kept: {safe}")


if __name__ == "__main__":
    main()
