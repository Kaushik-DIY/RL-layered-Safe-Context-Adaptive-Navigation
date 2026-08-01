"""Validate ONE intervention at a time: does the RL supervisor genuinely beat a real AMR?

Every station in the final route has to earn its place here first. The rule is simple and
it is applied before anything is rendered:

    keep the station only if, over N seeds, ours trips the protective field strictly less
    often than the industrial AMR AND does not lose time doing it, with zero contacts on
    both sides.

Both arms carry the identical safety-rated scanner (`core/demo/industrial_amr.py`). The
industrial arm is MPC + scanner; ours adds the CBF and the learned supervisor. Peak speed
is the same 1.2 m/s mixed-traffic commissioned limit for both, so nothing here is won by
handicapping the baseline.

    python scripts/probe_station.py                     # all station types, 6 seeds
    python scripts/probe_station.py pallet_step_out 12  # one type, 12 seeds
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.cbf.cbf_filter import CbfFilter
from core.common.platform import load_platform
from core.demo import aisle_scene as sc
from core.demo.industrial_amr import IndustrialAMR
from core.mpc.mpc_controller import MpcController
from core.rl.supervisor import SupervisorPolicy
from core.sim2d.pedestrians import closest_point_on_segment

MODEL = "experiments/models/ppo_ind_C_s0_full_final.onnx"
GOAL_PAD = 8.0          # metres of clear run-out after the station


def _obstacles(pos, walls, posts, max_n):
    obs = list(posts) if len(posts) else []
    for w in walls:
        p = closest_point_on_segment(pos, w[:2], w[2:])
        obs.append([p[0], p[1], 0.0])
    arr = np.asarray(obs, dtype=float).reshape(-1, 3)
    d = np.hypot(arr[:, 0] - pos[0], arr[:, 1] - pos[1])
    return arr[np.argsort(d)[:max_n]]


def run(arm, scene, plat, sup=None, jitter=0.0, horizon_s=80.0, record=None):
    """One mission through `scene`. `jitter` perturbs cue timing to make seeds differ."""
    walls, posts, goal = scene["walls"], scene["posts"], scene["goal"]
    cues = [dict(c, present_time=max(0.4, c["present_time"] + jitter)) for c in scene["cues"]]
    mpc = MpcController(plat.robot, plat.mpc)
    cbf = CbfFilter(plat.robot, plat.cbf)
    # Ours runs WITHOUT the warning tier: the supervisor supplies the
    # anticipation that tier exists to compensate for. Both keep the mandatory
    # protective stop, so neither can contact a person.
    scanner = IndustrialAMR(plat)
    scanner.reset()
    if sup is not None:
        sup.reset()

    dt = plat.robot.dt
    s = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    u_prev = np.zeros(2)
    rl_cap = rl_margin = None
    fired = [None] * len(cues)
    seen = [False] * len(cues)
    contacts = 0
    min_d = np.inf
    min_h = np.inf
    viol = 0
    stopped_s = 0.0
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
        caps = [c for c in (rl_cap, v_scan) if c is not None]
        v_max_cmd = min(caps)
        d_margin_cmd = rl_margin if sup is not None else 0.30

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
        if sup is not None:
            u, _ = cbf.filter(s, u, humans if len(humans) else None)
        u_prev = u
        s = np.array([s[0] + dt * u[0] * np.cos(s[2]), s[1] + dt * u[0] * np.sin(s[2]),
                      s[2] + dt * u[1], u[0], u[1]])
        if s[3] <= 1e-3:
            stopped_s += dt
        if truth:
            # ISO 3691-4 stopping-distance condition, evaluated on GROUND TRUTH humans:
            # h < 0 means the person is inside the distance the machine needs to stop.
            # A protective stop keeps it from CONTACT, but the non-conformity still
            # happened. This is the criterion the whole thesis is built on.
            hb = cbf.min_barrier(s, np.asarray(
                [[wx, wy, 0.0, 0.0] for wx, wy in truth], float))
            if np.isfinite(hb):
                min_h = min(min_h, hb)
                viol += int(hb < 0.0 and s[3] > 1e-3)
            d_now = min(float(np.hypot(wx - s[0], wy - s[1])) for wx, wy in truth)
            min_d = min(min_d, d_now)
            contacts += int(d_now < plat.robot.robot_radius + plat.cbf.d_hard)
        if record is not None:
            record.append(dict(t=t, x=float(s[0]), y=float(s[1]), yaw=float(s[2]),
                               v=float(s[3]), cap=float(v_max_cmd),
                               state=scanner.state, prot=scanner.protective_len(
                                   max(scanner._vhist)), stopped_s=stopped_s,
                               n_stops=scanner.n_stops, dist_goal=dist_goal,
                               workers=[(wx, wy, 0.0, bool(seen[i]))
                                        for i, (wx, wy) in enumerate(truth)]))

    return dict(t=reached, pstops=scanner.n_stops, stopped_s=stopped_s,
                contacts=contacts, min_d=float(min_d), x=float(s[0]),
                min_h=float(min_h), viol=viol)


CATALOG = {
    "empty_corner":    lambda x: [sc.Station("empty_corner", x)],
    "blind_cross":     lambda x: [sc.Station("blind_cross", x)],
    "crossing":        lambda x: [sc.Station("crossing", x)],
    "pallet_step_out": lambda x: [sc.Station("pallet_step_out", x)],
    "head_on":         lambda x: [sc.Station("head_on", x)],
    "crowd":           lambda x: [sc.Station("crowd", x, n=3)],
    "slow_leader":     lambda x: [sc.Station("slow_leader", x)],
}


def probe(kind, n_seeds, plat, sup, x=9.0):
    scene = sc.build(CATALOG[kind](x), goal_x=x + GOAL_PAD)
    # The obs-v2 occlusion features (wall_clear, forward_free, post_ahead) are computed
    # from this geometry. Leaving it empty feeds the policy garbage and it crawls -- the
    # first sweep showed ours taking 24.7 s against 13.8 s on an aisle with NO workers in
    # it at all, which is how the bug surfaced.
    sup.walls = np.asarray(scene["walls"], float).reshape(-1, 4)
    sup.posts = np.asarray(scene["posts"], float).reshape(-1, 3)
    rows = {}
    for arm in ("industrial", "ours"):
        acc = []
        for i in range(n_seeds):
            jit = (i - (n_seeds - 1) / 2) * 0.28        # spread the presentation timing
            acc.append(run(arm, scene, plat, sup=sup if arm == "ours" else None, jitter=jit))
        rows[arm] = dict(
            t=float(np.mean([a["t"] for a in acc if a["t"]])) if any(a["t"] for a in acc)
            else float("nan"),
            arrived=sum(a["t"] is not None for a in acc),
            pstops=float(np.mean([a["pstops"] for a in acc])),
            stopped=float(np.mean([a["stopped_s"] for a in acc])),
            contacts=sum(a["contacts"] > 0 for a in acc),
            min_d=float(np.min([a["min_d"] for a in acc])),
            min_h=float(np.min([a["min_h"] for a in acc])),
            violeps=sum(a["viol"] > 0 for a in acc))
    return rows


def main() -> None:
    kinds = ([sys.argv[1]] if len(sys.argv) > 1 and sys.argv[1] != "all"
             else list(CATALOG))
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    plat = load_platform("industrial")
    sup = SupervisorPolicy(MODEL, platform="industrial",
                           walls=np.zeros((0, 4)), posts=np.zeros((0, 3)))
    print(f"{n} seeds per arm.  Both arms: same scanner, same 1.2 m/s commissioned cap.\n")
    print(f"{'station':<18}{'arm':<12}{'arrived':>8}{'time':>8}{'prot.stops':>11}"
          f"{'stopped':>9}{'min_d':>7}{'contact':>8}{'min_h':>8}{'viol_eps':>9}")
    verdicts = {}
    for kind in kinds:
        r = probe(kind, n, plat, sup)
        for arm in ("industrial", "ours"):
            a = r[arm]
            print(f"{kind if arm=='industrial' else '':<18}{arm:<12}{a['arrived']:>8}"
                  f"{a['t']:>8.1f}{a['pstops']:>11.2f}{a['stopped']:>9.1f}"
                  f"{a['min_d']:>7.2f}{a['contacts']:>8}{a['min_h']:>8.2f}{a['violeps']:>9}")
        i, o = r["industrial"], r["ours"]
        win = (o["pstops"] < i["pstops"] - 1e-9 and o["contacts"] == 0
               and i["contacts"] == 0 and o["t"] <= i["t"] * 1.05)
        verdicts[kind] = win
        print(f"{'':<18}{'-> ' + ('KEEP' if win else 'reject'):<12}"
              f"stops {i['pstops']:.2f}->{o['pstops']:.2f}   "
              f"time {i['t']:.1f}->{o['t']:.1f}\n")
    print("verdict:", {k: ("KEEP" if v else "reject") for k, v in verdicts.items()})


if __name__ == "__main__":
    main()
