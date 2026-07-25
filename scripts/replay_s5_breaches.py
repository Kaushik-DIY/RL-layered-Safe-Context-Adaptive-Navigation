"""Replay + classify the S5 breach episodes (week-4 audit follow-up).

The S5 stress battery flagged 18 breach episodes out of 1000 adversarial runs.
None is a claim-killer on its own, but a clean S5 safety statement needs each one
attributed: did the CBF FAIL (leave shed-able speed on the table while the robot was
itself closing on a human), or was the breach UNAVOIDABLE for a non-reversing robot
(v_min = 0) already braking as hard as physics allows while a fast SFM pedestrian
out-closed it -- the constant-velocity-vs-social-force crowd mismatch that the sigma
inflation absorbs in the open but not in a dense squeeze?

The discriminator is NOT the robot's total speed (in a crowd it rolls toward its
goal while a pedestrian closes from another bearing, which the CBF's closing-only
cap correctly does not forbid). It is whether the filter was on the maximal-braking
trajectory (dv <= -a_brake*dt, or at rest, or in a protective stop) throughout the
descent of the barrier h into the breach:

    * filter braking maximally / stopped as h<0   -> UNAVOIDABLE (no better command)
    * robot closing (v_los>0) yet NOT braking hard -> FILTER-ACCOUNTABLE

We reconstruct each episode bit-for-bit (env seed = seed_base + i, adversary rng =
10000 + i, exactly as run_s5.py drives them), record the per-step trace, locate the
deepest barrier dip, and quantify who shut the gap (robot vs human closing).

    python scripts/replay_s5_breaches.py            # all 18 from s5_2d.csv
    python scripts/replay_s5_breaches.py random open_hall 31   # one episode, verbose
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from core.common.params import CbfParams, RlParams, RobotParams, load_yaml
from core.rl.nav_env import NavEnv
from scripts.run_s5 import make_adversary

RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results"


def replay(adversary: str, scenario: str, episode: int, seed_base: int, rl: RlParams,
           robot: RobotParams, cbf: CbfParams, verbose: bool = False) -> dict:
    """Re-run one S5 episode with recording; return the breach mechanism."""
    env = NavEnv(scenarios=[scenario], use_cbf=True, record=True)
    rng = np.random.default_rng(10_000 + episode)
    policy = make_adversary(adversary, rl, rng)
    obs, _ = env.reset(seed=seed_base + episode)
    ep = None
    while ep is None:
        obs, _, term, trunc, info = env.step(policy(obs))
        if term or trunc:
            ep = info["episode_metrics"]
    traj = env.trajectory

    d = np.array([r["d_human"] for r in traj])
    h = np.array([r["h"] for r in traj])
    v = np.array([r["v_safe"] for r in traj])
    i_dmin = int(np.argmin(d))          # closest physical approach
    i_hmin = int(np.argmin(h))          # deepest barrier dip (the breach instant)

    # v_min = 0: the robot CANNOT reverse. Its single best defensive move is to brake
    # at a_brake and stop. So the filter is "doing everything admissible" iff it is on
    # the maximal-braking trajectory (dv <= -a_brake*dt) or already at rest / in a
    # protective stop, throughout the descent of h into the breach. If so, no other
    # command would have kept h >= 0 -- the pedestrian's own approach carried the gap
    # shut faster than a non-reversing robot can yield (the CV-vs-SFM crowd mismatch
    # the sigma inflation absorbs in the open but not in a dense squeeze). The filter
    # is only ACCOUNTABLE if, while the robot was itself closing (v_los > 0), it left
    # shed-able speed on the table instead of braking hard as h fell.
    max_brake = cbf.a_brake * robot.dt
    dv = np.diff(v, prepend=v[0])
    brake_or_stopped = np.array([
        (dv[j] <= -max_brake + 5e-3) or (v[j] <= 5e-3) or traj[j]["protective_stop"]
        for j in range(len(traj))])

    # descent window: from the last h>=0 before the breach up to the breach
    j0 = i_hmin
    while j0 > 0 and h[j0 - 1] < 0.0:
        j0 -= 1
    j0 = max(0, j0 - 1)
    descent = range(j0, i_hmin + 1)
    # who shut the gap over the descent: integrate robot vs human closing speed
    robot_closed = sum(traj[j]["v_los"] * robot.dt for j in descent)
    human_closed = sum(max(0.0, traj[j]["human_closing"]) * robot.dt for j in descent)
    # was the robot ever failing to brake hard while itself closing during the descent?
    slack_step = any((traj[j]["v_los"] > 0.03) and (not brake_or_stopped[j])
                     and (h[j] < 0.0) for j in descent)

    v_los_at_breach = float(traj[i_hmin]["v_los"])
    human_closing_at_breach = float(traj[i_hmin]["human_closing"])
    v_total_at_breach = float(traj[i_hmin]["v_safe"])
    braking_at_breach = bool(brake_or_stopped[i_hmin])

    if slack_step:
        verdict = "FILTER-ACCOUNTABLE (robot closing, not braking maximally as h<0)"
    else:
        verdict = "UNAVOIDABLE (filter braking maximally / stopped; human out-closed it)"

    result = {
        "adversary": adversary, "scenario": scenario, "episode": episode,
        "min_d": float(d[i_dmin]), "min_h": float(h[i_hmin]),
        "footprint_contact": bool(d[i_dmin] < robot.robot_radius),
        "v_at_breach": v_total_at_breach,
        "v_los_at_breach": v_los_at_breach,
        "human_closing_at_breach": human_closing_at_breach,
        "robot_closed_m": float(robot_closed), "human_closed_m": float(human_closed),
        "braking_at_breach": braking_at_breach,
        "pstops": ep["protective_stops"], "full_stops": ep["full_stops"],
        "verdict": verdict,
    }

    if verbose:
        print(f"\n=== {adversary}/{scenario} ep {episode} "
              f"(env seed {seed_base + episode}, adv rng {10_000 + episode}) ===")
        print(f"  min_d={d[i_dmin]:.3f} @t={traj[i_dmin]['t']:.1f}s   "
              f"min_h={h[i_hmin]:+.3f} @t={traj[i_hmin]['t']:.1f}s")
        print(f"  footprint(r={robot.robot_radius}) contact: {d[i_dmin] < robot.robot_radius}"
              f"   d_hard={cbf.d_hard}")
        print(f"  @ breach:  v={v_total_at_breach:.3f}  v_los(robot->human)="
              f"{v_los_at_breach:.3f}  human_closing={human_closing_at_breach:+.3f} m/s"
              f"  braking_maximally={braking_at_breach}")
        print(f"  gap closed over descent: robot {robot_closed:.3f} m vs human "
              f"{human_closed:.3f} m")
        print(f"  protective_stops={ep['protective_stops']}  full_stops={ep['full_stops']}")
        print("  trace around breach (t | d | h | v_mpc->v_safe | v_los | h_close | pstop):")
        for j in range(max(0, i_hmin - 5), min(len(traj), i_hmin + 4)):
            r = traj[j]
            mark = " <-- min h" if j == i_hmin else ""
            print(f"    {r['t']:5.1f} | d={r['d_human']:.3f} | h={r['h']:+.3f} | "
                  f"{r['v_mpc']:.3f}->{r['v_safe']:.3f} | vlos={r['v_los']:.3f}"
                  f" | hcl={r['human_closing']:+.3f} | ps={int(r['protective_stop'])}{mark}")
        print(f"  --> {verdict}")
    return result


def main() -> None:
    cfg = load_yaml("scenarios")
    seed_base = cfg["seed_base"]
    rl, robot, cbf = RlParams.from_yaml(), RobotParams.from_yaml(), CbfParams.from_yaml()

    if len(sys.argv) == 4:                       # single episode, verbose
        adversary, scenario, episode = sys.argv[1], sys.argv[2], int(sys.argv[3])
        replay(adversary, scenario, episode, seed_base, rl, robot, cbf, verbose=True)
        return

    df = pd.read_csv(RESULTS / "s5_2d.csv")
    breaches = df[(df["violation_steps"] > 0) | (df["collision"])]
    print(f"Replaying {len(breaches)} S5 breach episodes...\n")
    rows = []
    for _, b in breaches.iterrows():
        rows.append(replay(b["adversary"], b["scenario"], int(b["episode"]),
                           seed_base, rl, robot, cbf, verbose=False))

    out = pd.DataFrame(rows)
    print(out[["adversary", "scenario", "episode", "min_d", "min_h", "footprint_contact",
               "v_at_breach", "robot_closed_m", "human_closed_m", "braking_at_breach",
               "verdict"]].to_string(index=False))
    out.to_csv(RESULTS / "s5_breach_classification.csv", index=False)

    n_filter = int(out["verdict"].str.startswith("FILTER").sum())
    n_unavoid = len(out) - n_filter
    print(f"\n  unavoidable (filter braking maximally; human out-closed it) : {n_unavoid}")
    print(f"  filter-accountable (shed-able speed left on the table)      : {n_filter}")
    print(f"  footprint contacts (d < r_robot)                            : {int(out['footprint_contact'].sum())}")
    print(f"  braking maximally at the breach instant                     : {int(out['braking_at_breach'].sum())}/{len(out)}")
    print(f"\nrows -> {RESULTS / 's5_breach_classification.csv'}")


if __name__ == "__main__":
    main()
