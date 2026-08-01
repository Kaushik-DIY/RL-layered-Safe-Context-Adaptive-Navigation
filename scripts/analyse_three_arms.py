"""Classify every ISO breach, then compare efficiency. THE GATE for the whole claim.

Raw `h < 0` counts are the wrong compliance metric. A certified AMR trips its protective
field exactly at `h = 0` (the field IS the stopping distance) and then decelerates -- during
which `h` stays negative while the machine is still moving. Counting those as violations
would fail a machine that is behaving perfectly.

The criterion the S5 battery was signed off with (`scripts/replay_s5_breaches.py`, 0
filter-accountable / 18 unavoidable in 1000 adversarial runs):

    FILTER-ACCOUNTABLE   the robot was closing on a human (v_los > 0) while h < 0 and it
                         was NOT on the maximal-braking trajectory -- it left shed-able
                         speed on the table. A machine failure.
    UNAVOIDABLE          throughout the descent of h the filter was braking at a_brake,
                         at rest, or in a protective stop. No admissible command would
                         have kept h >= 0: a non-reversing robot was out-closed by the
                         pedestrian. Residual risk under ISO 3691-4, not a machine fault.

This is STRICTER than the S5 script, which classified only the deepest dip per episode:
here every contiguous h < 0 run is classified and an episode counts as accountable if ANY
run is.

    python scripts/analyse_three_arms.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core.common.platform import load_platform
from core.sim2d.scenarios import HOLDOUT_CLEAN, HOLDOUT_SCENARIO

RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results"
REAL = ["blind_corner", "corridor_passby", "doorway_negotiation", "open_hall",
        "perpendicular_crossing"]
ARMS = ["A_commissioned", "B_rated", "C_rl"]
LABEL = {"A_commissioned": "A  commissioned AMR (0.5 m/s)",
         "B_rated": "B  rated speed (1.5 m/s)",
         "C_rl": "C  MPC + CBF + RL (ours)"}


def breach_runs(h: np.ndarray) -> list[range]:
    """Contiguous h < 0 runs, each extended one step back to catch the descent."""
    bad = h < 0.0
    runs, j = [], 0
    while j < len(bad):
        if bad[j]:
            k = j
            while k + 1 < len(bad) and bad[k + 1]:
                k += 1
            runs.append(range(max(0, j - 1), k + 1))
            j = k + 1
        else:
            j += 1
    return runs


def classify(g: pd.DataFrame, a_brake: float, dt: float, lat: int) -> dict:
    """Attribute every breach. Four mechanisms, three of them the machine's problem.

    Built on the two quantities `replay_s5_breaches.py` integrates -- how much of the gap
    the ROBOT closed versus how much the HUMAN closed -- because that is what separates
    "drove into someone" from "was walked into".

      sight_limited  h < 0 while h_seen >= 0: the tracker never saw them. Not a filter
                     fault, but a machine fault under ISO 3691-4, which requires speed
                     limited to sight distance.
      filter_fault   the filter left admissible braking unused while the gap was closing.
                     Should be ~0 for a correct CBF (S5: 0 in 1000 runs).
      speed_fault    no admissible action remained when the hazard became detectable, and
                     the ROBOT closed most of the gap: it arrived too fast. This is what
                     commissioning speed zones fix by hand and what the RL should fix by
                     anticipation.
      unavoidable    the HUMAN closed most of the gap on a robot already braking or at
                     rest. Residual risk under ISO 3691-4, not a machine fault.

    Latency is the trap. `tau_latency = 0.5 s` = `lat` steps of command pipeline, and
    `KinematicSim` has no acceleration state, so the speed in effect at step j is the
    command issued at j - lat. Measured directly: the filter cuts its command from 0.80 to
    0.20 in one step while the applied speed keeps RISING to 0.84 for five more steps, h
    plunges to -0.57 during exactly that window, and recovers the instant the brake lands.
    Asking "was it braking at step j" of a command already sitting at its floor reports a
    fault where the machine did everything it could -- so admissibility is evaluated over
    the window [j-lat, j] whose commands govern step j.
    """
    h = g["h"].to_numpy()
    h_seen = g["h_seen"].to_numpy()
    v = g["v_safe"].to_numpy()
    v_los = g["v_los_crit"].to_numpy()
    closing = g["closing_crit"].to_numpy()
    pstop = g["protective_stop"].to_numpy().astype(bool)
    out = {"breached": False, "sight_limited": False, "filter_fault": False,
           "speed_fault": False, "unavoidable": False, "machine_fault": False,
           "n_runs": 0, "min_h": float(h.min())}
    if not (h < 0.0).any():
        return out

    dv = np.diff(v, prepend=v[0])
    ok = (dv <= -a_brake * dt + 5e-3) | (v <= 5e-3) | pstop
    # admissible somewhere in the latency window that governs this instant
    ok_eff = np.array([ok[max(0, j - lat): j + 1].any() for j in range(len(ok))])
    net = v_los + closing                       # > 0 means the gap is actually shrinking

    tally = {"sight_limited": 0, "filter_fault": 0, "speed_fault": 0, "unavoidable": 0}
    for run in breach_runs(h):
        neg = [j for j in run if h[j] < 0.0]
        if all(h_seen[j] >= 0.0 for j in neg):
            tally["sight_limited"] += 1
        elif any((net[j] > 0.03) and (not ok_eff[j]) for j in neg):
            tally["filter_fault"] += 1
        else:
            robot_closed = sum(max(0.0, v_los[j]) * dt for j in run)
            human_closed = sum(max(0.0, closing[j]) * dt for j in run)
            tally["speed_fault" if robot_closed > human_closed
                  else "unavoidable"] += 1
    out.update(breached=True, n_runs=sum(tally.values()),
               **{k: v > 0 for k, v in tally.items()})
    out["machine_fault"] = bool(out["sight_limited"] or out["filter_fault"]
                                or out["speed_fault"])
    return out


def dynamics(g: pd.DataFrame, a_brake: float, dt: float) -> dict:
    """Efficiency / ride-quality metrics that the episode summary cannot give.

    Deceleration is differenced from `v_safe`, the COMMAND. `v_applied` is that same
    command replayed 0.5 s later through the latency buffer (KinematicSim has no
    acceleration state), so differencing it measures the command sequence with a delay and
    reports impossible peaks.
    """
    v = g["v_safe"].to_numpy()
    x, y = g["x"].to_numpy(), g["y"].to_numpy()
    d = g["d_human"].to_numpy()
    decel = -np.diff(v, prepend=v[0]) / dt            # >0 while slowing
    hard = decel > 0.5 * a_brake
    # count RUNS, not steps: one long brake is one event, not fifteen
    n_hard = int(np.sum(hard & ~np.r_[False, hard[:-1]]))
    path = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
    # anticipation: speed while the nearest hazard is still 3 m away -- before any
    # occluded pedestrian could possibly be visible (reveal is 1.2 m)
    far = (d > 2.8) & (d < 3.2)
    return {"peak_decel": float(decel.max()), "hard_brakes": n_hard,
            "path_m": path, "v_at_3m": float(v[far].mean()) if far.any() else np.nan}


def main() -> None:
    plat = load_platform("industrial")
    a_brake, dt = plat.cbf.a_brake, plat.robot.dt
    eps = pd.read_csv(RESULTS / "three_arms.csv")
    tr = pd.read_csv(RESULTS / "three_arms_traces.csv.gz")

    lat = int(round(plat.robot.tau_latency / dt))
    rows = []
    for (arm, scen, i), g in tr.groupby(["arm", "scenario", "episode"], sort=False):
        g = g.sort_values("step")
        rows.append({"arm": arm, "scenario": scen, "episode": i,
                     **classify(g, a_brake, dt, lat), **dynamics(g, a_brake, dt)})
    cls = pd.DataFrame(rows)
    cls = cls.merge(eps[["arm", "scenario", "episode", "success", "time_to_goal",
                         "protective_stops", "full_stops", "energy", "collision",
                         "violation_steps"]],
                    on=["arm", "scenario", "episode"], how="left")
    cls.to_csv(RESULTS / "three_arms_classified.csv", index=False)

    real = cls[cls.scenario.isin(REAL)]

    print("=" * 78)
    print("THE GATE -- ISO 3691-4 compliance, 5 industrial scenarios, 12 paired seeds")
    print("=" * 78)
    gate = real.groupby("arm").agg(
        episodes=("breached", "size"),
        raw_breach_eps=("breached", "sum"),
        MACHINE_FAULT=("machine_fault", "sum"),
        speed_fault=("speed_fault", "sum"),
        filter_fault=("filter_fault", "sum"),
        sight_limited=("sight_limited", "sum"),
        unavoidable=("unavoidable", "sum"),
        collisions=("collision", "sum"),
        min_h=("min_h", "min"))
    print(gate.reindex(ARMS).to_string())
    print("\n  speed_fault   arrived too fast: nothing admissible left when detected")
    print("  filter_fault  admissible braking left unused (should be ~0)")
    print("  sight_limited tracker never saw them (ISO: speed <= sight distance)")
    print("  unavoidable   the HUMAN closed the gap on an already-braking robot")
    print("\nCompliance = zero MACHINE_FAULT episodes (the first three).")
    for arm in ARMS:
        n = int(gate.loc[arm, "MACHINE_FAULT"])
        print(f"  {LABEL[arm]:32s} {'CERTIFIABLE' if n == 0 else f'FAILS ({n}/60)'}")

    print("\n" + "=" * 78)
    print("EFFICIENCY -- the actual contribution")
    print("=" * 78)
    eff = real.groupby("arm").agg(
        success=("success", "mean"),
        t_goal=("time_to_goal", "mean"),
        prot_stops=("protective_stops", "mean"),
        full_stops=("full_stops", "mean"),
        peak_decel=("peak_decel", "mean"),
        hard_brakes=("hard_brakes", "mean"),
        v_at_3m=("v_at_3m", "mean"),
        energy_per_m=("energy", "mean"))
    eff["energy_per_m"] = (real.groupby("arm").energy.mean()
                           / real.groupby("arm").path_m.mean())
    print(eff.reindex(ARMS).round(3).to_string())

    print("\n" + "=" * 78)
    print("AVAILABILITY -- missions completed within the 60 s timeout")
    print("=" * 78)
    print(cls.pivot_table(index="arm", columns="scenario",
                          values="success").reindex(ARMS).round(2).to_string())

    print("\n" + "=" * 78)
    print("GENERALIZATION -- held out, never trained on")
    print("=" * 78)
    for scen, note in ((HOLDOUT_CLEAN, "CLEAN: unseen combination, nothing adversarial"),
                       (HOLDOUT_SCENARIO, "probe: same junction WITH a seeking bystander")):
        g = cls[cls.scenario == scen]
        if g.empty:
            continue
        print(f"\n{scen}   ({note})")
        print(g.groupby("arm").agg(
            success=("success", "mean"),
            raw_breach=("breached", "sum"),
            machine_fault=("machine_fault", "sum"),
            speed_fault=("speed_fault", "sum"),
            filter_fault=("filter_fault", "sum"),
            t_goal=("time_to_goal", "mean")).reindex(ARMS).round(3).to_string())

    a, c = eff.loc["A_commissioned"], eff.loc["C_rl"]
    print("\n" + "=" * 78)
    print("HEADLINE")
    print("=" * 78)
    for arm in ("A_commissioned", "C_rl"):
        print(f"  {LABEL[arm]:32s} machine-fault "
              f"{int(gate.loc[arm,'MACHINE_FAULT'])}/60 missions")
    print(f"  time-to-goal    A {a.t_goal:5.1f} s  ->  C {c.t_goal:5.1f} s   "
          f"({100*(a.t_goal-c.t_goal)/a.t_goal:+.0f} % faster)")
    print(f"  mission success A {a.success:5.2f}    ->  C {c.success:5.2f}")
    print(f"  protective stops A {a.prot_stops:.2f}/mission -> C {c.prot_stops:.2f}")


if __name__ == "__main__":
    main()
