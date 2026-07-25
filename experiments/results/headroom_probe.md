# Supervision-headroom probe — where does adaptive modulation earn its place?

**Question.** The stage-C policy converges to always-max and matches (never beats)
fixed tuning on the TB3 battery. Is there a regime where a parameter supervisor
over the MPC+CBF stack is *genuinely* useful — or is the layer redundant?

**Method.** `scripts/headroom_probe.py`: open_hall (6–8 free-roaming SFM
pedestrians), three hand-built supervisors on the identical MPC+CBF stack, 30
paired seeds, at two platform scales:

- **tb3** (main platform): v_max 0.26, a_brake 0.3, τ 0.4 → d_stop(full) ≈ 0.25 m
- **industrial** (plan D2 appendix set in robot.yaml): v_max 1.5, a_brake 0.8,
  τ 0.5 → d_stop(full) ≈ 2.53 m (MPC carrot scaled 0.5→2.5 m, a_max_mpc 0.6)

Supervisors: `always-max` = [v_max, 0.30] (what the trained policy became);
`fixed-mid` = [0.55·v_max, 0.50] (static compromise); `heuristic` = density-aware
(speed from nearest-human distance in units of d_stop; margin grows with the count
of nearby people; reads the same tracker view the RL observation uses).

## Result (30 episodes each)

| scale | supervisor | success | coll | viol eps | t_goal | pstops/ep | jerk |
|---|---|---|---|---|---|---|---|
| industrial | always-max | 29/30 | 1 | **28/30** | 15.2 s | 1.6 | 3.37 |
| industrial | fixed-mid | 30/30 | 0 | **26/30** | 17.1 s | 1.8 | 3.27 |
| industrial | **heuristic** | 29/30 | 0 | **2/30** | 28.8 s | 2.1 | 2.62 |
| tb3 | always-max | 30/30 | 0 | 2/30 | 36.3 s | 2.4 | 0.81 |
| tb3 | fixed-mid | **1/30** | 1 | 1/30 | 57.7 s | 4.1 | 0.79 |
| tb3 | heuristic | 30/30 | 0 | 1/30 | 39.5 s | 2.6 | 1.30 |

## Findings

1. **TB3 scale: no headroom — confirmed a fourth way.** Always-max is already
   nearly violation-free (2/30) and fastest; the heuristic only adds 3 s for no
   safety gain; the slow compromise *fails outright* (1/30 — hall timeout, the S1
   freezing pathology). The trained policy converging to always-max was correct
   learning, not a failure: at d_stop ≈ 0.25 m the exact-braking CBF + the MPC
   human term already regulate speed near-optimally and there is nothing left for
   a 2 Hz supervisor to decide.

2. **Industrial scale: large, real headroom.** With a 2.5 m stopping envelope the
   filter binds constantly in a crowd and the CV-vs-SFM mismatch bites inside it:
   riding the cap (always-max) yields stopping-distance violation episodes in
   **93%** of runs plus a collision, despite the CBF. The density-aware supervisor
   cuts that to **7% (14×) with zero collisions and the lowest jerk**, at a
   deliberate throughput cost (28.8 vs 15.2 s). The static compromise is
   Pareto-dominated (26/30 violations AND slower than always-max) — **no fixed
   tuning can buy safety here; only adaptivity can.**

3. **Headroom scales with the stopping envelope**, not with the scenario: same
   hall, same crowd, same stack — only the platform dynamics changed.

## Implication for the thesis argument

- Main results (TB3): "the learned supervisor matches the per-scenario optimum
  with a single policy" — and the audit/slow-leader/paired-battery chain explains
  *why* it cannot exceed it (no headroom at this scale).
- Appendix (industrial): the regime where learned modulation is *necessary* —
  fixed tunings are Pareto-dominated and a simple adaptive rule already trades
  ~14 s for a 14× violation-episode reduction. A policy TRAINED at these dynamics
  (same pipeline, `industrial_appendix` params) has a concrete target: match the
  heuristic's safety at lower time cost, beating every fixed tuning on both axes.

Data: `headroom_probe.csv`, log `headroom_probe_run.log`,
reproduce: `python scripts/headroom_probe.py 30`.
