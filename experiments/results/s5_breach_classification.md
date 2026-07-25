# S5 breach classification — the 18 flagged episodes

**Question.** The S5 adversarial/random stress battery (1000 episodes through the
full MPC→CBF stack) flagged 18 episodes with a stopping-distance barrier dip
(`h<0` while moving) and/or a collision flag. A clean S5 safety claim needs every
one attributed: a **filter logic failure** (the layer left shed-able speed on the
table while the robot was itself closing on a human — the class of bug the S4
pass-through hole belonged to), or a breach that is **unavoidable** for a
non-reversing robot (`v_min = 0`) already braking as hard as physics allows while a
fast social-force pedestrian out-closes it.

**Method.** Each episode reconstructed bit-for-bit (`env seed = 1000+i`, adversary
`rng = 10000+i`, exactly as `run_s5.py` drives them) with `NavEnv(record=True)`,
instrumented for the line-of-sight closing geometry at every 10 Hz inner step
(`scripts/replay_s5_breaches.py`). The discriminator is **not** the robot's total
speed — in a crowd it rolls toward its goal while a pedestrian closes from another
bearing, which the CBF's closing-only cap correctly does not forbid — but whether
the filter was on the **maximal-braking trajectory** (`dv ≤ −a_brake·dt`, or at
rest, or in a protective stop) as `h` descended into the breach, plus the split of
who actually shut the gap (integrated robot vs human closing speed over the descent).

## Result

| metric | value |
|---|---|
| filter-accountable (shed-able speed left on the table) | **0 / 18** |
| unavoidable (filter braking maximally; human out-closed it) | **18 / 18** |
| braking maximally at the breach instant | **18 / 18** |
| footprint contacts (`d < r_robot = 0.22 m`) | **1 / 18** (ep 31, 0.216 m) |
| worst barrier dip `min_h` | **−0.084 m** (ep 31) |
| range of `min_h` | −0.084 … −0.0006 m |

**Who closed the gap.** In every episode the pedestrian's own approach dominated:
- 7 of 18 the robot contributed **0 m** — it was fully stopped and the human walked
  in. This includes ep 31, the only footprint graze (0.216 m, i.e. 4 mm inside the
  robot radius): robot displacement over the whole descent was zero.
- The rest: the robot closed a few cm (max 0.042 m, ep 89) against the human's tens
  of cm (e.g. ep 82: robot 0.034 m vs human 0.179 m).

**Distribution.** 13 open_hall (6–8-person free-roaming crowds), 3 doorway, 2
corridor. All under the adversarial (`[v_max, d_margin_floor]`) or random supervisor
— i.e. a policy actively trying to hurt the robot, which the deployed trained policy
is not.

## Interpretation

None of the 18 is a filter logic failure. The S4 interval-sampling fix stays intact
(the pass-through hole is closed and its regression tests hold). Every residual dip
is the **constant-velocity-vs-social-force model mismatch**: a pedestrian
accelerating or curving into the robot inside the `τ = 0.4 s` prediction window,
faster than the `σ = 1.1` inflation buffers, against a robot that **cannot reverse**
and is already braking maximally. The barrier definition `h = d − d_stop(σ·v·c) −
d_hard` goes slightly negative the instant `d < d_hard = 0.30 m`, and a
non-reversing robot has no admissible command that keeps a striding human out of a
0.30 m disc once the human commits — the robot's own contribution to the breach is
1–4 cm of residual creep (0 in 7 cases).

This is the exact limitation the G2 SFM battery + σ-inflation section documents,
now quantified under the deployed stack: **zero filter faults; residual barrier
dips ≤ 0.084 m and a single 4 mm footprint graze in 1000 adversarial episodes,
dominated by pedestrian motion.**

## Options (not applied)

- **σ bump** (e.g. 1.1 → 1.25): buys margin on both the robot cap and the ISO-13855
  human-approach β-term, but makes the robot more conservative in *every*
  interaction (throughput / G4 time cost). Not warranted for worst-case adversarial
  episodes the trained policy will not reproduce.
- **Short-horizon human acceleration** in the filter's prediction: targets the root
  cause (CV vs SFM) but adds model and tuning surface to the frozen layer.
- **Leave as a documented residual** (recommended): the deployed system uses the
  trained RL supervisor, not an adversary; stage-C crowd training reduces the
  policy-side exposure directly.

Data: `experiments/results/s5_breach_classification.csv`,
reproduce with `python scripts/replay_s5_breaches.py`.
