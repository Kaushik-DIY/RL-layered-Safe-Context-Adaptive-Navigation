# The head-on demo

An oncoming picker in a wide shared aisle: does the machine slow down, or step aside?

```
PYTHONPATH=$PWD .venv-navrl/bin/python scripts/verify_headon.py 4       # the gate
PYTHONPATH=$PWD .venv-navrl/bin/python scripts/render_headon_video.py
```

Outputs `experiments/results/headon_demo.mp4` (1920x1080, 20 fps, 29.6 s). Replay cached
in `headon_traj.pkl`; `--fresh` after changing anything upstream.

**Deliberately separate from the commissioning demo.** This route has no cross-aisles, so
there is nothing for an integrator to mark and the commissioning argument does not apply.
The only question here is behavioural.

## Why the aisle is wider

The commissioning route is a 3.5 m aisle — the industrial standard for one-way transport.
Passing an oncoming pedestrian in one leaves almost no lateral room, so both machines are
forced into the same manoeuvre and the comparison measures the aisle rather than the
controller. This route is a **5.0 m two-way aisle**, the width a site uses where AMRs and
people share a route in both directions, which creates a genuine choice between slowing
down and stepping aside.

`aisle_scene.build()` now takes `half_w`, defaulting to the 3.5 m aisle every existing
result was measured on. Each cue carries the width it was built with, so occlusion stays
consistent with the geometry instead of a module constant. **Verified: the commissioning
numbers are bit-identical after the refactor** (28.8 / 33.1 / 33.2 s, same `min_h`).

## The three arms

| arm | what it is |
|---|---|
| `industrial` | MPC + scanner. No zones to mark, so the **warning tier is its entire answer**: anything inside the ~5 m x 2.2 m forward box drops it to a flat 0.60 m/s, whether or not that person is on a collision course. |
| `ours` | MPC + relaxed governor + learned supervisor + sight floor. Same mandatory protective field, no warning tier. |
| `ours_lateral` | **The same stack**, with the supervisor's lateral request floored at the room the walls already show. Not a different controller — a different request. |

## Measured (4 presentations)

| | industrial | ours, as trained | ours, using the aisle |
|---|---|---|---|
| passing clearance | 0.77 m | 0.76 m | **1.99 m** |
| speed at the pass | 0.60 m/s | 0.58 m/s | **1.20 m/s** |
| lateral offset used | 0.04 m | 0.07 m | **1.30 m** |
| ISO barrier margin | +0.36 m | +0.41 m | **+1.69 m** |
| mission time | 24.4 s | 24.5 s | **23.4 s** |

Zero contacts and zero stopping-distance violations on all three.

## The finding, stated the way it should be

**As trained, ours is the industrial machine.** Same clearance to within a centimetre,
same speed at the pass, same mission time, and both use essentially none of a 2.5 m
half-aisle. This is not a rendering failure — it is the result, and it confirms what the
project has measured repeatedly: head-on is the geometry where the anticipation comes from
the **MPC's human-cost term and the CBF**, and it survives deleting the supervisor
entirely.

**But the stack can step aside, and the supervisor already has the actuator.** Measured
directly by driving the same stack at fixed `d_margin`:

| `d_margin` | lateral offset | passing gap | time |
|---|---|---|---|
| 0.30 (what the policy asks) | 0.02 m | 0.77 m | 21.1 s |
| 0.60 | 0.08 m | 0.83 m | 20.8 s |
| 1.00 | 0.36 m | 1.08 m | 20.2 s |
| 2.00 | 1.32 m | 2.01 m | 20.1 s |

Going round is **both safer and quicker** — it avoids having to shed speed at all. And the
trained policy pins `d_margin` at **0.30 m for the whole encounter** (min 0.30, max 0.59
over the run) against an action-box range of 0.1–2.0 m. It never asks.

So: **a gap in the policy, not in the architecture.** Closing it is a training question —
the reward has no term that rewards lateral clearance, so nothing ever taught it to ask.

Note this is a genuinely different result from the `slow_leader` case, where lateral
avoidance was found to be *structurally* impossible (directly behind a leader the radial
human potential points backward, not sideways, and forcing `d_margin` to 2.0 moved the
robot 0.01 m). Head-on has a lateral component in the gradient, so the authority is real
here.

## `passing_margin` is opt-in, and stays that way

`core/demo/sight_limit.passing_margin()` floors the lateral request at the half width less
the footprint and a wall keep-out, capped by the action box — derived from wall geometry,
nothing marked. It is **not enabled on the commissioning route**: it is a behavioural
change that would move numbers already measured and reported there. Rolling it in means
re-running that gate.
