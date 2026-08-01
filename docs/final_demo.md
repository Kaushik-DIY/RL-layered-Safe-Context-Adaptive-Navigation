# The final demo

The commissioning route, plus the same pedestrian pass twice — once where stepping aside
is safe and once where it is not.

```
PYTHONPATH=$PWD .venv-navrl/bin/python scripts/verify_final.py 6      # the gate
PYTHONPATH=$PWD .venv-navrl/bin/python scripts/render_final_video.py
```

Outputs `experiments/results/final_demo.mp4`. Replay cached in `final_traj.pkl`;
`--fresh` after changing anything upstream.

This supersedes `commissioning_demo` as the presentation cut. The earlier two are kept:
`commissioning_demo` is the pure commissioning argument on a 3.5 m aisle, and
`headon_demo` is the three-way diagnostic that established the lateral behaviour.

## The route

A 5.0 m two-way aisle, 31 m, three encounters. The picker at A and the picker at C are
**the same person doing the same thing** — same side, same walking speed, same closing
geometry. The only thing that differs is what the map says about the space the machine
would have to move into.

| | what is there | what ours does |
|---|---|---|
| **A** x=7.5 | Picker head-on, **and a blind cross-aisle on the south side** — the side it would swerve into. | **Refuses to use the width and slows.** Going round means crossing the mouth of an opening nobody can see into: it would be trading a person it can see for a person it cannot. |
| **B** x=16.0 | A true **4-way junction** — openings on both sides — with an occluded worker descending the north arm and crossing to the south one. | Slows for the corner as before. He now walks out through a real gap instead of appearing to step through the racking. |
| **C** x=24.5 | The same picker head-on, **solid racking both sides**. | **Steps aside 1.12 m and carries its speed through.** Nothing can emerge, so the room is real. |

The commissioned machine slows at both passes, because slowing is all a warning tier can
do. It never moves off the centreline anywhere on the route.

## Measured, nominal run

| station | commissioned (v@pass / mean / min / offset) | ours (v@pass / mean / min / offset) |
|---|---|---|
| A picker + blind opening | 0.60 / 0.89 / 0.60 / 0.02 | 0.58 / 0.82 / 0.36 / **0.02** |
| B 4-way junction | 0.60 / 0.90 / 0.60 / 0.02 | 0.80 / 0.90 / 0.60 / 0.01 |
| C picker + solid racking | 0.60 / 0.98 / 0.60 / 0.02 | **1.20 / 1.20 / 1.19 / 1.12** |

**Read `v@pass`, not `v_min`.** The minimum is misleading and misled once: ours dips to
0.36 m/s at A while the picker is still **1.67 m away**, then recovers to 0.58 by the time
they are actually alongside. That dip is a 1.4 s brake pulse, not the speed it passes at.
`encounters()` reports speed at closest approach and mean through the station for exactly
this reason.

### Why ours pulses at A and the commissioned machine does not

The commissioned machine's 0.60 m/s is a **hand-set constant** — the warning tier's
reduced speed. Nothing on it computes a speed from the closing geometry, because it has no
CBF; it applies 0.60 and relies on the protective field as backstop.

Ours runs the CBF, which enforces the ISO stopping-distance condition **including the
pedestrian's own approach speed** (the ISO 13855 human-approach term). A picker walking
into it at 1.25 m/s in a corridor where it may not step aside consumes a real part of the
budget, so the admissible speed drops. Both finish the pass at h = +0.46 m, so on this
encounter the extra braking buys no extra margin — that residual is genuine conservatism
and is stated rather than hidden. The substantive difference is that 0.60 is not computed
from anything: a faster pedestrian would not change it.

## The rule that decides A vs C

`core/demo/sight_limit.lateral_room()`:

> person on my left, solid racking on my right → **go round**
> person on my left, an open cross-aisle on my right → **do not go round, slow down**

Both read off the map the robot already has: the person's bearing from the tracker, and
the mapped jambs (`posts`) that mark every opening. Nothing marked, nothing configured. It
returns `None` when the machine should not use the width, which leaves the policy's own
request untouched.

This is the safe half of the finding in [[headon_demo]]: the trained policy pins
`d_margin` at 0.30 m and never asks for room, but asking unconditionally would be wrong
too. The map is what makes the difference decidable.

## Scene plumbing

* `aisle_scene.build()` places openings **per side** — `Station(..., side=-1.0)` puts the
  cross-aisle south. The wall/post append order is deliberate and documented: `_obstacles`
  keeps the N nearest and ties break on list order, so reshuffling moves results by a
  timestep for no reason.
* New station kind **`junction`**: openings on both sides plus the crossing worker, so he
  enters and leaves through real gaps.
* `head_on` carries `arc_to_lane`, so the cue fires such that the two actually meet at the
  station rather than immediately (the generic arc-to-lane measure runs the whole path for
  a walker who never crosses `y = 0`).
* **Verified: the commissioning route is unchanged by all of this** — 28.8 / 33.1 / 33.2 s
  with identical `min_h`. Re-run that check when touching `aisle_scene`.
