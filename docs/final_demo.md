# The final demo

The commissioning route, plus the same pedestrian pass twice — once where stepping aside
is safe and once where it is not.

```
PYTHONPATH=$PWD .venv-navrl/bin/python scripts/verify_final.py 6      # the gate
PYTHONPATH=$PWD .venv-navrl/bin/python scripts/render_final_video.py
```

Outputs `experiments/results/final_demo.mp4` (1920x1080, 20 fps, 32.9 s) **and**
`final_demo_summary.png` — the results table is a separate 1920x1080 image, not a card
tacked onto the end of the video, so it drops straight into slides. Regenerate just the
image with `--card-image`. Replay cached in `final_traj.pkl`; `--fresh` after changing
anything upstream.

Titled **"Safe Context-Adaptive Navigation for Industrial AMRs"** / *RL-supervised AMR
against a hand-commissioned industrial AMR on an identical shared warehouse aisle*.

The footer is a **legend, not a commentary** — four lines, each explaining something
actually on screen: the two fields plus the red/blue markings, then the two panel readouts
(`ISO margin h` and `lateral offset`). Anything that narrates what is about to happen was
removed; the panels do that themselves.

**Footer length is not free.** The panels are aspect-equal and height-limited, so map
width = (x-range / y-range) x panel height — every extra footer line and every bit of title
margin comes straight off the width of the picture. Five lines cost about 12 % of the map;
four does not.

## The 13 configured parameters

`commissioning_ledger()` reads them out of the configuration the simulated commissioned
machine actually runs on, so the counter on screen cannot drift from the model.

| # | parameter | value | what the integrator establishes |
|---|---|---|---|
| 1 | site speed limit | 1.20 m/s | aisle width, traffic mix, B56.5 hazard-zone rule |
| 2 | service braking rate | 0.80 m/s² | measured loaded, on this floor |
| 3 | system response time | 0.50 s | scanner + controller + brake engagement |
| 4 | speed-measurement factor | 1.10 | odometry tolerance, ISO 13855 chain |
| 5 | hard keep-out | 0.30 m | footprint tolerance + localisation error |
| 6 | protective field length | 2.05 m | stopping distance at 1.20 m/s, re-sized per tier |
| 7 | protective field width | 0.94 m | footprint + load overhang + tracking tolerance |
| 8 | warning field length | 5.12 m | pre-slow before the protective tier trips |
| 9 | warning field width | 2.20 m | see the aisle, ignore the side racking |
| 10 | warning-tier speed | 0.60 m/s | must fit the reduced field set |
| 11 | resume dwell | 3.0 s | site rule after a protective stop clears |
| 12 | zone J-01 speed / extent | 0.79 m/s over 3.3 m | corner sight line surveyed |
| 13 | zone J-02 speed / extent | 0.79 m/s over 3.3 m | corner sight line surveyed |

**The count is per-route and was wrong once.** `commissioning_ledger()` used to take its
zone rows from the *commissioning* route's three cross-aisles, so this video claimed 14
where it configures 13 — station C is plain aisle and has nothing to mark. It now takes the
zones of the route being shown. Every additional junction adds a row, which is the point.

This is the presentation cut. Earlier demos (a commissioning-only argument on a 3.5 m
aisle, and a three-way head-on diagnostic) were superseded by this route and removed;
they are in the git history if the working is ever needed.

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

**This is the safe half of a measured policy gap.** Driven at a fixed `d_margin`, the same
stack offsets 0.02 m at 0.30, 0.36 m at 1.00 and 1.32 m at 2.00 — and the widest setting is
also the *quickest*, because going round costs less than slowing. The trained policy
nevertheless pins `d_margin` at 0.30 for a whole encounter, against an action-box range of
0.1–2.0: it never asks. Closing that is a training question, not an architecture one.
But asking unconditionally would be wrong too — beside an open cross-aisle the room is not
real. The map is what makes the difference decidable.

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
