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

## The portfolio comparison figure

`scripts/plot_final_comparison.py` → `experiments/results/final_comparison.png`
(2312 × 1563). Four panels, each answering one question a sceptic would actually ask.

| panel | question | metric | result |
|---|---|---|---|
| 1 | What does it save? | site parameters configured by hand | 11 / 13 / **0** |
| 2 | Is it as safe? | worst stopping-distance margin vs the limit | +0.36 / +0.36 / **+0.37 m**, zero contacts and zero violations on all three |
| 3 | What does it cost? | mission time, 6 presentations, mean ± spread | 31.4 ± 1.3 / 32.9 ± 0.1 / **32.6 ± 0.1 s**, plus the Gazebo run at 31.8 s |
| 4 | Does it actually adapt? | speed at the same pedestrian pass, twice | 0.60 / 0.60 / **0.58** with a blind escape; 0.60 / 0.60 / **1.20** with a clear one |

**Why these four and not others.** Panel 4 is the one the whole thing rests on: the same
person, the same closing speed, and only the map differs. It is also the only panel where
the three machines separate — which is the honest shape of the result, and why panels 1–3
are there to establish that nothing was given up to get it.

**Deliberately not plotted:** `min_h` as a bare "safety score" (the three are equal to
within 0.01 m, so a bar chart of it would invite a difference that is not there — it is
drawn against the *limit* instead); speed-at-pass on its own (flatters us at station C and
says nothing at A); and anything from the `crowd` encounter, which is off this route and
where ours is beaten outright — stated in the figure's own footnote rather than omitted.

**Colour** is slots 1–3 of the data-viz reference palette, unchanged, because those three
are documented as clearing the all-pairs colour-blindness floors in both modes — which is
the case that applies to small multiples. They are *not* re-picked to match the video's
green and blue: the palette is validated as a set and re-stepping it by eye is the mistake
the method exists to prevent. Every bar is direct-labelled, so identity never rests on
colour alone. **Note:** the skill's validator is a Node script and there is no JS runtime
on this machine, so the palette was taken pre-validated rather than re-checked — if you
change a colour, install Node and re-run `scripts/validate_palette.js` before shipping.
