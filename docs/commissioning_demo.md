# The commissioning demo

The final 2D presentation video, and the measurement gate behind it.

```
PYTHONPATH=$PWD .venv-navrl/bin/python scripts/verify_commissioning.py 6   # the gate
PYTHONPATH=$PWD .venv-navrl/bin/python scripts/render_commissioning_video.py
```

Outputs `experiments/results/commissioning_demo.mp4` (1920x1080, 20 fps) plus stills via
`--still <t>`. The replay is cached in `commissioning_traj.pkl`; pass `--fresh` after
changing anything upstream of the controllers.

## The claim

> Equivalent ISO **compliance** with **zero per-site commissioning** — no zone marking,
> no field-set sizing, no re-validation on layout change — at a throughput cost of about
> fifteen percent.

Note the word: **compliance**, not safety-in-general. Measured over 6 randomised
presentations, ours matches the commissioned machine on the things compliance is made of
— zero contacts, zero stopping-distance violations, every mission completed — but it runs
**closer to the limit** (worst barrier margin +0.02 m against +0.81 m) and takes the
occasional protective stop the commissioned machine does not (0.17 against 0.00 per run).
The closing card states both. Do not let the nominal run, which is clean on every count,
stand in for the distribution.

Not "safer", and not "faster". Both of those were tried and both are dead:

* **"Safer" is a strawman.** A certified AMR does not violate ISO — that is the
  precondition for deployment. It buys compliance by crawling and stopping.
* **"Faster" is mathematically impossible in the obvious configuration.** With the same
  safety-rated scanner on both machines the applied speed is `min(rl_cap, scanner_cap)`,
  and `min(a, b) <= b`. Parity is the ceiling. See the negative-result write-up.

What is left is the *commissioning* argument, and it is a real one: the machine that has
to be told where the corners are costs engineering to deploy and re-engineering to move.

## The three arms

All three carry the identical **protective field**, sized from the service brake, because
it is mandatory equipment. All three are capped at the same 1.20 m/s mixed-traffic
commissioned speed. All three are scored on the same **strict** barrier.

| arm | what it is | why it is in the comparison |
|---|---|---|
| `scanner` | MPC + scanner only: warning tier -> 0.60 m/s, protective tier -> stop. No marked zones. | The fastest defensible configuration, and the baseline the earlier negative result was measured against. Kept visible so that result is not quietly replaced. |
| `commissioned` | The same machine **as actually deployed**: plus a hand-marked reduced-speed zone at every mapped cross-aisle. | A scanner cannot see round a corner. What closes that gap on a real site is an integrator marking the junction on the map. This is the machine the commissioning claim is about. |
| `ours` | MPC + relaxed CBF governor + learned supervisor + the same protective tier. No warning tier, no zones. | The supervisor is claimed to supply the anticipation the other two buy with a blunt warning tier and hand-marked zones. |

### Nothing about the baseline is a number we chose

The zone speed is **derived**, in `core/demo/site_zones.py`:

```
d_hard + d_stop(sigma * v)  <=  corner sight line
```

At the industrial platform's parameters and this scene's 1.20 m reveal that gives
**0.79 m/s** — the same calculation used everywhere else in the thesis. Zone entry is the
distance needed to shed speed at the service brake plus the system response distance; zone
exit clears the mouth plus the footprint. The extents are kept to the derived **minimum**;
real marked zones are drawn generously, which would only slow the baseline further.

### Why ours drops the warning tier but keeps the protective one

The warning field is near-universal in practice but is **not a certified safety function** —
only the protective field contributes to certification. So a machine may legitimately run
without it if something else supplies the anticipation. That is precisely the claim being
tested, and it is tested honestly: ours keeps the mandatory tier and is marked on the strict
barrier, not on its own relaxed governor.

## The route

Three cross-aisles at 8.5 m centres, then a clear run-out to the pick station
(`scripts/verify_commissioning.py`, `STATION_X`). Every one is an identical **site
feature** — same mouth, same jambs, same lost sight line — which is the point: the
integrator has to survey and configure for the feature whether or not anybody is in it.

1. `blind_clear` — a blind cross-aisle with **nobody there**. The control case: anything
   either machine does here is a response to the layout, not to a person.
2. `blind_cross` — a worker descends the side aisle, occluded until the reveal.
3. `crossing` — a worker crosses from the south, visible the whole way.

Each station type was validated on its own first, in `scripts/probe_station.py` and
`scripts/relax_sweep.py`, before earning a place here.

**The `crowd` station is deliberately absent, and the gate says so out loud.** Ours is
beaten there outright — 1.50 protective stops against 0.25, `min_h -0.01`, in every
governor setting. It is a known open failure, reported, not hidden by scene choice.

## What is on screen

| drawn | meaning |
|---|---|
| solid rectangle | the protective field: the room the machine needs to stop from the speed it is doing. **Both** machines carry the identical one. A *smaller* field means a *slower* robot, not a safer one. |
| dashed amber rectangle | the warning field. Only the commissioned machine has one. |
| red bands, labels, dimensions | engineering an integrator had to survey and enter for this site. |
| blue dashed ray | `post_ahead` — the along-heading distance to the next mapped constriction, read straight off the map the robot already had. This is a real observation feature the policy consumes, not an illustration. |

Fields are drawn as **rectangles** because that is what `IndustrialAMR._occupied`
actually tests. A disc would be a prettier lie, and would trip on side-aisle workers a
real field ignores.

## Measured result (6 presentations)

Reproduce with `scripts/verify_commissioning.py 6`. The raw log is a regenerable output
and is not tracked, so the numbers live here.


| arm | time | protective stops | contacts | worst `min_h` | ISO violations |
|---|---|---|---|---|---|
| `scanner` | 29.1 s | 0.00 | 0 | +0.73 m | 0 |
| `commissioned` | 33.4 s | 0.00 | 0 | +0.81 m | 0 |
| `ours` | 38.7 s | 0.17 | 0 | +0.02 m | 0 |

Ours is **+15.7 %** against the commissioned machine and **+32.9 %** against the zone-less
scanner machine.

### What the gate checks, and what it deliberately does not

Hard checks are the conditions the compliance claim rests on, and every one is evaluated
over the whole battery rather than over the nominal run: all missions completed, zero
contacts on every arm, ours holds `min_h >= 0`, ours logs no stopping-distance violation,
ours is not claimed faster, and the throughput cost stays inside the stated band.

Protective stops are **reported, not gated**. A protective stop is the safety device doing
its job, not a non-conformity — it costs *availability*, which is a claim this video does
not make. So when ours takes more of them than the baseline the gate prints a
`[REPORT ON THE CARD]` line, and the card has to carry it. That is the mechanism that
stops the number being quietly dropped; it is not a softened check.

## Reading the result honestly

Ours is **slower**. It slows at all three cross-aisles including the empty one, because it
cannot know the empty one is empty — and neither can a hand-marked zone, which is why the
commissioned machine has one there too. The trade is that throughput cost against every
per-site zone, field set and re-validation the other machine needs.

The closing card carries the measured rows from the run being shown, plus two industry
context figures (≈30 % of equipment spend on solution design and deployment; 8–14 weeks
contract-to-production). **Those two are cited context, not measurements from this work,
and the card labels them that way.** Add the citations to the caption before the video is
published anywhere.

## Gotchas paid for already

* The aspect-equal panel is **height-limited**, so map width = (x-range / y-range) x panel
  height. Widening either range eats the left gutter the instruments live in. `Y_LO/Y_HI`
  carry headroom above the side aisles purely so the station callouts do not land on the
  wall geometry they describe.
* Nothing decorative may sit where the robot drives. The pick station is drawn **beyond**
  the goal.
* `arrive_s` accumulates as `k*dt`, so `t_now >= arrive_s` misses its own frame — it needs
  an epsilon or the DELIVERED badge never fires.
* Feeding `SupervisorPolicy` empty `walls`/`posts` makes the policy crawl: the obs-v2
  occlusion features go to garbage. The empty-aisle case is the sanity check that catches
  it.
