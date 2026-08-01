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

> The same transport time and the same ISO **compliance** with **zero per-site
> commissioning** — no zone marking, no field-set sizing, no re-validation on layout
> change.

Note the word: **compliance**, not safety-in-general. Measured over 6 randomised
presentations, ours matches the commissioned machine on every count compliance is made of
— zero contacts, zero stopping-distance violations, zero protective stops, every mission
completed, 33.2 s against 33.1 s. It does run **closer to the limit** (worst barrier
margin +0.37 m against +0.73 m), and the closing card says so. Do not let the nominal run
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

### The supervisor is floored at the speed its own map justifies

Measured, the policy was **systematically slower than the geometry required**: 0.57 m/s
inside a blind cross-aisle where 0.79 was provably safe, 0.74 m/s with the nearest corner
6.7 m away, 0.64 m/s on a stretch with no corner at all. Over the 31 m route it spent
**15.2 of 37.7 s below 0.95 m/s with no human tracked at all**, where the barrier margin
was 6–16 m. That was the entire throughput gap, and none of it was buying safety.

The cause is a calibration mismatch, not a bad policy: it was trained against the
**strict** governor (σ 1.1, service brake 0.8) on a 1.5 m/s platform, and is deployed
against the **relaxed** governor (σ 1.0, physical brake 1.2) at 1.2 m/s. The same
clearance supports a higher speed under the chain it now runs on, so every cap it emits is
low by construction. Retraining would fix it properly; `core/demo/sight_limit.py` fixes it
without invalidating the trained artefact.

The floor is the same argument the hand-marked zone rests on, evaluated continuously
instead of surveyed once:

> the machine may travel at whatever speed it can still stop from inside the distance it
> can actually see

with sight distance taken from `post_ahead` — the map feature the policy already
consumes — and the protective field required to clear that sight line by one `d_hard`.
That clearance matters: at *exactly* the sight-limited speed the stopping distance equals
the sight line and the field IS the stopping distance, so somebody stepping out at the
edge of vision lands precisely on the field boundary and trips a protective stop. A
fielded AMR escapes this because its warning tier has already pre-slowed it; ours carries
no warning tier by design, so the clearance has to come from the speed. Measured over 3
presentations:

| clearance | time | protective stops |
|---|---|---|
| 0.00 m | 34.1 s | 1.00 |
| 0.15 m | 34.0 s | 0.67 |
| **0.30 m (= `d_hard`)** | **33.2 s** | **0.00** |
| 0.45 m | 34.0 s | 0.00 |

The derived value is also the quick one, because a protective stop costs a 3 s dwell.

**Three properties keep this honest:**

* **It is a floor, never a ceiling.** `cap = max(policy, floor)`. The policy stays free to
  go slower for anything the geometry does not express.
* **It cannot relax into a breach.** The floor is clamped by the STRICT barrier against
  every tracked human before it is applied — the same barrier the result is scored on. In
  practice this pulls the floor *below* the policy near a worker, so the relaxation never
  speeds the machine up around a person.
* **The split is measured, not assumed.** The floor sets the cap on **12 % of steps** — the
  empty cross-aisle and open aisle, where geometry is the only input. The learned policy
  sets it on the other 88 %, including everywhere a human is involved. The gate prints
  this every run.

**One machine constant is required** and the write-up must not skip it:
`SIGHT_PAST_OCCLUDER`, how far beyond a mapped occluder the sensors resolve a person. That
is a property of the sensor and its mounting, fixed for a vehicle across every site it is
ever deployed to — whereas the commissioned machine needs that *same physical quantity
surveyed per junction*, and then a zone speed, extent and polygon derived and validated
from it. The distinction is real, but it is not nothing, and the closing card states it.

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


| arm | time | protective stops | contacts | worst `min_h` | ISO violations | peak decel |
|---|---|---|---|---|---|---|
| `scanner` | 29.2 s | 0.17 | 0 | +0.49 m | 0 | −1.20 m/s² |
| `commissioned` | 33.1 s | 0.00 | 0 | +0.73 m | 0 | −0.60 m/s² |
| `ours` | 33.2 s | 0.00 | 0 | +0.37 m | 0 | −0.60 m/s² |

Ours is **+0.4 %** against the commissioned machine and **+13.7 %** against the zone-less
scanner machine, with the run-to-run spread down to 0.1 s.

## The plant is modelled, and this changed the numbers

Until 2026-08-01 the demo harnesses integrated the controller's output straight into the
state (`v_new = u[0]`), so speed could change by any amount in one 0.1 s step. Measured,
**every arm was braking at −4.5 to −5.8 m/s² on a 1.2 m/s² machine** while accelerating
correctly at +0.6. That is not a rounding artefact — it shortened every mission time, and
it made a zone boundary look like a step change in speed rather than something a vehicle
has to decelerate for.

The cause is a known interface defect: `v_max_cmd` enters the MPC as a hard bound
`v_k <= v_max_cmd` while the same program carries `|dv| <= a_max_mpc*dt`. When the cap
drops by more than one window's worth of deceleration the constraints contradict, the
program is infeasible, and the solver's infeasible iterate was being applied as a real
command.

`core/demo/plant.py` fixes it in two places, and `core/demo/site_zones.py` fixes the
third:

1. **`reachable_cap`** never asks for a cap the machine could not reach this step, so the
   program stays feasible. A protective stop is exempt — a real safety controller cuts the
   drives directly, and the field is sized on the service brake, not on the planner's
   comfort limit.
2. **`apply_plant`** bounds whatever finally comes out by `a_max_physical`, so no layer —
   MPC, CBF, scanner or supervisor — can command a speed change the vehicle cannot make.
3. **`zone_cap(..., a_dec=...)`** makes the marked limit something the machine *approaches*
   rather than steps into: outside a zone the cap is `sqrt(vz^2 + 2*a*d)`, so it is at the
   limit when it crosses the line. A machine that only obeys the limit once it is over the
   line spends the first stretch of every marked zone above the limit it was marked with.

**The correction cost us, which is the right sign.** Ours went 38.7 → 39.0 s and the
throughput gap widened from +15.7 % to +17.7 %; the commissioned machine got slightly
*faster* (33.4 → 33.1 s) because its approach is now smooth instead of an erratic
infeasible-iterate transient. The scanner arm picked up protective stops it did not have
before (0.00 → 0.17), because it can no longer shed its warning-tier speed instantly.

Two of the gate's checks now police this directly: no arm may brake harder than the
platform can, and the commissioned machine must honour its own marked zones. The residual
zone excess is **+0.041 m/s** (5 %) for about 0.1 m at one boundary, from the MPC lagging
the cap by ~2 control steps. The approach is planned on 80 % of available deceleration
(`APPROACH_MARGIN`); lowering it to 0.40 removes the residual entirely but slows the
baseline by 0.4 s, which would flatter us — so the baseline-favouring value is kept and the
residual is reported instead of removed.

**Not re-run:** `probe_station.py` and `relax_sweep.py` have their own copies of the
integration loop and were left alone, so their recorded per-station numbers are not
directly comparable to these.

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
