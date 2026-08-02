"""Hand-marked speed zones: the per-site commissioning a fielded AMR actually gets.

A safety scanner cannot see round a corner, so a scanner alone does not make a machine
safe at a blind cross-aisle -- it only makes it safe against what it can already see.
What closes that gap on a real site is an integrator walking the floor, marking each
junction, corner and doorway on the robot's map, and entering a reduced speed for it.
This module is that step, and it is deliberately DERIVED rather than tuned, so nothing
about the baseline's performance is a number chosen by us:

    zone speed  : the fastest speed whose stopping distance still fits inside the
                  sight line the corner leaves -- d_hard + d_stop(sigma*v) <= reveal.
                  This is the same calculation the thesis uses everywhere else; at
                  reveal 1.20 m on the industrial platform it gives 0.79 m/s.
    zone entry  : far enough back to shed the speed at the service brake, plus the
                  distance covered during the system response time.
    zone exit   : far enough past the mouth that the machine is fully clear of the
                  crossing before it resumes.

Every one of those depends on THIS site and THIS machine: change the aisle width, the
racking height, the floor, the load, or the truck, and each zone has to be re-derived
and re-validated. That is the cost the learned supervisor is claimed to remove -- it
reads the same corner off the map geometry it already has, with nothing marked.

The extents are kept to the derived MINIMUM. Real marked zones are drawn generously,
which would slow the baseline further; taking the minimum keeps the comparison
conservative in the baseline's favour.
"""
from __future__ import annotations

import numpy as np

from core.cbf.cbf_filter import d_stop

APPROACH_MARGIN = 0.80    # fraction of the available deceleration the approach plans on


class SpeedZone:
    """One marked zone on the site map: [x0, x1] along the aisle, capped at `v`."""

    def __init__(self, x0: float, x1: float, v: float, label: str):
        self.x0, self.x1, self.v, self.label = x0, x1, v, label

    def contains(self, x: float) -> bool:
        return self.x0 <= x <= self.x1

    def __repr__(self) -> str:
        return f"SpeedZone({self.label}: {self.x0:.2f}..{self.x1:.2f} m @ {self.v:.2f})"


def zone_speed(plat, reveal: float, v_cap: float) -> float:
    """Fastest speed that still stops inside `reveal` metres of sight line."""
    c = plat.cbf
    v = np.arange(0.05, v_cap + 1e-9, 0.001)
    fits = v[c.d_hard + d_stop(c.sigma * v, c.tau, c.a_brake) <= reveal]
    return float(fits.max()) if len(fits) else 0.05


def mark_zones(mouth_x, plat, reveal: float, v_cap: float, mouth_w: float):
    """The integrator's output: one reduced-speed zone per mapped cross-aisle.

    The zone has to START at least a stopping distance before the mouth, or the machine
    is inside the zone but still unable to stop for somebody emerging from it. At the
    zone speed that stopping distance IS the sight line, by the construction of
    `zone_speed` above -- so the lead is simply the surveyed reveal, which is also the
    quantity the integrator measured to get here.

    Note this is where the zone is MARKED, not where the machine starts slowing. Getting
    down to the zone speed by the time it crosses the line is the vehicle's problem, and
    `zone_cap` below is what makes it solve it.
    """
    vz = zone_speed(plat, reveal, v_cap)
    c, r = plat.cbf, plat.robot
    tail = mouth_w / 2.0 + r.robot_radius + c.d_hard
    return [SpeedZone(x - mouth_w / 2.0 - reveal, x + tail, vz, f"J-{i + 1:02d}")
            for i, x in enumerate(sorted(mouth_x))]


def zone_cap(zones, x: float, v_cap: float, a_dec: float | None = None) -> float:
    """Speed limit the marked map imposes at position `x`.

    With `a_dec` given, the limit is APPROACHED rather than stepped into: outside a zone
    the cap is the fastest speed from which the machine can still be down to the zone
    speed by the boundary, `sqrt(vz^2 + 2*a_dec*distance)`. A machine that only obeyed
    the limit once it was over the line would spend the first stretch of every marked
    zone above the limit it was marked with, which is not a configuration that passes
    commissioning -- and it would make a zone boundary look like a step change in speed
    rather than something a vehicle has to decelerate for.

    Without `a_dec` the old step behaviour is kept, so the difference the look-ahead
    makes can be measured rather than asserted.
    """
    out = v_cap
    for z in zones:
        if x > z.x1:
            continue
        if x >= z.x0:
            out = min(out, z.v)
        elif a_dec:
            # Planned against a fraction of the available deceleration. Sizing the
            # approach on the full rate leaves nothing for discretisation or tracking
            # error, and measured that way the machine was still 0.09 m/s over the
            # limit 0.14 m INSIDE the zone -- compliant everywhere except the one place
            # the zone exists for. Real approach controllers carry the same margin.
            out = min(out, float(np.sqrt(z.v ** 2 + 2.0 * APPROACH_MARGIN * a_dec
                                         * (z.x0 - x))))
        elif z.contains(x):
            out = min(out, z.v)
    return float(out)
