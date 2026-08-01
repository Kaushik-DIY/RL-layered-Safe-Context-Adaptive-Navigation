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
    """The integrator's output: one reduced-speed zone per mapped cross-aisle."""
    vz = zone_speed(plat, reveal, v_cap)
    c, r = plat.cbf, plat.robot
    lead = c.tau * v_cap + (v_cap ** 2 - vz ** 2) / (2.0 * c.a_brake)
    tail = mouth_w / 2.0 + r.robot_radius + c.d_hard
    return [SpeedZone(x - mouth_w / 2.0 - lead, x + tail, vz, f"J-{i + 1:02d}")
            for i, x in enumerate(sorted(mouth_x))]


def zone_cap(zones, x: float, v_cap: float) -> float:
    """Speed limit the marked map imposes at position `x`."""
    return min([v_cap] + [z.v for z in zones if z.contains(x)])
