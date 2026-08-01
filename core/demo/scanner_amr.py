"""The commissioned industrial AMR: speed-dependent protective-field switching.

This is the baseline the whole comparison rests on, so it lives here rather than inside one
script: `scripts/eval_three_arms.py` drives it through `NavEnv`, and
`scripts/verify_showcase.py` drives it through the showcase replay loop. One
implementation, one behaviour, no drift between the numbers and the video.

A certified AMR is not compliant because its planner is clever. It is compliant because a
safety-rated scanner trips a protective stop, and because an integrator hand-capped the
speed at commissioning so the stopping distance fits inside what the scanner can see. It
buys compliance by CRAWLING and STOPPING:

    d > warning field       ->  commissioned 0.50 m/s   (field 0.76 m)
    d < warning field       ->  creep        0.30 m/s   (field 0.53 m)
    d < the creep field     ->  protective stop, latched until RELEASE clear

Why 0.50 m/s is the right commissioned cap and not an arbitrary handicap: the stopping
distance there is 0.76 m, which fits inside the 1.2 m reveal distance at a blind corner. At
the platform's rated 1.5 m/s it is 2.83 m, which does not. That is the calculation an
integrator does by hand, and the fixed-tuning sweep agrees -- `pareto_industrial.csv` has
(v=0.5, m=0.9) at 100 % raw compliance across all five scenarios.

Two details that are easy to get wrong, both learned by measurement:

  * The creep tier is not optional. Without it the machine simply freezes whenever anyone
    lingers near it and times out -- 0/12 mission completion on both interferer scenarios
    -- which overstates the baseline's weakness. A real AMR switches to a smaller field at
    low speed and inches past.
  * Fields are sized at each tier's OWN speed and held fixed there, which is how real field
    switching works (a few configured fields). Recomputing them from the instantaneous
    speed would shrink the field to nothing while stopped, and the robot would resume
    straight into the person it had just stopped for.
"""
from __future__ import annotations

import numpy as np

from core.cbf.cbf_filter import d_stop

SPEED = 0.50            # m/s   shared-area cap set at commissioning
CREEP = 0.30            # m/s   creep speed, with its own smaller field
MARGIN = 0.90           # m     MPC human margin that goes with it (sweep point)
WARN_FACTOR = 1.5       #       warning field / commissioned protective field
RELEASE = 0.25          # m     hysteresis before a latched protective stop releases

NORMAL, WARNING, STOPPED = "NORMAL", "WARNING", "PROTECTIVE STOP"


class ScannerAMR:
    """Commissioned-AMR supervisor. Call with the robot's position and tracked humans."""

    def __init__(self, plat):
        def field(v):
            return plat.cbf.d_hard + d_stop(
                plat.cbf.sigma * v, plat.cbf.tau, plat.cbf.a_brake)
        self.r_prot = field(SPEED)             # 0.76 m at the commissioned speed
        self.r_creep = field(CREEP)            # 0.53 m at creep speed
        self.r_warn = WARN_FACTOR * self.r_prot
        self.latched = False
        self.state = NORMAL

    def reset(self) -> None:
        self.latched = False
        self.state = NORMAL

    def __call__(self, xy, humans) -> tuple:
        """-> (v_max_cmd, d_margin_cmd) for the next decision window.

        `xy` is the robot's (x, y); `humans` an (N, 4) array of TRACKED humans -- a scanner
        sees exactly what the tracker sees, occlusions included.
        """
        humans = np.asarray(humans, dtype=float).reshape(-1, 4)
        d_min = (float(np.min(np.hypot(humans[:, 0] - xy[0], humans[:, 1] - xy[1])))
                 if len(humans) else np.inf)

        if self.latched:
            if d_min > self.r_creep + RELEASE:
                self.latched = False
        elif d_min < self.r_creep:
            self.latched = True

        if self.latched:
            self.state = STOPPED
            return 0.0, MARGIN
        if d_min < self.r_warn:
            self.state = WARNING
            return CREEP, MARGIN
        self.state = NORMAL
        return SPEED, MARGIN
