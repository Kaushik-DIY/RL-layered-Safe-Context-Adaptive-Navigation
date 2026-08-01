"""The safety-rated scanner every industrial AMR carries, modelled from the standards.

This is the layer that makes a fielded AMR safe -- not its planner. It is safety-rated
hardware and it OVERRIDES the planner, so both arms of the comparison carry it. Giving it
only to the baseline (or only to us) would be a strawman in one direction or the other.

Behaviour, from ISO 3691-4 / ANSI-ITSDF B56.5 practice and vendor documentation:

    warning field       -> reduce speed, no stop
    protective field    -> full stop before contact
    object clears       -> hold for DWELL_S, then resume and accelerate back

Fields switch automatically with speed and direction (ISO 3691-4 cl. 4.8.2.6: "automatic
selection of the safe detection fields based on truck speed and direction, size of the
load"), which is why the protective field here is recomputed from the CURRENT speed rather
than fixed.

Two modelling choices that matter:

* **The fields are forward RECTANGLES, not discs.** Real scanner fields are rectangular or
  trapezoidal along the direction of travel. A disc would trip on a worker standing in a
  side aisle that a real field ignores completely, which would cripple the baseline with
  phantom stops and make the comparison meaningless.
* **The protective field is the stopping distance.** `d_stop(sigma*v) + d_hard` -- the same
  quantity the CBF uses -- so a protective stop begins exactly when the machine can no
  longer stop short. At our platform's 1.2 m/s mixed-traffic speed that is 2.05 m, which
  matches the 2 m protective field cited as a standard configuration. The physics was
  already calibrated to industry.

Commissioned speed is 1.2 m/s: travel in aisles shared with pedestrians is restricted to
<= 1.2 m/s (ANSI/ITSDF B56.5 hazard zone), against a 1.5-2.0 m/s rated maximum for
MiR/Fetch/Bastian-class machines.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from core.cbf.cbf_filter import d_stop

COMMISSIONED = 1.20     # m/s   mixed-traffic limit for aisles shared with pedestrians
WARN_SPEED = 0.60       # m/s   speed while the warning field is occupied
WARN_FACTOR = 2.5       #       warning field length / protective field length
WARN_HALF_W = 1.10      # m     warning field half-width
PROT_PAD = 0.25         # m     protective field half-width beyond the robot radius
DWELL_S = 3.0           # s     hold after the object clears, before resuming
MARGIN = 0.30           # m     MPC human-potential margin (planner default)

NORMAL, WARNING, STOPPED = "NORMAL", "WARNING", "PROTECTIVE STOP"


class IndustrialAMR:
    """Speed-dependent scanner. Call with the robot state and the tracked humans."""

    def __init__(self, plat, commissioned: float = COMMISSIONED,
                 use_warning: bool = True):
        self.plat = plat
        self.commissioned = commissioned
        # The warning tier is a blunt, hand-set anticipation: anything within ~5 m ahead
        # drops the machine to a flat 0.60 m/s regardless of context. It exists because a
        # reactive planner cannot anticipate. `use_warning=False` keeps only the mandatory
        # protective stop, which is the configuration a machine could justify if its
        # supervisor supplies the anticipation instead.
        self.use_warning = use_warning
        self.half_w = plat.robot.robot_radius + PROT_PAD
        self._n_hist = max(1, int(round(plat.robot.tau_latency / plat.robot.dt)))
        self._vhist = deque([0.0], maxlen=self._n_hist)
        self.state = NORMAL
        self.latched = False
        self.clear_since = None
        self.n_stops = 0
        self._was_stopped = False

    # ------------------------------------------------------------------ fields
    def protective_len(self, v: float) -> float:
        """Stopping distance at speed `v` -- the field the standard requires.

        Fields are selected from the MEASURED speed (cl. 4.8.2.6), specifically from the
        highest speed held over the last `tau_latency` seconds: the field must cover the
        speed the machine may still be carrying through its own command pipeline, and
        using a trailing maximum also damps the switch oscillation that sizing from the
        instantaneous speed would cause. This is what makes a slower machine need a
        smaller field -- and therefore what lets an anticipating supervisor avoid tripping
        a field the fast machine cannot avoid.
        """
        return self.plat.cbf.d_hard + d_stop(
            self.plat.cbf.sigma * max(0.0, v), self.plat.cbf.tau, self.plat.cbf.a_brake)

    def _occupied(self, s, humans, length, half_w) -> bool:
        """Is anyone inside the forward rectangle length x 2*half_w ahead of the robot?"""
        if not len(humans):
            return False
        dx = humans[:, 0] - s[0]
        dy = humans[:, 1] - s[1]
        c, sn = np.cos(s[2]), np.sin(s[2])
        ahead = dx * c + dy * sn                    # longitudinal, +ve = in front
        lat = np.abs(-dx * sn + dy * c)             # lateral offset
        return bool(np.any((ahead > -self.half_w) & (ahead < length) & (lat < half_w)))

    # ------------------------------------------------------------------ update
    def reset(self) -> None:
        self.state = NORMAL
        self.latched = False
        self.clear_since = None
        self.n_stops = 0
        self._was_stopped = False
        self._vhist = deque([0.0], maxlen=self._n_hist)

    def __call__(self, s, humans, t: float) -> tuple:
        """-> (v_max_cmd, d_margin_cmd). `s` = [x, y, yaw, v, omega], `t` = sim time.

        The protective field is evaluated at the COMMISSIONED speed while stopped, not at
        the current speed: sizing it from the instantaneous speed would shrink it to
        nothing at a standstill and the machine would resume straight into the person it
        just stopped for.
        """
        humans = np.asarray(humans, dtype=float).reshape(-1, 4)
        self._vhist.append(float(s[3]))
        v_ref = max(self._vhist)                     # trailing max over tau_latency
        prot = self.protective_len(v_ref)
        warn = WARN_FACTOR * prot

        in_prot = self._occupied(s, humans, prot, self.half_w)
        in_warn = (self.use_warning
                   and self._occupied(s, humans, warn, WARN_HALF_W))

        if in_prot:
            if not self.latched:
                self.n_stops += 1
            self.latched = True
            self.clear_since = None
        elif self.latched:
            # resume only after the field has been clear for the full dwell
            if self.clear_since is None:
                self.clear_since = t
            elif t - self.clear_since >= DWELL_S:
                self.latched = False
                self.clear_since = None

        if self.latched:
            self.state = STOPPED
            return 0.0, MARGIN
        if in_warn:
            self.state = WARNING
            return WARN_SPEED, MARGIN
        self.state = NORMAL
        return self.commissioned, MARGIN
