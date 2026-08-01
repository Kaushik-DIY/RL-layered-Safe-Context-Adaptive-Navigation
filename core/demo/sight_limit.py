"""The speed the robot's OWN MAP already justifies, used as a floor under the policy.

THE MEASURED PROBLEM. The supervisor was systematically slower than the geometry
required -- 0.57 m/s where 0.79 was provably safe inside a blind cross-aisle, 0.74 m/s
with the nearest corner 6.7 m away, 0.64 m/s on a stretch with no corner at all. Over a
31 m route it spent **15.2 of 37.7 s below 0.95 m/s with no human tracked at all**, in
places where the barrier margin was 6-16 m. That is not caution, it is a miscalibration,
and it was the whole of the throughput gap.

WHY IT HAPPENS. The policy was trained against the STRICT governor (sigma 1.1, service
brake 0.8) and a 1.5 m/s platform. It is deployed against the RELAXED governor (sigma
1.0, physical brake 1.2) at a 1.2 m/s commissioned cap. The same clearance supports a
higher speed under the relaxed chain than under the one the policy learned, so every cap
it emits is low by construction. Retraining would fix it properly; this fixes it without
invalidating the trained artefact.

THE FLOOR. Exactly the argument the hand-marked zone rests on, evaluated continuously
instead of surveyed once:

    the machine may travel at whatever speed it can still stop from
    inside the distance it can actually see

Sight distance comes from the same map-derived feature the policy itself consumes --
`post_ahead`, the along-heading distance to the next mapped constriction -- floored by
how far the sensors see past a mapped occluder once the machine is on top of it. Nothing
is marked, surveyed or entered for the site: given a map, this is computed everywhere.

WHAT THE FLOOR IS NOT. It is a LOWER bound, never an upper one. The policy remains free
to go slower whenever it has learned a reason the geometry does not express -- somebody
closing, a crowd, a context it was trained on -- and `binding_fraction` in the gate
reports how often the floor rather than the policy set the speed, so the policy's actual
contribution stays measurable rather than assumed.

SAFETY. Raising a cap can only be justified if it cannot raise it into a breach, so the
floor is clamped by the STRICT stopping-distance barrier against every tracked human
before it is applied. Ours is scored on that same strict barrier, so this relaxation
cannot buy throughput with the margin the claim depends on.

ONE MACHINE CONSTANT is required: `sight_past_occluder`, how far beyond a mapped
occluder the sensors resolve a person. That is a property of the sensor and its mounting,
fixed for a vehicle across every site it is ever deployed to -- unlike the zone speed,
extent and polygon the integrator derives per junction from the same physical quantity.
The distinction is real but it is not nothing, and the write-up states it.
"""
from __future__ import annotations

import numpy as np

from core.cbf.cbf_filter import d_stop
from core.common.observation import geometry_features

SIGHT_PAST_OCCLUDER = 1.20      # m, machine constant (sensor reach past a mapped edge)
FIELD_MARGIN = None             # m; None = the derived default, one hard keep-out


def field_margin(plat) -> float:
    """Clearance the protective field must keep inside the sight line.

    WHY IT IS NEEDED AT ALL. At exactly the sight-limited speed the stopping distance
    EQUALS the sight line, and the protective field IS the stopping distance -- so
    somebody stepping out at the edge of vision lands precisely on the field boundary
    and trips a protective stop. A fielded AMR does not suffer this because its warning
    tier has already pre-slowed it, shrinking its field; ours carries no warning tier by
    design, so the clearance has to come from the speed instead.

    THE VALUE IS DERIVED, not tuned: the field must clear the sight line by the same
    hard keep-out the barrier already reserves around a person, `d_hard`. Measured over
    3 presentations, that criterion is also where the trade actually sits --
    0.00 m: 34.1 s with 1.00 protective stops | 0.15: 34.0 s, 0.67 stops
    0.30 m (= d_hard): 33.2 s, 0.00 stops     | 0.45: 34.0 s, 0.00 stops
    -- because a protective stop costs a 3 s dwell, so the compliant speed is also the
    quick one. Above d_hard the clearance stops buying anything and just costs speed.
    """
    return float(plat.cbf.d_hard)


def speed_for_sight(plat, sight: float, v_cap: float, margin: float = 0.0) -> float:
    """Fastest speed whose stopping distance still fits inside `sight` metres.

    Same chain as the CBF and the protective field: d_hard + d_stop(sigma*v) <= sight.
    Evaluated on the STRICT parameters, not the relaxed governor -- the floor has to be
    defensible against the barrier the result is scored on.
    """
    c = plat.cbf
    v = np.arange(0.05, v_cap + 1e-9, 0.005)
    fits = v[c.d_hard + d_stop(c.sigma * v, c.tau, c.a_brake) <= sight - margin]
    return float(fits.max()) if len(fits) else 0.05


def sight_distance(state, walls, posts,
                   sight_past_occluder: float = SIGHT_PAST_OCCLUDER) -> float:
    """How far ahead the machine can see, from the map alone.

    `post_ahead` goes monotonically to zero as the machine reaches a constriction, but
    the sight line does not: once it is level with the corner it sees past it. Flooring
    at the sensor's reach is what stops the limit collapsing to a crawl at exactly the
    moment the corner opens up.
    """
    _, _, post_ahead = geometry_features(state, walls, posts)
    return max(float(post_ahead), float(sight_past_occluder))


def floor_speed(state, walls, posts, plat, v_cap, humans=None, scorer=None,
                sight_past_occluder: float = SIGHT_PAST_OCCLUDER,
                margin: float | None = None) -> float:
    """The floor: geometry-justified speed, clamped so it cannot relax into a breach.

    `margin` is read at CALL time, not bound as a default, so a sweep can actually move
    it -- binding it as a default silently produced four identical rows once already.
    """
    if margin is None:
        margin = field_margin(plat) if FIELD_MARGIN is None else FIELD_MARGIN
    v = speed_for_sight(plat, sight_distance(state, walls, posts,
                                             sight_past_occluder), v_cap, margin)
    if scorer is None or humans is None or not len(humans):
        return v
    # Never raise the cap to a speed at which the STRICT barrier would already be
    # negative for somebody currently tracked. Descending scan rather than a solve:
    # the barrier is monotone in v and this is a handful of evaluations per step.
    probe = np.asarray(state, dtype=float).copy()
    while v > plat.rl.v_max_low:
        probe[3] = v
        if scorer.min_barrier(probe, humans) >= 0.0:
            return float(v)
        v -= 0.02
    return float(plat.rl.v_max_low)
