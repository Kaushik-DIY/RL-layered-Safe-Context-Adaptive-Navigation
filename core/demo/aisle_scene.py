"""Parameterised warehouse aisle: stations composed from validated intervention types.

Replaces the old hand-tuned showcase scene, which was calibrated for a comparison we no
longer make (rated-speed baseline) and which does not discriminate against a properly
modelled industrial AMR -- measured 25.7 s for the baseline against 29.7 s for ours.

THE MECHANISM THIS SCENE IS BUILT AROUND
----------------------------------------
A fielded AMR is protected by a two-tier scanner, and the warning tier normally pre-empts
the protective tier: something enters the 5 m warning field, the machine drops to 0.6 m/s,
its protective field shrinks with its speed, and by the time the person is close the field
is too small to trip. Measured on the old route: a worker 1.80 m dead ahead in the lane
did not trip a protective field that had shrunk to 0.90 m. That is the scanner working as
designed, and it is why the baseline is strong.

The protective field therefore only fires when a person goes from INVISIBLE to inside it
faster than the warning tier can pre-slow the machine. That needs an occluder close to the
lane -- a pallet stack staged at the aisle edge, a stillage, a parked trolley -- not a wall
1.75 m away. Those obstructions are ordinary warehouse clutter, they are registered as
static obstacles so both planners see them, and the difference is that the supervisor has
learned that a close occluder means "someone may step out" while the MPC only sees a thing
to drive around.

Everything is presented on TIME-TO-ARRIVAL rather than distance, so both machines get the
same reaction time regardless of their speed. Presenting at a fixed distance gave the
slower machine roughly double the time and its worker finished crossing 5.7 s before it
arrived.
"""
from __future__ import annotations

import numpy as np

HALF_W = 1.75                 # 3.5 m industrial aisle
AISLE_TOP = 4.0               # side aisles run from HALF_W to here
MOUTH = 1.6                   # side-aisle opening width
X_MIN = -1.5
ROBOT_START = (0.0, 0.0, 0.0)

OCCLUSION_Y = HALF_W          # workers above the aisle line are hidden by racking
REVEAL_DISTANCE = 1.2


def _rect_walls(x0, y0, x1, y1):
    return [[x0, y0, x1, y0], [x0, y1, x1, y1]]


class Station:
    """One intervention. `kind` selects the mechanism; `x` is its centre along the aisle."""

    def __init__(self, kind, x, **kw):
        self.kind, self.x, self.kw = kind, x, kw


def build(stations, goal_x, half_w=HALF_W):
    """-> dict(walls, posts, cues, goal, half_w). Geometry is derived from the stations
    so the drawn scene and the planner's obstacle set can never disagree.

    `half_w` defaults to the 3.5 m industrial aisle every existing scene was measured on
    -- changing the default would silently move every recorded result. A scene that wants
    a wider aisle (a two-way run where an AMR and a picker have to pass each other) passes
    its own, and every cue carries the value it was built with so occlusion stays
    consistent with the geometry rather than with a module constant.
    """
    mouths = [s.x for s in stations
              if s.kind in ("blind_cross", "crossing", "blind_clear")]
    walls, posts, cues = [], [], []

    # main aisle walls, broken by each side-aisle mouth
    top = max(AISLE_TOP, half_w + 2.25)
    edges = [X_MIN]
    for mx in sorted(mouths):
        edges += [mx - MOUTH / 2, mx + MOUTH / 2]
    edges.append(goal_x + 1.5)
    for i in range(0, len(edges) - 1, 2):
        walls.append([edges[i], half_w, edges[i + 1], half_w])
    walls.append([X_MIN, -half_w, goal_x + 1.5, -half_w])
    for mx in mouths:                                   # side-aisle side walls
        walls.append([mx - MOUTH / 2, half_w, mx - MOUTH / 2, top])
        walls.append([mx + MOUTH / 2, half_w, mx + MOUTH / 2, top])
        posts.append([mx - MOUTH / 2, half_w, 0.12])
        posts.append([mx + MOUTH / 2, half_w, 0.12])

    for st in stations:
        cues.extend(_station_cues(st, posts, half_w))
    for c in cues:
        c.setdefault("occ_y", half_w)        # travels with the cue, not a module global

    return dict(walls=np.asarray(walls, float), posts=np.asarray(posts, float),
                cues=cues, goal=np.array([goal_x, 0.0]), half_w=half_w)


def _station_cues(st, posts, half_w=HALF_W):
    """Build the worker cues for one station and register any occluding clutter."""
    x, kw = st.x, st.kw
    HW = half_w

    if st.kind == "empty_corner":
        return []                                        # no geometry, nobody there

    if st.kind == "blind_clear":
        # A real blind cross-aisle -- mouth, jambs, no sight line -- with NOBODY in it.
        # `build` gives it the same geometry as `blind_cross`; the difference is only
        # that no worker is staged. It is the control case for the commissioning claim:
        # the site feature is identical, so anything either machine does here is a
        # response to the LAYOUT and not to a person.
        return []

    if st.kind == "blind_cross":
        # worker descends the side aisle and crosses the robot's lane
        return [dict(name=f"cross@{x:.0f}", lane_x=x, speed=kw.get("speed", 1.40),
                     path=[(x, HW + 1.3), (x, -HW - 2.5)],
                     present_time=kw.get("present_time", 1.9), occludes=True)]

    if st.kind == "crossing":
        # visible crossing from the south side, in view the whole way
        return [dict(name=f"vis@{x:.0f}", lane_x=x, speed=kw.get("speed", 1.35),
                     path=[(x, -HW - 3.0), (x, HW + 1.5)],
                     present_time=kw.get("present_time", 2.1), occludes=False)]

    if st.kind == "pallet_step_out":
        # A pallet stack staged AT THE AISLE EDGE. It is registered as a static obstacle,
        # so both planners avoid it -- but only a supervisor that has learned what a close
        # occluder implies will slow for what might be behind it.
        side = kw.get("side", 1.0)
        oy = kw.get("occluder_y", 0.95)                 # inner face, close to the lane
        posts.append([x, side * (oy + 0.25), 0.42])
        return [dict(name=f"step@{x:.0f}", lane_x=x + kw.get("lead", 0.9),
                     speed=kw.get("speed", 1.30),
                     path=[(x + 0.25, side * (oy + 0.15)), (x + 1.9, -side * 2.4)],
                     present_time=kw.get("present_time", 1.25),
                     occludes=True, occluder=(x, side * oy))]

    if st.kind == "head_on":
        # A picker walking the aisle towards the robot, keeping to his own side. At 0.35 m
        # off centre both machines squeezed past at 0.45 m -- inside the robot's own
        # footprint plus keep-out -- so the worker walks a realistic 1.0 m off centre and
        # there is a genuine passing lane for the AMR to use.
        lane = kw.get("lane", 1.00)
        return [dict(name=f"head@{x:.0f}", lane_x=x, speed=kw.get("speed", 1.25),
                     path=[(x + kw.get("lead", 5.5), lane),
                           (x - kw.get("trail", 8.0), lane)],
                     present_time=kw.get("present_time", 3.0), occludes=False)]

    if st.kind == "slow_leader":
        # A picker walking DOWN the aisle ahead of the robot, slower than it. This is the
        # one regime where the scanner's blunt warning tier is itself the bottleneck: the
        # worker sits inside the 5 m x 2.2 m forward box for as long as the machine stays
        # behind him, pinning it at 0.60 m/s indefinitely -- the "AMR stuck behind a
        # person" complaint. Escaping it needs a lateral offset big enough to put him
        # outside the box (|dy| > WARN_HALF_W = 1.1 m), which a fixed-margin planner will
        # not take and a supervisor that can widen d_margin might.
        lane = kw.get("lane", 0.45)
        return [dict(name=f"leader@{x:.0f}", lane_x=x, speed=kw.get("speed", 0.55),
                     path=[(x, lane), (x + kw.get("run", 16.0), lane)],
                     present_time=kw.get("present_time", 1.2), occludes=False)]

    if st.kind == "crowd":
        # A picking zone: several workers crossing the aisle in sequence, so the machine
        # meets one after another with no clear run between them. This is the documented
        # congestion case -- AMRs deliver 2-3x throughput in free flow but only ~3 % more
        # than human workers in crowded aisles.
        n = kw.get("n", 3)
        spacing = kw.get("spacing", 3.2)
        out = []
        for i in range(n):
            side = 1.0 if i % 2 == 0 else -1.0
            cx = x + spacing * i
            out.append(dict(name=f"crowd{i}@{cx:.0f}", lane_x=cx,
                            speed=kw.get("speed", 1.05),
                            path=[(cx, side * (HW + 1.2)), (cx, -side * (HW + 1.2))],
                            present_time=kw.get("present_time", 2.0),
                            occludes=False))
        return out

    raise ValueError(f"unknown station kind {st.kind!r}")


# ------------------------------------------------------------------ cue helpers
def path_len_to_lane(cue):
    """Arc length walked until the worker first reaches the robot's lane (y = 0)."""
    pts, arc = cue["path"], 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        seg = float(np.hypot(x1 - x0, y1 - y0))
        if y0 * y1 <= 0.0 and abs(y1 - y0) > 1e-9:
            return arc + seg * abs(y0) / abs(y1 - y0)
        arc += seg
    return arc


def time_to_lane(cue):
    return path_len_to_lane(cue) / cue["speed"]


def should_fire(cue, robot_x, v_robot):
    """Time-based presentation: the worker reaches the lane `present_time` seconds before
    the robot would arrive at the crossing point, whatever speed the robot is doing."""
    gap = cue["lane_x"] - robot_x
    if gap <= 0.0:
        return True
    v = max(float(v_robot), 0.25)          # a stopped robot must still get its cue
    return gap / v <= cue["present_time"] + time_to_lane(cue)


def walk_path(cue, t):
    pts, spd = cue["path"], cue["speed"]
    remaining = spd * max(0.0, t)
    for i in range(len(pts) - 1):
        (x0, y0), (x1, y1) = pts[i], pts[i + 1]
        seg = float(np.hypot(x1 - x0, y1 - y0))
        if remaining <= seg or i == len(pts) - 2:
            f = min(1.0, remaining / seg) if seg > 1e-9 else 1.0
            yaw = float(np.arctan2(y1 - y0, x1 - x0))
            return x0 + f * (x1 - x0), y0 + f * (y1 - y0), yaw, remaining >= seg
        remaining -= seg
    return pts[-1][0], pts[-1][1], 0.0, True


def staged_pose(cue):
    """Where a worker waits before his cue fires, facing the way he is about to walk.
    (Callers that only want the position ignore the yaw; it exists so a rendered worker
    does not stand facing backwards and then snap round when he sets off.)"""
    (x, y), (nx, ny) = cue["path"][0], cue["path"][1]
    return x, y, float(np.arctan2(ny - y, nx - x))


def visible(cue, wx, wy, robot_xy):
    """Tracker model. A worker is hidden while he is behind racking (above the aisle line)
    or behind his station's occluder, until he is within REVEAL_DISTANCE."""
    d = float(np.hypot(wx - robot_xy[0], wy - robot_xy[1]))
    if d <= REVEAL_DISTANCE:
        return True
    occ_y = cue.get("occ_y", OCCLUSION_Y)
    if wy > occ_y or wy < -occ_y:
        return False
    occ = cue.get("occluder")
    if occ is not None:
        ox, oy = occ
        # hidden while still behind the stack: same side of the aisle and not yet past its
        # inner face, and the robot has not drawn level with it
        if abs(wy) > abs(oy) - 0.10 and np.sign(wy) == np.sign(oy) and robot_xy[0] < ox:
            return False
    return True
