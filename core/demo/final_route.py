"""THE final route, in one place, so nothing that draws it can disagree with it.

Four things consume this module and they must never drift apart:

    scripts/verify_final.py          the 2D gate and the video
    scripts/gen_final_world.py       the Gazebo world (generated, never hand-edited)
    scene_director_node              the workers in Gazebo
    rl_supervisor_node               the walls/posts the policy's map features come from

The route asks one question twice, with the answer flipped by geometry alone:

  A  x = 7.5   picker walking head-on, AND a blind cross-aisle on the SOUTH side -- the
               side the machine would have to swerve into to get round him. It must
               refuse and slow, because going round trades a person it can see for a
               person it cannot.
  B  x = 16.0  a true 4-way junction, openings both sides, occluded worker descending
               the north arm and crossing to the south one.
  C  x = 24.5  the SAME picker head-on, solid racking both sides. Nothing can emerge, so
               the room is real: offset and carry the speed through.

Both pickers walk the NORTH half of the aisle, so in both cases the escape is SOUTH.
That is the whole design -- same person, same side, same closing speed, and only the map
differs.
"""
from __future__ import annotations

import numpy as np

from core.demo import aisle_scene as sc
from core.demo.industrial_amr import COMMISSIONED
from core.demo.site_zones import mark_zones

HALF_W = 2.50            # 5.0 m two-way aisle: passing a pedestrian needs real room
GOAL_X = 31.0
STATION_X = (7.5, 16.0, 24.5)
PICKER_LANE = 0.75       # both pickers walk the north half, so both escapes are south
PICKER_SPEED = 1.25
PICKER_LEAD, PICKER_TRAIL = 7.0, 9.0

# only the two mapped openings can be marked; C has nothing to mark
ZONE_X = (STATION_X[0], STATION_X[1])

STATIONS = [
    sc.Station("blind_clear", STATION_X[0], side=-1.0),
    sc.Station("head_on", STATION_X[0], lane=PICKER_LANE, speed=PICKER_SPEED,
               lead=PICKER_LEAD, trail=PICKER_TRAIL),
    sc.Station("junction", STATION_X[1]),
    sc.Station("head_on", STATION_X[2], lane=PICKER_LANE, speed=PICKER_SPEED,
               lead=PICKER_LEAD, trail=PICKER_TRAIL),
]
STATION_LABEL = ["picker head-on AND a blind\ncross-aisle on the escape side",
                 "4-way junction, occluded\nworker crosses",
                 "picker head-on, solid\nracking both sides"]

ROBOT_START = (0.0, 0.0, 0.0)


def build():
    return sc.build(STATIONS, goal_x=GOAL_X, half_w=HALF_W)


def site_zones(plat):
    """What an integrator marks on this site's map: one zone per mapped cross-aisle."""
    return mark_zones(ZONE_X, plat, sc.REVEAL_DISTANCE, COMMISSIONED, sc.MOUTH)


# Built once at import so the world generator, the director and the ROS supervisor all
# get literally the same arrays. Do not rebuild these per-caller.
SCENE = build()
WALLS = np.asarray(SCENE["walls"], float)
POSTS = np.asarray(SCENE["posts"], float)
CUES = SCENE["cues"]
GOAL = np.asarray(SCENE["goal"], float)
X_MIN, X_MAX = sc.X_MIN, GOAL_X + 1.5
MOUTH = sc.MOUTH

# which side each opening is on, for the world generator's racking and floor paint
NORTH_MOUTHS = sorted({st.x for st in STATIONS if st.kind == "junction"})
SOUTH_MOUTHS = sorted({st.x for st in STATIONS
                       if st.kind in ("blind_clear", "blind_cross", "crossing")
                       and st.kw.get("side", 1.0) < 0}
                      | set(NORTH_MOUTHS))
