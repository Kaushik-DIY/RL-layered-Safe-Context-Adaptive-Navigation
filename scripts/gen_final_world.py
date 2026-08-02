"""Generate the final Gazebo world from `core.demo.final_route`.

The world is GENERATED, never hand-edited, so the geometry Gazebo renders and the geometry
the policy and MPC are told about cannot drift apart. Every wall in the world comes from
the same `WALLS` array the supervisor's map features are computed from.

    PYTHONPATH=$PWD .venv-navrl/bin/python scripts/gen_final_world.py

What is different from the showcase world: a 5.0 m two-way aisle instead of 3.5 m, the
first cross-aisle opening SOUTH rather than north, a true 4-way in the middle, and plain
racking at the third station. Those three facts are the whole argument of the demo, and
all three are read off `final_route` rather than restated here.
"""
from __future__ import annotations

import os

from core.demo.final_route import (CUES, GOAL, HALF_W, MOUTH, NORTH_MOUTHS, ROBOT_START,
                                   SOUTH_MOUTHS, STATION_X, WALLS, X_MAX, X_MIN)
from core.demo.sdf import (RACK_H, WALL_H, WALL_T, amr_model, box,
                           safe_name, worker_model)

OUT = "ros2_ws/src/navrl_nodes/worlds/final_demo.world"
RACK_SETBACK = 0.55


def wall_models() -> str:
    out = ""
    for i, (x1, y1, x2, y2) in enumerate(WALLS):
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if abs(y2 - y1) < 1e-9:
            sx, sy = abs(x2 - x1), WALL_T
        else:
            sx, sy = WALL_T, abs(y2 - y1)
        out += box(f"wall_{i}", cx, cy, WALL_H / 2.0, sx, sy, WALL_H,
                   "0.62 0.63 0.66 1")
    return out


def _bays(mouths):
    """Clear spans of wall between this side's openings."""
    edges = [X_MIN] + [e for m in sorted(mouths) for e in m] + [X_MAX]
    return [(edges[i] + 0.4, edges[i + 1] - 0.4) for i in range(0, len(edges) - 1, 2)]


def _mouth_pairs(xs):
    return [(x - MOUTH / 2, x + MOUTH / 2) for x in xs]


def racking() -> str:
    """Low pallet racking set back behind the walls: uprights, shelves, loads."""
    out, n = "", 0
    runs = []
    for mouths, y in ((_mouth_pairs(NORTH_MOUTHS), HALF_W + RACK_SETBACK + 0.6),
                      (_mouth_pairs(SOUTH_MOUTHS), -(HALF_W + RACK_SETBACK + 0.6))):
        for lo, hi in _bays(mouths):
            if hi - lo > 1.2:
                runs.append((lo, hi, y))
    for lo, hi, y in runs:
        depth, length = 1.1, hi - lo
        for z in (0.45, 1.05):
            out += box(f"rack_shelf_{n}", (lo + hi) / 2.0, y, z, length, depth, 0.06,
                       "0.30 0.34 0.40 1", collide=False)
            n += 1
        k, step = 0, max(1.6, length / max(1, round(length / 2.2)))
        while lo + k * step <= hi + 1e-6:
            out += box(f"rack_post_{n}", lo + k * step, y, RACK_H / 2.0, 0.09, depth,
                       RACK_H, "0.75 0.45 0.10 1", collide=False)
            n += 1
            k += 1
        for z, c in ((0.72, "0.55 0.42 0.28 1"), (1.32, "0.50 0.38 0.25 1")):
            out += box(f"rack_load_{n}", (lo + hi) / 2.0, y, z, length * 0.9,
                       depth * 0.8, 0.45, c, collide=False)
            n += 1
    return out


def floor_markings() -> str:
    """Painted lane edges, broken at each opening the way a real aisle is painted, plus
    hazard hatching under every opening. Purely visual -- the planner sees the walls."""
    out = ""
    for tag, y, mouths in (("n", HALF_W - 0.18, _mouth_pairs(NORTH_MOUTHS)),
                           ("s", -(HALF_W - 0.18), _mouth_pairs(SOUTH_MOUTHS))):
        for k, (lo, hi) in enumerate(_bays(mouths)):
            out += box(f"lane_{tag}{k}", (lo + hi) / 2.0, y, 0.004, hi - lo, 0.10,
                       0.008, "0.90 0.85 0.20 1", collide=False)
    spots = ([(x, HALF_W - 0.95) for x in NORTH_MOUTHS]
             + [(x, -(HALF_W - 0.95)) for x in SOUTH_MOUTHS])
    for i, (x, y) in enumerate(spots):
        out += box(f"hazard_{i}", x, y, 0.003, MOUTH + 0.5, 1.8, 0.006,
                   "0.85 0.72 0.10 1", collide=False)
        for j in range(6):
            out += box(f"hatch_{i}_{j}", x - MOUTH / 2 - 0.1 + j * 0.42, y, 0.005,
                       0.13, 1.8, 0.010, "0.10 0.10 0.10 1", collide=False, yaw=0.5)
    return out


def pick_station() -> str:
    """Drawn well BEYOND the goal, and VISUAL ONLY.

    Both of those are load-bearing. At GOAL + 1.1 m the base kerb spanned x 31.5-32.7
    while the machine finishes with its nose at 31.5, and it drove into it: the check
    script caught a -7.55 m/s^2 spike at x = 31.05 on a 1.20 m/s^2 machine. And the
    planner is never told about this model -- it is not in WALLS -- so anything it can
    collide with is a bug by construction. Clear of the stopping pose AND non-colliding.
    """
    sx = GOAL[0] + 1.9
    out = box("pick_base", sx, 0.0, 0.05, 1.2, 2.6, 0.10, "0.55 0.57 0.60 1",
              collide=False)
    for i, z in enumerate((0.7, 1.3)):
        out += box(f"pick_shelf_{i}", sx, 0.0, z, 1.1, 2.4, 0.06,
                   "0.42 0.45 0.50 1", collide=False)
    return out


def main() -> None:
    body = wall_models() + racking() + floor_markings() + pick_station()
    for cue in CUES:
        x, y = cue["path"][0]
        nx, ny = cue["path"][1]
        import math
        body += worker_model(safe_name(cue["name"]), x, y,
                             math.atan2(ny - y, nx - x))

    world = f"""<?xml version="1.0" ?>
<!-- FINAL DEMO - generated by scripts/gen_final_world.py, DO NOT hand-edit.

     A {GOAL[0]:.0f} m mission down a {2 * HALF_W:.1f} m two-way aisle, three encounters:
       x={STATION_X[0]:.1f}  picker head-on AND a blind cross-aisle opening SOUTH, the side
              the machine would swerve into. It must refuse and slow instead.
       x={STATION_X[1]:.1f}  a true 4-way junction, occluded worker crossing north to south.
       x={STATION_X[2]:.1f}  the SAME picker head-on, solid racking both sides: the room is
              real, so it offsets and carries its speed through.

     Same picker, same side, same closing speed at the first and third. Only the map
     differs, which is the entire point of the demo.

     Workers are visual-only kinematic models driven by scene_director_node, which fires
     each cue on the ROBOT'S POSITION rather than the clock, so the hazard is presented at
     the same distance however fast the machine happens to be going. -->
<sdf version="1.6">
  <world name="final_demo">

    <include><uri>model://sun</uri></include>
    <include><uri>model://ground_plane</uri></include>
    <scene><ambient>0.65 0.65 0.68 1</ambient><background>0.78 0.82 0.86 1</background>
      <shadows>true</shadows></scene>

    <plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so">
      <ros><namespace>/gazebo</namespace></ros>
      <update_rate>50.0</update_rate>
    </plugin>

{body}
{amr_model(ROBOT_START)}
    <gui>
      <camera name="user_camera">
        <pose>-7 -7 9 0 0.62 0.62</pose>
        <track_visual>
          <name>amr</name><static>true</static><use_model_frame>true</use_model_frame>
          <xyz>-9.0 -5.5 9.5</xyz><inherit_yaw>false</inherit_yaw>
        </track_visual>
      </camera>
    </gui>
  </world>
</sdf>
"""
    # An XML comment may not contain a double hyphen, and the header of this file is
    # prose. Parse before writing so a stray "--" fails here instead of inside Gazebo.
    import xml.etree.ElementTree as ET
    ET.fromstring(world)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write(world)
    print(f"wrote {OUT}  ({len(world.splitlines())} lines, {len(WALLS)} walls, "
          f"{len(CUES)} workers)")


if __name__ == "__main__":
    main()
