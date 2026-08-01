"""Generate the industrial showcase Gazebo world from core.demo.showcase_scene.

The world is GENERATED, never hand-edited, so the Gazebo geometry and the geometry the
policy/MPC are told about can never drift apart.

    PYTHONPATH=$PWD .venv-navrl/bin/python scripts/gen_showcase_world.py

Design notes (learned the hard way over three failed demo attempts):
  * The AMR's model z MUST equal the wheel radius or the wheels float and it rides on
    its frictionless casters with no traction.
  * The wheel LINKS must stay unrotated: SDF expresses <axis><xyz> in the joint/child
    frame, so a rolled link turns "0 1 0" into a VERTICAL axis and the wheels spin like
    turntables. The cylinders are laid down by their geometry pose instead.
  * Workers are visual-only kinematic MODELS, not <actor>s, because scene_director_node
    drives their pose from the robot's position (see core/demo/showcase_scene).
  * Racking is low and set back; the previous 2 m solid blocks filled the frame and hid
    the aisle. The GUI camera tracks the AMR so it stays framed over the whole 24 m run.
"""
from __future__ import annotations

import os

from core.demo.showcase_scene import (AISLE_TOP, CUES, EVENT_X, GOAL, HALF_W, MOUTH,
                                      POSTS, ROBOT_START, WALLS, X_MAX, X_MIN,
                                      staged_pose)

OUT = "ros2_ws/src/navrl_nodes/worlds/industrial_showcase.world"

WALL_H, WALL_T = 1.1, 0.12
RACK_H, RACK_SETBACK = 1.5, 0.55


def box(name, cx, cy, cz, sx, sy, sz, rgba, collide=True, yaw=0.0):
    col = (f'        <collision name="c"><geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}'
           f'</size></box></geometry></collision>\n') if collide else ""
    return f"""    <model name="{name}">
      <static>true</static><pose>{cx:.3f} {cy:.3f} {cz:.3f} 0 0 {yaw:.4f}</pose>
      <link name="l">
{col}        <visual name="v"><geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry>
          <material><ambient>{rgba}</ambient><diffuse>{rgba}</diffuse></material></visual>
      </link>
    </model>
"""


def wall_models():
    out = ""
    for i, (x1, y1, x2, y2) in enumerate(WALLS):
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if abs(y2 - y1) < 1e-9:
            sx, sy = abs(x2 - x1), WALL_T
        else:
            sx, sy = WALL_T, abs(y2 - y1)
        out += box(f"wall_{i}", cx, cy, WALL_H / 2.0, sx, sy, WALL_H, "0.62 0.63 0.66 1")
    return out


def _bays(mouths):
    """Clear spans of wall between the aisle openings."""
    edges = [X_MIN] + [e for m in mouths for e in m] + [X_MAX]
    return [(edges[i] + 0.4, edges[i + 1] - 0.4) for i in range(0, len(edges) - 1, 2)]


def racking():
    """Low pallet racking behind the walls: uprights + two shelf slabs, visual only."""
    out, n = "", 0
    runs = []
    north_mouths = [(x - MOUTH / 2, x + MOUTH / 2) for x in EVENT_X]
    south_mouths = [(x - MOUTH / 2, x + MOUTH / 2) for x in EVENT_X[1:]]
    for lo, hi in _bays(south_mouths):              # south bays, around the cross-aisle
        if hi - lo > 1.2:
            runs.append((lo, hi, -(HALF_W + RACK_SETBACK + 0.6)))
    for lo, hi in _bays(north_mouths):              # north bays, between the aisles
        if hi - lo > 1.2:
            runs.append((lo, hi, HALF_W + RACK_SETBACK + 0.6))
    for lo, hi, y in runs:
        depth, length = 1.1, hi - lo
        for z in (0.45, 1.05):                       # shelf slabs
            out += box(f"rack_shelf_{n}", (lo + hi) / 2.0, y, z, length, depth, 0.06,
                       "0.30 0.34 0.40 1", collide=False); n += 1
        k, step = 0, max(1.6, length / max(1, round(length / 2.2)))
        while lo + k * step <= hi + 1e-6:            # uprights
            out += box(f"rack_post_{n}", lo + k * step, y, RACK_H / 2.0, 0.09, depth,
                       RACK_H, "0.75 0.45 0.10 1", collide=False); n += 1
            k += 1
        for z, c in ((0.72, "0.55 0.42 0.28 1"), (1.32, "0.50 0.38 0.25 1")):
            out += box(f"rack_load_{n}", (lo + hi) / 2.0, y, z, length * 0.9, depth * 0.8,
                       0.45, c, collide=False); n += 1
    return out


def floor_markings():
    """Painted lane edges + yellow hazard hatching at each blind corner."""
    out = ""
    # lane edge lines, broken at each opening the way a real aisle is painted
    north_mouths = [(x - MOUTH / 2, x + MOUTH / 2) for x in EVENT_X]
    south_mouths = [(x - MOUTH / 2, x + MOUTH / 2) for x in EVENT_X[1:]]
    for s, y, mouths in (("n", HALF_W - 0.18, north_mouths),
                         ("s", -(HALF_W - 0.18), south_mouths)):
        for k, (lo, hi) in enumerate(_bays(mouths)):
            out += box(f"lane_{s}{k}", (lo + hi) / 2.0, y, 0.004, hi - lo, 0.10,
                       0.008, "0.90 0.85 0.20 1", collide=False)
    # hazard hatching under each opening; B and C are 4-ways, so they get both sides
    zones = ([(x, HALF_W - 0.95) for x in EVENT_X]
         + [(x, -(HALF_W - 0.95)) for x in EVENT_X[1:]])
    for i, (x, y) in enumerate(zones):
        out += box(f"hazard_{i}", x, y, 0.003, MOUTH + 0.5, 1.8,
                   0.006, "0.85 0.72 0.10 1", collide=False)
        for j in range(6):                            # black hatch stripes
            out += box(f"hatch_{i}_{j}", x - MOUTH / 2 - 0.1 + j * 0.42, y,
                       0.005, 0.13, 1.8, 0.010, "0.10 0.10 0.10 1",
                       collide=False, yaw=0.5)
    return out


def worker(cue):
    """Visual-only kinematic worker: hi-vis torso, head, legs. Driven by the director."""
    x, y, yaw = staged_pose(cue)
    parts = [
        ("legs_l", 0.0, 0.10, 0.38, 0.16, 0.16, 0.76, "0.13 0.16 0.32 1"),
        ("legs_r", 0.0, -0.10, 0.38, 0.16, 0.16, 0.76, "0.13 0.16 0.32 1"),
        ("torso", 0.0, 0.0, 1.06, 0.30, 0.46, 0.60, "0.95 0.45 0.05 1"),
        ("band", 0.0, 0.0, 1.13, 0.32, 0.48, 0.10, "0.85 0.90 0.95 1"),
        ("head", 0.0, 0.0, 1.50, 0.21, 0.21, 0.24, "0.86 0.70 0.56 1"),
        ("helmet", 0.0, 0.0, 1.62, 0.25, 0.25, 0.12, "0.95 0.80 0.10 1"),
    ]
    vis = "".join(
        f"""        <visual name="{n}"><pose>{px:.3f} {py:.3f} {pz:.3f} 0 0 0</pose>
          <geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry>
          <material><ambient>{c}</ambient><diffuse>{c}</diffuse></material></visual>
"""
        for n, px, py, pz, sx, sy, sz, c in parts)
    return f"""    <model name="{cue['name']}">
      <pose>{x:.3f} {y:.3f} 0 0 0 {yaw:.4f}</pose>
      <link name="body">
        <kinematic>true</kinematic>
        <gravity>false</gravity>
        <inertial><mass>70.0</mass>
          <inertia><ixx>1</ixx><iyy>1</iyy><izz>1</izz>
                   <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
{vis}      </link>
    </model>
"""


AMR = f"""    <model name="amr">
      <!-- z MUST equal the wheel radius (0.1): at 0.11 the wheels floated 1 cm and the
           AMR rode on its frictionless casters with no traction. -->
      <pose>{ROBOT_START[0]:.2f} {ROBOT_START[1]:.2f} 0.1 0 0 {ROBOT_START[2]:.2f}</pose>
      <link name="base_footprint">
        <inertial><mass>60.0</mass>
          <inertia><ixx>1.70</ixx><iyy>3.26</iyy><izz>4.06</izz>
                   <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
        <!-- 0.75 x 0.50 m: the planner models a 0.22 m disc (mpc.yaml r_robot), so a
             0.9 x 0.58 body made on-camera clearances look tighter than the numbers. -->
        <collision name="body_c"><pose>0 0 0.14 0 0 0</pose>
          <geometry><box><size>0.75 0.50 0.28</size></box></geometry></collision>
        <visual name="body_v"><pose>0 0 0.14 0 0 0</pose>
          <geometry><box><size>0.75 0.50 0.28</size></box></geometry>
          <material><ambient>0.10 0.28 0.62 1</ambient><diffuse>0.15 0.40 0.85 1</diffuse></material></visual>
        <visual name="deck_v"><pose>0 0 0.29 0 0 0</pose>
          <geometry><box><size>0.70 0.46 0.03</size></box></geometry>
          <material><ambient>0.20 0.20 0.22 1</ambient></material></visual>
        <visual name="nose_v"><pose>0.35 0 0.14 0 0 0</pose>
          <geometry><box><size>0.06 0.30 0.14</size></box></geometry>
          <material><ambient>0.95 0.80 0.10 1</ambient><diffuse>0.95 0.80 0.10 1</diffuse></material></visual>
        <visual name="beacon_v"><pose>0 0 0.40 0 0 0</pose>
          <geometry><cylinder><radius>0.05</radius><length>0.10</length></cylinder></geometry>
          <material><ambient>0.90 0.55 0.05 1</ambient><diffuse>1.0 0.65 0.1 1</diffuse></material></visual>
        <!-- casters 5 mm clear so the WHEELS carry the load, not these -->
        <collision name="caster_f"><pose>0.28 0 -0.045 0 0 0</pose>
          <geometry><sphere><radius>0.05</radius></sphere></geometry>
          <surface><friction><ode><mu>0</mu><mu2>0</mu2></ode></friction></surface></collision>
        <collision name="caster_r"><pose>-0.28 0 -0.045 0 0 0</pose>
          <geometry><sphere><radius>0.05</radius></sphere></geometry>
          <surface><friction><ode><mu>0</mu><mu2>0</mu2></ode></friction></surface></collision>
      </link>

      <!-- Link frames stay UNROTATED. SDF expresses <axis><xyz> in the joint (child)
           frame, so rolling the link turned "0 1 0" into a VERTICAL axis and the wheels
           spun like turntables. The cylinder is laid down by its geometry pose. -->
      <link name="left_wheel">
        <pose>0 0.22 0 0 0 0</pose>
        <inertial><mass>1.5</mass>
          <inertia><ixx>0.005</ixx><iyy>0.008</iyy><izz>0.005</izz>
                   <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
        <collision name="c"><pose>0 0 0 -1.5708 0 0</pose>
          <geometry><cylinder><radius>0.1</radius><length>0.05</length></cylinder></geometry>
          <surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface></collision>
        <visual name="v"><pose>0 0 0 -1.5708 0 0</pose>
          <geometry><cylinder><radius>0.1</radius><length>0.05</length></cylinder></geometry>
          <material><ambient>0.08 0.08 0.08 1</ambient></material></visual>
      </link>
      <link name="right_wheel">
        <pose>0 -0.22 0 0 0 0</pose>
        <inertial><mass>1.5</mass>
          <inertia><ixx>0.005</ixx><iyy>0.008</iyy><izz>0.005</izz>
                   <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
        <collision name="c"><pose>0 0 0 -1.5708 0 0</pose>
          <geometry><cylinder><radius>0.1</radius><length>0.05</length></cylinder></geometry>
          <surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface></collision>
        <visual name="v"><pose>0 0 0 -1.5708 0 0</pose>
          <geometry><cylinder><radius>0.1</radius><length>0.05</length></cylinder></geometry>
          <material><ambient>0.08 0.08 0.08 1</ambient></material></visual>
      </link>

      <joint name="left_wheel_joint" type="revolute">
        <parent>base_footprint</parent><child>left_wheel</child>
        <axis><xyz>0 1 0</xyz><limit><lower>-1e16</lower><upper>1e16</upper></limit></axis>
      </joint>
      <joint name="right_wheel_joint" type="revolute">
        <parent>base_footprint</parent><child>right_wheel</child>
        <axis><xyz>0 1 0</xyz><limit><lower>-1e16</lower><upper>1e16</upper></limit></axis>
      </joint>

      <plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so">
        <ros><namespace>/</namespace></ros>
        <update_rate>50</update_rate>
        <left_joint>left_wheel_joint</left_joint>
        <right_joint>right_wheel_joint</right_joint>
        <wheel_separation>0.44</wheel_separation>
        <wheel_diameter>0.2</wheel_diameter>
        <max_wheel_torque>200</max_wheel_torque>
        <!-- rad/s^2. Linear accel limit = alpha * wheel_radius = alpha * 0.1.
             At 3.0 the plant managed only 0.30 m/s^2 while the CBF plans its stops
             assuming a_brake = 0.8, so the robot could brake at 38% of what the safety
             filter believed. Both runs then over-ran their envelopes (baseline min_h
             was 0.67 below zero against 0.03 predicted) and both were plant-limited
             rather than policy-limited, which flattened the whole comparison.
             12.0 gives 1.2 m/s^2 = robot.yaml a_max_physical, so the CONTROLLER's
             limits bind, as they do in the 2D sim the policy was trained in. -->
        <max_wheel_acceleration>12.0</max_wheel_acceleration>
        <command_topic>cmd_vel</command_topic>
        <odometry_topic>odom</odometry_topic>
        <odometry_frame>odom</odometry_frame>
        <robot_base_frame>base_footprint</robot_base_frame>
        <publish_odom>true</publish_odom>
        <publish_odom_tf>true</publish_odom_tf>
        <publish_wheel_tf>true</publish_wheel_tf>
      </plugin>
    </model>
"""


def main() -> None:
    body = wall_models() + racking() + floor_markings()
    body += "".join(worker(c) for c in CUES)
    world = f"""<?xml version="1.0" ?>
<!-- INDUSTRIAL SHOWCASE - generated by scripts/gen_showcase_world.py, DO NOT hand-edit.

     One continuous {GOAL[0]:.0f} m mission through three hazard stations:
       x={EVENT_X[0]:.1f}  blind corner, NO worker  - the supervisor slows on map geometry alone
       x={EVENT_X[1]:.1f}  occluded ONCOMING worker - steps out of the aisle and walks at the AMR
       x={EVENT_X[2]:.1f}  blind corner + CROSSING worker - emerges and crosses the lane
     Main aisle y in [{-HALF_W:.0f}, {HALF_W:.0f}] (4 m), side aisles opening north.

     The workers are visual-only kinematic models driven by scene_director_node, which
     fires each cue on the ROBOT'S POSITION (not the clock) so the hazard is presented at
     the same distance in the RL and baseline runs. A time script cannot do that: the two
     runs' arrival times diverge by up to ~13 s over this route. -->
<sdf version="1.6">
  <world name="industrial_showcase">

    <include><uri>model://sun</uri></include>
    <include><uri>model://ground_plane</uri></include>
    <scene><ambient>0.65 0.65 0.68 1</ambient><background>0.78 0.82 0.86 1</background>
      <shadows>true</shadows></scene>

    <plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so">
      <ros><namespace>/gazebo</namespace></ros>
      <update_rate>50.0</update_rate>
    </plugin>

{body}
{AMR}
    <!-- camera rides with the AMR so it stays framed across the whole run -->
    <gui>
      <camera name="user_camera">
        <pose>-7 -6 8 0 0.62 0.62</pose>
        <track_visual>
          <!-- Higher and less oblique than before. The old low offset kept the near
               racking across the bottom half of frame; the sight line to the robot was
               clear, but the shot was cluttered. -->
          <name>amr</name><static>true</static><use_model_frame>true</use_model_frame>
          <xyz>-8.0 -4.5 8.5</xyz><inherit_yaw>false</inherit_yaw>
        </track_visual>
      </camera>
    </gui>
  </world>
</sdf>
"""
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write(world)
    print(f"wrote {OUT}  ({len(world.splitlines())} lines)")


if __name__ == "__main__":
    main()
