"""SDF building blocks for the generated Gazebo worlds.

Canonical home for the AMR model and the worker model. Every generated world imports from
here, so the physics fixes below cannot be re-broken by copy-paste into a second copy.

THE FOUR THINGS THAT COST A DEMO EACH, all encoded below -- do not "tidy" them away:

1. The AMR's model z MUST equal the wheel radius. At 0.11 with a 0.10 radius the wheels
   floated a centimetre and the machine rode on its frictionless casters with no traction.
2. The wheel LINK frames must stay UNROTATED. SDF expresses `<axis><xyz>` in the joint's
   child frame, so rolling the link turns "0 1 0" into a VERTICAL axis and the wheels spin
   like turntables. The cylinders are laid down by their geometry pose instead.
3. `max_wheel_acceleration` must be at least `a_brake / wheel_radius`. At 3.0 rad/s^2 the
   plant managed 0.30 m/s^2 while the CBF planned its stops assuming 0.8, so the machine
   could brake at 38 % of what the safety filter believed and every run was plant-limited
   rather than policy-limited. 12.0 gives 1.2 m/s^2 = `robot.yaml` a_max_physical.
4. Workers are visual-only KINEMATIC models, not `<actor>`s, because the scene director
   drives their pose from the robot's position. An actor follows a fixed time script,
   which cannot present a hazard at the same distance to runs that arrive at different
   times.
"""
from __future__ import annotations

def safe_name(name: str) -> str:
    """Gazebo entity names go through SDF, topic names and the SetEntityState service.
    The cue names carry an '@' (`head@24`), which is asking for trouble in all three, so
    every world and every node that addresses a model goes through this."""
    return name.replace("@", "_at_").replace(".", "_").replace("-", "_")


WALL_H, WALL_T = 1.1, 0.12
RACK_H = 1.5
WHEEL_R = 0.10


def box(name, cx, cy, cz, sx, sy, sz, rgba, collide=True, yaw=0.0) -> str:
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


def worker_model(name, x, y, yaw) -> str:
    """Hi-vis worker: legs, torso, band, head, helmet. Kinematic, gravity off."""
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
    return f"""    <model name="{name}">
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


def amr_model(start=(0.0, 0.0, 0.0), max_wheel_accel=12.0) -> str:
    """The differential-drive AMR. See the module header before changing any number."""
    return f"""    <model name="amr">
      <!-- z MUST equal the wheel radius ({WHEEL_R}); higher and the wheels float and the
           machine rides on its frictionless casters with no traction. -->
      <pose>{start[0]:.2f} {start[1]:.2f} {WHEEL_R} 0 0 {start[2]:.2f}</pose>
      <link name="base_footprint">
        <inertial><mass>60.0</mass>
          <inertia><ixx>1.70</ixx><iyy>3.26</iyy><izz>4.06</izz>
                   <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
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
        <collision name="caster_f"><pose>0.28 0 -0.045 0 0 0</pose>
          <geometry><sphere><radius>0.05</radius></sphere></geometry>
          <surface><friction><ode><mu>0</mu><mu2>0</mu2></ode></friction></surface></collision>
        <collision name="caster_r"><pose>-0.28 0 -0.045 0 0 0</pose>
          <geometry><sphere><radius>0.05</radius></sphere></geometry>
          <surface><friction><ode><mu>0</mu><mu2>0</mu2></ode></friction></surface></collision>
      </link>

      <!-- Link frames stay UNROTATED; the cylinder is laid down by its geometry pose. -->
      <link name="left_wheel">
        <pose>0 0.22 0 0 0 0</pose>
        <inertial><mass>1.5</mass>
          <inertia><ixx>0.005</ixx><iyy>0.008</iyy><izz>0.005</izz>
                   <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
        <collision name="c"><pose>0 0 0 -1.5708 0 0</pose>
          <geometry><cylinder><radius>{WHEEL_R}</radius><length>0.05</length></cylinder></geometry>
          <surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface></collision>
        <visual name="v"><pose>0 0 0 -1.5708 0 0</pose>
          <geometry><cylinder><radius>{WHEEL_R}</radius><length>0.05</length></cylinder></geometry>
          <material><ambient>0.08 0.08 0.08 1</ambient></material></visual>
      </link>
      <link name="right_wheel">
        <pose>0 -0.22 0 0 0 0</pose>
        <inertial><mass>1.5</mass>
          <inertia><ixx>0.005</ixx><iyy>0.008</iyy><izz>0.005</izz>
                   <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
        <collision name="c"><pose>0 0 0 -1.5708 0 0</pose>
          <geometry><cylinder><radius>{WHEEL_R}</radius><length>0.05</length></cylinder></geometry>
          <surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface></collision>
        <visual name="v"><pose>0 0 0 -1.5708 0 0</pose>
          <geometry><cylinder><radius>{WHEEL_R}</radius><length>0.05</length></cylinder></geometry>
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
        <wheel_diameter>{2 * WHEEL_R}</wheel_diameter>
        <max_wheel_torque>200</max_wheel_torque>
        <!-- rad/s^2; linear limit = this x wheel radius. Must be >= a_brake / radius
             or the plant cannot deliver the stop the safety filter is planning. -->
        <max_wheel_acceleration>{max_wheel_accel:.1f}</max_wheel_acceleration>
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
