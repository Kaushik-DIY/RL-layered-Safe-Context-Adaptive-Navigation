"""Position-triggered worker choreography for the showcase demo.

Gazebo `<actor>`s follow a fixed TIME script, which cannot work here: the RL-supervised
run and the fixed-parameter baseline traverse the same 24 m mission at very different
speeds, so their arrival times at the three hazards diverge by up to ~13 s. A time script
either blocks the slow run or is out-run by the fast one -- both were observed on video.

So the workers are visual-only kinematic MODELS and this node drives their pose, firing
each cue on the ROBOT'S POSITION. That reproduces the industrial 2D scenarios'
`("robot_x_ge", x)` trigger (core/sim2d/scenarios.py) and makes the A/B fair: the hazard
is presented at the SAME distance in both runs, so only approach speed differs.

All cue geometry lives in core.demo.showcase_scene -- shared with the world generator and
the offline verifier, so it cannot drift.
"""
from __future__ import annotations

import math

import rclpy
from gazebo_msgs.srv import SetEntityState

from core.demo.sdf import safe_name
from nav_msgs.msg import Odometry
from rclpy.node import Node

def _load_scene(name):
    """`scene:=final` drives the final route; anything else keeps the showcase, so the
    previously-recorded demo still launches unchanged."""
    if name == "final":
        from core.demo.aisle_scene import should_fire, staged_pose, walk_path
        from core.demo.final_route import CUES
        return CUES, should_fire, staged_pose, walk_path
    from core.demo.showcase_scene import (CUES, should_fire, staged_pose,  # noqa: E501
                                          walk_path)
    return CUES, should_fire, staged_pose, walk_path


class SceneDirectorNode(Node):
    def __init__(self):
        super().__init__("scene_director_node")
        self.declare_parameter("scene", "showcase")
        self.declare_parameter("rate_hz", 50.0)   # match libgazebo_ros_state so pose steps stay small
        self.declare_parameter("bob_amplitude", 0.035)   # fake walk cycle
        self.declare_parameter("bob_hz", 1.9)
        self.declare_parameter("enabled", True)

        global CUES, should_fire, staged_pose, walk_path
        CUES, should_fire, staged_pose, walk_path = _load_scene(
            str(self.get_parameter("scene").value))
        self._fired = [None] * len(CUES)
        self._state = None            # (x, v)
        self._t0 = None

        self.cli = self.create_client(SetEntityState, "/gazebo/set_entity_state")
        self.create_subscription(Odometry, "/odom", self._on_odom, 10)
        rate = float(self.get_parameter("rate_hz").value)
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            f"scene director up: {len(CUES)} position-triggered cues "
            f"({', '.join(c['name'] for c in CUES)})")

    def _on_odom(self, msg: Odometry) -> None:
        self._state = (msg.pose.pose.position.x, msg.twist.twist.linear.x)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _place(self, name: str, x: float, y: float, yaw: float, z: float = 0.0) -> None:
        if not self.cli.service_is_ready():
            return
        req = SetEntityState.Request()
        req.state.name = name
        req.state.pose.position.x = float(x)
        req.state.pose.position.y = float(y)
        req.state.pose.position.z = float(z)
        req.state.pose.orientation.z = math.sin(yaw / 2.0)
        req.state.pose.orientation.w = math.cos(yaw / 2.0)
        req.state.reference_frame = "world"
        self.cli.call_async(req)          # fire-and-forget: next tick corrects any miss

    def _tick(self) -> None:
        if not bool(self.get_parameter("enabled").value) or self._state is None:
            return
        robot_x, robot_v = self._state
        now = self._now()
        amp = float(self.get_parameter("bob_amplitude").value)
        bob_hz = float(self.get_parameter("bob_hz").value)

        for i, cue in enumerate(CUES):
            if self._fired[i] is None:
                # the ISO presentation test: fire so the worker reaches the lane with the
                # robot present_distance away, whatever speed the robot is doing
                if should_fire(cue, robot_x, robot_v):
                    self._fired[i] = now
                    # the two scenes describe presentation differently (showcase by
                    # DISTANCE, the aisle scenes by TIME); log whichever this cue has
                    # rather than assuming, which used to raise KeyError on first fire
                    # and take the whole director timer down with it
                    how = ("present_distance" if "present_distance" in cue
                           else "present_time")
                    self.get_logger().info(
                        f"cue '{cue['name']}' fired: robot x={robot_x:.2f} "
                        f"v={robot_v:.2f} -> {how} {cue[how]:.2f}")
                else:
                    x, y, yaw = staged_pose(cue)
                    self._place(safe_name(cue["name"]), x, y, yaw)
                    continue
            el = now - self._fired[i]
            x, y, yaw, done = walk_path(cue, el)
            # a small vertical bob + yaw sway reads as walking; these are rigid models,
            # so there is no skeletal animation to play
            z = 0.0 if done else abs(math.sin(2.0 * math.pi * bob_hz * el)) * amp
            sway = 0.0 if done else math.sin(2.0 * math.pi * bob_hz * el) * 0.05
            self._place(safe_name(cue["name"]), x, y, yaw + sway, z)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SceneDirectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
