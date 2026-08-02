"""LAYER 3 as a ROS node: the trained supervisor plus the two map-derived guards.

Publishes (v_max_cmd, d_margin_cmd) on /navrl/params, which mpc_node consumes to modulate
the NMPC. Not launching this node == the fixed-parameter baseline: mpc_node falls back to
platform defaults when no /navrl/params arrives.

THIS NODE IS A FAITHFUL PORT OF THE 2D LOOP IN `scripts/verify_final.py`, and the rates
matter. In 2D the policy is queried at 2 Hz (`rl.decision_every` steps of 0.1 s) but the
geometry guards and the reachable-cap clamp run at the FULL 10 Hz control rate. Running
the guards at 2 Hz instead would let the cap fall 0.30 m/s between clamps, which is five
times the deceleration the machine can actually deliver -- so this node ticks at 10 Hz and
calls the ONNX policy on every fifth tick.

The three things layered on top of the raw policy output, all from the map the robot
already has and none of them configured per site:

  sight floor    `sight_limit.floor_speed` -- never slower than the speed it could still
                 stop from inside the distance it can see. The policy was trained against
                 the strict governor and is systematically slow under the relaxed one.
  lateral rule   `sight_limit.lateral_room` -- ask for room to pass somebody ONLY when the
                 side it would move into is solid. Beside an open cross-aisle it must not,
                 because that trades a person it can see for one it cannot.
  reachable cap  `plant.reachable_cap` -- never command a cap the machine could not reach
                 this step, which is what keeps the MPC program feasible.

Parameters
----------
model_path : absolute path to the ONNX exported by scripts/export_onnx.py.
platform   : 'industrial' (35-dim obs v2 + MiR action box) or 'tb3'.
walls,posts: flat geometry of THIS world, for the obs-v2 map features and the guards.
half_w     : aisle half width, for the lateral rule's usable-room calculation.
commissioned : site speed limit the cap is bounded by.
rate_hz    : control rate (10.0). The policy itself runs at rate_hz / policy_every.
use_floor / use_lateral : guards on, for ablation.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from core.cbf.cbf_filter import CbfFilter
from core.common.platform import load_platform
from core.demo.plant import reachable_cap
from core.demo.sight_limit import floor_speed, lateral_room
from core.rl.supervisor import SupervisorPolicy

from navrl_nodes import yaw_from_quaternion


class RlSupervisorNode(Node):
    def __init__(self):
        super().__init__("rl_supervisor_node")
        self.declare_parameter("model_path", "")
        self.declare_parameter("platform", "industrial")
        self.declare_parameter("walls", [0.0])
        self.declare_parameter("posts", [0.0])
        self.declare_parameter("half_w", 2.5)
        self.declare_parameter("commissioned", 1.2)
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("policy_every", 5)
        self.declare_parameter("use_floor", True)
        self.declare_parameter("use_lateral", True)

        model_path = str(self.get_parameter("model_path").value)
        platform = str(self.get_parameter("platform").value)
        if not model_path:
            raise RuntimeError("rl_supervisor_node requires the 'model_path' parameter "
                               "(path to the exported .onnx policy)")
        self.walls = self._reshape(self.get_parameter("walls").value, 4)
        self.posts = self._reshape(self.get_parameter("posts").value, 3)
        self.half_w = float(self.get_parameter("half_w").value)
        self.v_site = float(self.get_parameter("commissioned").value)
        self.use_floor = bool(self.get_parameter("use_floor").value)
        self.use_lateral = bool(self.get_parameter("use_lateral").value)
        self.policy_every = int(self.get_parameter("policy_every").value)

        self.plat = load_platform(platform)
        # STRICT parameters: the floor may never relax into a breach of the barrier the
        # result is scored on, so it is clamped against this one, not the governor.
        self.scorer = CbfFilter(self.plat.robot, self.plat.cbf)
        self.policy = SupervisorPolicy(model_path, platform=platform,
                                       walls=self.walls, posts=self.posts)

        self._state = None
        self._goal = None
        self._humans = np.zeros((0, 4))
        self._k = 0
        self._cap = None
        self._margin = None

        self.create_subscription(Odometry, "/odom", self._on_odom, 10)
        self.create_subscription(PoseStamped, "/goal_pose", self._on_goal, 10)
        self.create_subscription(Float32MultiArray, "/navrl/humans", self._on_humans, 10)
        self.pub = self.create_publisher(Float32MultiArray, "/navrl/params", 10)
        # published for the recorder so the Gazebo run can be compared with the 2D one
        self.pub_dbg = self.create_publisher(Float32MultiArray, "/navrl/supervisor", 10)
        rate = float(self.get_parameter("rate_hz").value)
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            f"supervisor up: {platform}, {0 if self.walls is None else len(self.walls)} "
            f"walls, {0 if self.posts is None else len(self.posts)} posts, "
            f"floor={self.use_floor} lateral={self.use_lateral}, "
            f"policy at {rate / self.policy_every:.1f} Hz of {rate:.0f} Hz control")

    @staticmethod
    def _reshape(flat, width):
        arr = np.asarray([float(v) for v in flat], dtype=float)
        if arr.size < width:            # sentinel [0.0] == "none"
            return None
        return arr[: width * (arr.size // width)].reshape(-1, width)

    def _on_odom(self, msg: Odometry) -> None:
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self._state = np.array([p.x, p.y, yaw_from_quaternion(q),
                                msg.twist.twist.linear.x, msg.twist.twist.angular.z])

    def _on_goal(self, msg: PoseStamped) -> None:
        self._goal = np.array([msg.pose.position.x, msg.pose.position.y])
        self.policy.reset()             # new mission: back to the conservative floor
        self._k = 0
        self.get_logger().info(f"goal set: {self._goal.round(2).tolist()}")

    def _on_humans(self, msg: Float32MultiArray) -> None:
        self._humans = np.asarray(msg.data, dtype=float).reshape(-1, 4)

    def _tick(self) -> None:
        if self._state is None or self._goal is None:
            return                      # nothing to modulate yet; MPC waits for a goal too
        s, humans = self._state, self._humans

        if self._k % self.policy_every == 0 or self._cap is None:
            self._cap, self._margin = self.policy.compute(s, self._goal, humans)
        self._k += 1
        cap, margin = float(self._cap), float(self._margin)

        floor = blind = 0.0
        if self.use_floor:
            floor = floor_speed(s, self.walls, self.posts, self.plat, self.v_site,
                                humans=humans, scorer=self.scorer)
            cap = max(cap, floor)
        if self.use_lateral:
            lat = lateral_room(s, self.walls, self.posts, self.plat, humans, self.half_w)
            blind = 1.0 if lat["blind"] else 0.0
            if lat["margin"] is not None:
                margin = max(margin, lat["margin"])

        # never ask for a change the machine cannot make this step
        cap = reachable_cap(min(cap, self.v_site), float(s[3]), self.plat)
        self.pub.publish(Float32MultiArray(data=[cap, margin]))
        self.pub_dbg.publish(Float32MultiArray(
            data=[cap, margin, float(floor), blind, float(self._cap)]))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RlSupervisorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
