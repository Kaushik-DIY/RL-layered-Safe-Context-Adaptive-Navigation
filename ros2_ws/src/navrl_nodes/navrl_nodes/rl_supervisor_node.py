"""LAYER 3 as a ROS node (plan D8, week 5): the trained context supervisor.

Runs the exported ONNX policy at 2 Hz and publishes (v_max_cmd, d_margin_cmd) on
/navrl/params, which mpc_node consumes to MODULATE the NMPC. All the logic lives in
the ROS-free `core.rl.supervisor.SupervisorPolicy` (torch-free, onnxruntime only,
unit-tested against the 2D env); this node only moves messages -- the same core/ vs
ros2_ws/ split as the MPC and CBF layers.

Not launching this node == the always-max baseline: mpc_node falls back to platform
defaults when no /navrl/params arrives. That is exactly the demo's A/B toggle.

Parameters
----------
model_path : absolute path to the ONNX exported by scripts/export_onnx.py.
platform   : 'industrial' (35-dim obs v2 + MiR action box) or 'tb3'.
walls      : flat [x1,y1,x2,y2]*n static wall segments of THIS world (obs-v2 geometry).
posts      : flat [x,y,r]*m constriction/corner circles of THIS world (obs-v2 geometry).
rate_hz    : supervisor frequency (2.0, matching training).
"""
from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from core.rl.supervisor import SupervisorPolicy

from navrl_nodes import yaw_from_quaternion


class RlSupervisorNode(Node):
    def __init__(self):
        super().__init__("rl_supervisor_node")
        self.declare_parameter("model_path", "")
        self.declare_parameter("platform", "industrial")
        self.declare_parameter("walls", [0.0])
        self.declare_parameter("posts", [0.0])
        self.declare_parameter("rate_hz", 2.0)

        model_path = str(self.get_parameter("model_path").value)
        platform = str(self.get_parameter("platform").value)
        if not model_path:
            raise RuntimeError("rl_supervisor_node requires the 'model_path' parameter "
                               "(path to the exported .onnx policy)")
        walls = self._reshape(self.get_parameter("walls").value, 4)
        posts = self._reshape(self.get_parameter("posts").value, 3)

        self.policy = SupervisorPolicy(model_path, platform=platform,
                                       walls=walls, posts=posts)
        self.get_logger().info(
            f"supervisor up: {platform} policy, {0 if walls is None else len(walls)} walls, "
            f"{0 if posts is None else len(posts)} posts")

        self._state = None
        self._goal = None
        self._humans = np.zeros((0, 4))

        self.create_subscription(Odometry, "/odom", self._on_odom, 10)
        self.create_subscription(PoseStamped, "/goal_pose", self._on_goal, 10)
        self.create_subscription(Float32MultiArray, "/navrl/humans", self._on_humans, 10)
        self.pub = self.create_publisher(Float32MultiArray, "/navrl/params", 10)
        self.create_timer(1.0 / float(self.get_parameter("rate_hz").value), self._tick)

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
        self.get_logger().info(f"goal set: {self._goal.round(2).tolist()}")

    def _on_humans(self, msg: Float32MultiArray) -> None:
        self._humans = np.asarray(msg.data, dtype=float).reshape(-1, 4)

    def _tick(self) -> None:
        if self._state is None or self._goal is None:
            return                      # nothing to modulate yet; MPC waits for a goal too
        v_max_cmd, d_margin_cmd = self.policy.compute(
            self._state, self._goal, self._humans)
        self.pub.publish(Float32MultiArray(data=[float(v_max_cmd), float(d_margin_cmd)]))


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
