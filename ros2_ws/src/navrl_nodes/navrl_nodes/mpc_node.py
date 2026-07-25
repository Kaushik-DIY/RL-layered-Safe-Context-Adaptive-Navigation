"""LAYER 2 as a ROS node (plan D8): 10 Hz NMPC proposing /navrl/cmd_mpc.

Pure wrapper: state in, Twist out. The controller itself is core.mpc (ROS-free),
identical to what the 2D sim runs -- that identity is the transfer claim.

The RL parameters arrive on /navrl/params; until a supervisor exists, platform
defaults apply (v_max, default_margin), which makes this node the S2-style fixed
stack Gate G3 requires ("Gazebo runs the MPC+CBF stack closed-loop").
"""
from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from core.common.params import MpcParams, RobotParams
from core.mpc.mpc_controller import MpcController

from navrl_nodes import yaw_from_quaternion

GOAL_TOL = 0.15  # m


class MpcNode(Node):
    def __init__(self):
        super().__init__("mpc_node")
        self.robot = RobotParams.from_yaml()
        self.mpc_cfg = MpcParams.from_yaml()
        self.mpc = MpcController(self.robot, self.mpc_cfg)

        # flat [x, y, r] * n static obstacles, settable per scenario
        self.declare_parameter("static_obstacles", [0.0])
        self._u_prev = np.zeros(2)
        self._state = None            # [x, y, yaw, v, omega]
        self._goal = None
        self._humans = np.zeros((0, 4))
        self._params = (None, None)   # (v_max_cmd, d_margin_cmd)

        self.create_subscription(Odometry, "/odom", self._on_odom, 10)
        self.create_subscription(PoseStamped, "/goal_pose", self._on_goal, 10)
        self.create_subscription(Float32MultiArray, "/navrl/humans", self._on_humans, 10)
        self.create_subscription(Float32MultiArray, "/navrl/params", self._on_params, 10)
        self.pub_cmd = self.create_publisher(Twist, "/navrl/cmd_mpc", 10)
        self.pub_info = self.create_publisher(Float32MultiArray, "/navrl/mpc_info", 10)
        self.create_timer(self.robot.dt, self._tick)   # 10 Hz control loop

    # ------------------------------------------------------------- callbacks
    def _on_odom(self, msg: Odometry) -> None:
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self._state = np.array([p.x, p.y, yaw_from_quaternion(q),
                                msg.twist.twist.linear.x, msg.twist.twist.angular.z])

    def _on_goal(self, msg: PoseStamped) -> None:
        self._goal = np.array([msg.pose.position.x, msg.pose.position.y])
        self.get_logger().info(f"goal set: {self._goal.round(2).tolist()}")

    def _on_humans(self, msg: Float32MultiArray) -> None:
        self._humans = np.asarray(msg.data, dtype=float).reshape(-1, 4)

    def _on_params(self, msg: Float32MultiArray) -> None:
        self._params = (float(msg.data[0]), float(msg.data[1]))

    # ------------------------------------------------------------------ loop
    def _static_obs(self):
        flat = [float(v) for v in self.get_parameter("static_obstacles").value]
        if len(flat) < 3:
            return None
        return np.asarray(flat[: 3 * (len(flat) // 3)]).reshape(-1, 3)

    def _carrot(self) -> np.ndarray:
        """Straight-line carrot toward the goal (same policy as the 2D env)."""
        to_goal = self._goal - self._state[:2]
        d = float(np.hypot(*to_goal))
        if d < 1e-9:
            return self._goal
        return self._state[:2] + to_goal / d * min(self.mpc_cfg.carrot_lookahead, d)

    def _tick(self) -> None:
        if self._state is None or self._goal is None:
            return  # wait for odom + goal
        if float(np.hypot(*(self._goal - self._state[:2]))) < GOAL_TOL:
            self.pub_cmd.publish(Twist())        # arrived: propose zero
            self._u_prev = np.zeros(2)
            return

        humans = self._humans[
            np.argsort(np.hypot(self._humans[:, 0] - self._state[0],
                                self._humans[:, 1] - self._state[1]))
            [: self.mpc_cfg.max_humans]] if len(self._humans) else None
        u, info = self.mpc.solve(
            x0=self._state[:3], carrot=self._carrot(), static_obs=self._static_obs(),
            humans=humans, v_max_cmd=self._params[0], d_margin_cmd=self._params[1],
            u_prev=self._u_prev)
        self._u_prev = u

        cmd = Twist()
        cmd.linear.x, cmd.angular.z = float(u[0]), float(u[1])
        self.pub_cmd.publish(cmd)
        self.pub_info.publish(Float32MultiArray(
            data=[float(info["solve_ms"]), float(info["success"])]))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MpcNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
