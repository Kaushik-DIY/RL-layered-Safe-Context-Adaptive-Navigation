"""LAYER 1 as a ROS node (plan D8): the CBF safety filter between the MPC proposal
(/navrl/cmd_mpc) and the robot (/cmd_vel).

Fail-safe behaviour is part of the safety story:
  * no MPC proposal yet            -> command zero
  * odometry stale (> `stale_s`)   -> command zero (perception loss = stop)
The filter itself is core.cbf (ROS-free, frozen constants) -- identical to the one
verified by the G2 batteries. NOTHING here takes input from the RL layer.
"""
from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from core.cbf.cbf_filter import CbfFilter
from core.common.platform import load_platform

from navrl_nodes import yaw_from_quaternion


class CbfNode(Node):
    def __init__(self):
        super().__init__("cbf_node")
        self.declare_parameter("platform", "tb3")   # 'industrial' = MiR-class stack
        plat = load_platform(str(self.get_parameter("platform").value))
        self.robot = plat.robot
        self.cbf = plat.cbf
        self.filt = CbfFilter(self.robot, self.cbf)
        self.declare_parameter("stale_s", 0.5)

        self._state = None
        self._odom_t = None
        self._u_mpc = None
        self._humans = np.zeros((0, 4))

        self.create_subscription(Odometry, "/odom", self._on_odom, 10)
        self.create_subscription(Twist, "/navrl/cmd_mpc", self._on_cmd, 10)
        self.create_subscription(Float32MultiArray, "/navrl/humans", self._on_humans, 10)
        self.pub_cmd = self.create_publisher(Twist, "/cmd_vel", 10)
        self.pub_info = self.create_publisher(Float32MultiArray, "/navrl/cbf_info", 10)
        self.create_timer(self.robot.dt, self._tick)   # 10 Hz, same period as MPC

    def _on_odom(self, msg: Odometry) -> None:
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self._state = np.array([p.x, p.y, yaw_from_quaternion(q),
                                msg.twist.twist.linear.x, msg.twist.twist.angular.z])
        self._odom_t = self.get_clock().now()

    def _on_cmd(self, msg: Twist) -> None:
        self._u_mpc = np.array([msg.linear.x, msg.angular.z])

    def _on_humans(self, msg: Float32MultiArray) -> None:
        self._humans = np.asarray(msg.data, dtype=float).reshape(-1, 4)

    def _tick(self) -> None:
        stale = float(self.get_parameter("stale_s").value)
        now = self.get_clock().now()
        odom_ok = (self._state is not None and self._odom_t is not None
                   and (now - self._odom_t).nanoseconds * 1e-9 < stale)
        if not odom_ok or self._u_mpc is None:
            self.pub_cmd.publish(Twist())        # fail-safe: stop
            self.filt.reset()
            return

        u_safe, info = self.filt.filter(self._state, self._u_mpc, self._humans)
        cmd = Twist()
        cmd.linear.x, cmd.angular.z = float(u_safe[0]), float(u_safe[1])
        self.pub_cmd.publish(cmd)
        h = info["h_min"] if np.isfinite(info["h_min"]) else 1e6
        self.pub_info.publish(Float32MultiArray(
            data=[float(info["intervention"]), float(info["protective_stop"]),
                  float(h), float(info["n_active"])]))
        if info["protective_stop"]:
            self.get_logger().warn("protective stop", throttle_duration_sec=1.0)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CbfNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
