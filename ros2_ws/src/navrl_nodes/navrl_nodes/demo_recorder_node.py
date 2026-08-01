"""Record the demo's telemetry to CSV so the video can show the numbers, not just motion.

Subscribes to everything the story needs and writes one row per tick:
  * /odom             -> where the AMR is and how fast it is actually going
  * /navrl/params     -> (v_max_cmd, d_margin_cmd), i.e. the RL supervisor's ACTION.
                         Silent on the baseline run (no supervisor is launched), which is
                         itself the point -- the column is simply the platform max there.
  * /navrl/cbf_info   -> [intervention, protective_stop, h_min, n_active]; h_min < 0 is a
                         stopping-distance violation, the headline metric.
  * /navrl/humans     -> tracked workers, so clearance can be plotted (respects occlusion:
                         a worker still up the aisle is genuinely absent here)

scripts/render_showcase_video.py turns two of these CSVs into the side-by-side panel.
"""
from __future__ import annotations

import csv
import os

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from navrl_nodes import yaw_from_quaternion

FIELDS = ["t", "x", "y", "yaw", "v", "omega",
          "v_max_cmd", "d_margin_cmd", "intervention", "protective_stop",
          "h_min", "n_active", "n_humans", "nearest_human_d"]


class DemoRecorderNode(Node):
    def __init__(self):
        super().__init__("demo_recorder_node")
        self.declare_parameter("out_path", "")
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("v_max_default", 1.5)   # baseline has no supervisor

        path = str(self.get_parameter("out_path").value)
        if not path:
            raise RuntimeError("demo_recorder_node requires 'out_path'")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._fh = open(path, "w", newline="")
        self._w = csv.DictWriter(self._fh, fieldnames=FIELDS)
        self._w.writeheader()
        self._path = path

        self._state = None
        self._params = None
        self._cbf = None
        self._humans = np.zeros((0, 4))
        self._t0 = None
        self._moving = False

        self.create_subscription(Odometry, "/odom", self._on_odom, 10)
        self.create_subscription(Float32MultiArray, "/navrl/params", self._on_params, 10)
        self.create_subscription(Float32MultiArray, "/navrl/cbf_info", self._on_cbf, 10)
        self.create_subscription(Float32MultiArray, "/navrl/humans", self._on_humans, 10)
        self.create_timer(1.0 / float(self.get_parameter("rate_hz").value), self._tick)
        self.get_logger().info(f"recording telemetry -> {path}")

    def _on_odom(self, msg: Odometry) -> None:
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self._state = (p.x, p.y, yaw_from_quaternion(q),
                       msg.twist.twist.linear.x, msg.twist.twist.angular.z)

    def _on_params(self, msg: Float32MultiArray) -> None:
        self._params = (float(msg.data[0]), float(msg.data[1]))

    def _on_cbf(self, msg: Float32MultiArray) -> None:
        self._cbf = [float(v) for v in msg.data]

    def _on_humans(self, msg: Float32MultiArray) -> None:
        self._humans = np.asarray(msg.data, dtype=float).reshape(-1, 4)

    def _tick(self) -> None:
        if self._state is None:
            return
        x, y, yaw, v, omega = self._state
        # t = 0 at the moment the AMR starts the mission, so the two runs align
        if not self._moving and abs(v) > 0.02:
            self._moving = True
            self._t0 = self.get_clock().now().nanoseconds * 1e-9
        if self._t0 is None:
            return
        t = self.get_clock().now().nanoseconds * 1e-9 - self._t0

        vmax = self._params[0] if self._params else float(
            self.get_parameter("v_max_default").value)
        dmar = self._params[1] if self._params else float("nan")
        cbf = self._cbf or [float("nan")] * 4
        nd = float("nan")
        if len(self._humans):
            nd = float(np.min(np.hypot(self._humans[:, 0] - x, self._humans[:, 1] - y)))

        self._w.writerow(dict(
            t=round(t, 3), x=round(x, 4), y=round(y, 4), yaw=round(yaw, 4),
            v=round(v, 4), omega=round(omega, 4),
            v_max_cmd=round(vmax, 4), d_margin_cmd=round(dmar, 4),
            intervention=round(cbf[0], 4), protective_stop=int(cbf[1] == 1.0),
            h_min=round(cbf[2], 4), n_active=int(cbf[3]) if cbf[3] == cbf[3] else 0,
            n_humans=len(self._humans), nearest_human_d=round(nd, 4)))
        self._fh.flush()

    def destroy_node(self):
        try:
            self._fh.close()
            self.get_logger().info(f"telemetry written: {self._path}")
        finally:
            return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DemoRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
