"""Visualization-only node: draws the whole scene for rviz.

This is what you RECORD. Speed reads badly on a 3D camera; the ISO 3691-4 stopping
envelope does not, because it scales with v^2 -- 2.83 m at 1.5 m/s against 1.03 m at
0.8 m/s. The envelope is `d_stop(sigma*v) + d_hard`, exactly the quantity the CBF
barrier is built from, and it turns red the moment a worker is inside it.

It publishes the ROBOT and the WALLS too, not just the envelope. rviz had no way to draw
either: there is no RobotModel display and no `robot_description`, because the AMR is
defined inside the world SDF. With only a translucent disc and some odometry arrows in an
otherwise empty viewport, the disc covered the screen and the robot was invisible.

Pure visualization: subscribes to /odom, /navrl/cbf_info, /navrl/humans, /navrl/params
and publishes markers. It takes NO part in control and cannot affect the robot.
"""
from __future__ import annotations

import math

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray

from core.common.platform import load_platform

from navrl_nodes import yaw_from_quaternion

C_SAFE = (0.10, 0.72, 0.30)
C_WARN = (0.95, 0.62, 0.10)
C_BAD = (0.85, 0.13, 0.10)


class FieldVizNode(Node):
    def __init__(self):
        super().__init__("field_viz_node")
        self.declare_parameter("platform", "tb3")   # 'industrial' = MiR-class stack
        self.declare_parameter("frame_id", "odom")
        # flat [x1,y1,x2,y2]*n wall segments of THIS world, so rviz shows the aisle
        self.declare_parameter("walls", [0.0])
        # The envelope is drawn as a RING. A filled translucent disc is available but off
        # by default: with nothing else in the scene it covered the whole viewport.
        self.declare_parameter("fill_envelope", False)
        plat = load_platform(str(self.get_parameter("platform").value))
        self.robot = plat.robot
        self.cbf = plat.cbf

        self._v = 0.0
        self._pose = (0.0, 0.0)
        self._yaw = 0.0
        self._pstop = 0.0
        self._h = float("nan")
        self._cap = float(self.robot.v_max)      # no supervisor -> platform max
        self._humans = np.zeros((0, 4))
        self._viol_s = 0.0
        self._was_breach = False
        self._marks = []

        self.create_subscription(Odometry, "/odom", self._on_odom, 10)
        self.create_subscription(Float32MultiArray, "/navrl/cbf_info", self._on_info, 10)
        self.create_subscription(Float32MultiArray, "/navrl/humans", self._on_humans, 10)
        self.create_subscription(Float32MultiArray, "/navrl/params", self._on_params, 10)
        self.pub = self.create_publisher(MarkerArray, "/navrl/field", 10)
        self.create_timer(0.1, self._tick)   # 10 Hz

    # ------------------------------------------------------------- callbacks
    def _on_odom(self, msg: Odometry) -> None:
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self._pose = (p.x, p.y)
        self._yaw = yaw_from_quaternion(q)
        self._v = float(msg.twist.twist.linear.x)

    def _on_info(self, msg: Float32MultiArray) -> None:
        d = list(msg.data)
        self._pstop = d[1] if len(d) > 1 else 0.0
        self._h = d[2] if len(d) > 2 else float("nan")

    def _on_params(self, msg: Float32MultiArray) -> None:
        self._cap = float(msg.data[0])

    def _on_humans(self, msg: Float32MultiArray) -> None:
        self._humans = np.asarray(msg.data, dtype=float).reshape(-1, 4)

    # ----------------------------------------------------------------- draw
    def _stopping_radius(self) -> float:
        """d_stop(sigma*v) + d_hard -- exactly the barrier the CBF enforces.

        robot_radius is deliberately NOT included: it does not appear in the barrier
        (h = d - d_stop - d_hard, centre-to-centre).
        """
        s = self.cbf.sigma * max(self._v, 0.0)
        return self.cbf.d_hard + s * self.cbf.tau + s * s / (2.0 * self.cbf.a_brake)

    def _mk(self, ns, mid, mtype, frame, stamp):
        m = Marker()
        m.header.frame_id, m.header.stamp = frame, stamp
        m.ns, m.id, m.type, m.action = ns, mid, mtype, Marker.ADD
        m.pose.orientation.w = 1.0
        m.color.a = 1.0
        return m

    def _wall_marker(self, frame, stamp):
        flat = [float(v) for v in self.get_parameter("walls").value]
        if len(flat) < 4:
            return None
        m = self._mk("walls", 0, Marker.LINE_LIST, frame, stamp)
        m.scale.x = 0.10
        m.color.r, m.color.g, m.color.b = 0.35, 0.37, 0.42
        for i in range(0, len(flat) - 3, 4):
            for j in (0, 2):
                p = Point()
                p.x, p.y, p.z = flat[i + j], flat[i + j + 1], 0.02
                m.points.append(p)
        return m

    def _tick(self) -> None:
        frame = str(self.get_parameter("frame_id").value)
        stamp = self.get_clock().now().to_msg()
        arr = MarkerArray()
        x, y = self._pose
        r = self._stopping_radius()

        breach = bool(np.isfinite(self._h) and self._h < 0.0) or self._pstop > 0.5
        if breach:
            col = C_BAD
        elif np.isfinite(self._h) and self._h < 0.35:
            col = C_WARN
        else:
            col = C_SAFE

        walls = self._wall_marker(frame, stamp)
        if walls is not None:
            arr.markers.append(walls)

        # --- the AMR itself: nothing else in rviz draws it -------------------
        body = self._mk("robot", 0, Marker.CUBE, frame, stamp)
        body.pose.position.x, body.pose.position.y, body.pose.position.z = x, y, 0.14
        body.pose.orientation.z = math.sin(self._yaw / 2.0)
        body.pose.orientation.w = math.cos(self._yaw / 2.0)
        body.scale.x, body.scale.y, body.scale.z = 0.75, 0.50, 0.28
        body.color.r, body.color.g, body.color.b = 0.12, 0.35, 0.75
        arr.markers.append(body)

        nose = self._mk("robot", 1, Marker.ARROW, frame, stamp)
        nose.scale.x, nose.scale.y, nose.scale.z = 0.06, 0.13, 0.13
        for dx in (0.30, 0.90):
            p = Point()
            p.x = x + dx * math.cos(self._yaw)
            p.y = y + dx * math.sin(self._yaw)
            p.z = 0.30
            nose.points.append(p)
        nose.color.r, nose.color.g, nose.color.b = 0.97, 0.80, 0.10
        arr.markers.append(nose)

        # --- stopping envelope, as a RING ------------------------------------
        ring = self._mk("envelope", 0, Marker.LINE_STRIP, frame, stamp)
        ring.scale.x = 0.07
        ring.color.r, ring.color.g, ring.color.b = col
        for k in range(49):
            a = 2.0 * math.pi * k / 48.0
            p = Point()
            p.x, p.y, p.z = x + r * math.cos(a), y + r * math.sin(a), 0.03
            ring.points.append(p)
        arr.markers.append(ring)

        if bool(self.get_parameter("fill_envelope").value):
            disc = self._mk("envelope", 1, Marker.CYLINDER, frame, stamp)
            disc.pose.position.x, disc.pose.position.y, disc.pose.position.z = x, y, 0.01
            disc.scale.x = disc.scale.y = 2.0 * r
            disc.scale.z = 0.01
            disc.color.r, disc.color.g, disc.color.b = col
            disc.color.a = 0.18
            arr.markers.append(disc)

        # --- violations: one persistent mark per event -----------------------
        if breach:
            self._viol_s += 0.1                       # this timer runs at 10 Hz
            if not self._was_breach:
                self._marks.append((x, y))
        self._was_breach = breach
        for i, (mx, my) in enumerate(self._marks):
            m = self._mk("violations", i, Marker.CUBE, frame, stamp)
            m.pose.position.x, m.pose.position.y, m.pose.position.z = mx, my, 0.04
            m.scale.x = m.scale.y = 0.5
            m.scale.z = 0.03
            m.color.r, m.color.g, m.color.b = C_BAD
            arr.markers.append(m)

        # --- tracked workers ---------------------------------------------------
        for i, h in enumerate(self._humans):
            m = self._mk("humans", i, Marker.CYLINDER, frame, stamp)
            m.pose.position.x = float(h[0])
            m.pose.position.y = float(h[1])
            m.pose.position.z = 0.9
            m.scale.x = m.scale.y = 0.45
            m.scale.z = 1.8
            m.color.r, m.color.g, m.color.b = 0.97, 0.45, 0.05
            arr.markers.append(m)
        for i in range(len(self._humans), len(self._humans) + 8):
            m = self._mk("humans", i, Marker.CYLINDER, frame, stamp)
            m.action = Marker.DELETE
            arr.markers.append(m)

        # --- readout ------------------------------------------------------------
        txt = self._mk("readout", 0, Marker.TEXT_VIEW_FACING, frame, stamp)
        txt.pose.position.x, txt.pose.position.y, txt.pose.position.z = x, y, 2.4
        txt.scale.z = 0.40
        txt.color.r = txt.color.g = txt.color.b = 0.10
        hs = "--" if not np.isfinite(self._h) else f"{self._h:+.2f} m"
        txt.text = (f"speed {self._v:.2f} m/s    cap {self._cap:.2f} m/s\n"
                    f"protective field {r:.2f} m    barrier h {hs}\n"
                    f"time in violation {self._viol_s:.2f} s")
        arr.markers.append(txt)

        self.pub.publish(arr)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FieldVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
