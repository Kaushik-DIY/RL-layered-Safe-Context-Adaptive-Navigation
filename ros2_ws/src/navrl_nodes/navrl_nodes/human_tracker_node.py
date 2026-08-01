"""Ground-truth human tracker (plan D8): publishes /navrl/humans.

Abstracted behind a fixed array interface so a real detector (or HuNavSim's agent
topics) can replace this node without touching the control stack. This
implementation reads Gazebo model states and selects models whose names match the
`name_prefixes` parameter (default: actor/human/person/pedestrian).

Gazebo Classic note: /gazebo/model_states only exists if the world loads the
`libgazebo_ros_state.so` plugin. Without it this node publishes an empty array at
10 Hz (the stack then runs human-free -- still a valid G3 closed-loop check).
Injected observation noise for the robustness study is a parameter here (off by
default), NOT hidden inside the filter.
"""
from __future__ import annotations

import numpy as np
import rclpy
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class HumanTrackerNode(Node):
    def __init__(self):
        super().__init__("human_tracker_node")
        self.declare_parameter("name_prefixes", ["actor", "human", "person", "pedestrian"])
        self.declare_parameter("vel_noise_std", 0.0)   # m/s, robustness study only
        # Minimum baseline for the finite-difference velocity estimate.
        # /gazebo/model_states is published faster than a scripted/animated pedestrian's
        # pose actually changes, so differencing CONSECUTIVE samples aliases badly: with a
        # 30 Hz pose source sampled at 50 Hz, a 1.35 m/s walker reads as 0 on one sample
        # and 2.25 m/s on the next. The MPC propagates that over its 2 s horizon (4.5 m
        # instead of 2.7 m) and swerves hard at a crossing -- in Gazebo this turned the
        # robot 90 degrees into a side aisle, where it deadlocked. Differencing over a
        # fixed window instead makes the estimate stable.
        self.declare_parameter("vel_window_s", 0.20)
        # Blind-corner occlusion, reproducing core.sim2d ScenarioSpec.visible_humans
        # (the model the policy TRAINED under): a worker is hidden until it enters the
        # main corridor (y <= occlusion_y) OR the robot closes within reveal_distance,
        # then it LATCHES (a real tracker keeps the track). Disabled by default
        # (reveal_distance <= 0), so the plain G3 tracker is unchanged.
        self.declare_parameter("reveal_distance", 0.0)     # m; <=0 disables occlusion
        self.declare_parameter("occlusion_y", 1e9)         # corridor line: y<=this is open
        self._named = []         # [(name, x, y, vx, vy)] latest frame
        self._seen = set()       # names revealed at least once (latch)
        self._robot_xy = None
        self._prev = {}          # name -> (x, y, t), the velocity-window anchor
        self._vel = {}           # name -> (vx, vy), last good estimate
        self._rng = np.random.default_rng(0)

        # both topic spellings seen across gazebo_ros_state configurations
        self.create_subscription(ModelStates, "/gazebo/model_states", self._on_states, 10)
        self.create_subscription(ModelStates, "/model_states", self._on_states, 10)
        self.create_subscription(Odometry, "/odom", self._on_odom, 10)
        self.pub = self.create_publisher(Float32MultiArray, "/navrl/humans", 10)
        self.create_timer(0.1, self._tick)   # 10 Hz

    def _on_odom(self, msg: Odometry) -> None:
        self._robot_xy = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def _on_states(self, msg: ModelStates) -> None:
        prefixes = tuple(self.get_parameter("name_prefixes").value)
        t = self.get_clock().now().nanoseconds * 1e-9
        named = []
        for name, pose, twist in zip(msg.name, msg.pose, msg.twist):
            if not name.startswith(prefixes):
                continue
            x, y = pose.position.x, pose.position.y
            vx, vy = twist.linear.x, twist.linear.y
            # Gazebo <actor>s report ZERO twist (animated, not physics-driven), so a
            # crossing pedestrian would look static and the stack could not anticipate
            # its path. Estimate velocity by finite-differencing the pose (what a real
            # tracker does) whenever the reported twist is ~0.
            if abs(vx) < 1e-3 and abs(vy) < 1e-3:
                win = float(self.get_parameter("vel_window_s").value)
                if name in self._prev:
                    px, py, pt = self._prev[name]
                    dt = t - pt
                    if dt >= win:
                        vx, vy = (x - px) / dt, (y - py) / dt
                        self._vel[name] = (vx, vy)
                        self._prev[name] = (x, y, t)
                    else:
                        vx, vy = self._vel.get(name, (0.0, 0.0))   # hold last good
                else:
                    self._prev[name] = (x, y, t)
            else:
                self._prev[name] = (x, y, t)         # physics twist is usable as-is
            named.append((name, x, y, vx, vy))
        self._named = named

    def _visible_rows(self):
        reveal = float(self.get_parameter("reveal_distance").value)
        occ_y = float(self.get_parameter("occlusion_y").value)
        rows = []
        for name, x, y, vx, vy in self._named:
            if reveal > 0.0 and name not in self._seen:
                in_open = y <= occ_y
                close = (self._robot_xy is not None
                         and np.hypot(x - self._robot_xy[0], y - self._robot_xy[1]) <= reveal)
                if in_open or close:
                    self._seen.add(name)
                if reveal > 0.0 and name not in self._seen:
                    continue                      # still occluded this frame
            rows.append([x, y, vx, vy])
        return np.asarray(rows, dtype=float).reshape(-1, 4)

    def _tick(self) -> None:
        out = self._visible_rows()
        noise = float(self.get_parameter("vel_noise_std").value)
        if noise > 0.0 and len(out):
            out[:, 2:] += self._rng.normal(0.0, noise, size=out[:, 2:].shape)
        self.pub.publish(Float32MultiArray(data=out.ravel().tolist()))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HumanTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
