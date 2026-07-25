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
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class HumanTrackerNode(Node):
    def __init__(self):
        super().__init__("human_tracker_node")
        self.declare_parameter("name_prefixes", ["actor", "human", "person", "pedestrian"])
        self.declare_parameter("vel_noise_std", 0.0)   # m/s, robustness study only
        self._humans = np.zeros((0, 4))
        self._rng = np.random.default_rng(0)

        # both topic spellings seen across gazebo_ros_state configurations
        self.create_subscription(ModelStates, "/gazebo/model_states", self._on_states, 10)
        self.create_subscription(ModelStates, "/model_states", self._on_states, 10)
        self.pub = self.create_publisher(Float32MultiArray, "/navrl/humans", 10)
        self.create_timer(0.1, self._tick)   # 10 Hz

    def _on_states(self, msg: ModelStates) -> None:
        prefixes = tuple(self.get_parameter("name_prefixes").value)
        rows = []
        for name, pose, twist in zip(msg.name, msg.pose, msg.twist):
            if name.startswith(prefixes):
                rows.append([pose.position.x, pose.position.y,
                             twist.linear.x, twist.linear.y])
        self._humans = np.asarray(rows, dtype=float).reshape(-1, 4)

    def _tick(self) -> None:
        out = self._humans.copy()
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
