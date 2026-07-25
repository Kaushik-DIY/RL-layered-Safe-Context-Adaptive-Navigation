"""Sim-agnostic navigation core: MPC + CBF + 2D sim + RL env.

CRITICAL ARCHITECTURE RULE (plan D8): this package has ZERO ROS imports.
The identical control code runs in the 2D training sim and in Gazebo (wrapped by
thin ROS 2 nodes under ros2_ws/). Keep it that way.
"""

__version__ = "0.0.1"
