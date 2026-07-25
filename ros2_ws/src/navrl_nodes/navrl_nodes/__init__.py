"""Thin ROS 2 wrappers around the sim-agnostic MPC+CBF core (plan D8).

Topic conventions (all nodes):
    /odom               nav_msgs/Odometry          robot state (from TB3 / Gazebo)
    /goal_pose          geometry_msgs/PoseStamped  navigation goal (rviz2-compatible)
    /navrl/humans       std_msgs/Float32MultiArray flat [x, y, vx, vy] * n humans
    /navrl/params       std_msgs/Float32MultiArray [v_max_cmd, d_margin_cmd] (RL, 2 Hz)
    /navrl/cmd_mpc      geometry_msgs/Twist        MPC proposal (Layer 2 output)
    /cmd_vel            geometry_msgs/Twist        CBF-filtered command (robot input)
    /navrl/cbf_info     std_msgs/Float32MultiArray [intervention, pstop, h_min, n_active]
    /navrl/mpc_info     std_msgs/Float32MultiArray [solve_ms, success]

Humans ride a Float32MultiArray rather than a custom message on purpose: custom
interfaces need a CMake rosidl package, and the tracker interface is expected to be
replaced by HuNavSim/a real tracker later -- freeze the array layout, not a msg type.
"""

import numpy as np


def yaw_from_quaternion(q) -> float:
    """Yaw (Z) from a geometry_msgs Quaternion -- avoids a tf_transformations dep."""
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return float(np.arctan2(siny, cosy))
