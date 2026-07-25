"""Launch the MPC + CBF + tracker stack (plan D8).

Gazebo/TB3 are launched separately (turtlebot3_gazebo), keeping this file about
OUR stack only:

    export TURTLEBOT3_MODEL=waffle
    ros2 launch turtlebot3_gazebo empty_world.launch.py      # or any TB3 world
    ros2 launch navrl_nodes nav_stack.launch.py
    ros2 topic pub -1 /goal_pose geometry_msgs/PoseStamped \
        '{pose: {position: {x: 2.0, y: 0.0}}}'
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        Node(package="navrl_nodes", executable="human_tracker_node",
             name="human_tracker", output="screen"),
        Node(package="navrl_nodes", executable="mpc_node",
             name="mpc_controller", output="screen"),
        Node(package="navrl_nodes", executable="cbf_node",
             name="cbf_filter", output="screen"),
    ])
