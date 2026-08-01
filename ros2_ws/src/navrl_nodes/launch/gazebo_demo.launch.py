"""One-shot sim-to-sim transfer demo (Gate G5).

Brings up EVERYTHING for the portfolio video in a single command:
  Gazebo Classic + our pedestrian_crossing.world (crossing actor + state plugin)
  -> spawns the TurtleBot3 waffle at the origin
  -> starts the MPC + CBF + human_tracker stack (3 s after spawn, so /odom exists)

Usage:
    export TURTLEBOT3_MODEL=waffle
    ros2 launch navrl_nodes gazebo_demo.launch.py
    # then, once Gazebo is up, publish the goal:
    ros2 topic pub -1 /goal_pose geometry_msgs/PoseStamped \
        '{header: {frame_id: "odom"}, pose: {position: {x: 4.0, y: 0.0}}}'

Composition follows the stock turtlebot3_gazebo world launches, so it works on a
standard ROS 2 Humble + Gazebo Classic 11 install. If spawn/robot_state_publisher
launch filenames differ on your TB3 package, adjust the two paths flagged below.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    gazebo_ros = get_package_share_directory("gazebo_ros")
    tb3_gazebo = get_package_share_directory("turtlebot3_gazebo")
    navrl = get_package_share_directory("navrl_nodes")

    world = os.path.join(navrl, "worlds", "pedestrian_crossing.world")
    tb3_launch = os.path.join(tb3_gazebo, "launch")

    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gazebo_ros, "launch", "gzserver.launch.py")),
        launch_arguments={"world": world}.items(),
    )
    gzclient = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gazebo_ros, "launch", "gzclient.launch.py")),
    )
    # --- adjust these two filenames if your turtlebot3_gazebo names them differently ---
    robot_state_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(tb3_launch, "robot_state_publisher.launch.py")),
        launch_arguments={"use_sim_time": "true"}.items(),
    )
    spawn_tb3 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(tb3_launch, "spawn_turtlebot3.launch.py")),
        launch_arguments={"x_pose": "0.0", "y_pose": "0.0"}.items(),
    )

    sim_time = {"use_sim_time": True}
    nav_stack = TimerAction(period=3.0, actions=[
        Node(package="navrl_nodes", executable="human_tracker_node",
             name="human_tracker", output="screen", parameters=[sim_time]),
        Node(package="navrl_nodes", executable="mpc_node",
             name="mpc_controller", output="screen", parameters=[sim_time]),
        Node(package="navrl_nodes", executable="cbf_node",
             name="cbf_filter", output="screen", parameters=[sim_time]),
        Node(package="navrl_nodes", executable="field_viz_node",
             name="field_viz", output="screen", parameters=[sim_time]),
    ])

    return LaunchDescription([gzserver, gzclient, robot_state_publisher, spawn_tb3, nav_stack])
