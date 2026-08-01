"""Headline portfolio demo: the INDUSTRIAL learned system across a full warehouse run.

One ~30 s mission through THREE hazard types, so the clip shows the supervisor
adapting repeatedly instead of reacting once:

    x ~  6.25   blind corner      (worker occluded up a north racking aisle)
    x ~ 13.25   4-way intersection (worker crossing from the south aisle, visible)
    x ~ 20.25   blind corner      (second occluded aisle, different approach speed)

The A/B money shot is still a single argument:
    ros2 launch navrl_nodes warehouse_demo.launch.py                  # RL supervisor ON
    ros2 launch navrl_nodes warehouse_demo.launch.py always_max:=true # baseline (no RL)

`always_max:=true` simply doesn't launch the supervisor, so mpc_node falls back to the
platform's max speed -- the always-max baseline, no separate code path.

The goal is published AUTOMATICALLY `goal_delay` seconds after launch (default 8 s,
which lets Gazebo finish loading). That keeps the two runs comparable: hand-publishing
the goal made the encounter timing luck-of-the-draw. Set `auto_goal:=false` to publish
it yourself instead.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            IncludeLaunchDescription, TimerAction)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# --- 2D warehouse geometry, passed to BOTH the obs-v2 policy and the MPC ---
# main corridor y in [-2, 2]; north aisles at A/B/C; south aisle at the intersection
WALLS = [-1.0, -2.0, 12.5, -2.0,   14.0, -2.0, 25.0, -2.0,
         12.5, -2.0, 12.5, -4.5,   14.0, -2.0, 14.0, -4.5,
         -1.0,  2.0,  5.5,  2.0,    7.0,  2.0, 12.5,  2.0,
         14.0,  2.0, 19.5,  2.0,   21.0,  2.0, 25.0,  2.0,
          5.5,  2.0,  5.5,  4.5,    7.0,  2.0,  7.0,  4.5,
         12.5,  2.0, 12.5,  4.5,   14.0,  2.0, 14.0,  4.5,
         19.5,  2.0, 19.5,  4.5,   21.0,  2.0, 21.0,  4.5]
POSTS = [ 5.5,  2.0, 0.12,   7.0,  2.0, 0.12,
         12.5,  2.0, 0.12,  14.0,  2.0, 0.12,
         12.5, -2.0, 0.12,  14.0, -2.0, 0.12,
         19.5,  2.0, 0.12,  21.0,  2.0, 0.12]

REVEAL_DISTANCE = 1.2
OCCLUSION_Y = 2.0            # corridor's north edge: workers above it are occluded
GOAL_XY = (24.0, 0.0)
DEFAULT_MODEL = os.path.expanduser(
    "~/Context_adaptive_navigation/experiments/models/ppo_ind_C_s0_full_final.onnx")

GOAL_MSG = ("{header: {frame_id: 'odom'}, pose: {position: "
            f"{{x: {GOAL_XY[0]}, y: {GOAL_XY[1]}}}}}}}")


def generate_launch_description() -> LaunchDescription:
    gazebo_ros = get_package_share_directory("gazebo_ros")
    navrl = get_package_share_directory("navrl_nodes")
    world = os.path.join(navrl, "worlds", "industrial_warehouse.world")

    always_max = LaunchConfiguration("always_max")
    model_path = LaunchConfiguration("model_path")
    goal_delay = LaunchConfiguration("goal_delay")
    industrial = {"platform": "industrial", "use_sim_time": True}

    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros, "launch", "gzserver.launch.py")),
        launch_arguments={"world": world}.items())
    gzclient = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros, "launch", "gzclient.launch.py")))

    tracker = Node(package="navrl_nodes", executable="human_tracker_node",
                   name="human_tracker", output="screen",
                   parameters=[{"use_sim_time": True,
                                "reveal_distance": REVEAL_DISTANCE,
                                "occlusion_y": OCCLUSION_Y}])
    # the MPC needs the corridor geometry too (NavEnv._mpc_obstacles equivalent),
    # otherwise the controller is blind to the walls and drives into them
    mpc = Node(package="navrl_nodes", executable="mpc_node",
               name="mpc_controller", output="screen",
               parameters=[industrial, {"walls": WALLS, "posts": POSTS}])
    cbf = Node(package="navrl_nodes", executable="cbf_node",
               name="cbf_filter", output="screen", parameters=[industrial])
    viz = Node(package="navrl_nodes", executable="field_viz_node",
               name="field_viz", output="screen", parameters=[industrial])
    supervisor = Node(package="navrl_nodes", executable="rl_supervisor_node",
                      name="rl_supervisor", output="screen",
                      condition=UnlessCondition(always_max),
                      parameters=[{"use_sim_time": True, "platform": "industrial",
                                   "model_path": model_path,
                                   "walls": WALLS, "posts": POSTS}])

    stack = TimerAction(period=4.0, actions=[tracker, mpc, cbf, viz, supervisor])
    send_goal = TimerAction(
        period=goal_delay,
        actions=[ExecuteProcess(
            cmd=["ros2", "topic", "pub", "-1", "/goal_pose",
                 "geometry_msgs/PoseStamped", GOAL_MSG],
            condition=IfCondition(LaunchConfiguration("auto_goal")),
            output="screen")])

    return LaunchDescription([
        DeclareLaunchArgument("always_max", default_value="false",
                              description="true = baseline (no RL supervisor)"),
        DeclareLaunchArgument("auto_goal", default_value="true",
                              description="publish the goal automatically"),
        DeclareLaunchArgument("goal_delay", default_value="8.0",
                              description="seconds after launch to publish the goal"),
        DeclareLaunchArgument("model_path", default_value=DEFAULT_MODEL,
                              description="absolute path to the exported ONNX policy"),
        gzserver, gzclient, stack, send_goal,
    ])
