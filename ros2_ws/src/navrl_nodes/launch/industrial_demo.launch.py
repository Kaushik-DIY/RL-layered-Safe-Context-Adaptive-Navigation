"""Headline sim-to-sim demo (Gate G5): the INDUSTRIAL learned system in Gazebo.

Brings up the warehouse blind-corner world (MiR-class AMR @1.5 m/s, blind side
aisle, occluded worker) and the full stack on the industrial platform. The RL
supervisor modulates the MPC; the CBF is the frozen backstop.

The A/B money shot is a single argument:
    ros2 launch navrl_nodes industrial_demo.launch.py                  # RL supervisor ON
    ros2 launch navrl_nodes industrial_demo.launch.py always_max:=true # baseline (no RL)
then, once Gazebo is up, send the goal down the corridor:
    ros2 topic pub -1 /goal_pose geometry_msgs/PoseStamped \
        '{header: {frame_id: "odom"}, pose: {position: {x: 12.0, y: 0.0}}}'

`always_max:=true` simply doesn't launch the supervisor, so mpc_node falls back to
the platform's max speed -- the always-max baseline, no separate code path.

model_path defaults to the repo's exported ONNX; override if yours lives elsewhere:
    ros2 launch navrl_nodes industrial_demo.launch.py model_path:=/abs/path/policy.onnx
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# --- 2D blind_corner geometry (platform=industrial), passed to the obs-v2 policy ---
WALLS = [-0.5, -1.75, 13.0, -1.75,
         -0.5,  1.75,  6.0,  1.75,
          7.5,  1.75, 13.0,  1.75,
          6.0,  1.75,  6.0,  3.95,
          7.5,  1.75,  7.5,  3.95]
POSTS = [6.0, 1.75, 0.12,  7.5, 1.75, 0.12]
REVEAL_DISTANCE = 1.2
OCCLUSION_Y = 1.75
DEFAULT_MODEL = os.path.expanduser(
    "~/Context_adaptive_navigation/experiments/models/ppo_ind_C_s0_full_final.onnx")


def generate_launch_description() -> LaunchDescription:
    gazebo_ros = get_package_share_directory("gazebo_ros")
    navrl = get_package_share_directory("navrl_nodes")
    world = os.path.join(navrl, "worlds", "industrial_blind_corner.world")

    always_max = LaunchConfiguration("always_max")
    model_path = LaunchConfiguration("model_path")
    industrial = {"platform": "industrial", "use_sim_time": True}

    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gazebo_ros, "launch", "gzserver.launch.py")),
        launch_arguments={"world": world}.items())
    gzclient = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gazebo_ros, "launch", "gzclient.launch.py")))

    tracker = Node(package="navrl_nodes", executable="human_tracker_node",
                   name="human_tracker", output="screen",
                   parameters=[{"use_sim_time": True,
                                "reveal_distance": REVEAL_DISTANCE,
                                "occlusion_y": OCCLUSION_Y}])
    # the MPC needs the corridor geometry too (NavEnv._mpc_obstacles equivalent) --
    # without it the controller is blind to the walls and drives into them
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

    # start the stack after Gazebo has the world + AMR up (/odom exists)
    stack = TimerAction(period=4.0, actions=[tracker, mpc, cbf, viz, supervisor])

    return LaunchDescription([
        DeclareLaunchArgument("always_max", default_value="false",
                              description="true = baseline (no RL supervisor)"),
        DeclareLaunchArgument("model_path", default_value=DEFAULT_MODEL,
                              description="absolute path to the exported ONNX policy"),
        gzserver, gzclient, stack,
    ])
