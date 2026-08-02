"""THE FINAL DEMO in Gazebo: the same stack and the same route as the 2D video, in 3D.

    ros2 launch navrl_nodes final_demo.launch.py                 # Gazebo + rviz
    ros2 launch navrl_nodes final_demo.launch.py rviz:=false      # Gazebo only
    ros2 launch navrl_nodes final_demo.launch.py run:=fixed       # no supervisor
    ros2 launch navrl_nodes final_demo.launch.py gui:=false rviz:=false   # headless

A 31 m mission down a 5.0 m two-way aisle, three encounters:

    x =  7.5   picker head-on AND a blind cross-aisle opening SOUTH, the side the machine
               would have to swerve into. It must refuse and slow.
    x = 16.0   a true 4-way junction, occluded worker crossing north to south.
    x = 24.5   the SAME picker head-on, solid racking both sides: the room is real, so it
               offsets and carries its speed through.

WHAT THIS IS FOR. The 2D gate is the measurement; this is the proof that the identical
control stack -- same ONNX policy, same MPC, same CBF, same map-derived guards -- drives a
physically simulated differential-drive robot through the same route. Nothing here is a
re-implementation for the demo: the nodes import `core.*` exactly as the 2D harness does,
and the world is generated from the same `core.demo.final_route`.

Telemetry lands in `experiments/results/final_gz_<run>.csv`; compare it against the 2D
numbers with `scripts/check_final_gazebo.py`.

Requires the repo root on PYTHONPATH (the nodes import `core.*`):
    export PYTHONPATH=$HOME/Context_adaptive_navigation:$PYTHONPATH
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            IncludeLaunchDescription, TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (LaunchConfiguration, PathJoinSubstitution,
                                  PythonExpression)
from launch_ros.actions import Node

from core.demo.aisle_scene import REVEAL_DISTANCE
from core.demo.final_route import GOAL, HALF_W, POSTS, WALLS
from core.demo.industrial_amr import COMMISSIONED

REPO = os.path.expanduser("~/Context_adaptive_navigation")
DEFAULT_MODEL = os.path.join(REPO, "experiments/models/ppo_ind_C_s0_full_final.onnx")
GOAL_MSG = ("{header: {frame_id: 'odom'}, pose: {position: "
            "{x: %.1f, y: %.1f}}}" % (GOAL[0], GOAL[1]))


def generate_launch_description() -> LaunchDescription:
    gazebo_ros = get_package_share_directory("gazebo_ros")
    navrl = get_package_share_directory("navrl_nodes")
    world = os.path.join(navrl, "worlds", "final_demo.world")

    run = LaunchConfiguration("run")
    model_path = LaunchConfiguration("model_path")
    goal_delay = LaunchConfiguration("goal_delay")
    out_csv = LaunchConfiguration("out_csv")
    is_rl = IfCondition(PythonExpression(["'", run, "' == 'rl'"]))

    industrial = {"platform": "industrial", "use_sim_time": True}
    geom = {"walls": WALLS.ravel().tolist(), "posts": POSTS.ravel().tolist()}

    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros, "launch", "gzserver.launch.py")),
        launch_arguments={"world": world}.items())
    gzclient = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros, "launch", "gzclient.launch.py")),
        condition=IfCondition(LaunchConfiguration("gui")))

    tracker = Node(package="navrl_nodes", executable="human_tracker_node",
                   name="human_tracker", output="screen",
                   parameters=[{"use_sim_time": True,
                                "reveal_distance": REVEAL_DISTANCE,
                                # a worker is hidden by racking above THIS aisle's wall line; the
                                # module default is the 3.5 m aisle and would reveal
                                # the junction worker 0.75 m early
                                "occlusion_y": float(HALF_W),
                                "name_prefixes": ["head_at_", "cross_at_", "worker", "human"]}])
    mpc = Node(package="navrl_nodes", executable="mpc_node", name="mpc_controller",
               output="screen", parameters=[industrial, geom])
    # relaxed governor: what every recent 2D result drives by
    cbf = Node(package="navrl_nodes", executable="cbf_node", name="cbf_filter",
               output="screen", parameters=[industrial, {"relaxed": True}])
    viz = Node(package="navrl_nodes", executable="field_viz_node", name="field_viz",
               output="screen",
               parameters=[industrial, {"walls": WALLS.ravel().tolist()}])
    director = Node(package="navrl_nodes", executable="scene_director_node",
                    name="scene_director", output="screen",
                    parameters=[{"use_sim_time": True}])
    # rviz is where the SAFETY layer is legible: speed differences read badly in 3D
    # (1.2 vs 0.6 m/s looks similar on camera) but the protective field does not, because
    # it scales with v^2. Run it alongside Gazebo rather than instead of it.
    rviz = Node(package="rviz2", executable="rviz2", name="rviz2", output="log",
                arguments=["-d", os.path.join(navrl, "rviz", "demo.rviz")],
                parameters=[{"use_sim_time": True}],
                condition=IfCondition(LaunchConfiguration("rviz")))
    recorder = Node(package="navrl_nodes", executable="demo_recorder_node",
                    name="demo_recorder", output="screen",
                    parameters=[{"use_sim_time": True, "out_path": out_csv}])
    supervisor = Node(package="navrl_nodes", executable="rl_supervisor_node",
                      name="rl_supervisor", output="screen", condition=is_rl,
                      parameters=[{"use_sim_time": True, "platform": "industrial",
                                   "model_path": model_path,
                                   "half_w": float(HALF_W),
                                   "commissioned": float(COMMISSIONED),
                                   "walls": WALLS.ravel().tolist(),
                                   "posts": POSTS.ravel().tolist()}])

    stack = TimerAction(period=4.0,
                        actions=[tracker, mpc, cbf, viz, director, recorder,
                                 supervisor, rviz])
    send_goal = TimerAction(period=goal_delay, actions=[ExecuteProcess(
        cmd=["ros2", "topic", "pub", "-1", "/goal_pose",
             "geometry_msgs/PoseStamped", GOAL_MSG],
        condition=IfCondition(LaunchConfiguration("auto_goal")), output="screen")])

    return LaunchDescription([
        DeclareLaunchArgument("run", default_value="rl",
                              description="'rl' = supervised; anything else = fixed"),
        DeclareLaunchArgument("gui", default_value="true",
                              description="false runs gzserver only (headless)"),
        DeclareLaunchArgument("rviz", default_value="true",
                              description="open rviz with the safety-field view"),
        DeclareLaunchArgument("auto_goal", default_value="true"),
        DeclareLaunchArgument("goal_delay", default_value="8.0"),
        DeclareLaunchArgument("model_path", default_value=DEFAULT_MODEL),
        DeclareLaunchArgument(
            "out_csv",
            default_value=PathJoinSubstitution(
                [REPO, "experiments/results",
                 PythonExpression(["'final_gz_' + '", run, "' + '.csv'"])])),
        gzserver, gzclient, stack, send_goal,
    ])
