"""INDUSTRIAL SHOWCASE -- the portfolio demo.

One continuous 27 m mission through three hazard stations:

    x ~  6.3   blind corner, NOBODY there   -> the supervisor slows on map geometry alone
    x ~ 14.3   worker crosses the 4-way intersection, visible the whole time
    x ~ 22.3   occluded worker steps out of a blind aisle and crosses

Record it twice; only one flag changes:

    ros2 launch navrl_nodes showcase_demo.launch.py run:=rl
    ros2 launch navrl_nodes showcase_demo.launch.py run:=always_max

Anything other than `run:=rl` simply does not launch the supervisor, so mpc_node falls
back to the platform's max speed. Same world, same workers, same goal, same code path.
The telemetry filename follows the flag, so `run:=always_max` writes
`experiments/results/showcase_always_max.csv`.

The goal is published automatically `goal_delay` seconds after launch, and the workers
are triggered by the ROBOT'S POSITION (scene_director_node), so both runs are repeatable
and see the hazard presented at the same distance. Telemetry lands in
`experiments/results/showcase_<run>.csv` for scripts/check_gazebo_run.py.

Requires PYTHONPATH to include the repo root (the nodes import `core.*`):
    export PYTHONPATH=$HOME/Context_adaptive_navigation:$PYTHONPATH
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            IncludeLaunchDescription, TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node

from core.demo.showcase_scene import GOAL, OCCLUSION_Y, POSTS, REVEAL_DISTANCE, WALLS

REPO = os.path.expanduser("~/Context_adaptive_navigation")
DEFAULT_MODEL = os.path.join(
    REPO, "experiments/models/ppo_ind_C_s0_full_final.onnx")

GOAL_MSG = ("{header: {frame_id: 'odom'}, pose: {position: "
            "{x: %.1f, y: %.1f}}}" % (GOAL[0], GOAL[1]))


def generate_launch_description() -> LaunchDescription:
    gazebo_ros = get_package_share_directory("gazebo_ros")
    navrl = get_package_share_directory("navrl_nodes")
    world = os.path.join(navrl, "worlds", "industrial_showcase.world")

    run = LaunchConfiguration("run")
    model_path = LaunchConfiguration("model_path")
    goal_delay = LaunchConfiguration("goal_delay")
    out_csv = LaunchConfiguration("out_csv")
    # run:=rl launches the supervisor; anything else (baseline) does not
    is_rl = IfCondition(PythonExpression(["'", run, "' == 'rl'"]))

    industrial = {"platform": "industrial", "use_sim_time": True}
    geom = {"walls": WALLS.ravel().tolist(), "posts": POSTS.ravel().tolist()}

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
                                "occlusion_y": OCCLUSION_Y,
                                "name_prefixes": ["worker", "actor", "human",
                                                  "person", "pedestrian"]}])
    # the MPC needs the corridor geometry too (NavEnv._mpc_obstacles equivalent);
    # without it the controller is blind to the walls and drives into them
    mpc = Node(package="navrl_nodes", executable="mpc_node",
               name="mpc_controller", output="screen",
               parameters=[industrial, geom])
    cbf = Node(package="navrl_nodes", executable="cbf_node",
               name="cbf_filter", output="screen", parameters=[industrial])
    # field_viz draws the robot AND the walls: rviz has no RobotModel display and no
    # robot_description (the AMR lives in the world SDF), so without this the viewport
    # shows a safety disc floating in empty space.
    viz = Node(package="navrl_nodes", executable="field_viz_node",
               name="field_viz", output="screen",
               parameters=[industrial, {"walls": WALLS.ravel().tolist()}])
    director = Node(package="navrl_nodes", executable="scene_director_node",
                    name="scene_director", output="screen",
                    parameters=[{"use_sim_time": True}])
    recorder = Node(package="navrl_nodes", executable="demo_recorder_node",
                    name="demo_recorder", output="screen",
                    parameters=[{"use_sim_time": True, "out_path": out_csv}])
    supervisor = Node(package="navrl_nodes", executable="rl_supervisor_node",
                      name="rl_supervisor", output="screen", condition=is_rl,
                      parameters=[{"use_sim_time": True, "platform": "industrial",
                                   "model_path": model_path,
                                   "walls": WALLS.ravel().tolist(),
                                   "posts": POSTS.ravel().tolist()}])

    stack = TimerAction(period=4.0,
                        actions=[tracker, mpc, cbf, viz, director, recorder, supervisor])
    send_goal = TimerAction(period=goal_delay, actions=[ExecuteProcess(
        cmd=["ros2", "topic", "pub", "-1", "/goal_pose",
             "geometry_msgs/PoseStamped", GOAL_MSG],
        condition=IfCondition(LaunchConfiguration("auto_goal")), output="screen")])

    return LaunchDescription([
        DeclareLaunchArgument("run", default_value="rl",
                              description="'rl' (supervisor on); anything else runs fixed-parameter"),
        DeclareLaunchArgument("auto_goal", default_value="true"),
        DeclareLaunchArgument("goal_delay", default_value="8.0",
                              description="seconds after launch to publish the goal"),
        DeclareLaunchArgument("model_path", default_value=DEFAULT_MODEL),
        DeclareLaunchArgument(
            "out_csv",
            default_value=PathJoinSubstitution(
                [REPO, "experiments/results",
                 PythonExpression(["'showcase_' + '", run, "' + '.csv'"])]),
            description="telemetry CSV path"),
        gzserver, gzclient, stack, send_goal,
    ])
