# The final demo in Gazebo

3D proof that the **same** control stack that produced the 2D video drives a physically
simulated differential-drive AMR through the same route.

## Run it

**One-time, per machine** (or after any change to the nodes, the world or `core/`):

```bash
cd ~/Context_adaptive_navigation/.claude/worktrees/commissioning-video
source /opt/ros/humble/setup.bash
PYTHONPATH=$PWD .venv-navrl/bin/python scripts/gen_final_world.py   # world is GENERATED
cd ros2_ws && colcon build --packages-select navrl_nodes --symlink-install && cd ..
```

**Every terminal** that runs a node needs all three of these, or the nodes cannot
`import core.*` and the launch file itself will not load:

```bash
cd ~/Context_adaptive_navigation/.claude/worktrees/commissioning-video
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
export PYTHONPATH=$PWD:$PYTHONPATH
```

**Then, a single command** — Gazebo and rviz together:

```bash
ros2 launch navrl_nodes final_demo.launch.py
```

Variants:

| command | what it does |
|---|---|
| `... final_demo.launch.py` | Gazebo GUI + rviz, supervised. The demo. |
| `... final_demo.launch.py rviz:=false` | Gazebo only |
| `... final_demo.launch.py gui:=false` | rviz only, no Gazebo window |
| `... final_demo.launch.py gui:=false rviz:=false` | headless, for the check below |
| `... final_demo.launch.py run:=fixed` | no supervisor, for contrast |
| `... final_demo.launch.py auto_goal:=false` | hold at the start; publish the goal by hand |

The goal is published automatically 8 s after launch (`goal_delay:=N` to change it). Then:

```bash
PYTHONPATH=$PWD .venv-navrl/bin/python scripts/check_final_gazebo.py
```

### What each window shows

**Gazebo** is the physical truth: the AMR, the racking, the aisle, the three workers. The
camera tracks the AMR over the whole 31 m.

**rviz is where the safety layer becomes legible**, and it is the more useful of the two.
Speed differences read badly in 3D — 1.2 against 0.6 m/s looks much the same on camera —
but the protective field does not, because it scales with v². The `demo.rviz` view rides
with `base_footprint`, draws the walls and the robot from `field_viz_node` (rviz has no
`RobotModel` here: the AMR lives in the world SDF, not a URDF), and colours the field ring
by the barrier `h`. Watch the ring grow and shrink as the machine changes speed.

## Nothing here is a re-implementation

The ROS nodes import `core.*` exactly as the 2D harness does — same ONNX policy, same
`MpcController`, same `CbfFilter`, same `sight_limit` guards. The world is generated from
the same `core.demo.final_route` the 2D gate builds its scene from, so the geometry Gazebo
renders and the geometry the policy is told about cannot drift apart. That is the whole
value of the 3D run: it is evidence about the *implementation*, not a second model of it.

`rl_supervisor_node` is a faithful port of the 2D loop, **and the rates matter**: the
policy is queried at 2 Hz but the sight floor, the lateral rule and the reachable-cap
clamp run at the full 10 Hz control rate. Running the guards at 2 Hz would let the cap
fall 0.30 m/s between clamps — five times the deceleration the machine can deliver.

## Measured (headless run, checked against the 2D gate)

| | 2D | Gazebo |
|---|---|---|
| mission time | 32.5 s | **31.8 s** (−2 %) |
| worst barrier margin | +0.41 m | **+0.44 m** |
| protective stops | 0 | **0** |
| closest approach | — | 0.77 m |
| peak decel / accel | — | −1.12 / +0.87 m/s² (limit 1.20) |

| station | 2D v@pass / offset | Gazebo v@pass / offset |
|---|---|---|
| A blind cross-aisle on the escape side | 0.58 / 0.02 | **0.52 / 0.01** |
| B 4-way junction, occluded worker | 0.80 / 0.01 | 0.90 / 0.01 |
| C plain aisle, solid racking | 1.20 / 1.12 | **1.19 / 1.13** |

**The behaviour transfers.** It refuses the width beside the blind opening and slows; it
steps aside 1.13 m in the plain aisle and keeps its speed. All 10 checks pass.
