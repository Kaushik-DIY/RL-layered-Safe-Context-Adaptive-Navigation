# Gazebo sim-to-sim demos (portfolio video)

Record **the showcase** — it is the portfolio headline. The other two are kept as
narrower single-event clips.

| | **Showcase (record this)** | Warehouse | Single blind corner | TB3 |
|---|---|---|---|---|
| Launch | `showcase_demo.launch.py` | `warehouse_demo.launch.py` | `industrial_demo.launch.py` | `gazebo_demo.launch.py` |
| Robot | MiR-class AMR, 1.5 m/s | same | same | TurtleBot3, 0.26 m/s |
| Shows | three hazard types in one ~27 s mission, with telemetry | earlier 3-event attempt | one blind-corner A/B | MPC+CBF transfer |

These need **your** ROS 2 Humble + Gazebo Classic 11 — I can't run Gazebo from the dev
box, so build/run on your machine and we iterate on any breakage.

---

# The showcase demo

A 27 m mission down a 3.5 m industrial aisle past **three hazard types**, chosen because
each one isolates a different reason the supervisor helps:

| # | x | event | what it isolates |
|---|---|---|---|
| A | 6.3 | **blind corner, nobody there** | pure map-geometry anticipation — it slows for a corner with no human present at all |
| B | 14.3 | **worker crosses the 4-way intersection**, visible throughout | the CBF's structural blind spot — its cap divides by the closing cosine, so a walker crossing at ~90° barely registers |
| C | 22.3 | **occluded worker steps out** of a blind aisle and crosses | late reveal: at 1.5 m/s the stopping distance (2.53 m) exceeds what the reveal leaves |

Station B is the subtle one and worth explaining on camera: the safety filter is *not*
what saves the supervised run there. Its speed cap is `v_c_max / (σ·cos φ)`, which goes
to infinity for a perpendicular walker, so the filter barely reacts until he is nearly
dead ahead. The supervisor slows because it **sees** him in the observation (relative
position, relative velocity, time-to-closest-approach). That is a capability the
hand-tuned stack structurally does not have.

## Build

```bash
cd ~/Context_adaptive_navigation/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select navrl_nodes
source install/setup.bash
export PYTHONPATH=$HOME/Context_adaptive_navigation:$PYTHONPATH
```

`.venv-navrl` must **not** be active — the ROS nodes run under system `python3`.

## Record the two runs

```bash
# 1) the RL-supervised run
ros2 launch navrl_nodes showcase_demo.launch.py run:=rl

# 2) the fixed-parameter baseline — same world, same workers, same goal
ros2 launch navrl_nodes showcase_demo.launch.py run:=baseline
```

Nothing to type mid-run. The goal is published automatically 8 s after launch, and the
workers are triggered by the **robot's position**, so both runs are repeatable and see
each hazard presented at the same distance. Telemetry is written to
`experiments/results/showcase_{rl,baseline}.csv`.

`run:=baseline` simply does not launch `rl_supervisor_node`, so `mpc_node` falls back to
the platform max. There is no separate baseline code path.

## Expected (verified offline through the real MPC + CBF + ONNX policy)

| | mission | cap at A / B / C | violation steps | min barrier h | closest worker |
|---|---|---|---|---|---|
| RL-supervised | **27.4 s** | 0.76 / 0.56 / 0.49 m/s | **0** | **+0.57 / +0.68** | 1.79 / 1.82 m |
| fixed-parameter | 20.6 s | 1.50 / 1.50 / 1.50 m/s | **2 at B, 1 at C** | **−0.01 / −0.03** | 1.92 / 1.40 m |

Read the middle column first: the supervised cap steps *down at every station and back up
between them*, while the baseline is a flat 1.50 m/s line. That is the adaptivity, visible
as a single trace.

If the Gazebo run disagrees with this by more than ~15 %, something in the **plant** is
usually the cause before the cues are. The first recorded run came in at 45 s / 35 s with
24 baseline violations because `max_wheel_acceleration` was 3.0 rad/s² = 0.30 m/s² linear,
against a CBF that plans stops at 0.8 m/s². Check that first, then the cue distances. Do
not skip this comparison; every earlier attempt failed precisely because it was never made.

## Offline video — presentable without Gazebo

If Gazebo is unavailable or its capture doesn't read well, this renders the whole result
from the verified offline simulation. Same `core/` stack, same scene, same ONNX policy —
just drawn top-down instead of in 3D:

```bash
PYTHONPATH=$PWD .venv-navrl/bin/python scripts/render_showcase_demo.py           # mp4
PYTHONPATH=$PWD .venv-navrl/bin/python scripts/render_showcase_demo.py --gif     # + gif
PYTHONPATH=$PWD .venv-navrl/bin/python scripts/render_showcase_demo.py --still 15.6
```

Output: `experiments/results/showcase_demo.mp4` — 1920×1080, 20 fps, 27.4 s, both runs
stacked (supervised above, baseline below), playing on a common clock.

What carries the argument is the shaded **ISO stopping-distance envelope** drawn around
each robot: `d_stop(σ·v) + d_hard`, the room that robot needs to stop from the speed it is
doing right now. 2.83 m at 1.5 m/s, 0.76 m at 0.5 m/s. It turns **red** the moment a
worker is inside it, a banner appears, and a red **✗** is left on the path. Breaches last
only ~0.15 s in real time, so the red state is held ~0.9 s and the marks persist —
otherwise a genuine safety failure is literally invisible at 20 fps.

Result shown: **supervised 27.4 s, 0 s in violation; baseline 20.6 s, breaching at both
the intersection and the blind corner.**

Label it honestly as the **2D simulation** — it is the environment the policy was trained
and evaluated in, running the identical control stack, not a 3D render.

## Build the telemetry panel

```bash
cd ~/Context_adaptive_navigation
PYTHONPATH=$PWD .venv-navrl/bin/python scripts/render_showcase_video.py            # mp4
PYTHONPATH=$PWD .venv-navrl/bin/python scripts/render_showcase_video.py --static   # png
```

It plots the commanded speed cap (the supervisor's actual action) against achieved speed,
the ISO barrier `h` with violations shaded red, and clearance to the nearest worker, with
the three stations marked and the 12-seed statistics printed alongside. Compose it under
the two Gazebo captures.

## Why the workers are position-triggered

The supervised run averages ~0.9 m/s and the baseline ~1.3 m/s, so over 27 m their
arrival times at the three stations diverge by up to ~13 s. **No time script, and no
single goal-delay offset, can present the same hazard to both** — a time-scripted worker
either blocks the slow run or is out-run by the fast one. Both were observed on video.

`scene_director_node` therefore fires each cue on the robot's position, reproducing the
`("robot_x_ge", x)` trigger the policy was trained and evaluated under. It also makes the
comparison fair, and that is a line worth putting in the video: *the hazard is presented
at an identical distance in both runs, so only the approach speed differs.*

Consequence: the workers are visual-only kinematic models, not Gazebo `<actor>`s, so they
have no skeletal walk cycle. The director adds a vertical bob and yaw sway to suggest one.

## Reality check (do this once)

```bash
PYTHONPATH=$PWD .venv-navrl/bin/python scripts/verify_showcase.py
```

This replays the exact scene offline through the real stack and gates on the contrast.
Run it before Gazebo; then compare the recorded CSVs against it (arrival time per station,
commanded cap at each corner, `min_h`).

## Tuning knobs

| symptom | knob |
|---|---|
| **both runs slow, long, and similar-looking** | the plant, not the policy: `max_wheel_acceleration` in the world must be >= `a_brake / wheel_radius` = 8.0 (it is 12.0). At 3.0 the robot braked at 0.30 m/s² while the CBF planned stops assuming 0.8, so both runs were plant-limited and the comparison flattened |
| a worker crosses too early/late | `present_distance` in `core/demo/showcase_scene.py` (smaller = tighter presentation) |
| baseline no longer breaches | reduce that station's `present_distance`, or raise the worker's `speed` (warning time is `HALF_W / speed`) |
| supervised run also breaches | raise `present_distance` — it is too tight for *any* approach speed |
| want a different route | edit `X_A/X_B/X_C`, `GOAL`; regenerate with `scripts/gen_showcase_world.py` |

Geometry lives in **one** place (`core/demo/showcase_scene.py`) and is imported by the
world generator, the verifier, the director node and the launch file, so it cannot drift.
Regenerate the world after any change:

```bash
PYTHONPATH=$PWD .venv-navrl/bin/python scripts/gen_showcase_world.py
```

## Say this in the video description

- Both runs use the **same** MPC + CBF; only the supervisor's `(v_max, d_margin)`
  modulation differs. This is not "RL vs no controller".
- Headline statistic, 12 seeds (`experiments/results/s4_industrial_full.csv`): ISO
  stopping-distance violations **8/12 → 0/12** at blind corners and **6/12 → 0/12** at
  crossings. The clip is one instance of that, not the evidence for it.
- The supervised run is **slower** (~27 s vs ~22 s). That is the price of compliance and
  should be shown, not hidden.
- Sim-to-sim: a MiR-class AMR simulated at 1.5 m/s, not hardware.
- The supervised policy does **not** stop less often than the baseline — it stops
  *earlier and further away*. Its win is eliminating envelope breaches, not stop-and-go.

---

# Demo B — TB3 pedestrian yield (secondary safety clip)

Fixed-parameter stack at TB3 scale; shows the same `core/` code transferring and the CBF
yielding to a crossing pedestrian. (No RL: at 0.26 m/s the policy ≈ always-max, proven —
so this clip is about *transfer + safety*, not adaptivity.)

```bash
cd ~/Context_adaptive_navigation/ros2_ws
colcon build --packages-select navrl_nodes && source install/setup.bash
export TURTLEBOT3_MODEL=waffle

# Stage 0 — sanity-check the baseline first (empty world)
ros2 launch turtlebot3_gazebo empty_world.launch.py            # terminal A
ros2 launch navrl_nodes nav_stack.launch.py                    # terminal B
ros2 topic pub -1 /goal_pose geometry_msgs/PoseStamped \
    '{header: {frame_id: "odom"}, pose: {position: {x: 2.0, y: 0.0}}}'

# Stage 1 — pedestrian world (one command)
ros2 launch navrl_nodes gazebo_demo.launch.py
ros2 topic pub -1 /goal_pose geometry_msgs/PoseStamped \
    '{header: {frame_id: "odom"}, pose: {position: {x: 4.0, y: 0.0}}}'
```
`gazebo_demo.launch.py` composes the stock turtlebot3_gazebo launches; if your TB3
package names `robot_state_publisher.launch.py` / `spawn_turtlebot3.launch.py`
differently, adjust the two flagged paths in that file.

---

## Troubleshooting (both demos)

| Symptom | Cause / fix |
|---|---|
| no `/odom`, AMR doesn't move (Demo A) | diff-drive plugin didn't load — check `ros2 topic list \| grep odom` and `gz` console for `libgazebo_ros_diff_drive`; needs `ros-humble-gazebo-ros-pkgs` |
| `/navrl/humans` always empty | state plugin missing (`ros2 topic echo /gazebo/model_states`) or actor name doesn't start with `actor` |
| worker never appears in `/navrl/humans` even when crossing | occlusion latch: it reveals only at `y ≤ occlusion_y` (1.75) or within `reveal_distance` (1.2 m) — correct; if it *never* appears, the actor isn't reaching the corridor (check `walk.dae` loaded + `delay_start`) |
| worker has `vx=vy=0` | actor twist is zero by design — the tracker finite-differences pose; if still zero the actor isn't moving |
| `rl_supervisor` exits immediately | `model_path` wrong — pass `model_path:=/abs/path.onnx`; export step (A.0) must have run |
| AMR reaches goal but never slowed | RL run: confirm `/navrl/params` is publishing and `v_max_cmd` drops; if flat, the supervisor didn't load the world geometry (walls/posts params) |
| robot stops and never resumes | `protective_stop` latched on a too-close worker; resumes when it clears `d_hard` — correct fail-safe |
| rviz empty | Fixed Frame must be `odom`; the stack must be running |
| actor `walk.dae` not found | `ls /usr/share/gazebo-11/media/models/walk.dae`; install `gazebo11` media if missing |

## Honest framing (put in the video description)

The industrial AMR is a **simulation** of a MiR-class platform (1.5 m/s) — a faithful
sim-to-sim transfer of the trained system, not physical hardware. The rigorous adaptivity
evidence is the 2D battery (many seeds + statistics); this video *illustrates* it running
as a live ROS 2 system. TB3 hardware cannot do 1.5 m/s, which is exactly why Demo B (real
TB3 scale) shows transfer + safety while Demo A (AMR sim) shows the learned adaptivity.
