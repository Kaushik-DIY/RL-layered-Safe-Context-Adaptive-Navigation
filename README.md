# Safe Context-Adaptive Navigation for Industrial AMRs

An autonomous warehouse robot normally learns where to be careful from a **person**. An
integrator walks the site, marks the blind corners and junctions on the robot's map, sizes
its safety scanner fields, and enters a reduced speed for each zone. That work has to be
redone for every site, and again whenever the layout changes.

This project replaces that step with a learned supervisor. A reinforcement-learning policy
sits on top of a conventional MPC + control-barrier-function stack and decides, from the
map the robot already has, **how fast to go and how much room to leave** — with nothing
marked, surveyed or configured for the site.

Measured against the same machine as actually commissioned, on the same 31 m shared-aisle
route:

| | Commissioned AMR | This work |
|---|---|---|
| site parameters configured by hand | **13** | **0** |
| mission time | 32.9 s | 32.6 s |
| ISO stopping-distance violations | 0 | 0 |
| contacts / protective stops | 0 / 0 | 0 / 0 |

Same compliance, same transport time, no commissioning. The interesting part is *how* it
gets there — see the demo.

---

## The demo

**▶ [Download the demo video](https://github.com/Kaushik-DIY/RL-layered-Safe-Context-Adaptive-Navigation/raw/main/experiments/results/final_demo.mp4)** — 33 s, 1920×1080
(`experiments/results/final_demo.mp4`)

Two machines run the same route with the same people in it: a hand-commissioned industrial
AMR on top, this work below. The route puts **the same pedestrian encounter twice**, and
only the geometry differs:

- **A blind cross-aisle on the escape side.** Getting round the picker would mean swinging
  across the mouth of an opening nobody can see into — trading a person it can see for one
  it cannot. It refuses the width and slows instead.
- **Plain aisle, solid racking both sides.** Nothing can emerge, so the room is real: it
  offsets 1.12 m and carries its speed straight through.

The commissioned machine slows at both, because slowing is all a warning field can do.

**Honest limits.** Measured on this route only. The supervisor needs one *machine* constant
(how far its sensors see past a mapped occluder) plus the map — the commissioned machine
needs that same quantity surveyed per junction, then a zone speed, extent and polygon
derived and validated from it. And there is one encounter type, a crowded picking zone,
where this approach is beaten outright; it is documented in `docs/` rather than omitted.

---

## Running the Gazebo simulation

The 3D build runs the **identical** control stack — same policy, same MPC, same CBF — on a
physically simulated differential-drive robot. It reproduces the 2D result to within 2 %.

### Requirements

ROS 2 Humble, Gazebo Classic, and Python 3.10 with `numpy`, `onnxruntime`, `casadi`,
`proxsuite`, `pyyaml` available to the **system** interpreter (the ROS nodes run under it,
not under a virtualenv).

### 1. Build, once

```bash
git clone https://github.com/Kaushik-DIY/RL-layered-Safe-Context-Adaptive-Navigation.git
cd RL-layered-Safe-Context-Adaptive-Navigation

source /opt/ros/humble/setup.bash
PYTHONPATH=$PWD python3 scripts/gen_final_world.py     # the world is GENERATED, not stored
cd ros2_ws && colcon build --packages-select navrl_nodes --symlink-install && cd ..
```

### 2. Every terminal that runs a node

All three lines. Without `PYTHONPATH` the nodes cannot `import core.*` and the launch file
itself will not load.

```bash
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
export PYTHONPATH=$PWD:$PYTHONPATH
```

### 3. Run it

```bash
ros2 launch navrl_nodes final_demo.launch.py
```

That opens Gazebo and rviz together and drives the route. The goal is published
automatically 8 seconds after launch.

| variant | effect |
|---|---|
| `rviz:=false` | Gazebo only |
| `gui:=false` | rviz only, no Gazebo window |
| `gui:=false rviz:=false` | fully headless |
| `run:=fixed` | no supervisor, for contrast |
| `goal_delay:=15` | more time to set up a screen recording |

### 4. Check the run reproduced the 2D result

```bash
PYTHONPATH=$PWD python3 scripts/check_final_gazebo.py
```

Prints mission time, worst safety margin, protective stops and the per-encounter response
against the 2D reference, and exits non-zero if any of them drifted.

> **Watch rviz, not just Gazebo.** Speed differences read badly in 3D — 1.2 against
> 0.6 m/s looks much the same on camera — but the protective field does not, because it
> scales with v². The rviz view rides with the robot and colours the field by the safety
> margin; watch it swell and shrink as the machine changes speed.

---

## Reproducing the measurements

Everything on screen comes from a gate that must pass before anything is rendered. These
use the project virtualenv (`.venv-navrl`), not the system interpreter:

```bash
PYTHONPATH=$PWD .venv-navrl/bin/python scripts/verify_final.py 6        # the 2D gate
PYTHONPATH=$PWD .venv-navrl/bin/python scripts/render_final_video.py    # the video
PYTHONPATH=$PWD .venv-navrl/bin/python scripts/plot_final_comparison.py # the figure
```

Outputs land in `experiments/results/` and are **not** tracked — only the finished demo
video is. Everything else regenerates from the scripts above.

---

## Layout

| path | what is in it |
|---|---|
| `core/mpc`, `core/cbf` | the conventional stack: NMPC, and the safety filter that enforces the ISO stopping-distance condition |
| `core/rl` | the learned supervisor, and its deployment-form ONNX runner |
| `core/demo` | the route, the scanner model, the site-zone derivation, the map-derived guards |
| `ros2_ws/src/navrl_nodes` | the same stack as ROS 2 nodes, plus the generated Gazebo world |
| `scripts/verify_*.py` | the gates — measurement, not rendering |
| `scripts/render_*.py`, `plot_*.py` | the video and the figures |
| `docs/` | how each demo was designed, what was measured, and what did not survive |

## Design notes worth reading first

- **`docs/final_demo.md`** — the route, the metrics, and why the minimum speed is a
  misleading statistic.
- **`docs/final_gazebo.md`** — the 3D build, and four bugs it surfaced that are easy to
  reintroduce.
- **`docs/architecture.md`** — the three layers and why the safety filter is not tunable
  by the policy.
