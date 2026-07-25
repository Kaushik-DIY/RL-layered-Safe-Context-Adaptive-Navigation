"""The five fixed, seeded evaluation scenario generators (plan 4.2), plus
randomized free-roam for training.

    1 corridor_passby        - anticipatory slowdown
    2 perpendicular_crossing - yield-vs-proceed judgment
    3 doorway_negotiation    - deadlock / freezing
    4 open_hall              - throughput under crowd
    5 blind_corner           - filter under worst-case surprise (occlusion)

Parameters come from experiments/configs/scenarios.yaml. Every randomized quantity
is drawn from np.random.default_rng(seed) alone, so the same (name, seed) builds the
identical episode in the 2D sim and (later) Gazebo.

Conventions:
  * walls            (n,4) segments [x1,y1,x2,y2]: repel SFM pedestrians, define the
                     arena for collision checks / rendering. Long corridor walls are
                     NOT converted to MPC obstacle circles here -- keeping the robot
                     off walls is the env's job (week 3); `static_obstacles` carries
                     only compact critical geometry (doorway posts, corner posts)
                     sized to fit the MPC's 6-slot capacity.
  * visible_humans() is what the robot's TRACKER sees; crowd.state() is ground truth.
                     They differ only in blind_corner, where the occluded pedestrian
                     is withheld until revealed (then stays tracked -- a tracker does
                     not forget). Reveal rule: the pedestrian has entered the main
                     corridor (genuinely in the open) OR is within reveal_distance
                     of the robot (peeking past the corner). The reveal_distance
                     trigger is the plan's worst case ("appears at 1.2 m"); true
                     line-of-sight would reveal earlier, so this is conservative.
  * events           [(trigger, ped_idx, new_goal)] scripted goal switches applied
                     by tick(). trigger is either a TIME (float, seconds -- the
                     TB3-track form, frozen) or a POSITION predicate
                     ("robot_x_ge", x): fire when the robot's x passes x. The
                     position form is the industrial-track emergence: the hazard is
                     presented at a fixed DISTANCE from the robot (ISO-style
                     presentation test), so approach speed alone decides
                     survivability -- a time script tuned to nominal arrival
                     desynchronizes at 1.5 m/s (fast robots outrun it, measured).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.common.params import RobotParams, load_yaml
from core.sim2d.pedestrians import SfmParams, SocialForceCrowd


@dataclass
class ScenarioSpec:
    """One seeded, fully-specified episode: arena + crowd + robot task."""

    name: str
    seed: int
    robot_start: np.ndarray            # [x, y, theta]
    goal: np.ndarray                   # [x, y]
    walls: np.ndarray                  # (n, 4) segments
    static_obstacles: np.ndarray       # (m, 3) circles [x, y, r] for the MPC
    crowd: SocialForceCrowd
    timeout_s: float = 60.0            # plan D6: 60 s episode timeout
    reveal_distance: float | None = None   # blind corner only
    occlusion_y: float | None = None       # blind corner: main-corridor wall line
    events: list = field(default_factory=list)     # [(t_s, ped_idx, goal)]
    _seen: set = field(default_factory=set)

    def tick(self, t_s: float, robot_xy=None) -> None:
        """Apply scripted events (call once per sim step). Time triggers fire at
        t_s; position triggers fire on the robot's x-progress (robot_xy needed)."""
        due = []
        for ev in self.events:
            trig = ev[0]
            if isinstance(trig, tuple):
                if (robot_xy is not None and trig[0] == "robot_x_ge"
                        and float(robot_xy[0]) >= trig[1]):
                    due.append(ev)
            elif trig <= t_s:
                due.append(ev)
        for ev in due:
            self.crowd.goals[ev[1]] = np.asarray(ev[2], dtype=float)
            self.events.remove(ev)

    def visible_humans(self, robot_xy) -> np.ndarray:
        """(k, 4) humans as seen by the robot's tracker (occlusion applied)."""
        state = self.crowd.state()
        if self.reveal_distance is None:
            return state
        rx, ry = float(robot_xy[0]), float(robot_xy[1])
        vis = []
        for i, h in enumerate(state):
            if i not in self._seen:
                in_open = self.occlusion_y is not None and h[1] <= self.occlusion_y
                close = np.hypot(h[0] - rx, h[1] - ry) <= self.reveal_distance
                if in_open or close:
                    self._seen.add(i)
            if i in self._seen:
                vis.append(h)
        return np.asarray(vis, dtype=float).reshape(-1, 4)


# --------------------------------------------------------------------- helpers
def _t_robot_reaches(x: float, robot: RobotParams) -> float:
    """Nominal time for the robot to reach distance x from a standing start:
    latency deadtime + a_max_mpc ramp + v_max cruise. Used only to TIME scripted
    pedestrians against the robot -- coarse is fine (jitter is added on top)."""
    t_ramp = robot.v_max / robot.a_max_mpc
    d_ramp = 0.5 * robot.a_max_mpc * t_ramp ** 2
    if x <= d_ramp:
        return robot.tau_latency + np.sqrt(2.0 * x / robot.a_max_mpc)
    return robot.tau_latency + t_ramp + (x - d_ramp) / robot.v_max


def _rect_walls(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    return np.array([[x0, y0, x1, y0], [x1, y0, x1, y1],
                     [x1, y1, x0, y1], [x0, y1, x0, y0]])


# ------------------------------------------------------------------- builders
# Every builder takes an optional `geom` override dict (industrial_geometry block
# in scenarios.yaml, merged by make_scenario). Defaults ARE the historical TB3
# numbers, so tb3-platform episodes stay bit-for-bit identical.
def corridor_passby(seed: int, cfg: dict, sfm: SfmParams,
                    robot: RobotParams, geom: dict | None = None) -> ScenarioSpec:
    """Corridor, one oncoming pedestrian. Tests anticipatory slowdown."""
    sc = {**cfg["scenarios"]["corridor_passby"], **(geom or {})}
    rng = np.random.default_rng(seed)
    half = sc["corridor_width"] / 2.0
    length = sc.get("length", 6.5)
    goal_x = length - 0.5
    walls = np.array([[-0.5, -half, length, -half], [-0.5, half, length, half]])
    y0 = rng.uniform(-0.35, 0.35)
    x0 = rng.uniform(length - 1.5, length - 0.2)
    speed = rng.uniform(*sfm.desired_speed_range)
    crowd = SocialForceCrowd(sfm, [[x0, y0]], [[-1.5, y0]], [speed],
                             walls=walls, rng=rng)
    return ScenarioSpec("corridor_passby", seed, np.array([0.0, 0.0, 0.0]),
                        np.array([goal_x, 0.0]), walls, np.zeros((0, 3)), crowd)


def perpendicular_crossing(seed: int, cfg: dict, sfm: SfmParams,
                           robot: RobotParams, geom: dict | None = None) -> ScenarioSpec:
    """Open floor; pedestrian crosses the robot's path at randomized timing."""
    sc = {**cfg["scenarios"]["perpendicular_crossing"], **(geom or {})}
    rng = np.random.default_rng(seed)
    jit_lo, jit_hi = sc["crossing_timing_jitter"]
    goal_x = sc.get("goal_x", 6.0)
    x_c = rng.uniform(goal_x / 3.0, 2.0 * goal_x / 3.0)
    side = float(rng.choice([-1.0, 1.0]))
    speed = rng.uniform(*sfm.desired_speed_range)
    # pedestrian reaches the robot's line (y=0) at t_robot(x_c) + jitter offset,
    # centred so half the draws cross ahead of the robot and half behind
    t_meet = _t_robot_reaches(x_c, robot) + rng.uniform(jit_lo, jit_hi) - (jit_hi - jit_lo) / 2.0
    standoff = speed * max(t_meet, 0.5)
    crowd = SocialForceCrowd(sfm, [[x_c, side * standoff]],
                             [[x_c, -side * (standoff + 1.0)]], [speed], rng=rng)
    return ScenarioSpec("perpendicular_crossing", seed, np.array([0.0, 0.0, 0.0]),
                        np.array([goal_x, 0.0]), np.zeros((0, 4)), np.zeros((0, 3)), crowd)


def doorway_negotiation(seed: int, cfg: dict, sfm: SfmParams,
                        robot: RobotParams, geom: dict | None = None) -> ScenarioSpec:
    """Narrow gap; pedestrian arrives simultaneously from the far side."""
    sc = {**cfg["scenarios"]["doorway_negotiation"], **(geom or {})}
    rng = np.random.default_rng(seed)
    gap, door_x = sc["gap_width"], sc.get("door_x", 3.0)
    goal_x = sc.get("goal_x", 6.0)
    walls = np.array([[door_x, gap / 2, door_x, 2.5],
                      [door_x, -2.5, door_x, -gap / 2]])
    posts = np.array([[door_x, gap / 2, 0.12], [door_x, -gap / 2, 0.12]])
    speed = rng.uniform(*sfm.desired_speed_range)
    # start the pedestrian so it hits the doorway when the robot nominally does (+-1.5 s)
    t_arrive = _t_robot_reaches(door_x, robot) + rng.uniform(-1.5, 1.5)
    x0 = door_x + speed * max(t_arrive, 1.0)
    y0 = rng.uniform(-0.2, 0.2)
    crowd = SocialForceCrowd(sfm, [[x0, y0]], [[-1.5, 0.0]], [speed],
                             walls=walls, rng=rng)
    return ScenarioSpec("doorway_negotiation", seed, np.array([0.0, 0.0, 0.0]),
                        np.array([goal_x, 0.0]), walls, posts, crowd)


def open_hall(seed: int, cfg: dict, sfm: SfmParams,
              robot: RobotParams, geom: dict | None = None) -> ScenarioSpec:
    """Hall with free-roaming pedestrians. Tests throughput under crowd."""
    sc = {**cfg["scenarios"]["open_hall"], **(geom or {})}
    rng = np.random.default_rng(seed)
    n_lo, n_hi = sc["n_pedestrians"]
    n = int(rng.integers(n_lo, n_hi + 1))
    x1, y_half = sc.get("x1", 8.0), sc.get("y_half", 3.5)
    goal_x = sc.get("goal_x", 7.0)
    walls = _rect_walls(-1.0, -y_half, x1, y_half)
    start, goal = np.array([0.0, 0.0, 0.0]), np.array([goal_x, 0.0])

    def sample_point(rng_):
        return np.array([rng_.uniform(0.0, x1 - 0.5),
                         rng_.uniform(-(y_half - 0.7), y_half - 0.7)])

    positions: list[np.ndarray] = []
    while len(positions) < n:  # keep spawns off the robot and off each other
        p = sample_point(rng)
        if np.hypot(*(p - start[:2])) < 1.2:
            continue
        if any(np.hypot(*(p - q)) < 0.6 for q in positions):
            continue
        positions.append(p)
    goals = [sample_point(rng) for _ in range(n)]
    speeds = rng.uniform(*sfm.desired_speed_range, size=n)
    crowd = SocialForceCrowd(sfm, positions, goals, speeds, walls=walls, rng=rng,
                             goal_sampler=lambda r, i: sample_point(r))  # free-roam
    return ScenarioSpec("open_hall", seed, start, goal, walls, np.zeros((0, 3)), crowd)


def _corner_arena(sc: dict):
    """Shared T-geometry: main corridor + side passage joining from above."""
    half = sc.get("half_width", 1.0)
    open_x = sc.get("open_x", 3.0)
    pw = sc.get("passage_width", 1.0)
    length = sc.get("length", 6.5)
    p_top = half + 2.2
    walls = np.array([
        [-0.5, -half, length, -half],                 # bottom wall
        [-0.5, half, open_x, half],                   # top wall, left of opening
        [open_x + pw, half, length, half],            # top wall, right of opening
        [open_x, half, open_x, p_top],                # side-passage walls
        [open_x + pw, half, open_x + pw, p_top],
    ])
    posts = np.array([[open_x, half, 0.12], [open_x + pw, half, 0.12]])
    return half, open_x, pw, length, walls, posts


def blind_corner(seed: int, cfg: dict, sfm: SfmParams,
                 robot: RobotParams, geom: dict | None = None) -> ScenarioSpec:
    """Occluded pedestrian steps into the corridor from a side passage; the tracker
    only sees it at reveal_distance (plan: 1.2 m). At TB3 scale the CBF alone saves
    this; at industrial scale reveal < d_stop, so ONLY slowing before the corner
    can be clean -- the supervision headroom demo."""
    sc = {**cfg["scenarios"]["blind_corner"], **(geom or {})}
    rng = np.random.default_rng(seed)
    half, open_x, pw, length, walls, posts = _corner_arena(sc)
    goal_x = sc.get("goal_x", 6.0)
    px = rng.uniform(open_x + 0.25 * pw, open_x + 0.75 * pw)
    py = rng.uniform(half + 0.6, half + 1.6)
    if geom is None:
        # TB3 (frozen): any-speed pedestrian, time trigger against the nominal
        # arrival -- enters the corridor when the robot is nominally 0.8-1.6 m short
        speed = rng.uniform(*sfm.desired_speed_range)
        crowd = SocialForceCrowd(sfm, [[px, py]], [[px, py]], [speed],
                                 walls=walls, rng=rng)
        x_emerge = px - rng.uniform(0.8, 1.6)
        t_start = max(_t_robot_reaches(x_emerge, robot) - (py - half) / speed, 0.0)
        events = [(t_start, 0, np.array([px, -half + 0.4]))]
    else:
        # industrial: a STRIDING worker (1.0-1.5 m/s -- slow crossers are visible
        # through their whole descent in a 3.5 m aisle, no surprise left) with a
        # POSITION trigger (ISO-style presentation): entry onto the corridor line
        # happens when a full-speed robot is d_present short of the crossing point
        # -> reveal at hypot(d_present, half) ~2.1 m < d_stop(cruise) 2.53 m:
        # violation guaranteed at cruise. A slower approach sits farther back at
        # entry (position trigger + descent delay), so approach speed ALONE
        # decides survivability. (A time script tuned to nominal arrival
        # desynchronizes at 1.5 m/s; a slow crosser de-cloaks early -- both
        # measured in P1.4 iterations.)
        speed = rng.uniform(1.0, 1.5)
        crowd = SocialForceCrowd(sfm, [[px, py]], [[px, py]], [speed],
                                 walls=walls, rng=rng)
        # anchor the ped's arrival IN THE ROBOT'S LANE (y~0), not at the corridor
        # line: entry-anchoring degenerates to a harmless side-swipe (measured).
        # A full-speed robot is d_front short of the crossing when the ped is in
        # its lane -- squarely inside the envelope, head-on.
        d_front = rng.uniform(0.8, 1.3)
        delay = py / speed                                # descent time to lane
        trigger_x = px - d_front - robot.v_max * delay
        events = [(("robot_x_ge", trigger_x), 0, np.array([px, -half + 0.4]))]
    return ScenarioSpec("blind_corner", seed, np.array([0.0, 0.0, 0.0]),
                        np.array([goal_x, 0.0]), walls, posts, crowd,
                        reveal_distance=sc["reveal_distance"], occlusion_y=half,
                        events=events)


def interferer(seed: int, cfg: dict, sfm: SfmParams,
               robot: RobotParams, geom: dict | None = None) -> ScenarioSpec:
    """A curious bystander SEEKS the robot while normal pedestrians roam (the
    documented people-crowd-the-robot AMR problem). The context-correct response
    is to KEEP MOVING (the MPC routes around); a density heuristic slows down,
    invites the hover-mob, and bleeds time/protective stops."""
    sc = {**cfg["scenarios"]["interferer"], **(geom or {})}
    rng = np.random.default_rng(seed)
    x1, y_half = sc.get("x1", 8.0), sc.get("y_half", 3.5)
    goal_x = sc.get("goal_x", 7.0)
    walls = _rect_walls(-1.0, -y_half, x1, y_half)
    start = np.array([0.0, 0.0, 0.0])

    def sample_point(rng_):
        return np.array([rng_.uniform(0.0, x1 - 0.5),
                         rng_.uniform(-(y_half - 0.7), y_half - 0.7)])

    n_roam = sc["n_pedestrians"]
    positions = [np.array([rng.uniform(x1 * 0.3, x1 * 0.6),
                           rng.uniform(1.0, y_half - 1.0) * rng.choice([-1, 1])])]
    while len(positions) < n_roam + 1:
        p = sample_point(rng)
        if np.hypot(*(p - start[:2])) < 1.5:
            continue
        positions.append(p)
    goals = [p.copy() for p in positions]           # seeker goal re-aimed per step
    speeds = rng.uniform(*sfm.desired_speed_range, size=n_roam + 1)
    crowd = SocialForceCrowd(sfm, positions, goals, speeds, walls=walls, rng=rng,
                             goal_sampler=lambda r, i: sample_point(r),
                             seekers={0})
    return ScenarioSpec("interferer", seed, start, np.array([goal_x, 0.0]),
                        walls, np.zeros((0, 3)), crowd)


def t_junction_interferer(seed: int, cfg: dict, sfm: SfmParams,
                          robot: RobotParams, geom: dict | None = None) -> ScenarioSpec:
    """HOLDOUT (never trained on): the corner arena WITH an occluded emerging
    pedestrian AND a robot-seeking bystander in the main corridor -- the unseen
    combination that tests whether learned anticipation generalizes."""
    sc = {**cfg["scenarios"]["t_junction_interferer"], **(geom or {})}
    rng = np.random.default_rng(seed)
    half, open_x, pw, length, walls, posts = _corner_arena(sc)
    goal_x = sc.get("goal_x", 6.0)
    # occluded pedestrian in the passage (same emergence logic as blind_corner)
    px = rng.uniform(open_x + 0.25 * pw, open_x + 0.75 * pw)
    py = rng.uniform(half + 0.6, half + 1.6)
    if geom is None:
        speed_o = rng.uniform(*sfm.desired_speed_range)
    else:
        speed_o = rng.uniform(1.0, 1.5)      # striding worker (see blind_corner)
    # seeker in the main corridor, ahead and offset
    sx = rng.uniform(length * 0.55, length * 0.8)
    sy = rng.uniform(-half + 0.5, half - 0.5)
    speeds = [speed_o, rng.uniform(*sfm.desired_speed_range)]
    crowd = SocialForceCrowd(sfm, [[px, py], [sx, sy]],
                             [[px, py], [sx, sy]], speeds,
                             walls=walls, rng=rng, seekers={1})
    if geom is None:
        x_emerge = px - rng.uniform(0.8, 1.6)
        t_start = max(_t_robot_reaches(x_emerge, robot) - (py - half) / speed_o, 0.0)
        events = [(t_start, 0, np.array([px, -half + 0.4]))]
    else:                                    # position trigger (see blind_corner)
        d_front = rng.uniform(0.8, 1.3)
        delay = py / speed_o                 # lane-anchored, not entry-anchored
        trigger_x = px - d_front - robot.v_max * delay
        events = [(("robot_x_ge", trigger_x), 0, np.array([px, -half + 0.4]))]
    return ScenarioSpec("t_junction_interferer", seed, np.array([0.0, 0.0, 0.0]),
                        np.array([goal_x, 0.0]), walls, posts, crowd,
                        reveal_distance=sc["reveal_distance"], occlusion_y=half,
                        events=events)


def free_roam(seed: int, n_pedestrians: int = 4,
              arena: tuple = (-1.0, -3.0, 7.0, 3.0)) -> ScenarioSpec:
    """Randomized free-roam (training data generator, plan D7): like open_hall but
    with caller-chosen crowd size and arena, for curriculum stages B/C."""
    sfm = SfmParams.from_yaml()
    robot = RobotParams.from_yaml()
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = arena
    walls = _rect_walls(x0, y0, x1, y1)
    start = np.array([x0 + 1.0, 0.0, 0.0])
    goal = np.array([x1 - 1.0, 0.0])

    def sample_point(rng_):
        return np.array([rng_.uniform(x0 + 0.5, x1 - 0.5),
                         rng_.uniform(y0 + 0.5, y1 - 0.5)])

    positions: list[np.ndarray] = []
    while len(positions) < n_pedestrians:
        p = sample_point(rng)
        if np.hypot(*(p - start[:2])) < 1.2:
            continue
        positions.append(p)
    goals = [sample_point(rng) for _ in positions]
    speeds = rng.uniform(*sfm.desired_speed_range, size=n_pedestrians)
    crowd = SocialForceCrowd(sfm, positions, goals, speeds, walls=walls, rng=rng,
                             goal_sampler=lambda r, i: sample_point(r))
    return ScenarioSpec("free_roam", seed, start, goal, walls, np.zeros((0, 3)), crowd)


_BUILDERS = {
    "corridor_passby": corridor_passby,
    "perpendicular_crossing": perpendicular_crossing,
    "doorway_negotiation": doorway_negotiation,
    "open_hall": open_hall,
    "blind_corner": blind_corner,
    "interferer": interferer,
    "t_junction_interferer": t_junction_interferer,
}

# The historical five-scenario evaluation battery (TB3 track) -- unchanged, so
# every existing battery/CSV keeps its meaning. The industrial track evaluates
# INDUSTRIAL_SCENARIOS (adds the interferer; t_junction_interferer is HOLDOUT).
SCENARIO_NAMES = ("corridor_passby", "perpendicular_crossing",
                  "doorway_negotiation", "open_hall", "blind_corner")
INDUSTRIAL_SCENARIOS = SCENARIO_NAMES + ("interferer",)
HOLDOUT_SCENARIO = "t_junction_interferer"


def make_scenario(name: str, seed: int, platform: str = "tb3") -> ScenarioSpec:
    """Build the named scenario for a seed (deterministic). platform='industrial'
    swaps in MiR-class arena geometry (scenarios.yaml `industrial_geometry`) and
    times scripted pedestrians against the industrial robot's motion."""
    if name not in _BUILDERS:
        raise ValueError(f"unknown scenario '{name}'; choose from {tuple(_BUILDERS)}")
    cfg = load_yaml("scenarios")
    if platform == "tb3":
        robot, geom = RobotParams.from_yaml(), None
    else:
        from core.common.platform import load_platform
        robot = load_platform(platform).robot
        geom = cfg["industrial_geometry"].get(name, {})
    return _BUILDERS[name](seed, cfg, SfmParams.from_yaml(), robot, geom)
