"""LAYER 3, deployment form: the trained context supervisor as a ROS-free,
torch-free policy runner (plan D8, week 5).

The 2 Hz supervisor step in `NavEnv` is: build the normalized observation, run the
policy, clip to the action box, feed the chosen (v_max_cmd, d_margin_cmd) back into
the next observation. This class reproduces that EXACTLY -- same `build_observation`
call, same `spec_.goal` target, same previous-action feedback initialised to the
conservative floor -- but consumes the exported ONNX via onnxruntime, so the ROS
`rl_supervisor_node` needs neither torch nor stable-baselines3. Keeping it ROS-free
lets it be unit-tested against the 2D env's behaviour without a ROS install (mirrors
the core/ vs ros2_ws/ split used by the MPC and CBF layers).

Observation parity is the transfer claim, so this file must stay a thin mirror of
`NavEnv._observe`; if that changes, change this too.
"""
from __future__ import annotations

import numpy as np

from core.common.observation import build_observation
from core.common.platform import load_platform


class SupervisorPolicy:
    """obs -> (v_max_cmd, d_margin_cmd), matching NavEnv's 2 Hz supervisor step.

    Parameters
    ----------
    onnx_path : path to the policy exported by scripts/export_onnx.py (the clip to
                the action box is baked into the graph).
    platform  : parameter stack; 'industrial' = 35-dim obs v2 + MiR action box.
    walls, posts : the world's static geometry for the obs-v2 occlusion features
                (walls (n,4) segments, posts (m,3) circles). Fixed for a run.
    """

    def __init__(self, onnx_path: str, platform: str = "industrial",
                 walls=None, posts=None):
        import onnxruntime as ort   # local import: keeps torch/onnx off the import path
        self.plat = load_platform(platform)
        self.rl = self.plat.rl
        self.walls = None if walls is None else np.asarray(walls, float).reshape(-1, 4)
        self.posts = None if posts is None else np.asarray(posts, float).reshape(-1, 3)
        self.sess = ort.InferenceSession(str(onnx_path),
                                         providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self) -> None:
        """Previous-action feedback before the first inference = NavEnv's reset
        floor (slowest speed, widest margin): conservative until proven otherwise."""
        self.v_max_cmd = self.rl.v_max_low
        self.d_margin_cmd = self.rl.d_margin_high

    def compute(self, state, goal_xy, humans) -> tuple[float, float]:
        """One supervisor step.

        state   : [x, y, theta, v, omega] (robot odometry)
        goal_xy : [x, y] mission goal (NOT the carrot -- matches NavEnv._observe)
        humans  : (n, 4) [x, y, vx, vy] as published by the tracker
        """
        humans = (np.zeros((0, 4)) if humans is None
                  else np.asarray(humans, float).reshape(-1, 4))
        obs = build_observation(
            np.asarray(state, float), np.asarray(goal_xy, float), humans,
            self.v_max_cmd, self.d_margin_cmd,
            k_nearest=self.rl.K_nearest, version=self.plat.obs_version,
            walls=self.walls, posts=self.posts, scale=self.plat.obs_scale)
        out = self.sess.run(None, {"obs": obs[None].astype(np.float32)})[0][0]
        self.v_max_cmd, self.d_margin_cmd = float(out[0]), float(out[1])
        return self.v_max_cmd, self.d_margin_cmd
