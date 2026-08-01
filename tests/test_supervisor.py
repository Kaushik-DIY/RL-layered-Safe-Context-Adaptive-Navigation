"""Regression test for the deployment-form supervisor (core.rl.supervisor).

Guards the transfer claim: the ROS-free SupervisorPolicy (ONNX, torch-free) must
reproduce the 2D env's anticipatory blind-corner slowdown. Skips cleanly when the
exported ONNX or onnxruntime is absent (both are optional, off the repo), so the
core test suite stays green without them.
"""
from pathlib import Path

import numpy as np
import pytest

ONNX = Path(__file__).resolve().parents[1] / "experiments" / "models" / "ppo_ind_C_s0_full_final.onnx"


@pytest.mark.skipif(not ONNX.exists(), reason="industrial ONNX not exported")
def test_supervisor_anticipates_blind_corner():
    pytest.importorskip("onnxruntime")
    from core.rl.supervisor import SupervisorPolicy
    from core.sim2d.scenarios import make_scenario

    spec = make_scenario("blind_corner", seed=21, platform="industrial")
    corner_x = float(np.min(spec.static_obstacles[:, 0]))
    y0 = float(spec.robot_start[1])

    pol = SupervisorPolicy(str(ONNX), platform="industrial",
                           walls=spec.walls, posts=spec.static_obstacles)
    pol.reset()

    cmds = []
    for x in np.arange(corner_x - 6.0, corner_x - 0.05, 0.4):
        state = np.array([x, y0, 0.0, pol.v_max_cmd, 0.0])
        v_cmd, _ = pol.compute(state, spec.goal, humans=np.zeros((0, 4)))
        cmds.append(v_cmd)

    # cruises far out, slows into the corner WITHOUT the (still-occluded) pedestrian,
    # and the descent is (near-)monotonic -- the learned anticipation.
    assert cmds[0] > 0.8, f"should start near cruise, got {cmds[0]:.2f}"
    assert cmds[-1] < 0.5, f"should be slow at the corner, got {cmds[-1]:.2f}"
    non_increasing = sum(cmds[i + 1] <= cmds[i] + 1e-3 for i in range(len(cmds) - 1))
    assert non_increasing >= len(cmds) - 3


@pytest.mark.skipif(not ONNX.exists(), reason="industrial ONNX not exported")
def test_supervisor_output_within_action_box():
    pytest.importorskip("onnxruntime")
    from core.common.platform import load_platform
    from core.rl.supervisor import SupervisorPolicy

    plat = load_platform("industrial")
    pol = SupervisorPolicy(str(ONNX), platform="industrial")
    rng = np.random.default_rng(0)
    for _ in range(20):
        state = np.array([rng.uniform(-5, 5), rng.uniform(-3, 3),
                          rng.uniform(-np.pi, np.pi), rng.uniform(0, 1.5), 0.0])
        goal = np.array([rng.uniform(5, 12), rng.uniform(-2, 2)])
        v_cmd, m_cmd = pol.compute(state, goal, humans=np.zeros((0, 4)))
        assert plat.rl.v_max_low - 1e-4 <= v_cmd <= plat.rl.v_max_high + 1e-4
        assert plat.rl.d_margin_low - 1e-4 <= m_cmd <= plat.rl.d_margin_high + 1e-4
