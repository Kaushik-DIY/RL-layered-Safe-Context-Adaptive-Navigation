"""Export the trained SB3 supervisor policy to ONNX (plan D8, week 5).

Evaluation and the ROS `rl_supervisor` node then run WITHOUT a torch dependency:
one 32-float observation in, one clipped [v_max_cmd, d_margin_cmd] out, via
onnxruntime. The exported graph is the DETERMINISTIC policy (Gaussian mean) with
the action-space clip baked in, so consumers need no SB3/gym knowledge either.

    python scripts/export_onnx.py experiments/models/ppo_B_s0_final.zip
    # -> experiments/models/ppo_B_s0_final.onnx  (+ parity check vs torch)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from core.common.observation import obs_dim
from core.common.params import RlParams

# reuse the cross-version-safe loader (Kaggle NumPy 2 -> local NumPy 1)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_policy import load_model  # noqa: E402


class DeterministicPolicy(torch.nn.Module):
    """obs (1, 32) float32 -> clipped action (1, 2): the 2 Hz supervisor step."""

    def __init__(self, policy, low, high):
        super().__init__()
        self.mlp = policy.mlp_extractor
        self.features = policy.features_extractor if hasattr(policy, "features_extractor") else None
        self.action_net = policy.action_net
        self.low = torch.tensor(low, dtype=torch.float32)
        self.high = torch.tensor(high, dtype=torch.float32)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        latent_pi = self.mlp.forward_actor(obs)
        mean = self.action_net(latent_pi)         # deterministic = Gaussian mean
        return torch.clamp(mean, self.low, self.high)


def main() -> None:
    model_path = Path(sys.argv[1])
    out_path = model_path.with_suffix(".onnx")
    rl = RlParams.from_yaml()
    low = np.array([rl.v_max_low, rl.d_margin_low], dtype=np.float32)
    high = np.array([rl.v_max_high, rl.d_margin_high], dtype=np.float32)

    model = load_model(str(model_path))
    wrapper = DeterministicPolicy(model.policy, low, high).eval()

    dummy = torch.zeros((1, obs_dim(rl.K_nearest)), dtype=torch.float32)
    torch.onnx.export(wrapper, (dummy,), str(out_path), input_names=["obs"],
                      output_names=["params"], opset_version=17,
                      dynamic_axes={"obs": {0: "batch"}, "params": {0: "batch"}},
                      dynamo=False)

    # ---- parity check: onnxruntime vs SB3 predict on random observations ----
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(256):
        obs = rng.normal(0, 1, size=(1, obs_dim(rl.K_nearest))).astype(np.float32)
        a_onnx = sess.run(None, {"obs": obs})[0][0]
        a_sb3, _ = model.predict(obs[0], deterministic=True)
        a_sb3 = np.clip(a_sb3, low, high)
        worst = max(worst, float(np.max(np.abs(a_onnx - a_sb3))))
    assert worst < 1e-5, f"ONNX/torch mismatch: {worst}"
    print(f"exported -> {out_path}")
    print(f"parity   : max |onnx - sb3| = {worst:.2e} over 256 random obs  (OK)")


if __name__ == "__main__":
    main()
